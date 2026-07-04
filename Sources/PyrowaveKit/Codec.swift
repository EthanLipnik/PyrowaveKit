import Foundation

public extension EncodedFrame {
    func packetized(maximumPacketBytes: Int) throws -> [EncodedPacket] {
        guard maximumPacketBytes >= 8 else {
            throw PyrowaveError.invalidDimensions
        }

        var reader = BinaryReader(data)
        _ = try PyrowaveSequenceHeader(reader: &reader)

        var packets = [EncodedPacket]()
        var packetStart = 0
        var packetSize = reader.offset

        while reader.offset < data.count {
            let blockStart = reader.offset
            _ = try PyrowaveCoefficientBlockCodec.decodeBlock(reader: &reader)
            let blockSize = reader.offset - blockStart

            if packetSize + blockSize > maximumPacketBytes, packetSize > 0 {
                packets.append(EncodedPacket(data: Data(data[packetStart..<(packetStart + packetSize)])))
                packetStart = blockStart
                packetSize = 0
            }

            packetSize += blockSize
        }

        if packetSize > 0 {
            packets.append(EncodedPacket(data: Data(data[packetStart..<(packetStart + packetSize)])))
        }

        return packets
    }
}

public final class PyrowavePacketStreamDecoder {
    private let codec: PyrowaveCodec
    private let expectedFrame: ExpectedFrame?
    private var sequenceHeader: PyrowaveSequenceHeader?
    private var blockPackets = [Int: Data]()
    private var decodedFrameForCurrentSequence = false
    private var lastSequence: UInt8?

    public init(useMetalAcceleration: Bool = true) throws {
        codec = try PyrowaveCodec(useMetalAcceleration: useMetalAcceleration)
        expectedFrame = nil
    }

    public init(
        width: Int,
        height: Int,
        chroma: ChromaSubsampling,
        videoSignal: VideoSignalMetadata = .default,
        useMetalAcceleration: Bool = true
    ) throws {
        codec = try PyrowaveCodec(useMetalAcceleration: useMetalAcceleration)
        let layout = try PyrowaveBlockLayout(width: width, height: height, chroma: chroma)
        expectedFrame = ExpectedFrame(
            width: width,
            height: height,
            chroma: chroma,
            videoSignal: videoSignal,
            totalBlocks: layout.descriptors.count
        )
    }

    public func clear() {
        sequenceHeader = nil
        blockPackets.removeAll(keepingCapacity: true)
        decodedFrameForCurrentSequence = false
        lastSequence = nil
    }

    public func pushPacket(_ packet: EncodedPacket) throws {
        try pushPacket(packet.data)
    }

    public func pushPacket(_ data: Data) throws {
        var reader = BinaryReader(data)
        while reader.offset < data.count {
            let packetStart = reader.offset
            let header = try PyrowavePacketHeader(reader: &reader)

            if header.extended {
                try reader.seek(to: packetStart)
                let sequence = try PyrowaveSequenceHeader(reader: &reader)
                if isStale(sequence.sequence) {
                    return
                }
                try validate(sequence)
                if lastSequence != sequence.sequence {
                    beginSequence(sequence)
                } else {
                    if let sequenceHeader, sequenceHeader != sequence {
                        throw PyrowaveError.invalidBitstream("sequence header changed within active sequence")
                    }
                    sequenceHeader = sequence
                }
                continue
            }

            let packetEnd = packetStart + Int(header.payloadWords) * 4
            guard packetEnd >= reader.offset else {
                throw PyrowaveError.invalidBitstream("payload_words is not large enough")
            }
            guard packetEnd <= data.count else {
                throw PyrowaveError.truncatedInput
            }
            if isStale(header.sequence) {
                return
            }
            if lastSequence != header.sequence {
                guard let expectedFrame else {
                    throw PyrowaveError.invalidBitstream("coefficient packet sequence has no matching sequence header")
                }
                try beginSequence(expectedFrame.sequenceHeader(sequence: header.sequence))
            }
            guard sequenceHeader != nil else {
                throw PyrowaveError.invalidBitstream("coefficient packet before sequence header")
            }

            if blockPackets[header.blockIndex] == nil {
                blockPackets[header.blockIndex] = Data(data[packetStart..<packetEnd])
            }
            try reader.seek(to: packetEnd)
        }
    }

    public func decodeIsReady(allowPartialFrame: Bool = false) -> Bool {
        guard let sequenceHeader, !decodedFrameForCurrentSequence else {
            return false
        }
        if blockPackets.count < sequenceHeader.totalBlocks {
            return allowPartialFrame && blockPackets.count > sequenceHeader.totalBlocks / 2
        }
        return true
    }

    public func decode(allowPartialFrame: Bool = false) throws -> YUVFrame {
        guard decodeIsReady(allowPartialFrame: allowPartialFrame), let sequenceHeader else {
            throw PyrowaveError.invalidBitstream("packet stream is not ready to decode")
        }

        var writer = BinaryWriter()
        sequenceHeader.write(to: &writer)
        for blockIndex in blockPackets.keys.sorted() {
            if let packet = blockPackets[blockIndex] {
                writer.append(data: packet)
            }
        }

        let frame = try codec.decode(EncodedFrame(data: writer.data), allowPartialFrame: allowPartialFrame)
        decodedFrameForCurrentSequence = true
        return frame
    }

    private func beginSequence(_ sequence: PyrowaveSequenceHeader) {
        sequenceHeader = sequence
        blockPackets.removeAll(keepingCapacity: true)
        decodedFrameForCurrentSequence = false
        lastSequence = sequence.sequence
    }

    private func validate(_ sequence: PyrowaveSequenceHeader) throws {
        guard let expectedFrame else {
            return
        }
        guard sequence.width == expectedFrame.width,
              sequence.height == expectedFrame.height,
              sequence.chroma == expectedFrame.chroma else {
            throw PyrowaveError.invalidBitstream("sequence header does not match decoder geometry")
        }
    }

    private func isStale(_ sequence: UInt8) -> Bool {
        guard let lastSequence else {
            return false
        }
        let diff = (Int(sequence) - Int(lastSequence)) & PyrowaveBitstream.sequenceCountMask
        return diff > PyrowaveBitstream.sequenceCountMask / 2
    }

    private struct ExpectedFrame {
        var width: Int
        var height: Int
        var chroma: ChromaSubsampling
        var videoSignal: VideoSignalMetadata
        var totalBlocks: Int

        func sequenceHeader(sequence: UInt8) throws -> PyrowaveSequenceHeader {
            try PyrowaveSequenceHeader(
                width: width,
                height: height,
                sequence: sequence,
                totalBlocks: totalBlocks,
                chroma: chroma,
                videoSignal: videoSignal
            )
        }
    }
}

public final class PyrowaveCodec: Sendable {
    private static let sparseBlockSize = 32

    private let metalBackend: MetalPyrowaveBackend?
    private let sequenceCounter = SequenceCounter()

    public init(useMetalAcceleration: Bool = true) throws {
        if useMetalAcceleration {
            metalBackend = try MetalPyrowaveBackend()
        } else {
            metalBackend = nil
        }
    }

    public func encode(_ frame: YUVFrame, configuration: CodecConfiguration = CodecConfiguration()) throws -> EncodedFrame {
        guard configuration.decompositionLevels == PyrowaveBitstream.decompositionLevels,
              configuration.quantizationStep > 0,
              configuration.maximumEncodedBytes == nil || configuration.maximumEncodedBytes! > 0 else {
            throw PyrowaveError.invalidDimensions
        }

        let sequenceNumber = sequenceCounter.next()
        let layout = try PyrowaveBlockLayout(width: frame.width, height: frame.height, chroma: frame.chroma)
        let encodedPlanes = [
            try encodePlane(frame.y, component: 0, frameWidth: frame.width, frameHeight: frame.height, chroma: frame.chroma, layout: layout, configuration: configuration),
            try encodePlane(frame.cb, component: 1, frameWidth: frame.width, frameHeight: frame.height, chroma: frame.chroma, layout: layout, configuration: configuration),
            try encodePlane(frame.cr, component: 2, frameWidth: frame.width, frameHeight: frame.height, chroma: frame.chroma, layout: layout, configuration: configuration)
        ]

        let rateControlPlan = try selectSparseRateControlPlan(
            planes: encodedPlanes,
            layout: layout,
            configuration: configuration
        )
        let planeBlocks = try encodedPlanes.enumerated().flatMap { index, plane in
            try sparseBlocks(
                plane,
                layout: layout,
                sequence: sequenceNumber,
                quantLevels: rateControlPlan.quantLevelsByPlane?[index],
                defaultQuantLevel: rateControlPlan.defaultQuantLevel
            )
        }
        let sortedBlocks = planeBlocks.sorted { $0.blockIndex < $1.blockIndex }

        var writer = BinaryWriter()
        let sequence = try PyrowaveSequenceHeader(
            width: frame.width,
            height: frame.height,
            sequence: sequenceNumber,
            totalBlocks: sortedBlocks.count,
            chroma: frame.chroma,
            videoSignal: frame.videoSignal
        )
        sequence.write(to: &writer)
        for block in sortedBlocks {
            writer.append(data: block.data)
        }

        if let maximumEncodedBytes = configuration.maximumEncodedBytes, writer.data.count > maximumEncodedBytes {
            throw PyrowaveError.processFailed("minimum sparse frame size \(writer.data.count) exceeds maximumEncodedBytes \(maximumEncodedBytes)")
        }

        return EncodedFrame(data: writer.data)
    }

    public func decode(_ frame: EncodedFrame) throws -> YUVFrame {
        try decode(frame, allowPartialFrame: false)
    }

    fileprivate func decode(_ frame: EncodedFrame, allowPartialFrame: Bool) throws -> YUVFrame {
        var reader = BinaryReader(frame.data)
        let sequence = try PyrowaveSequenceHeader(reader: &reader)
        let layout = try PyrowaveBlockLayout(width: sequence.width, height: sequence.height, chroma: sequence.chroma)
        var decodedPlanes = try (0..<PyrowaveBitstream.componentCount).map { component in
            try makeDecodedPlane(component: component, width: sequence.width, height: sequence.height, chroma: sequence.chroma, layout: layout)
        }
        var seenBlocks = Set<Int>()

        while reader.offset < frame.data.count {
            let block = try PyrowaveCoefficientBlockCodec.decodeBlock(reader: &reader)
            guard block.sequence == sequence.sequence else {
                throw PyrowaveError.invalidBitstream("coefficient packet sequence mismatch")
            }
            guard block.blockIndex >= 0, block.blockIndex < layout.descriptors.count, seenBlocks.insert(block.blockIndex).inserted else {
                throw PyrowaveError.invalidBitstream("bad sparse block index")
            }
            guard let target = decodedPlanes.indices.first(where: { decodedPlanes[$0].descriptorsByBlockIndex[block.blockIndex] != nil }),
                  let descriptor = decodedPlanes[target].descriptorsByBlockIndex[block.blockIndex] else {
                throw PyrowaveError.invalidBitstream("sparse block index has no plane mapping")
            }
            try applySparseBlock(block, descriptor: descriptor, decodedPlane: &decodedPlanes[target])
        }

        if seenBlocks.count < sequence.totalBlocks {
            guard allowPartialFrame, seenBlocks.count > sequence.totalBlocks / 2 else {
                throw PyrowaveError.invalidBitstream("expected \(sequence.totalBlocks) blocks, decoded \(seenBlocks.count)")
            }
        } else if seenBlocks.count > sequence.totalBlocks {
            throw PyrowaveError.invalidBitstream("expected \(sequence.totalBlocks) blocks, decoded \(seenBlocks.count)")
        }

        guard reader.offset == frame.data.count else {
            throw PyrowaveError.invalidBitstream("trailing bytes")
        }

        let y = try finishDecodedPlane(decodedPlanes[0])
        let cb = try finishDecodedPlane(decodedPlanes[1])
        let cr = try finishDecodedPlane(decodedPlanes[2])

        return try YUVFrame(
            width: sequence.width,
            height: sequence.height,
            chroma: sequence.chroma,
            y: y,
            cb: cb,
            cr: cr,
            videoSignal: sequence.videoSignal
        )
    }

    private struct EncodedPlane {
        var component: Int
        var paddedWidth: Int
        var paddedHeight: Int
        var levels: Int
        var quantCodesByBlockIndex: [Int: UInt8]
        var qScaleCodesByBlockIndex: [Int: [UInt8]]
        var coefficients: [Int16]
    }

    private struct SparseBlock {
        var blockIndex: Int
        var data: Data
    }

    private struct SparseRateControlPlan {
        var defaultQuantLevel: Int
        var quantLevelsByPlane: [[Int]]?
    }

    private struct PlaneBlockDescriptor {
        var blockIndex: Int
        var globalLevel: Int
        var level: Int
        var band: Int
        var originX: Int
        var originY: Int
        var validWidth: Int
        var validHeight: Int
    }

    private struct DecodedPlane {
        var visibleWidth: Int
        var visibleHeight: Int
        var paddedWidth: Int
        var paddedHeight: Int
        var levels: Int
        var samples: [Float]
        var descriptorsByBlockIndex: [Int: PlaneBlockDescriptor]
    }

    private struct PlaneGeometry {
        var visibleWidth: Int
        var visibleHeight: Int
        var paddedWidth: Int
        var paddedHeight: Int
        var requestedLevels: Int
    }

    private final class SequenceCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var value: UInt8 = 0

        func next() -> UInt8 {
            lock.lock()
            defer { lock.unlock() }
            value = (value + 1) & UInt8(PyrowaveBitstream.sequenceCountMask)
            return value
        }
    }

    private func encodePlane(
        _ plane: Plane8,
        component: Int,
        frameWidth: Int,
        frameHeight: Int,
        chroma: ChromaSubsampling,
        layout: PyrowaveBlockLayout,
        configuration: CodecConfiguration
    ) throws -> EncodedPlane {
        let geometry = planeGeometry(component: component, frameWidth: frameWidth, frameHeight: frameHeight, chroma: chroma, requestedLevels: configuration.decompositionLevels)
        var padded = Wavelet.padPlane(plane, paddedWidth: geometry.paddedWidth, paddedHeight: geometry.paddedHeight)
        let requestedLevels = geometry.requestedLevels
        let levels = Wavelet.usableLevels(width: padded.width, height: padded.height, requested: requestedLevels)
        padded.samples = try forwardWavelet(padded.samples, width: padded.width, height: padded.height, levels: levels)

        let descriptors = planeBlockDescriptors(
            component: component,
            paddedWidth: padded.width,
            paddedHeight: padded.height,
            levels: levels,
            layout: layout
        )
        let quantized = try quantize(padded.samples, stride: padded.width, descriptors: descriptors, component: component, configuration: configuration)

        return EncodedPlane(
            component: component,
            paddedWidth: padded.width,
            paddedHeight: padded.height,
            levels: levels,
            quantCodesByBlockIndex: quantized.quantCodesByBlockIndex,
            qScaleCodesByBlockIndex: quantized.qScaleCodesByBlockIndex,
            coefficients: quantized.coefficients
        )
    }

    private func selectSparseRateControlPlan(
        planes: [EncodedPlane],
        layout: PyrowaveBlockLayout,
        configuration: CodecConfiguration
    ) throws -> SparseRateControlPlan {
        guard let maximumEncodedBytes = configuration.maximumEncodedBytes else {
            return SparseRateControlPlan(defaultQuantLevel: 0, quantLevelsByPlane: nil)
        }

        let fixedHeaderBytes = frameHeaderSize
        let rateBlocksByPlane = try planes.map { try makeRateControlBlocks($0, layout: layout) }
        if let thresholdsByPlane = PyrowaveRateController.selectThresholds(
            blocksByPlane: rateBlocksByPlane,
            fixedHeaderBytes: fixedHeaderBytes,
            maximumEncodedBytes: maximumEncodedBytes
        ) {
            return SparseRateControlPlan(defaultQuantLevel: 0, quantLevelsByPlane: thresholdsByPlane)
        }

        var low = 0
        var high = PyrowaveBlockStats.candidateCount - 1
        while low < high {
            let mid = (low + high) / 2
            let size = try estimateFrameSize(planes: planes, layout: layout, quantLevel: mid)
            if size <= maximumEncodedBytes {
                high = mid
            } else {
                low = mid + 1
            }
        }

        return SparseRateControlPlan(defaultQuantLevel: low, quantLevelsByPlane: nil)
    }

    private var frameHeaderSize: Int {
        8
    }

    private func estimateFrameSize(planes: [EncodedPlane], layout: PyrowaveBlockLayout, quantLevel: Int) throws -> Int {
        var size = frameHeaderSize
        for plane in planes {
            let blocks = try sparseBlocks(plane, layout: layout, defaultQuantLevel: quantLevel)
            size += blocks.reduce(0) { $0 + $1.data.count }
        }
        return size
    }

    private func sparseBlocks(
        _ plane: EncodedPlane,
        layout: PyrowaveBlockLayout,
        sequence: UInt8 = 0,
        quantLevels: [Int]? = nil,
        defaultQuantLevel: Int
    ) throws -> [SparseBlock] {
        let descriptors = planeBlockDescriptors(plane: plane, layout: layout)
        if let quantLevels, quantLevels.count != descriptors.count {
            throw PyrowaveError.processFailed("quant level count \(quantLevels.count) does not match block count \(descriptors.count)")
        }

        var blocks = [SparseBlock]()
        blocks.reserveCapacity(descriptors.count)

        for (index, descriptor) in descriptors.enumerated() {
            let quantLevel = quantLevels?[index] ?? defaultQuantLevel
            if let data = try PyrowaveCoefficientBlockCodec.encodeBlock(
                blockIndex: descriptor.blockIndex,
                coefficients: plane.coefficients,
                stride: plane.paddedWidth,
                originX: descriptor.originX,
                originY: descriptor.originY,
                validWidth: descriptor.validWidth,
                validHeight: descriptor.validHeight,
                threshold: 0,
                quantLevel: quantLevel,
                sequence: sequence,
                quantCode: try quantCode(for: descriptor, plane: plane),
                qScaleCodes: try qScaleCodes(for: descriptor, plane: plane)
            ) {
                blocks.append(SparseBlock(blockIndex: descriptor.blockIndex, data: data))
            }
        }

        return blocks
    }

    private func makeRateControlBlocks(_ plane: EncodedPlane, layout: PyrowaveBlockLayout) throws -> [PyrowaveRateControlBlock] {
        let descriptors = planeBlockDescriptors(plane: plane, layout: layout)
        return try descriptors.map { descriptor in
            try PyrowaveRateController.makeBlock(
                blockIndex: descriptor.blockIndex,
                coefficients: plane.coefficients,
                stride: plane.paddedWidth,
                originX: descriptor.originX,
                originY: descriptor.originY,
                validWidth: descriptor.validWidth,
                validHeight: descriptor.validHeight,
                quantCode: try quantCode(for: descriptor, plane: plane),
                qScaleCodes: try qScaleCodes(for: descriptor, plane: plane),
                rdoDistortionScale: PyrowaveQuantization.rdoDistortionScale(
                    level: descriptor.globalLevel,
                    component: plane.component,
                    band: descriptor.band,
                    chroma: layout.chroma
                )
            )
        }
    }

    private func makeDecodedPlane(component: Int, width: Int, height: Int, chroma: ChromaSubsampling, layout: PyrowaveBlockLayout) throws -> DecodedPlane {
        let geometry = planeGeometry(component: component, frameWidth: width, frameHeight: height, chroma: chroma, requestedLevels: PyrowaveBitstream.decompositionLevels)
        let levels = Wavelet.usableLevels(width: geometry.paddedWidth, height: geometry.paddedHeight, requested: geometry.requestedLevels)
        let scratchPlane = EncodedPlane(
            component: component,
            paddedWidth: geometry.paddedWidth,
            paddedHeight: geometry.paddedHeight,
            levels: levels,
            quantCodesByBlockIndex: [:],
            qScaleCodesByBlockIndex: [:],
            coefficients: []
        )
        let descriptors = planeBlockDescriptors(plane: scratchPlane, layout: layout)
        return DecodedPlane(
            visibleWidth: geometry.visibleWidth,
            visibleHeight: geometry.visibleHeight,
            paddedWidth: geometry.paddedWidth,
            paddedHeight: geometry.paddedHeight,
            levels: levels,
            samples: Array(repeating: 0, count: geometry.paddedWidth * geometry.paddedHeight),
            descriptorsByBlockIndex: Dictionary(uniqueKeysWithValues: descriptors.map { ($0.blockIndex, $0) })
        )
    }

    private func applySparseBlock(
        _ block: PyrowaveCoefficientBlockCodec.DecodedBlock,
        descriptor: PlaneBlockDescriptor,
        decodedPlane: inout DecodedPlane
    ) throws {
        let blockSize = Self.sparseBlockSize
        for entry in block.coefficients {
            let localOffset = Int(entry.offset)
            let localX = localOffset % blockSize
            let localY = localOffset / blockSize
            let x = descriptor.originX + localX
            let y = descriptor.originY + localY
            guard localOffset < blockSize * blockSize,
                  localX < descriptor.validWidth,
                  localY < descriptor.validHeight,
                  x < decodedPlane.paddedWidth,
                  y < decodedPlane.paddedHeight else {
                throw PyrowaveError.invalidBitstream("sparse coefficient out of range")
            }
            decodedPlane.samples[y * decodedPlane.paddedWidth + x] = PyrowaveQuantization.dequantize(
                coefficient: entry.value,
                quantCode: block.quantCode,
                qScaleCode: entry.qScaleCode
            )
        }
    }

    private func finishDecodedPlane(_ plane: DecodedPlane) throws -> Plane8 {
        let reconstructed = try inverseWavelet(plane.samples, width: plane.paddedWidth, height: plane.paddedHeight, levels: plane.levels)
        return try Wavelet.cropPlane(reconstructed, paddedWidth: plane.paddedWidth, width: plane.visibleWidth, height: plane.visibleHeight)
    }

    private func planeBlockDescriptors(plane: EncodedPlane, layout: PyrowaveBlockLayout) -> [PlaneBlockDescriptor] {
        planeBlockDescriptors(
            component: plane.component,
            paddedWidth: plane.paddedWidth,
            paddedHeight: plane.paddedHeight,
            levels: plane.levels,
            layout: layout
        )
    }

    private func planeBlockDescriptors(
        component: Int,
        paddedWidth: Int,
        paddedHeight: Int,
        levels: Int,
        layout: PyrowaveBlockLayout
    ) -> [PlaneBlockDescriptor] {
        layout.descriptors.compactMap { global in
            guard global.component == component else {
                return nil
            }

            let localLevel = component != 0 && layout.chroma == .yuv420 ? global.level - 1 : global.level
            guard localLevel >= 0, localLevel < levels else {
                return nil
            }

            let subbandWidth = paddedWidth >> (localLevel + 1)
            let subbandHeight = paddedHeight >> (localLevel + 1)
            let origin = planeBandOrigin(level: localLevel, finalLevel: levels - 1, band: global.band, subbandWidth: subbandWidth, subbandHeight: subbandHeight)
            let x = global.blockX * Self.sparseBlockSize
            let y = global.blockY * Self.sparseBlockSize
            return PlaneBlockDescriptor(
                blockIndex: global.blockIndex,
                globalLevel: global.level,
                level: localLevel,
                band: global.band,
                originX: origin.x + x,
                originY: origin.y + y,
                validWidth: min(Self.sparseBlockSize, subbandWidth - x),
                validHeight: min(Self.sparseBlockSize, subbandHeight - y)
            )
        }
    }

    private func quantCode(for descriptor: PlaneBlockDescriptor, plane: EncodedPlane) throws -> UInt8 {
        guard let quantCode = plane.quantCodesByBlockIndex[descriptor.blockIndex] else {
            throw PyrowaveError.processFailed("missing quant code for block \(descriptor.blockIndex)")
        }
        return quantCode
    }

    private func qScaleCodes(for descriptor: PlaneBlockDescriptor, plane: EncodedPlane) throws -> [UInt8] {
        guard let qScaleCodes = plane.qScaleCodesByBlockIndex[descriptor.blockIndex] else {
            throw PyrowaveError.processFailed("missing 8x8 quant scale codes for block \(descriptor.blockIndex)")
        }
        return qScaleCodes
    }

    private func planeGeometry(component: Int, frameWidth: Int, frameHeight: Int, chroma: ChromaSubsampling, requestedLevels: Int) -> PlaneGeometry {
        let alignedWidth = Wavelet.alignedDimension(frameWidth)
        let alignedHeight = Wavelet.alignedDimension(frameHeight)
        let isSubsampledChroma = component != 0 && chroma == .yuv420
        let divisor = isSubsampledChroma ? 2 : 1
        let planeRequestedLevels = isSubsampledChroma ? max(1, requestedLevels - 1) : requestedLevels
        return PlaneGeometry(
            visibleWidth: frameWidth / divisor,
            visibleHeight: frameHeight / divisor,
            paddedWidth: alignedWidth / divisor,
            paddedHeight: alignedHeight / divisor,
            requestedLevels: planeRequestedLevels
        )
    }

    private func planeBandOrigin(level: Int, finalLevel: Int, band: Int, subbandWidth: Int, subbandHeight: Int) -> (x: Int, y: Int) {
        if level == finalLevel, band == 0 {
            return (0, 0)
        }

        switch band {
        case 1:
            return (subbandWidth, 0)
        case 2:
            return (0, subbandHeight)
        case 3:
            return (subbandWidth, subbandHeight)
        default:
            return (0, 0)
        }
    }

    private func forwardWavelet(_ samples: [Float], width: Int, height: Int, levels: Int) throws -> [Float] {
        if let metalBackend {
            return try metalBackend.forwardWavelet(samples, width: width, height: height, levels: levels)
        }

        var transformed = samples
        Wavelet.forward2D(&transformed, width: width, height: height, levels: levels)
        return transformed
    }

    private func inverseWavelet(_ samples: [Float], width: Int, height: Int, levels: Int) throws -> [Float] {
        if let metalBackend {
            return try metalBackend.inverseWavelet(samples, width: width, height: height, levels: levels)
        }

        var transformed = samples
        Wavelet.inverse2D(&transformed, width: width, height: height, levels: levels)
        return transformed
    }

    private func quantize(
        _ samples: [Float],
        stride: Int,
        descriptors: [PlaneBlockDescriptor],
        component: Int,
        configuration: CodecConfiguration
    ) throws -> (coefficients: [Int16], quantCodesByBlockIndex: [Int: UInt8], qScaleCodesByBlockIndex: [Int: [UInt8]]) {
        var coefficients = Array(repeating: Int16(0), count: samples.count)
        var quantCodes = [Int: UInt8]()
        var qScaleCodes = [Int: [UInt8]]()
        quantCodes.reserveCapacity(descriptors.count)
        qScaleCodes.reserveCapacity(descriptors.count)

        for descriptor in descriptors {
            let requestedStep = PyrowaveQuantization.quantizationStep(
                level: descriptor.globalLevel,
                component: component,
                band: descriptor.band,
                baseStep: configuration.quantizationStep
            )
            let quantCode = try PyrowaveQuantization.encodeBlockScale(requestedStep)
            let decodedStep = PyrowaveQuantization.decodeBlockScale(quantCode)
            let baseScale = 1.0 / decodedStep
            quantCodes[descriptor.blockIndex] = quantCode

            var blockQScaleCodes = Array(repeating: PyrowaveQuantization.identityQScaleCode, count: 16)
            for smallBlockY in 0..<4 {
                for smallBlockX in 0..<4 {
                    let smallBlock = smallBlockY * 4 + smallBlockX
                    let smallOriginX = descriptor.originX + smallBlockX * PyrowaveBitstream.smallBlockSize
                    let smallOriginY = descriptor.originY + smallBlockY * PyrowaveBitstream.smallBlockSize
                    let smallValidWidth = max(0, min(PyrowaveBitstream.smallBlockSize, descriptor.validWidth - smallBlockX * PyrowaveBitstream.smallBlockSize))
                    let smallValidHeight = max(0, min(PyrowaveBitstream.smallBlockSize, descriptor.validHeight - smallBlockY * PyrowaveBitstream.smallBlockSize))
                    guard smallValidWidth > 0, smallValidHeight > 0 else {
                        continue
                    }

                    var maxScaledCoefficient = Float(0)
                    for y in 0..<smallValidHeight {
                        let row = (smallOriginY + y) * stride + smallOriginX
                        for x in 0..<smallValidWidth {
                            maxScaledCoefficient = max(maxScaledCoefficient, abs(samples[row + x] * baseScale))
                        }
                    }

                    let qScaleCode = PyrowaveQuantization.encode8x8ScaleCode(maxScaledCoefficient: maxScaledCoefficient)
                    let quantScale = PyrowaveQuantization.quantScale(for8x8ScaleCode: qScaleCode)
                    blockQScaleCodes[smallBlock] = qScaleCode

                    for y in 0..<smallValidHeight {
                        let row = (smallOriginY + y) * stride + smallOriginX
                        for x in 0..<smallValidWidth {
                            let index = row + x
                            let quantized = Int((samples[index] * baseScale * quantScale).rounded(.towardZero))
                            coefficients[index] = Int16(max(Int(Int16.min), min(Int(Int16.max), quantized)))
                        }
                    }
                }
            }

            qScaleCodes[descriptor.blockIndex] = blockQScaleCodes
        }

        return (coefficients, quantCodes, qScaleCodes)
    }

}
