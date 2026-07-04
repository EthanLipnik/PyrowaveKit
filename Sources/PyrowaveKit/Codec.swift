import Foundation

#if canImport(Metal)
import Metal
#endif

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
                packetByteCosts: rateControlPlan.packetByteCostsByPlane?[index],
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

    #if canImport(Metal)
    public func encode(
        yTexture: MTLTexture,
        cbTexture: MTLTexture,
        crTexture: MTLTexture,
        configuration: CodecConfiguration = CodecConfiguration(),
        videoSignal: VideoSignalMetadata = .default
    ) throws -> EncodedFrame {
        guard configuration.decompositionLevels == PyrowaveBitstream.decompositionLevels,
              configuration.quantizationStep > 0,
              configuration.maximumEncodedBytes == nil || configuration.maximumEncodedBytes! > 0 else {
            throw PyrowaveError.invalidDimensions
        }
        guard yTexture.pixelFormat == .r8Unorm,
              cbTexture.pixelFormat == .r8Unorm,
              crTexture.pixelFormat == .r8Unorm,
              yTexture.width > 0,
              yTexture.height > 0 else {
            throw PyrowaveError.unsupportedFormat("Metal texture encode expects r8Unorm planes")
        }

        let chroma: ChromaSubsampling
        if cbTexture.width == yTexture.width / 2,
           cbTexture.height == yTexture.height / 2,
           crTexture.width == cbTexture.width,
           crTexture.height == cbTexture.height {
            chroma = .yuv420
        } else if cbTexture.width == yTexture.width,
                  cbTexture.height == yTexture.height,
                  crTexture.width == yTexture.width,
                  crTexture.height == yTexture.height {
            chroma = .yuv444
        } else {
            throw PyrowaveError.invalidDimensions
        }

        let sequenceNumber = sequenceCounter.next()
        let layout = try PyrowaveBlockLayout(width: yTexture.width, height: yTexture.height, chroma: chroma)
        let encodedPlanes = [
            try encodeTexturePlane(yTexture, component: 0, frameWidth: yTexture.width, frameHeight: yTexture.height, chroma: chroma, layout: layout, configuration: configuration),
            try encodeTexturePlane(cbTexture, component: 1, frameWidth: yTexture.width, frameHeight: yTexture.height, chroma: chroma, layout: layout, configuration: configuration),
            try encodeTexturePlane(crTexture, component: 2, frameWidth: yTexture.width, frameHeight: yTexture.height, chroma: chroma, layout: layout, configuration: configuration)
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
                packetByteCosts: rateControlPlan.packetByteCostsByPlane?[index],
                defaultQuantLevel: rateControlPlan.defaultQuantLevel
            )
        }
        let sortedBlocks = planeBlocks.sorted { $0.blockIndex < $1.blockIndex }

        var writer = BinaryWriter()
        let sequence = try PyrowaveSequenceHeader(
            width: yTexture.width,
            height: yTexture.height,
            sequence: sequenceNumber,
            totalBlocks: sortedBlocks.count,
            chroma: chroma,
            videoSignal: videoSignal
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
    #endif

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
        var pendingSparseBlocks = Array(repeating: [PendingSparseBlock](), count: PyrowaveBitstream.componentCount)
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
            pendingSparseBlocks[target].append(PendingSparseBlock(block: block, descriptor: descriptor))
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

        for index in decodedPlanes.indices {
            try applySparseBlocks(pendingSparseBlocks[index], decodedPlane: &decodedPlanes[index])
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

    private struct PendingSparseBlock {
        var block: PyrowaveCoefficientBlockCodec.DecodedBlock
        var descriptor: PlaneBlockDescriptor
    }

    private struct SparseRateControlPlan {
        var defaultQuantLevel: Int
        var quantLevelsByPlane: [[Int]]?
        var packetByteCostsByPlane: [[[Int]]]?
    }

    private struct MetalRateControlBucketData {
        var indicesByPlane: [[[Int]]]
        var cumulativeSavings: [Int]
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
        return try encodePaddedPlane(
            samples: try padPlane(plane, paddedWidth: geometry.paddedWidth, paddedHeight: geometry.paddedHeight),
            paddedWidth: geometry.paddedWidth,
            paddedHeight: geometry.paddedHeight,
            requestedLevels: geometry.requestedLevels,
            component: component,
            layout: layout,
            configuration: configuration
        )
    }

    #if canImport(Metal)
    private func encodeTexturePlane(
        _ texture: MTLTexture,
        component: Int,
        frameWidth: Int,
        frameHeight: Int,
        chroma: ChromaSubsampling,
        layout: PyrowaveBlockLayout,
        configuration: CodecConfiguration
    ) throws -> EncodedPlane {
        guard let metalBackend else {
            throw PyrowaveError.externalToolUnavailable("Metal")
        }
        let geometry = planeGeometry(component: component, frameWidth: frameWidth, frameHeight: frameHeight, chroma: chroma, requestedLevels: configuration.decompositionLevels)
        guard texture.width == geometry.visibleWidth,
              texture.height == geometry.visibleHeight else {
            throw PyrowaveError.invalidDimensions
        }
        return try encodePaddedPlane(
            samples: try metalBackend.padTexturePlane(texture, paddedWidth: geometry.paddedWidth, paddedHeight: geometry.paddedHeight),
            paddedWidth: geometry.paddedWidth,
            paddedHeight: geometry.paddedHeight,
            requestedLevels: geometry.requestedLevels,
            component: component,
            layout: layout,
            configuration: configuration
        )
    }
    #endif

    private func encodePaddedPlane(
        samples: [Float],
        paddedWidth: Int,
        paddedHeight: Int,
        requestedLevels: Int,
        component: Int,
        layout: PyrowaveBlockLayout,
        configuration: CodecConfiguration
    ) throws -> EncodedPlane {
        var transformed = samples
        let levels = Wavelet.usableLevels(width: paddedWidth, height: paddedHeight, requested: requestedLevels)
        transformed = try forwardWavelet(transformed, width: paddedWidth, height: paddedHeight, levels: levels)

        let descriptors = planeBlockDescriptors(
            component: component,
            paddedWidth: paddedWidth,
            paddedHeight: paddedHeight,
            levels: levels,
            layout: layout
        )
        let quantized = try quantize(transformed, stride: paddedWidth, descriptors: descriptors, component: component, configuration: configuration)

        return EncodedPlane(
            component: component,
            paddedWidth: paddedWidth,
            paddedHeight: paddedHeight,
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
            return SparseRateControlPlan(defaultQuantLevel: 0, quantLevelsByPlane: nil, packetByteCostsByPlane: nil)
        }

        let fixedHeaderBytes = frameHeaderSize
        let rateBlocksByPlane = try planes.map { try makeRateControlBlocks($0, layout: layout) }
        let packetByteCostsByPlane = rateBlocksByPlane.map { $0.map(\.packetByteCosts) }
        let metalBucketData = try metalRateControlBucketData(blocksByPlane: rateBlocksByPlane)
        if let thresholdsByPlane = PyrowaveRateController.selectThresholds(
            blocksByPlane: rateBlocksByPlane,
            fixedHeaderBytes: fixedHeaderBytes,
            maximumEncodedBytes: maximumEncodedBytes,
            bucketIndicesByPlane: metalBucketData?.indicesByPlane,
            cumulativeBucketSavings: metalBucketData?.cumulativeSavings
        ) {
            return SparseRateControlPlan(
                defaultQuantLevel: 0,
                quantLevelsByPlane: thresholdsByPlane,
                packetByteCostsByPlane: packetByteCostsByPlane
            )
        }

        var low = 0
        var high = PyrowaveBlockStats.candidateCount - 1
        while low < high {
            let mid = (low + high) / 2
            let thresholds = rateBlocksByPlane.map { Array(repeating: mid, count: $0.count) }
            let size = PyrowaveRateController.estimateFrameBytes(
                blocksByPlane: rateBlocksByPlane,
                thresholdsByPlane: thresholds,
                fixedHeaderBytes: fixedHeaderBytes
            )
            if size <= maximumEncodedBytes {
                high = mid
            } else {
                low = mid + 1
            }
        }

        return SparseRateControlPlan(
            defaultQuantLevel: low,
            quantLevelsByPlane: nil,
            packetByteCostsByPlane: packetByteCostsByPlane
        )
    }

    private func metalRateControlBucketData(blocksByPlane: [[PyrowaveRateControlBlock]]) throws -> MetalRateControlBucketData? {
        guard let metalBackend else {
            return nil
        }

        var bucketsByPlane = [[[Int]]]()
        bucketsByPlane.reserveCapacity(blocksByPlane.count)
        var flatBuckets = [[Int]]()
        var flatPacketByteCosts = [[Int]]()
        for blocks in blocksByPlane {
            let distortions = blocks.map { block in
                (0..<PyrowaveBlockStats.candidateCount).map { block.distortion(quantLevel: $0) }
            }
            let packetByteCosts = blocks.map(\.packetByteCosts)
            let buckets = try metalBackend.rateControlBucketIndices(
                distortions: distortions,
                packetByteCosts: packetByteCosts
            )
            guard buckets.count == blocks.count else {
                throw PyrowaveError.processFailed("Metal rate-control bucket pass returned \(buckets.count) blocks for \(blocks.count) inputs")
            }
            bucketsByPlane.append(buckets)
            flatBuckets.append(contentsOf: buckets)
            flatPacketByteCosts.append(contentsOf: packetByteCosts)
        }
        let cumulativeSavings = try metalBackend.rateControlCumulativeBucketSavings(
            bucketIndices: flatBuckets,
            packetByteCosts: flatPacketByteCosts
        )
        return MetalRateControlBucketData(indicesByPlane: bucketsByPlane, cumulativeSavings: cumulativeSavings)
    }

    private var frameHeaderSize: Int {
        8
    }

    private func sparseBlocks(
        _ plane: EncodedPlane,
        layout: PyrowaveBlockLayout,
        sequence: UInt8 = 0,
        quantLevels: [Int]? = nil,
        packetByteCosts: [[Int]]? = nil,
        defaultQuantLevel: Int
    ) throws -> [SparseBlock] {
        let descriptors = planeBlockDescriptors(plane: plane, layout: layout)
        if let quantLevels, quantLevels.count != descriptors.count {
            throw PyrowaveError.processFailed("quant level count \(quantLevels.count) does not match block count \(descriptors.count)")
        }
        if let packetByteCosts, packetByteCosts.count != descriptors.count {
            throw PyrowaveError.processFailed("packet byte-cost count \(packetByteCosts.count) does not match block count \(descriptors.count)")
        }

        var blocks = [SparseBlock]()
        blocks.reserveCapacity(descriptors.count)
        let selectedPacketByteCosts: [[Int]]?
        if let packetByteCosts {
            selectedPacketByteCosts = packetByteCosts
        } else {
            selectedPacketByteCosts = try metalSparsePacketByteCosts(
                plane: plane,
                descriptors: descriptors
            )
        }
        if let metalBackend {
            return try sparseBlocksWithMetal(
                plane,
                descriptors: descriptors,
                sequence: sequence,
                quantLevels: quantLevels,
                packetByteCosts: selectedPacketByteCosts,
                defaultQuantLevel: defaultQuantLevel,
                backend: metalBackend
            )
        }

        for (index, descriptor) in descriptors.enumerated() {
            let quantLevel = quantLevels?[index] ?? defaultQuantLevel
            if let selectedPacketByteCosts,
               quantLevel >= 0,
               quantLevel < selectedPacketByteCosts[index].count,
               selectedPacketByteCosts[index][quantLevel] == 0 {
                continue
            }
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

    private func sparseBlocksWithMetal(
        _ plane: EncodedPlane,
        descriptors: [PlaneBlockDescriptor],
        sequence: UInt8,
        quantLevels: [Int]?,
        packetByteCosts: [[Int]]?,
        defaultQuantLevel: Int,
        backend: MetalPyrowaveBackend
    ) throws -> [SparseBlock] {
        var packetDescriptors = [MetalSparsePacketEncodeDescriptor]()
        var packetQScaleCodes = [[UInt8]]()
        var blockIndices = [Int]()
        packetDescriptors.reserveCapacity(descriptors.count)
        packetQScaleCodes.reserveCapacity(descriptors.count)
        blockIndices.reserveCapacity(descriptors.count)

        for (index, descriptor) in descriptors.enumerated() {
            let quantLevel = quantLevels?[index] ?? defaultQuantLevel
            guard quantLevel >= 0 else {
                throw PyrowaveError.invalidBitstream("negative quant level")
            }
            if let packetByteCosts,
               quantLevel < packetByteCosts[index].count,
               packetByteCosts[index][quantLevel] == 0 {
                continue
            }
            guard quantLevel <= Int(UInt32.max),
                  descriptor.blockIndex <= Int(UInt32.max),
                  plane.paddedWidth <= Int(UInt32.max) else {
                throw PyrowaveError.invalidDimensions
            }
            packetDescriptors.append(MetalSparsePacketEncodeDescriptor(
                originX: UInt32(descriptor.originX),
                originY: UInt32(descriptor.originY),
                validWidth: UInt32(descriptor.validWidth),
                validHeight: UInt32(descriptor.validHeight),
                stride: UInt32(plane.paddedWidth),
                blockIndex: UInt32(descriptor.blockIndex),
                quantLevel: UInt32(quantLevel),
                sequence: UInt32(sequence),
                quantCode: UInt32(try quantCode(for: descriptor, plane: plane))
            ))
            packetQScaleCodes.append(try qScaleCodes(for: descriptor, plane: plane))
            blockIndices.append(descriptor.blockIndex)
        }

        let packets = try backend.encodeSparsePackets(
            coefficients: plane.coefficients,
            descriptors: packetDescriptors,
            qScaleCodes: packetQScaleCodes
        )
        guard packets.count == packetDescriptors.count else {
            throw PyrowaveError.processFailed("Metal sparse packet encode returned \(packets.count) packets for \(packetDescriptors.count) descriptors")
        }
        return packets.indices.compactMap { index in
            packets[index].map { SparseBlock(blockIndex: blockIndices[index], data: $0) }
        }
    }

    private func metalSparsePacketByteCosts(
        plane: EncodedPlane,
        descriptors: [PlaneBlockDescriptor]
    ) throws -> [[Int]]? {
        guard let metalBackend else {
            return nil
        }
        let packetCostDescriptors = metalPacketByteCostDescriptors(
            descriptors: descriptors,
            stride: plane.paddedWidth
        )
        let costs = try metalBackend.packetByteCosts(
            coefficients: plane.coefficients,
            descriptors: packetCostDescriptors
        )
        guard costs.count == descriptors.count else {
            throw PyrowaveError.processFailed("Metal packet byte-cost returned \(costs.count) block costs for \(descriptors.count) descriptors")
        }
        return costs
    }

    private func makeRateControlBlocks(_ plane: EncodedPlane, layout: PyrowaveBlockLayout) throws -> [PyrowaveRateControlBlock] {
        let descriptors = planeBlockDescriptors(plane: plane, layout: layout)
        if let metalBackend {
            return try makeRateControlBlocksWithMetal(plane, descriptors: descriptors, layout: layout, backend: metalBackend)
        }

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

    private func makeRateControlBlocksWithMetal(
        _ plane: EncodedPlane,
        descriptors: [PlaneBlockDescriptor],
        layout: PyrowaveBlockLayout,
        backend: MetalPyrowaveBackend
    ) throws -> [PyrowaveRateControlBlock] {
        var metalDescriptors = [MetalRateControlStatsDescriptor]()
        metalDescriptors.reserveCapacity(descriptors.count * 16)

        for descriptor in descriptors {
            let quantCode = try quantCode(for: descriptor, plane: plane)
            let qScaleCodes = try qScaleCodes(for: descriptor, plane: plane)
            let rdoDistortionScale = PyrowaveQuantization.rdoDistortionScale(
                level: descriptor.globalLevel,
                component: plane.component,
                band: descriptor.band,
                chroma: layout.chroma
            )
            for tileY in 0..<4 {
                for tileX in 0..<4 {
                    let tileIndex = tileY * 4 + tileX
                    metalDescriptors.append(MetalRateControlStatsDescriptor(
                        originX: UInt32(descriptor.originX + tileX * PyrowaveBitstream.smallBlockSize),
                        originY: UInt32(descriptor.originY + tileY * PyrowaveBitstream.smallBlockSize),
                        validWidth: UInt32(max(0, min(PyrowaveBitstream.smallBlockSize, descriptor.validWidth - tileX * PyrowaveBitstream.smallBlockSize))),
                        validHeight: UInt32(max(0, min(PyrowaveBitstream.smallBlockSize, descriptor.validHeight - tileY * PyrowaveBitstream.smallBlockSize))),
                        stride: UInt32(plane.paddedWidth),
                        quantCode: UInt32(quantCode),
                        qScaleCode: UInt32(qScaleCodes[tileIndex]),
                        distortionScale: rdoDistortionScale
                    ))
                }
            }
        }

        let tileStats = try backend.rateControlTileStats(coefficients: plane.coefficients, descriptors: metalDescriptors)
        guard tileStats.count == metalDescriptors.count else {
            throw PyrowaveError.processFailed("Metal rate-control returned \(tileStats.count) tile stats for \(metalDescriptors.count) descriptors")
        }
        let packetCostDescriptors = metalPacketByteCostDescriptors(descriptors: descriptors, stride: plane.paddedWidth)
        let packetByteCosts = try backend.packetByteCosts(coefficients: plane.coefficients, descriptors: packetCostDescriptors)
        guard packetByteCosts.count == descriptors.count else {
            throw PyrowaveError.processFailed("Metal packet byte-cost returned \(packetByteCosts.count) block costs for \(descriptors.count) descriptors")
        }

        var blocks = [PyrowaveRateControlBlock]()
        blocks.reserveCapacity(descriptors.count)
        for (descriptorIndex, descriptor) in descriptors.enumerated() {
            let firstTile = descriptorIndex * 16
            let eightByEightStats = try tileStats[firstTile..<(firstTile + 16)].map { tile -> PyrowaveBlockStats in
                guard tile.stats.count == PyrowaveBlockStats.candidateCount else {
                    throw PyrowaveError.processFailed("Metal rate-control returned \(tile.stats.count) quant stats")
                }
                return PyrowaveBlockStats(
                    numPlanes: Int(tile.numPlanes),
                    stats: tile.stats.map {
                        PyrowaveQuantStats(squareError: $0.squareError, encodeCostBits: Int($0.encodeCostBits))
                    }
                )
            }
            blocks.append(PyrowaveRateControlBlock(
                blockIndex: descriptor.blockIndex,
                eightByEightStats: eightByEightStats,
                packetByteCosts: packetByteCosts[descriptorIndex]
            ))
        }
        return blocks
    }

    private func metalPacketByteCostDescriptors(
        descriptors: [PlaneBlockDescriptor],
        stride: Int
    ) -> [MetalPacketByteCostDescriptor] {
        descriptors.map {
            MetalPacketByteCostDescriptor(
                originX: UInt32($0.originX),
                originY: UInt32($0.originY),
                validWidth: UInt32($0.validWidth),
                validHeight: UInt32($0.validHeight),
                stride: UInt32(stride)
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
        for entry in block.coefficients {
            let destinationOffset = try sparseDestinationOffset(
                entryOffset: entry.offset,
                descriptor: descriptor,
                decodedPlane: decodedPlane
            )
            decodedPlane.samples[destinationOffset] = PyrowaveQuantization.dequantize(
                coefficient: entry.value,
                quantCode: block.quantCode,
                qScaleCode: entry.qScaleCode
            )
        }
    }

    private func applySparseBlocks(
        _ blocks: [PendingSparseBlock],
        decodedPlane: inout DecodedPlane
    ) throws {
        guard let metalBackend else {
            for pending in blocks {
                try applySparseBlock(pending.block, descriptor: pending.descriptor, decodedPlane: &decodedPlane)
            }
            return
        }

        let entries = try metalSparseCoefficientEntries(blocks, decodedPlane: decodedPlane)
        decodedPlane.samples = try metalBackend.applySparseCoefficients(
            sampleCount: decodedPlane.paddedWidth * decodedPlane.paddedHeight,
            entries: entries
        )
    }

    private func metalSparseCoefficientEntries(
        _ blocks: [PendingSparseBlock],
        decodedPlane: DecodedPlane
    ) throws -> [MetalSparseCoefficientEntry] {
        let capacity = blocks.reduce(0) { $0 + $1.block.coefficients.count }
        var entries = [MetalSparseCoefficientEntry]()
        entries.reserveCapacity(capacity)

        for pending in blocks {
            for entry in pending.block.coefficients {
                let destinationOffset = try sparseDestinationOffset(
                    entryOffset: entry.offset,
                    descriptor: pending.descriptor,
                    decodedPlane: decodedPlane
                )
                entries.append(MetalSparseCoefficientEntry(
                    destinationOffset: UInt32(destinationOffset),
                    coefficient: Int32(entry.value),
                    quantCode: UInt32(pending.block.quantCode),
                    qScaleCode: UInt32(entry.qScaleCode)
                ))
            }
        }

        return entries
    }

    private func sparseDestinationOffset(
        entryOffset: UInt16,
        descriptor: PlaneBlockDescriptor,
        decodedPlane: DecodedPlane
    ) throws -> Int {
        let blockSize = Self.sparseBlockSize
        let localOffset = Int(entryOffset)
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
        return y * decodedPlane.paddedWidth + x
    }

    private func finishDecodedPlane(_ plane: DecodedPlane) throws -> Plane8 {
        let reconstructed = try inverseWavelet(plane.samples, width: plane.paddedWidth, height: plane.paddedHeight, levels: plane.levels)
        return try cropPlane(reconstructed, paddedWidth: plane.paddedWidth, width: plane.visibleWidth, height: plane.visibleHeight)
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

    private func padPlane(_ plane: Plane8, paddedWidth: Int, paddedHeight: Int) throws -> [Float] {
        if let metalBackend {
            return try metalBackend.padPlane(plane, paddedWidth: paddedWidth, paddedHeight: paddedHeight)
        }

        return Wavelet.padPlane(plane, paddedWidth: paddedWidth, paddedHeight: paddedHeight).samples
    }

    private func cropPlane(_ samples: [Float], paddedWidth: Int, width: Int, height: Int) throws -> Plane8 {
        if let metalBackend {
            return try metalBackend.cropPlane(samples, paddedWidth: paddedWidth, width: width, height: height)
        }

        return try Wavelet.cropPlane(samples, paddedWidth: paddedWidth, width: width, height: height)
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
        if let metalBackend {
            return try quantizeWithMetal(
                samples,
                stride: stride,
                descriptors: descriptors,
                component: component,
                configuration: configuration,
                backend: metalBackend
            )
        }

        return try quantizeOnCPU(
            samples,
            stride: stride,
            descriptors: descriptors,
            component: component,
            configuration: configuration
        )
    }

    private func quantizeOnCPU(
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

    private func quantizeWithMetal(
        _ samples: [Float],
        stride: Int,
        descriptors: [PlaneBlockDescriptor],
        component: Int,
        configuration: CodecConfiguration,
        backend: MetalPyrowaveBackend
    ) throws -> (coefficients: [Int16], quantCodesByBlockIndex: [Int: UInt8], qScaleCodesByBlockIndex: [Int: [UInt8]]) {
        var quantCodes = [Int: UInt8]()
        quantCodes.reserveCapacity(descriptors.count)
        let metalDescriptors = try descriptors.map { descriptor in
            let requestedStep = PyrowaveQuantization.quantizationStep(
                level: descriptor.globalLevel,
                component: component,
                band: descriptor.band,
                baseStep: configuration.quantizationStep
            )
            let quantCode = try PyrowaveQuantization.encodeBlockScale(requestedStep)
            let decodedStep = PyrowaveQuantization.decodeBlockScale(quantCode)
            quantCodes[descriptor.blockIndex] = quantCode

            return MetalPlaneQuantizationDescriptor(
                originX: UInt32(descriptor.originX),
                originY: UInt32(descriptor.originY),
                validWidth: UInt32(descriptor.validWidth),
                validHeight: UInt32(descriptor.validHeight),
                stride: UInt32(stride),
                quantCode: UInt32(quantCode),
                baseScale: 1.0 / decodedStep
            )
        }

        let result = try backend.quantizePlane(samples, stride: stride, descriptors: metalDescriptors)
        guard result.qScaleCodesByDescriptor.count == descriptors.count else {
            throw PyrowaveError.processFailed("Metal quantization returned \(result.qScaleCodesByDescriptor.count) q-scale rows for \(descriptors.count) descriptors")
        }

        var qScaleCodes = [Int: [UInt8]]()
        qScaleCodes.reserveCapacity(descriptors.count)
        for (index, descriptor) in descriptors.enumerated() {
            qScaleCodes[descriptor.blockIndex] = result.qScaleCodesByDescriptor[index]
        }
        return (result.coefficients, quantCodes, qScaleCodes)
    }

}
