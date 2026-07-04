import Foundation

public final class PyrowaveCodec: Sendable {
    private static let magic: [UInt8] = [0x50, 0x57, 0x4b, 0x53] // PWKS
    private static let sparseBlockSize = 32

    private let metalBackend: MetalPyrowaveBackend?

    public init(useMetalAcceleration: Bool = true) {
        if useMetalAcceleration {
            metalBackend = try? MetalPyrowaveBackend()
        } else {
            metalBackend = nil
        }
    }

    public func encode(_ frame: YUVFrame, configuration: CodecConfiguration = CodecConfiguration()) throws -> EncodedFrame {
        guard configuration.decompositionLevels > 0, configuration.quantizationStep > 0,
              configuration.maximumEncodedBytes == nil || configuration.maximumEncodedBytes! > 0 else {
            throw PyrowaveError.invalidDimensions
        }

        let encodedPlanes = [
            try encodePlane(frame.y, component: 0, chroma: frame.chroma, configuration: configuration),
            try encodePlane(frame.cb, component: 1, chroma: frame.chroma, configuration: configuration),
            try encodePlane(frame.cr, component: 2, chroma: frame.chroma, configuration: configuration)
        ]

        let rateControlPlan = try selectSparseRateControlPlan(
            planes: encodedPlanes,
            configuration: configuration
        )
        let planePayloads = try encodedPlanes.enumerated().map { index, plane in
            try makePlanePayload(
                plane,
                sparseThresholds: rateControlPlan.thresholdsByPlane?[index],
                defaultThreshold: rateControlPlan.defaultThreshold
            )
        }

        var writer = BinaryWriter()
        writer.append(bytes: Self.magic)
        writer.append(UInt16(3))
        writer.append(UInt16(0))
        writer.append(UInt32(frame.width))
        writer.append(UInt32(frame.height))
        writer.append(frame.chroma.rawValue)
        writer.append(UInt8(encodedPlanes.count))
        writer.append(UInt16(configuration.decompositionLevels))
        writer.append(configuration.quantizationStep)
        writer.append(UInt16(Self.sparseBlockSize))
        writer.append(UInt16(clamping: rateControlPlan.streamThresholdHint))

        for (plane, payload) in zip(encodedPlanes, planePayloads) {
            writer.append(UInt32(plane.visibleWidth))
            writer.append(UInt32(plane.visibleHeight))
            writer.append(UInt32(plane.paddedWidth))
            writer.append(UInt32(plane.paddedHeight))
            writer.append(UInt16(plane.levels))
            writer.append(UInt16(0))
            writer.append(UInt32(payload.coefficientCount))
            writer.append(UInt32(payload.blockCount))
            writer.append(data: payload.data)
        }

        if let maximumEncodedBytes = configuration.maximumEncodedBytes, writer.data.count > maximumEncodedBytes {
            throw PyrowaveError.processFailed("minimum sparse frame size \(writer.data.count) exceeds maximumEncodedBytes \(maximumEncodedBytes)")
        }

        return EncodedFrame(data: writer.data)
    }

    public func decode(_ frame: EncodedFrame) throws -> YUVFrame {
        var reader = BinaryReader(frame.data)
        let magic = try (0..<4).map { _ in try reader.readUInt8() }
        guard magic == Self.magic else {
            throw PyrowaveError.invalidBitstream("bad magic")
        }

        let version = try reader.readUInt16()
        guard version == 3 else {
            throw PyrowaveError.invalidBitstream("unsupported version \(version)")
        }
        _ = try reader.readUInt16()

        let width = Int(try reader.readUInt32())
        let height = Int(try reader.readUInt32())
        guard let chroma = ChromaSubsampling(rawValue: try reader.readUInt8()) else {
            throw PyrowaveError.invalidBitstream("bad chroma mode")
        }
        let planeCount = Int(try reader.readUInt8())
        guard planeCount == 3 else {
            throw PyrowaveError.invalidBitstream("expected three planes")
        }
        _ = try reader.readUInt16()
        let quantizationStep = try reader.readFloat()
        guard quantizationStep > 0 else {
            throw PyrowaveError.invalidBitstream("bad quantization step")
        }
        let blockSize = Int(try reader.readUInt16())
        guard blockSize == Self.sparseBlockSize else {
            throw PyrowaveError.invalidBitstream("unsupported sparse block size \(blockSize)")
        }
        _ = try reader.readUInt16()

        let y = try decodePlane(reader: &reader, quantizationStep: quantizationStep)
        let cb = try decodePlane(reader: &reader, quantizationStep: quantizationStep)
        let cr = try decodePlane(reader: &reader, quantizationStep: quantizationStep)

        guard reader.offset == frame.data.count else {
            throw PyrowaveError.invalidBitstream("trailing bytes")
        }

        return try YUVFrame(width: width, height: height, chroma: chroma, y: y, cb: cb, cr: cr)
    }

    private struct EncodedPlane {
        var component: Int
        var visibleWidth: Int
        var visibleHeight: Int
        var paddedWidth: Int
        var paddedHeight: Int
        var levels: Int
        var quantCode: UInt8
        var qScaleCode: UInt8
        var coefficients: [Int16]
    }

    private struct PlanePayload {
        var coefficientCount: Int
        var blockCount: Int
        var data: Data
    }

    private struct SparseBlock {
        var blockIndex: Int
        var data: Data
    }

    private struct SparseRateControlPlan {
        var defaultThreshold: Int
        var thresholdsByPlane: [[Int]]?

        var streamThresholdHint: Int {
            if let thresholdsByPlane {
                return thresholdsByPlane.flatMap { $0 }.max() ?? defaultThreshold
            }
            return defaultThreshold
        }
    }

    private struct PlaneBlockDescriptor {
        var blockIndex: Int
        var level: Int
        var band: Int
        var originX: Int
        var originY: Int
        var validWidth: Int
        var validHeight: Int
    }

    private func encodePlane(_ plane: Plane8, component: Int, chroma: ChromaSubsampling, configuration: CodecConfiguration) throws -> EncodedPlane {
        var padded = Wavelet.padPlane(plane)
        let requestedLevels = component == 0 || chroma == .yuv444 ?
            configuration.decompositionLevels :
            max(1, configuration.decompositionLevels - 1)
        let levels = Wavelet.usableLevels(width: padded.width, height: padded.height, requested: requestedLevels)
        padded.samples = try forwardWavelet(padded.samples, width: padded.width, height: padded.height, levels: levels)

        let coefficients = try quantize(padded.samples, quantizationStep: configuration.quantizationStep)
        let quantCode = try PyrowaveQuantization.encodeBlockScale(configuration.quantizationStep)

        return EncodedPlane(
            component: component,
            visibleWidth: plane.width,
            visibleHeight: plane.height,
            paddedWidth: padded.width,
            paddedHeight: padded.height,
            levels: levels,
            quantCode: quantCode,
            qScaleCode: PyrowaveQuantization.identityQScaleCode,
            coefficients: coefficients
        )
    }

    private func decodePlane(reader: inout BinaryReader, quantizationStep: Float) throws -> Plane8 {
        let visibleWidth = Int(try reader.readUInt32())
        let visibleHeight = Int(try reader.readUInt32())
        let paddedWidth = Int(try reader.readUInt32())
        let paddedHeight = Int(try reader.readUInt32())
        let levels = Int(try reader.readUInt16())
        _ = try reader.readUInt16()
        let count = Int(try reader.readUInt32())
        let blockCount = Int(try reader.readUInt32())

        guard visibleWidth > 0, visibleHeight > 0,
              paddedWidth >= visibleWidth, paddedHeight >= visibleHeight,
              count == paddedWidth * paddedHeight else {
            throw PyrowaveError.invalidBitstream("bad plane dimensions")
        }

        let samples = try readSparseSamples(reader: &reader, count: count, width: paddedWidth, height: paddedHeight, levels: levels, blockCount: blockCount)
        let reconstructed = try inverseWavelet(samples, width: paddedWidth, height: paddedHeight, levels: levels)
        return try Wavelet.cropPlane(reconstructed, paddedWidth: paddedWidth, width: visibleWidth, height: visibleHeight)
    }

    private func selectSparseRateControlPlan(
        planes: [EncodedPlane],
        configuration: CodecConfiguration
    ) throws -> SparseRateControlPlan {
        guard let maximumEncodedBytes = configuration.maximumEncodedBytes else {
            return SparseRateControlPlan(defaultThreshold: 0, thresholdsByPlane: nil)
        }

        let fixedHeaderBytes = frameHeaderSize + planes.count * planeHeaderSize
        let rateBlocksByPlane = try planes.map { try makeRateControlBlocks($0) }
        if let thresholdsByPlane = PyrowaveRateController.selectThresholds(
            blocksByPlane: rateBlocksByPlane,
            fixedHeaderBytes: fixedHeaderBytes,
            maximumEncodedBytes: maximumEncodedBytes
        ) {
            return SparseRateControlPlan(defaultThreshold: 0, thresholdsByPlane: thresholdsByPlane)
        }

        var low = 0
        var high = Int(Int16.max)
        while low < high {
            let mid = (low + high) / 2
            let size = try estimateFrameSize(planes: planes, sparseThreshold: mid)
            if size <= maximumEncodedBytes {
                high = mid
            } else {
                low = mid + 1
            }
        }

        return SparseRateControlPlan(defaultThreshold: low, thresholdsByPlane: nil)
    }

    private var frameHeaderSize: Int {
        4 + 2 + 2 + 4 + 4 + 1 + 1 + 2 + 4 + 2 + 2
    }

    private var planeHeaderSize: Int {
        4 + 4 + 4 + 4 + 2 + 1 + 1 + 4 + 4
    }

    private func estimateFrameSize(planes: [EncodedPlane], sparseThreshold: Int) throws -> Int {
        var size = frameHeaderSize
        for plane in planes {
            let payload = try makePlanePayload(plane, defaultThreshold: sparseThreshold)
            size += planeHeaderSize + payload.data.count
        }
        return size
    }

    private func makePlanePayload(
        _ plane: EncodedPlane,
        sparseThresholds: [Int]? = nil,
        defaultThreshold: Int
    ) throws -> PlanePayload {
        let blocks = try sparseBlocks(plane, thresholds: sparseThresholds, defaultThreshold: defaultThreshold)
        var writer = BinaryWriter()
        for block in blocks {
            writer.append(data: block.data)
        }

        return PlanePayload(
            coefficientCount: plane.coefficients.count,
            blockCount: blocks.count,
            data: writer.data
        )
    }

    private func sparseBlocks(_ plane: EncodedPlane, thresholds: [Int]?, defaultThreshold: Int) throws -> [SparseBlock] {
        let descriptors = planeBlockDescriptors(width: plane.paddedWidth, height: plane.paddedHeight, levels: plane.levels)
        if let thresholds, thresholds.count != descriptors.count {
            throw PyrowaveError.processFailed("sparse threshold count \(thresholds.count) does not match block count \(descriptors.count)")
        }

        var blocks = [SparseBlock]()
        blocks.reserveCapacity(descriptors.count)

        for (index, descriptor) in descriptors.enumerated() {
            let threshold = thresholds?[index] ?? defaultThreshold
            if let data = try PyrowaveCoefficientBlockCodec.encodeBlock(
                blockIndex: descriptor.blockIndex,
                coefficients: plane.coefficients,
                stride: plane.paddedWidth,
                originX: descriptor.originX,
                originY: descriptor.originY,
                validWidth: descriptor.validWidth,
                validHeight: descriptor.validHeight,
                threshold: threshold,
                quantCode: plane.quantCode,
                qScaleCode: plane.qScaleCode
            ) {
                blocks.append(SparseBlock(blockIndex: descriptor.blockIndex, data: data))
            }
        }

        return blocks
    }

    private func makeRateControlBlocks(_ plane: EncodedPlane) throws -> [PyrowaveRateControlBlock] {
        let descriptors = planeBlockDescriptors(width: plane.paddedWidth, height: plane.paddedHeight, levels: plane.levels)
        return try descriptors.map { descriptor in
            try PyrowaveRateController.makeBlock(
                blockIndex: descriptor.blockIndex,
                coefficients: plane.coefficients,
                stride: plane.paddedWidth,
                originX: descriptor.originX,
                originY: descriptor.originY,
                validWidth: descriptor.validWidth,
                validHeight: descriptor.validHeight,
                quantCode: plane.quantCode,
                qScaleCode: plane.qScaleCode
            )
        }
    }

    private func readSparseSamples(
        reader: inout BinaryReader,
        count: Int,
        width: Int,
        height: Int,
        levels: Int,
        blockCount: Int
    ) throws -> [Float] {
        guard blockCount >= 0 else {
            throw PyrowaveError.invalidBitstream("negative sparse block count")
        }

        let blockSize = Self.sparseBlockSize
        let descriptors = planeBlockDescriptors(width: width, height: height, levels: levels)
        var samples = Array(repeating: Float(0), count: count)
        var seenBlocks = Set<Int>()

        for _ in 0..<blockCount {
            let block = try PyrowaveCoefficientBlockCodec.decodeBlock(reader: &reader)
            let blockIndex = block.blockIndex

            guard blockIndex >= 0, blockIndex < descriptors.count, seenBlocks.insert(blockIndex).inserted else {
                throw PyrowaveError.invalidBitstream("bad sparse block index")
            }

            let descriptor = descriptors[blockIndex]
            for entry in block.coefficients {
                let localOffset = Int(entry.offset)
                let localX = localOffset % blockSize
                let localY = localOffset / blockSize
                let x = descriptor.originX + localX
                let y = descriptor.originY + localY
                guard localOffset < blockSize * blockSize,
                      localX < descriptor.validWidth,
                      localY < descriptor.validHeight,
                      x < width,
                      y < height else {
                    throw PyrowaveError.invalidBitstream("sparse coefficient out of range")
                }
                samples[y * width + x] = PyrowaveQuantization.dequantize(
                    coefficient: entry.value,
                    quantCode: block.quantCode,
                    qScaleCode: entry.qScaleCode
                )
            }
        }

        return samples
    }

    private func planeBlockDescriptors(width: Int, height: Int, levels: Int) -> [PlaneBlockDescriptor] {
        var descriptors = [PlaneBlockDescriptor]()
        let blockSize = Self.sparseBlockSize

        for level in stride(from: levels - 1, through: 0, by: -1) {
            let subbandWidth = width >> (level + 1)
            let subbandHeight = height >> (level + 1)
            let blockColumns = (subbandWidth + blockSize - 1) / blockSize
            let blockRows = (subbandHeight + blockSize - 1) / blockSize
            let firstBand = level == levels - 1 ? 0 : 1

            for band in firstBand..<PyrowaveBitstream.bandCount {
                let origin = planeBandOrigin(level: level, finalLevel: levels - 1, band: band, subbandWidth: subbandWidth, subbandHeight: subbandHeight)
                for blockY in 0..<blockRows {
                    for blockX in 0..<blockColumns {
                        descriptors.append(PlaneBlockDescriptor(
                            blockIndex: descriptors.count,
                            level: level,
                            band: band,
                            originX: origin.x + blockX * blockSize,
                            originY: origin.y + blockY * blockSize,
                            validWidth: min(blockSize, subbandWidth - blockX * blockSize),
                            validHeight: min(blockSize, subbandHeight - blockY * blockSize)
                        ))
                    }
                }
            }
        }

        return descriptors
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
        if let metalBackend, let accelerated = try? metalBackend.forwardWavelet(samples, width: width, height: height, levels: levels) {
            return accelerated
        }

        var transformed = samples
        Wavelet.forward2D(&transformed, width: width, height: height, levels: levels)
        return transformed
    }

    private func inverseWavelet(_ samples: [Float], width: Int, height: Int, levels: Int) throws -> [Float] {
        if let metalBackend, let accelerated = try? metalBackend.inverseWavelet(samples, width: width, height: height, levels: levels) {
            return accelerated
        }

        var transformed = samples
        Wavelet.inverse2D(&transformed, width: width, height: height, levels: levels)
        return transformed
    }

    private func quantize(_ samples: [Float], quantizationStep: Float) throws -> [Int16] {
        if let metalBackend, let accelerated = try? metalBackend.quantize(samples, quantizationStep: quantizationStep) {
            return accelerated
        }

        let invStep = 1.0 / quantizationStep
        return samples.map { sample -> Int16 in
            let quantized = Int((sample * invStep).rounded())
            return Int16(max(Int(Int16.min), min(Int(Int16.max), quantized)))
        }
    }

}
