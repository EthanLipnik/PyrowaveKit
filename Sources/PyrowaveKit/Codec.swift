import Foundation

public final class PyrowaveCodec: Sendable {
    private static let magic: [UInt8] = [0x50, 0x57, 0x4b, 0x53] // PWKS
    private static let densePlaneCoding: UInt8 = 0
    private static let sparsePlaneCoding: UInt8 = 1
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

        let threshold = try selectSparseThreshold(
            planes: encodedPlanes,
            frame: frame,
            configuration: configuration
        )
        let planePayloads = encodedPlanes.map { makePlanePayload($0, sparseThreshold: threshold, forceSparse: configuration.maximumEncodedBytes != nil) }

        var writer = BinaryWriter()
        writer.append(bytes: Self.magic)
        writer.append(UInt16(2))
        writer.append(UInt16(0))
        writer.append(UInt32(frame.width))
        writer.append(UInt32(frame.height))
        writer.append(frame.chroma.rawValue)
        writer.append(UInt8(encodedPlanes.count))
        writer.append(UInt16(configuration.decompositionLevels))
        writer.append(configuration.quantizationStep)
        writer.append(UInt16(Self.sparseBlockSize))
        writer.append(UInt16(threshold))

        for (plane, payload) in zip(encodedPlanes, planePayloads) {
            writer.append(UInt32(plane.visibleWidth))
            writer.append(UInt32(plane.visibleHeight))
            writer.append(UInt32(plane.paddedWidth))
            writer.append(UInt32(plane.paddedHeight))
            writer.append(UInt16(plane.levels))
            writer.append(payload.coding)
            writer.append(UInt8(0))
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
        guard version == 2 else {
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
        var coefficients: [Int16]
    }

    private struct PlanePayload {
        var coding: UInt8
        var coefficientCount: Int
        var blockCount: Int
        var data: Data
    }

    private struct SparseBlock {
        var blockIndex: Int
        var data: Data
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

    private struct RateControlHistogram {
        var coefficientCounts = Array(repeating: 0, count: Int(Int16.max) + 1)
        var nonEmptyBlockCounts = Array(repeating: 0, count: Int(Int16.max) + 1)

        func sparsePayloadSize(threshold: Int) -> Int {
            let start = min(max(threshold + 1, 0), coefficientCounts.count)
            var coefficients = 0
            var blocks = 0
            for magnitude in start..<coefficientCounts.count {
                coefficients += coefficientCounts[magnitude]
                blocks += nonEmptyBlockCounts[magnitude]
            }
            return coefficients * 4 + blocks * 8
        }
    }

    private func encodePlane(_ plane: Plane8, component: Int, chroma: ChromaSubsampling, configuration: CodecConfiguration) throws -> EncodedPlane {
        var padded = Wavelet.padPlane(plane)
        let requestedLevels = component == 0 || chroma == .yuv444 ?
            configuration.decompositionLevels :
            max(1, configuration.decompositionLevels - 1)
        let levels = Wavelet.usableLevels(width: padded.width, height: padded.height, requested: requestedLevels)
        padded.samples = try forwardWavelet(padded.samples, width: padded.width, height: padded.height, levels: levels)

        let coefficients = try quantize(padded.samples, quantizationStep: configuration.quantizationStep)

        return EncodedPlane(
            component: component,
            visibleWidth: plane.width,
            visibleHeight: plane.height,
            paddedWidth: padded.width,
            paddedHeight: padded.height,
            levels: levels,
            coefficients: coefficients
        )
    }

    private func decodePlane(reader: inout BinaryReader, quantizationStep: Float) throws -> Plane8 {
        let visibleWidth = Int(try reader.readUInt32())
        let visibleHeight = Int(try reader.readUInt32())
        let paddedWidth = Int(try reader.readUInt32())
        let paddedHeight = Int(try reader.readUInt32())
        let levels = Int(try reader.readUInt16())
        let coding = try reader.readUInt8()
        _ = try reader.readUInt8()
        let count = Int(try reader.readUInt32())
        let blockCount = Int(try reader.readUInt32())

        guard visibleWidth > 0, visibleHeight > 0,
              paddedWidth >= visibleWidth, paddedHeight >= visibleHeight,
              count == paddedWidth * paddedHeight else {
            throw PyrowaveError.invalidBitstream("bad plane dimensions")
        }

        let coefficients: [Int16]
        switch coding {
        case Self.densePlaneCoding:
            guard blockCount == 0 else {
                throw PyrowaveError.invalidBitstream("dense plane cannot declare sparse blocks")
            }
            coefficients = try reader.readInt16Array(count: count)
        case Self.sparsePlaneCoding:
            coefficients = try readSparseCoefficients(reader: &reader, count: count, width: paddedWidth, height: paddedHeight, levels: levels, blockCount: blockCount)
        default:
            throw PyrowaveError.invalidBitstream("unknown plane coding \(coding)")
        }

        var samples = try dequantize(coefficients, quantizationStep: quantizationStep)
        samples = try inverseWavelet(samples, width: paddedWidth, height: paddedHeight, levels: levels)
        return try Wavelet.cropPlane(samples, paddedWidth: paddedWidth, width: visibleWidth, height: visibleHeight)
    }

    private func selectSparseThreshold(
        planes: [EncodedPlane],
        frame: YUVFrame,
        configuration: CodecConfiguration
    ) throws -> Int {
        guard let maximumEncodedBytes = configuration.maximumEncodedBytes else {
            return 0
        }

        if estimateFrameSize(planes: planes, sparseThreshold: 0, forceSparse: true) <= maximumEncodedBytes {
            return 0
        }

        var low = 0
        var high = Int(Int16.max)
        while low < high {
            let mid = (low + high) / 2
            let size = estimateFrameSize(planes: planes, sparseThreshold: mid, forceSparse: true)
            if size <= maximumEncodedBytes {
                high = mid
            } else {
                low = mid + 1
            }
        }

        return low
    }

    private var frameHeaderSize: Int {
        4 + 2 + 2 + 4 + 4 + 1 + 1 + 2 + 4 + 2 + 2
    }

    private var planeHeaderSize: Int {
        4 + 4 + 4 + 4 + 2 + 1 + 1 + 4 + 4
    }

    private func makeRateControlHistogram(_ planes: [EncodedPlane]) -> RateControlHistogram {
        var histogram = RateControlHistogram()
        for plane in planes {
            for coefficient in plane.coefficients {
                let magnitude = abs(Int(coefficient))
                if magnitude > 0 {
                    histogram.coefficientCounts[magnitude] += 1
                }
            }

            for descriptor in planeBlockDescriptors(width: plane.paddedWidth, height: plane.paddedHeight, levels: plane.levels) {
                var maxMagnitude = 0
                for localY in 0..<descriptor.validHeight {
                    for localX in 0..<descriptor.validWidth {
                        let x = descriptor.originX + localX
                        let y = descriptor.originY + localY
                        maxMagnitude = max(maxMagnitude, abs(Int(plane.coefficients[y * plane.paddedWidth + x])))
                    }
                }
                if maxMagnitude > 0 {
                    histogram.nonEmptyBlockCounts[maxMagnitude] += 1
                }
            }
        }

        return histogram
    }

    private func estimateFrameSize(planes: [EncodedPlane], sparseThreshold: Int, forceSparse: Bool) -> Int {
        var size = frameHeaderSize
        for plane in planes {
            let payload = makePlanePayload(plane, sparseThreshold: sparseThreshold, forceSparse: forceSparse)
            size += planeHeaderSize + payload.data.count
        }
        return size
    }

    private func makePlanePayload(_ plane: EncodedPlane, sparseThreshold: Int, forceSparse: Bool) -> PlanePayload {
        let denseBytes = makeDensePayload(plane.coefficients)
        let sparse = makeSparsePayload(plane, threshold: sparseThreshold)

        if forceSparse || sparse.data.count < denseBytes.count {
            return sparse
        }

        return PlanePayload(
            coding: Self.densePlaneCoding,
            coefficientCount: plane.coefficients.count,
            blockCount: 0,
            data: denseBytes
        )
    }

    private func makeDensePayload(_ coefficients: [Int16]) -> Data {
        var writer = BinaryWriter()
        writer.append(contentsOf: coefficients)
        return writer.data
    }

    private func makeSparsePayload(_ plane: EncodedPlane, threshold: Int) -> PlanePayload {
        let blocks = sparseBlocks(plane, threshold: threshold)
        var writer = BinaryWriter()
        for block in blocks {
            writer.append(data: block.data)
        }

        return PlanePayload(
            coding: Self.sparsePlaneCoding,
            coefficientCount: plane.coefficients.count,
            blockCount: blocks.count,
            data: writer.data
        )
    }

    private func sparseBlocks(_ plane: EncodedPlane, threshold: Int) -> [SparseBlock] {
        let descriptors = planeBlockDescriptors(width: plane.paddedWidth, height: plane.paddedHeight, levels: plane.levels)
        var blocks = [SparseBlock]()
        blocks.reserveCapacity(descriptors.count)

        for descriptor in descriptors {
            if let data = try? PyrowaveCoefficientBlockCodec.encodeBlock(
                blockIndex: descriptor.blockIndex,
                coefficients: plane.coefficients,
                stride: plane.paddedWidth,
                originX: descriptor.originX,
                originY: descriptor.originY,
                validWidth: descriptor.validWidth,
                validHeight: descriptor.validHeight,
                threshold: threshold
            ) {
                blocks.append(SparseBlock(blockIndex: descriptor.blockIndex, data: data))
            }
        }

        return blocks
    }

    private func readSparseCoefficients(
        reader: inout BinaryReader,
        count: Int,
        width: Int,
        height: Int,
        levels: Int,
        blockCount: Int
    ) throws -> [Int16] {
        guard blockCount >= 0 else {
            throw PyrowaveError.invalidBitstream("negative sparse block count")
        }

        let blockSize = Self.sparseBlockSize
        let descriptors = planeBlockDescriptors(width: width, height: height, levels: levels)
        var coefficients = Array(repeating: Int16(0), count: count)
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
                let value = entry.value
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
                coefficients[y * width + x] = value
            }
        }

        return coefficients
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

    private func dequantize(_ coefficients: [Int16], quantizationStep: Float) throws -> [Float] {
        if let metalBackend, let accelerated = try? metalBackend.dequantize(coefficients, quantizationStep: quantizationStep) {
            return accelerated
        }

        return coefficients.map { Float($0) * quantizationStep }
    }
}
