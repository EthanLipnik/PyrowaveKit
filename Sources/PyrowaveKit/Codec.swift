import Foundation
import CoreVideo
import Metal

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

    public init() throws {
        codec = try PyrowaveCodec()
        expectedFrame = nil
    }

    public init(
        width: Int,
        height: Int,
        chroma: ChromaSubsampling,
        videoSignal: VideoSignalMetadata = .default
    ) throws {
        codec = try PyrowaveCodec()
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

    func decode(allowPartialFrame: Bool = false) throws -> YUVFrame {
        try codec.decode(try assembledFrame(allowPartialFrame: allowPartialFrame), allowPartialFrame: allowPartialFrame)
    }

    public func decodeToMetalTextures(
        allowPartialFrame: Bool = false,
        device: MTLDevice? = nil,
        usage: MTLTextureUsage = [.shaderRead, .shaderWrite]
    ) throws -> (y: MTLTexture, cb: MTLTexture, cr: MTLTexture) {
        try codec.decodeToMetalTextures(
            try assembledFrame(allowPartialFrame: allowPartialFrame),
            device: device,
            usage: usage
        )
    }

    public func decodeToCVPixelBuffer(
        allowPartialFrame: Bool = false,
        pixelFormat: OSType? = nil
    ) throws -> CVPixelBuffer {
        try codec.decodeToCVPixelBuffer(
            try assembledFrame(allowPartialFrame: allowPartialFrame),
            pixelFormat: pixelFormat
        )
    }

    private func assembledFrame(allowPartialFrame: Bool) throws -> EncodedFrame {
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

        decodedFrameForCurrentSequence = true
        return EncodedFrame(data: writer.data)
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

public final class PyrowaveCodec: @unchecked Sendable {
    private static let sparseBlockSize = 32

    private let metalBackend: MetalPyrowaveBackend
    let coreVideoTextureCache: CVMetalTextureCache
    private let sequenceCounter = SequenceCounter()

    public init() throws {
        metalBackend = try MetalPyrowaveBackend()
        var textureCache: CVMetalTextureCache?
        let status = CVMetalTextureCacheCreate(nil, nil, metalBackend.device, nil, &textureCache)
        guard status == kCVReturnSuccess, let textureCache else {
            throw PyrowaveError.processFailed("failed to create CoreVideo Metal texture cache")
        }
        coreVideoTextureCache = textureCache
    }

    func encode(_ frame: YUVFrame, configuration: CodecConfiguration = CodecConfiguration()) throws -> EncodedFrame {
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

        let layout = try PyrowaveBlockLayout(width: yTexture.width, height: yTexture.height, chroma: chroma)
        let encodedPlanes = [
            try encodeTexturePlane(yTexture, channel: 0, component: 0, frameWidth: yTexture.width, frameHeight: yTexture.height, chroma: chroma, layout: layout, configuration: configuration),
            try encodeTexturePlane(cbTexture, channel: 0, component: 1, frameWidth: yTexture.width, frameHeight: yTexture.height, chroma: chroma, layout: layout, configuration: configuration),
            try encodeTexturePlane(crTexture, channel: 0, component: 2, frameWidth: yTexture.width, frameHeight: yTexture.height, chroma: chroma, layout: layout, configuration: configuration)
        ]
        return try encodeFrame(
            width: yTexture.width,
            height: yTexture.height,
            chroma: chroma,
            videoSignal: videoSignal,
            encodedPlanes: encodedPlanes,
            layout: layout,
            configuration: configuration
        )
    }

    public func encode(
        yTexture: MTLTexture,
        cbCrTexture: MTLTexture,
        configuration: CodecConfiguration = CodecConfiguration(),
        videoSignal: VideoSignalMetadata = .default
    ) throws -> EncodedFrame {
        guard configuration.decompositionLevels == PyrowaveBitstream.decompositionLevels,
              configuration.quantizationStep > 0,
              configuration.maximumEncodedBytes == nil || configuration.maximumEncodedBytes! > 0 else {
            throw PyrowaveError.invalidDimensions
        }
        guard yTexture.pixelFormat == .r8Unorm,
              cbCrTexture.pixelFormat == .rg8Unorm,
              yTexture.width > 0,
              yTexture.height > 0,
              cbCrTexture.width == yTexture.width / 2,
              cbCrTexture.height == yTexture.height / 2 else {
            throw PyrowaveError.unsupportedFormat("Metal NV12 encode expects r8Unorm luma and rg8Unorm chroma planes")
        }

        let layout = try PyrowaveBlockLayout(width: yTexture.width, height: yTexture.height, chroma: .yuv420)
        let encodedPlanes = [
            try encodeTexturePlane(yTexture, channel: 0, component: 0, frameWidth: yTexture.width, frameHeight: yTexture.height, chroma: .yuv420, layout: layout, configuration: configuration),
            try encodeTexturePlane(cbCrTexture, channel: 0, component: 1, frameWidth: yTexture.width, frameHeight: yTexture.height, chroma: .yuv420, layout: layout, configuration: configuration),
            try encodeTexturePlane(cbCrTexture, channel: 1, component: 2, frameWidth: yTexture.width, frameHeight: yTexture.height, chroma: .yuv420, layout: layout, configuration: configuration)
        ]
        return try encodeFrame(
            width: yTexture.width,
            height: yTexture.height,
            chroma: .yuv420,
            videoSignal: videoSignal,
            encodedPlanes: encodedPlanes,
            layout: layout,
            configuration: configuration
        )
    }

    private func encodeFrame(
        width: Int,
        height: Int,
        chroma: ChromaSubsampling,
        videoSignal: VideoSignalMetadata,
        encodedPlanes: [EncodedPlane],
        layout: PyrowaveBlockLayout,
        configuration: CodecConfiguration
    ) throws -> EncodedFrame {
        let sequenceNumber = sequenceCounter.next()
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
            width: width,
            height: height,
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

    func decode(_ frame: EncodedFrame) throws -> YUVFrame {
        try decode(frame, allowPartialFrame: false)
    }

    fileprivate func decode(_ frame: EncodedFrame, allowPartialFrame: Bool) throws -> YUVFrame {
        try makeYUVFrame(from: decodePlanes(frame, allowPartialFrame: allowPartialFrame))
    }

    public func decodeToMetalTextures(
        _ frame: EncodedFrame,
        device: MTLDevice? = nil,
        usage: MTLTextureUsage = [.shaderRead, .shaderWrite]
    ) throws -> (y: MTLTexture, cb: MTLTexture, cr: MTLTexture) {
        let decoded = try decodeGPUPlanes(frame, allowPartialFrame: false)
        let outputDevice = device ?? metalBackend.device
        guard outputDevice.registryID == metalBackend.device.registryID else {
            throw PyrowaveError.invalidDimensions
        }
        var textureUsage = usage
        textureUsage.insert(.shaderWrite)
        let yTexture = try makeOutputTexture(width: decoded.planes[0].visibleWidth, height: decoded.planes[0].visibleHeight, device: outputDevice, usage: textureUsage)
        let cbTexture = try makeOutputTexture(width: decoded.planes[1].visibleWidth, height: decoded.planes[1].visibleHeight, device: outputDevice, usage: textureUsage)
        let crTexture = try makeOutputTexture(width: decoded.planes[2].visibleWidth, height: decoded.planes[2].visibleHeight, device: outputDevice, usage: textureUsage)
        try finishGPUDecodedPlane(decoded.planes[0], to: yTexture)
        try finishGPUDecodedPlane(decoded.planes[1], to: cbTexture)
        try finishGPUDecodedPlane(decoded.planes[2], to: crTexture)
        return (yTexture, cbTexture, crTexture)
    }

    func decodeToNV12Textures(_ frame: EncodedFrame, yTexture: MTLTexture, cbCrTexture: MTLTexture) throws {
        let decoded = try decodeGPUPlanes(frame, allowPartialFrame: false)
        guard decoded.sequence.chroma == .yuv420,
              yTexture.width == decoded.sequence.width,
              yTexture.height == decoded.sequence.height,
              cbCrTexture.width == decoded.sequence.width / 2,
              cbCrTexture.height == decoded.sequence.height / 2 else {
            throw PyrowaveError.invalidDimensions
        }

        let y = try inverseWaveletBuffer(decoded.planes[0])
        let cb = try inverseWaveletBuffer(decoded.planes[1])
        let cr = try inverseWaveletBuffer(decoded.planes[2])
        try metalBackend.cropPlanesToNV12Textures(
            yBuffer: y,
            ySampleCount: decoded.planes[0].sampleCount,
            yPaddedWidth: decoded.planes[0].paddedWidth,
            cbBuffer: cb,
            cbSampleCount: decoded.planes[1].sampleCount,
            crBuffer: cr,
            crSampleCount: decoded.planes[2].sampleCount,
            chromaPaddedWidth: decoded.planes[1].paddedWidth,
            width: decoded.sequence.width,
            height: decoded.sequence.height,
            yTexture: yTexture,
            cbCrTexture: cbCrTexture
        )
    }

    private func decodePlanes(_ frame: EncodedFrame, allowPartialFrame: Bool) throws -> DecodedFramePlanes {
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

        return DecodedFramePlanes(sequence: sequence, planes: decodedPlanes)
    }

    private func decodeGPUPlanes(_ frame: EncodedFrame, allowPartialFrame: Bool) throws -> GPUDecodedFramePlanes {
        var reader = BinaryReader(frame.data)
        let sequence = try PyrowaveSequenceHeader(reader: &reader)
        let layout = try PyrowaveBlockLayout(width: sequence.width, height: sequence.height, chroma: sequence.chroma)
        var decodedPlanes = try (0..<PyrowaveBitstream.componentCount).map { component in
            try makeGPUDecodedPlane(component: component, width: sequence.width, height: sequence.height, chroma: sequence.chroma, layout: layout)
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
            try applySparseBlocksToBuffer(pendingSparseBlocks[index], decodedPlane: &decodedPlanes[index])
        }

        return GPUDecodedFramePlanes(sequence: sequence, planes: decodedPlanes)
    }

    private func makeYUVFrame(from decoded: DecodedFramePlanes) throws -> YUVFrame {
        let y = try finishDecodedPlane(decoded.planes[0])
        let cb = try finishDecodedPlane(decoded.planes[1])
        let cr = try finishDecodedPlane(decoded.planes[2])

        return try YUVFrame(
            width: decoded.sequence.width,
            height: decoded.sequence.height,
            chroma: decoded.sequence.chroma,
            y: y,
            cb: cb,
            cr: cr,
            videoSignal: decoded.sequence.videoSignal
        )
    }

    private func makeOutputTexture(width: Int, height: Int, device: MTLDevice, usage: MTLTextureUsage) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = usage
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw PyrowaveError.processFailed("failed to allocate Metal decode texture")
        }
        return texture
    }

    private struct EncodedPlane {
        var component: Int
        var paddedWidth: Int
        var paddedHeight: Int
        var levels: Int
        var quantCodesByBlockIndex: [Int: UInt8]
        var qScaleCodesByBlockIndex: [Int: [UInt8]]
        var coefficients: [Int16]
        var coefficientBuffer: MTLBuffer?
        var coefficientCount: Int
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

    private struct DecodedFramePlanes {
        var sequence: PyrowaveSequenceHeader
        var planes: [DecodedPlane]
    }

    private struct GPUDecodedPlane {
        var visibleWidth: Int
        var visibleHeight: Int
        var paddedWidth: Int
        var paddedHeight: Int
        var levels: Int
        var sampleCount: Int
        var samples: MTLBuffer
        var descriptorsByBlockIndex: [Int: PlaneBlockDescriptor]
    }

    private struct GPUDecodedFramePlanes {
        var sequence: PyrowaveSequenceHeader
        var planes: [GPUDecodedPlane]
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

    private func encodeTexturePlane(
        _ texture: MTLTexture,
        channel: Int,
        component: Int,
        frameWidth: Int,
        frameHeight: Int,
        chroma: ChromaSubsampling,
        layout: PyrowaveBlockLayout,
        configuration: CodecConfiguration
    ) throws -> EncodedPlane {
        let geometry = planeGeometry(component: component, frameWidth: frameWidth, frameHeight: frameHeight, chroma: chroma, requestedLevels: configuration.decompositionLevels)
        guard texture.width == geometry.visibleWidth,
              texture.height == geometry.visibleHeight else {
            throw PyrowaveError.invalidDimensions
        }
        return try encodePaddedPlaneBuffer(
            samples: try metalBackend.padTexturePlaneBuffer(texture, channel: channel, paddedWidth: geometry.paddedWidth, paddedHeight: geometry.paddedHeight),
            sampleCount: geometry.paddedWidth * geometry.paddedHeight,
            paddedWidth: geometry.paddedWidth,
            paddedHeight: geometry.paddedHeight,
            requestedLevels: geometry.requestedLevels,
            component: component,
            layout: layout,
            configuration: configuration
        )
    }

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
            coefficients: quantized.coefficients,
            coefficientBuffer: nil,
            coefficientCount: quantized.coefficients.count
        )
    }

    private func encodePaddedPlaneBuffer(
        samples: MTLBuffer,
        sampleCount: Int,
        paddedWidth: Int,
        paddedHeight: Int,
        requestedLevels: Int,
        component: Int,
        layout: PyrowaveBlockLayout,
        configuration: CodecConfiguration
    ) throws -> EncodedPlane {
        let levels = Wavelet.usableLevels(width: paddedWidth, height: paddedHeight, requested: requestedLevels)
        let transformed = try metalBackend.forwardWaveletBuffer(samples, sampleCount: sampleCount, width: paddedWidth, height: paddedHeight, levels: levels)

        let descriptors = planeBlockDescriptors(
            component: component,
            paddedWidth: paddedWidth,
            paddedHeight: paddedHeight,
            levels: levels,
            layout: layout
        )
        let quantized = try quantizeResidentBuffer(transformed, sampleCount: sampleCount, stride: paddedWidth, descriptors: descriptors, component: component, configuration: configuration)

        return EncodedPlane(
            component: component,
            paddedWidth: paddedWidth,
            paddedHeight: paddedHeight,
            levels: levels,
            quantCodesByBlockIndex: quantized.quantCodesByBlockIndex,
            qScaleCodesByBlockIndex: quantized.qScaleCodesByBlockIndex,
            coefficients: [],
            coefficientBuffer: quantized.coefficientBuffer,
            coefficientCount: quantized.coefficientCount
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
            bucketIndicesByPlane: metalBucketData.indicesByPlane,
            cumulativeBucketSavings: metalBucketData.cumulativeSavings
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

    private func metalRateControlBucketData(blocksByPlane: [[PyrowaveRateControlBlock]]) throws -> MetalRateControlBucketData {
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

        let selectedPacketByteCosts: [[Int]]?
        if let packetByteCosts {
            selectedPacketByteCosts = packetByteCosts
        } else {
            selectedPacketByteCosts = try metalSparsePacketByteCosts(
                plane: plane,
                descriptors: descriptors
            )
        }
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

        let packets = try encodeSparsePackets(
            plane,
            descriptors: packetDescriptors,
            qScaleCodes: packetQScaleCodes,
            backend: backend
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
    ) throws -> [[Int]] {
        let packetCostDescriptors = metalPacketByteCostDescriptors(
            descriptors: descriptors,
            stride: plane.paddedWidth
        )
        let costs = try packetByteCosts(plane, descriptors: packetCostDescriptors, backend: metalBackend)
        guard costs.count == descriptors.count else {
            throw PyrowaveError.processFailed("Metal packet byte-cost returned \(costs.count) block costs for \(descriptors.count) descriptors")
        }
        return costs
    }

    private func makeRateControlBlocks(_ plane: EncodedPlane, layout: PyrowaveBlockLayout) throws -> [PyrowaveRateControlBlock] {
        let descriptors = planeBlockDescriptors(plane: plane, layout: layout)
        return try makeRateControlBlocksWithMetal(plane, descriptors: descriptors, layout: layout, backend: metalBackend)
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

        let tileStats = try rateControlTileStats(plane, descriptors: metalDescriptors, backend: backend)
        guard tileStats.count == metalDescriptors.count else {
            throw PyrowaveError.processFailed("Metal rate-control returned \(tileStats.count) tile stats for \(metalDescriptors.count) descriptors")
        }
        let packetCostDescriptors = metalPacketByteCostDescriptors(descriptors: descriptors, stride: plane.paddedWidth)
        let packetByteCosts = try packetByteCosts(plane, descriptors: packetCostDescriptors, backend: backend)
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

    private func rateControlTileStats(
        _ plane: EncodedPlane,
        descriptors: [MetalRateControlStatsDescriptor],
        backend: MetalPyrowaveBackend
    ) throws -> [MetalRateControlTileStats] {
        if let coefficientBuffer = plane.coefficientBuffer {
            return try backend.rateControlTileStats(
                coefficientBuffer: coefficientBuffer,
                coefficientCount: plane.coefficientCount,
                descriptors: descriptors
            )
        }
        return try backend.rateControlTileStats(coefficients: plane.coefficients, descriptors: descriptors)
    }

    private func packetByteCosts(
        _ plane: EncodedPlane,
        descriptors: [MetalPacketByteCostDescriptor],
        backend: MetalPyrowaveBackend
    ) throws -> [[Int]] {
        if let coefficientBuffer = plane.coefficientBuffer {
            return try backend.packetByteCosts(
                coefficientBuffer: coefficientBuffer,
                coefficientCount: plane.coefficientCount,
                descriptors: descriptors
            )
        }
        return try backend.packetByteCosts(coefficients: plane.coefficients, descriptors: descriptors)
    }

    private func encodeSparsePackets(
        _ plane: EncodedPlane,
        descriptors: [MetalSparsePacketEncodeDescriptor],
        qScaleCodes: [[UInt8]],
        backend: MetalPyrowaveBackend
    ) throws -> [Data?] {
        if let coefficientBuffer = plane.coefficientBuffer {
            return try backend.encodeSparsePackets(
                coefficientBuffer: coefficientBuffer,
                coefficientCount: plane.coefficientCount,
                descriptors: descriptors,
                qScaleCodes: qScaleCodes
            )
        }
        return try backend.encodeSparsePackets(
            coefficients: plane.coefficients,
            descriptors: descriptors,
            qScaleCodes: qScaleCodes
        )
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
            coefficients: [],
            coefficientBuffer: nil,
            coefficientCount: 0
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

    private func makeGPUDecodedPlane(component: Int, width: Int, height: Int, chroma: ChromaSubsampling, layout: PyrowaveBlockLayout) throws -> GPUDecodedPlane {
        let geometry = planeGeometry(component: component, frameWidth: width, frameHeight: height, chroma: chroma, requestedLevels: PyrowaveBitstream.decompositionLevels)
        let levels = Wavelet.usableLevels(width: geometry.paddedWidth, height: geometry.paddedHeight, requested: geometry.requestedLevels)
        let scratchPlane = EncodedPlane(
            component: component,
            paddedWidth: geometry.paddedWidth,
            paddedHeight: geometry.paddedHeight,
            levels: levels,
            quantCodesByBlockIndex: [:],
            qScaleCodesByBlockIndex: [:],
            coefficients: [],
            coefficientBuffer: nil,
            coefficientCount: 0
        )
        let descriptors = planeBlockDescriptors(plane: scratchPlane, layout: layout)
        let sampleCount = geometry.paddedWidth * geometry.paddedHeight
        guard let samples = metalBackend.device.makeBuffer(length: sampleCount * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate GPU decoded plane")
        }
        memset(samples.contents(), 0, sampleCount * MemoryLayout<Float>.stride)
        return GPUDecodedPlane(
            visibleWidth: geometry.visibleWidth,
            visibleHeight: geometry.visibleHeight,
            paddedWidth: geometry.paddedWidth,
            paddedHeight: geometry.paddedHeight,
            levels: levels,
            sampleCount: sampleCount,
            samples: samples,
            descriptorsByBlockIndex: Dictionary(uniqueKeysWithValues: descriptors.map { ($0.blockIndex, $0) })
        )
    }

    private func applySparseBlocks(
        _ blocks: [PendingSparseBlock],
        decodedPlane: inout DecodedPlane
    ) throws {
        let entries = try metalSparseCoefficientEntries(blocks, decodedPlane: decodedPlane)
        decodedPlane.samples = try metalBackend.applySparseCoefficients(
            sampleCount: decodedPlane.paddedWidth * decodedPlane.paddedHeight,
            entries: entries
        )
    }

    private func applySparseBlocksToBuffer(
        _ blocks: [PendingSparseBlock],
        decodedPlane: inout GPUDecodedPlane
    ) throws {
        let entries = try metalSparseCoefficientEntries(blocks, paddedWidth: decodedPlane.paddedWidth, paddedHeight: decodedPlane.paddedHeight)
        decodedPlane.samples = try metalBackend.applySparseCoefficientBuffer(
            sampleCount: decodedPlane.sampleCount,
            entries: entries
        )
    }

    private func metalSparseCoefficientEntries(
        _ blocks: [PendingSparseBlock],
        decodedPlane: DecodedPlane
    ) throws -> [MetalSparseCoefficientEntry] {
        try metalSparseCoefficientEntries(blocks, paddedWidth: decodedPlane.paddedWidth, paddedHeight: decodedPlane.paddedHeight)
    }

    private func metalSparseCoefficientEntries(
        _ blocks: [PendingSparseBlock],
        paddedWidth: Int,
        paddedHeight: Int
    ) throws -> [MetalSparseCoefficientEntry] {
        let capacity = blocks.reduce(0) { $0 + $1.block.coefficients.count }
        var entries = [MetalSparseCoefficientEntry]()
        entries.reserveCapacity(capacity)

        for pending in blocks {
            for entry in pending.block.coefficients {
                let destinationOffset = try sparseDestinationOffset(
                    entryOffset: entry.offset,
                    descriptor: pending.descriptor,
                    paddedWidth: paddedWidth,
                    paddedHeight: paddedHeight
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
        try sparseDestinationOffset(
            entryOffset: entryOffset,
            descriptor: descriptor,
            paddedWidth: decodedPlane.paddedWidth,
            paddedHeight: decodedPlane.paddedHeight
        )
    }

    private func sparseDestinationOffset(
        entryOffset: UInt16,
        descriptor: PlaneBlockDescriptor,
        paddedWidth: Int,
        paddedHeight: Int
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
              x < paddedWidth,
              y < paddedHeight else {
            throw PyrowaveError.invalidBitstream("sparse coefficient out of range")
        }
        return y * paddedWidth + x
    }

    private func finishDecodedPlane(_ plane: DecodedPlane) throws -> Plane8 {
        let reconstructed = try inverseWavelet(plane.samples, width: plane.paddedWidth, height: plane.paddedHeight, levels: plane.levels)
        return try cropPlane(reconstructed, paddedWidth: plane.paddedWidth, width: plane.visibleWidth, height: plane.visibleHeight)
    }

    private func finishDecodedPlane(_ plane: DecodedPlane, to texture: MTLTexture) throws {
        let reconstructed = try inverseWavelet(plane.samples, width: plane.paddedWidth, height: plane.paddedHeight, levels: plane.levels)
        try metalBackend.cropPlaneToTexture(
            reconstructed,
            paddedWidth: plane.paddedWidth,
            width: plane.visibleWidth,
            height: plane.visibleHeight,
            texture: texture
        )
    }

    private func finishGPUDecodedPlane(_ plane: GPUDecodedPlane, to texture: MTLTexture) throws {
        let reconstructed = try inverseWaveletBuffer(plane)
        try metalBackend.cropPlaneToTexture(
            reconstructed,
            sampleCount: plane.sampleCount,
            paddedWidth: plane.paddedWidth,
            width: plane.visibleWidth,
            height: plane.visibleHeight,
            texture: texture
        )
    }

    private func inverseWaveletBuffer(_ plane: GPUDecodedPlane) throws -> MTLBuffer {
        try metalBackend.inverseWaveletBuffer(
            plane.samples,
            sampleCount: plane.sampleCount,
            width: plane.paddedWidth,
            height: plane.paddedHeight,
            levels: plane.levels
        )
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
        try metalBackend.padPlane(plane, paddedWidth: paddedWidth, paddedHeight: paddedHeight)
    }

    private func cropPlane(_ samples: [Float], paddedWidth: Int, width: Int, height: Int) throws -> Plane8 {
        try metalBackend.cropPlane(samples, paddedWidth: paddedWidth, width: width, height: height)
    }

    private func forwardWavelet(_ samples: [Float], width: Int, height: Int, levels: Int) throws -> [Float] {
        try metalBackend.forwardWavelet(samples, width: width, height: height, levels: levels)
    }

    private func inverseWavelet(_ samples: [Float], width: Int, height: Int, levels: Int) throws -> [Float] {
        try metalBackend.inverseWavelet(samples, width: width, height: height, levels: levels)
    }

    private func quantize(
        _ samples: [Float],
        stride: Int,
        descriptors: [PlaneBlockDescriptor],
        component: Int,
        configuration: CodecConfiguration
    ) throws -> (coefficients: [Int16], quantCodesByBlockIndex: [Int: UInt8], qScaleCodesByBlockIndex: [Int: [UInt8]]) {
        try quantizeWithMetal(
            samples,
            stride: stride,
            descriptors: descriptors,
            component: component,
            configuration: configuration,
            backend: metalBackend
        )
    }

    private func quantizeBuffer(
        _ samples: MTLBuffer,
        sampleCount: Int,
        stride: Int,
        descriptors: [PlaneBlockDescriptor],
        component: Int,
        configuration: CodecConfiguration
    ) throws -> (coefficients: [Int16], quantCodesByBlockIndex: [Int: UInt8], qScaleCodesByBlockIndex: [Int: [UInt8]]) {
        try quantizeWithMetal(
            samples,
            sampleCount: sampleCount,
            stride: stride,
            descriptors: descriptors,
            component: component,
            configuration: configuration,
            backend: metalBackend
        )
    }

    private func quantizeResidentBuffer(
        _ samples: MTLBuffer,
        sampleCount: Int,
        stride: Int,
        descriptors: [PlaneBlockDescriptor],
        component: Int,
        configuration: CodecConfiguration
    ) throws -> (coefficientBuffer: MTLBuffer, coefficientCount: Int, quantCodesByBlockIndex: [Int: UInt8], qScaleCodesByBlockIndex: [Int: [UInt8]]) {
        try quantizeWithMetalResident(
            samples,
            sampleCount: sampleCount,
            stride: stride,
            descriptors: descriptors,
            component: component,
            configuration: configuration,
            backend: metalBackend
        )
    }

    private func quantizeWithMetal(
        _ samples: [Float],
        stride: Int,
        descriptors: [PlaneBlockDescriptor],
        component: Int,
        configuration: CodecConfiguration,
        backend: MetalPyrowaveBackend
    ) throws -> (coefficients: [Int16], quantCodesByBlockIndex: [Int: UInt8], qScaleCodesByBlockIndex: [Int: [UInt8]]) {
        let input = try makeQuantizationDescriptors(
            stride: stride,
            descriptors: descriptors,
            component: component,
            configuration: configuration
        )
        let result = try backend.quantizePlane(samples, stride: stride, descriptors: input.metalDescriptors)
        return try finishQuantizationResult(result, descriptors: descriptors, quantCodes: input.quantCodes)
    }

    private func quantizeWithMetal(
        _ samples: MTLBuffer,
        sampleCount: Int,
        stride: Int,
        descriptors: [PlaneBlockDescriptor],
        component: Int,
        configuration: CodecConfiguration,
        backend: MetalPyrowaveBackend
    ) throws -> (coefficients: [Int16], quantCodesByBlockIndex: [Int: UInt8], qScaleCodesByBlockIndex: [Int: [UInt8]]) {
        let input = try makeQuantizationDescriptors(
            stride: stride,
            descriptors: descriptors,
            component: component,
            configuration: configuration
        )
        let result = try backend.quantizePlaneBuffer(samples, sampleCount: sampleCount, stride: stride, descriptors: input.metalDescriptors)
        return try finishQuantizationResult(result, descriptors: descriptors, quantCodes: input.quantCodes)
    }

    private func quantizeWithMetalResident(
        _ samples: MTLBuffer,
        sampleCount: Int,
        stride: Int,
        descriptors: [PlaneBlockDescriptor],
        component: Int,
        configuration: CodecConfiguration,
        backend: MetalPyrowaveBackend
    ) throws -> (coefficientBuffer: MTLBuffer, coefficientCount: Int, quantCodesByBlockIndex: [Int: UInt8], qScaleCodesByBlockIndex: [Int: [UInt8]]) {
        let input = try makeQuantizationDescriptors(
            stride: stride,
            descriptors: descriptors,
            component: component,
            configuration: configuration
        )
        let result = try backend.quantizePlaneBufferResult(samples, sampleCount: sampleCount, stride: stride, descriptors: input.metalDescriptors)
        return try finishQuantizationBufferResult(result, descriptors: descriptors, quantCodes: input.quantCodes)
    }

    private func makeQuantizationDescriptors(
        stride: Int,
        descriptors: [PlaneBlockDescriptor],
        component: Int,
        configuration: CodecConfiguration
    ) throws -> (metalDescriptors: [MetalPlaneQuantizationDescriptor], quantCodes: [Int: UInt8]) {
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

        return (metalDescriptors, quantCodes)
    }

    private func finishQuantizationResult(
        _ result: MetalPlaneQuantizationResult,
        descriptors: [PlaneBlockDescriptor],
        quantCodes: [Int: UInt8]
    ) throws -> (coefficients: [Int16], quantCodesByBlockIndex: [Int: UInt8], qScaleCodesByBlockIndex: [Int: [UInt8]]) {
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

    private func finishQuantizationBufferResult(
        _ result: MetalPlaneQuantizationBufferResult,
        descriptors: [PlaneBlockDescriptor],
        quantCodes: [Int: UInt8]
    ) throws -> (coefficientBuffer: MTLBuffer, coefficientCount: Int, quantCodesByBlockIndex: [Int: UInt8], qScaleCodesByBlockIndex: [Int: [UInt8]]) {
        guard result.qScaleCodesByDescriptor.count == descriptors.count else {
            throw PyrowaveError.processFailed("Metal quantization returned \(result.qScaleCodesByDescriptor.count) q-scale rows for \(descriptors.count) descriptors")
        }

        var qScaleCodes = [Int: [UInt8]]()
        qScaleCodes.reserveCapacity(descriptors.count)
        for (index, descriptor) in descriptors.enumerated() {
            qScaleCodes[descriptor.blockIndex] = result.qScaleCodesByDescriptor[index]
        }
        return (result.coefficientBuffer, result.coefficientCount, quantCodes, qScaleCodes)
    }
}
