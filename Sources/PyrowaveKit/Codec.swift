import Foundation
import CoreVideo
import Metal

public extension EncodedFrame {
    var descriptor: PyrowaveEncodedFrameDescriptor {
        get throws {
            var reader = BinaryReader(data)
            let header = try PyrowaveSequenceHeader(reader: &reader)
            return PyrowaveEncodedFrameDescriptor(
                width: header.width,
                height: header.height,
                chroma: header.chroma,
                videoSignal: header.videoSignal
            )
        }
    }

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

        let packetBytes = blockPackets.values.reduce(0) { $0 + $1.count }
        var writer = BinaryWriter(capacity: 8 + packetBytes)
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

public final class PyrowaveGPUFrame: @unchecked Sendable {
    public let width: Int
    public let height: Int
    public let chroma: ChromaSubsampling
    public let videoSignal: VideoSignalMetadata
    public let sequenceNumber: UInt8

    let planes: [PyrowaveGPUFramePlane]

    init(
        width: Int,
        height: Int,
        chroma: ChromaSubsampling,
        videoSignal: VideoSignalMetadata,
        sequenceNumber: UInt8,
        planes: [PyrowaveGPUFramePlane]
    ) {
        self.width = width
        self.height = height
        self.chroma = chroma
        self.videoSignal = videoSignal
        self.sequenceNumber = sequenceNumber
        self.planes = planes
    }

    public var estimatedPacketCapacityBytes: Int {
        planes.reduce(0) { $0 + $1.encoded.descriptorCount * $1.encoded.maxPacketBytes }
    }

    public var selectedQuantLevelsByPlane: [[Int]] {
        planes.map { plane in
            guard plane.selectedQuantLevelCount > 0 else {
                return []
            }
            let pointer = plane.selectedQuantLevelBuffer.contents().bindMemory(to: UInt32.self, capacity: plane.selectedQuantLevelCount)
            return Array(UnsafeBufferPointer(start: pointer, count: plane.selectedQuantLevelCount)).map(Int.init)
        }
    }

    public func encodedByteCountForInspection() -> Int {
        planes.reduce(0) { total, plane in
            guard plane.encoded.descriptorCount > 0 else {
                return total
            }
            let pointer = plane.encoded.sizeBuffer.contents().bindMemory(to: UInt32.self, capacity: plane.encoded.descriptorCount)
            return total + (0..<plane.encoded.descriptorCount).reduce(0) { $0 + Int(pointer[$1]) }
        }
    }
}

struct PyrowaveGPUFramePlane {
    var component: Int
    var paddedWidth: Int
    var paddedHeight: Int
    var levels: Int
    var sampleCount: Int
    var encoded: MetalSparsePacketEncodedPlane
    var decodeDescriptorCount: Int
    var decodeDescriptorBuffer: MTLBuffer
    var blockIndexCount: Int
    var blockIndexBuffer: MTLBuffer
    var selectedQuantLevelCount: Int
    var selectedQuantLevelBuffer: MTLBuffer

    func blockIndicesForInspection() -> [Int] {
        guard blockIndexCount > 0 else {
            return []
        }
        let pointer = blockIndexBuffer.contents().bindMemory(to: UInt32.self, capacity: blockIndexCount)
        return Array(UnsafeBufferPointer(start: pointer, count: blockIndexCount)).map(Int.init)
    }
}

struct PyrowaveContiguousNV12DecodeMetrics: Equatable, Sendable {
    let packetParseMilliseconds: Double
    let metalDecodeMilliseconds: Double
}

public final class PyrowaveCodec: @unchecked Sendable {
    private static let sparseBlockSize = 32

    private let metalBackend: MetalPyrowaveBackend
    private let gpuDecodedPlanePlaceholder: MTLBuffer
    let coreVideoTextureCache: CVMetalTextureCache
    private let sequenceCounter = SequenceCounter()
    private let geometryCacheLock = NSLock()
    private var layoutCache = [LayoutCacheKey: PyrowaveBlockLayout]()
    private var planeDescriptorCache = [PlaneDescriptorCacheKey: [PlaneBlockDescriptor]]()
    private var quantizationDescriptorCache = [QuantizationDescriptorCacheKey: QuantizationDescriptorCacheEntry]()
    private var sparsePacketEncodeDescriptorCache = [SparsePacketEncodeDescriptorCacheKey: SparsePacketEncodeDescriptorCacheEntry]()
    private var decodedPlaneTemplateCache = [DecodedPlaneTemplateCacheKey: DecodedPlaneTemplate]()
    private var sparseBlockTargetCache = [LayoutCacheKey: [SparseBlockTarget?]]()

    public init() throws {
        metalBackend = try MetalPyrowaveBackend()
        guard let placeholder = metalBackend.device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate GPU decoded plane placeholder")
        }
        gpuDecodedPlanePlaceholder = placeholder
        var textureCache: CVMetalTextureCache?
        let status = CVMetalTextureCacheCreate(nil, nil, metalBackend.device, nil, &textureCache)
        guard status == kCVReturnSuccess, let textureCache else {
            throw PyrowaveError.processFailed("failed to create CoreVideo Metal texture cache")
        }
        coreVideoTextureCache = textureCache
    }

    func encode(_ frame: YUVFrame, configuration: CodecConfiguration = CodecConfiguration()) throws -> EncodedFrame {
        guard configuration.decompositionLevels == PyrowaveBitstream.decompositionLevels,
              configuration.quantizationStep > 0 else {
            throw PyrowaveError.invalidDimensions
        }

        let sequenceNumber = sequenceCounter.next()
        let layout = try cachedLayout(width: frame.width, height: frame.height, chroma: frame.chroma)
        let encodedPlanes = [
            try encodePlane(frame.y, component: 0, frameWidth: frame.width, frameHeight: frame.height, chroma: frame.chroma, layout: layout, configuration: configuration),
            try encodePlane(frame.cb, component: 1, frameWidth: frame.width, frameHeight: frame.height, chroma: frame.chroma, layout: layout, configuration: configuration),
            try encodePlane(frame.cr, component: 2, frameWidth: frame.width, frameHeight: frame.height, chroma: frame.chroma, layout: layout, configuration: configuration)
        ]
        return try encodeFrame(
            width: frame.width,
            height: frame.height,
            chroma: frame.chroma,
            videoSignal: frame.videoSignal,
            sequenceNumber: sequenceNumber,
            encodedPlanes: encodedPlanes,
            layout: layout,
            configuration: configuration
        )
    }

    public func encode(
        yTexture: MTLTexture,
        cbTexture: MTLTexture,
        crTexture: MTLTexture,
        configuration: CodecConfiguration = CodecConfiguration(),
        videoSignal: VideoSignalMetadata = .default
    ) throws -> EncodedFrame {
        guard configuration.decompositionLevels == PyrowaveBitstream.decompositionLevels,
              configuration.quantizationStep > 0 else {
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

        let layout = try cachedLayout(width: yTexture.width, height: yTexture.height, chroma: chroma)
        let encodedPlanes = try encodeTexturePlanes(
            [
                (texture: yTexture, channel: 0, component: 0),
                (texture: cbTexture, channel: 0, component: 1),
                (texture: crTexture, channel: 0, component: 2)
            ],
            frameWidth: yTexture.width,
            frameHeight: yTexture.height,
            chroma: chroma,
            layout: layout,
            configuration: configuration
        )
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
        try exportGPUFrame(try encodeGPUFrame(
            yTexture: yTexture,
            cbCrTexture: cbCrTexture,
            configuration: configuration,
            videoSignal: videoSignal
        ))
    }

    public func encodeGPUFrame(
        yTexture: MTLTexture,
        cbCrTexture: MTLTexture,
        configuration: CodecConfiguration = CodecConfiguration(),
        videoSignal: VideoSignalMetadata = .default
    ) throws -> PyrowaveGPUFrame {
        try encodeGPUFrame(
            yTexture: yTexture,
            cbCrTexture: cbCrTexture,
            configuration: configuration,
            videoSignal: videoSignal,
            reusesPacketBuffers: false
        )
    }

    func encodeGPUFrame(
        yTexture: MTLTexture,
        cbCrTexture: MTLTexture,
        configuration: CodecConfiguration = CodecConfiguration(),
        videoSignal: VideoSignalMetadata = .default,
        reusesPacketBuffers: Bool,
        packetOutputStorageMode: MTLStorageMode = .private
    ) throws -> PyrowaveGPUFrame {
        guard configuration.decompositionLevels == PyrowaveBitstream.decompositionLevels,
              configuration.quantizationStep > 0 else {
            throw PyrowaveError.invalidDimensions
        }
        guard yTexture.pixelFormat == .r8Unorm,
              cbCrTexture.pixelFormat == .rg8Unorm,
              yTexture.width > 0,
              yTexture.height > 0,
              cbCrTexture.width == yTexture.width / 2,
              cbCrTexture.height == yTexture.height / 2 else {
            throw PyrowaveError.unsupportedFormat("Metal NV12 GPU encode expects r8Unorm luma and rg8Unorm chroma planes")
        }

        let layout = try cachedLayout(width: yTexture.width, height: yTexture.height, chroma: .yuv420)
        let encodedPlanes = try encodeTexturePlanes(
            [
                (texture: yTexture, channel: 0, component: 0),
                (texture: cbCrTexture, channel: 0, component: 1),
                (texture: cbCrTexture, channel: 1, component: 2)
            ],
            frameWidth: yTexture.width,
            frameHeight: yTexture.height,
            chroma: .yuv420,
            layout: layout,
            configuration: configuration,
            readsQScaleCodes: false,
            waitsForDWTCompletion: false
        )
        let sequenceNumber = sequenceCounter.next()
        let planes = try makeGPUFramePlanes(
            encodedPlanes,
            layout: layout,
            sequence: sequenceNumber,
            quantLevelsByPlane: nil,
            packetByteCostsByPlane: nil,
            defaultQuantLevel: 0,
            reusesPacketBuffers: reusesPacketBuffers,
            packetOutputStorageMode: packetOutputStorageMode
        )
        return PyrowaveGPUFrame(
            width: yTexture.width,
            height: yTexture.height,
            chroma: .yuv420,
            videoSignal: videoSignal,
            sequenceNumber: sequenceNumber,
            planes: planes
        )
    }

    public func decodeGPUFrameToNV12Textures(
        _ frame: PyrowaveGPUFrame,
        yTexture: MTLTexture,
        cbCrTexture: MTLTexture
    ) throws {
        guard frame.chroma == .yuv420,
              frame.planes.count == PyrowaveBitstream.componentCount,
              yTexture.width == frame.width,
              yTexture.height == frame.height,
              cbCrTexture.width == frame.width / 2,
              cbCrTexture.height == frame.height / 2 else {
            throw PyrowaveError.invalidDimensions
        }
        try metalBackend.decodeSparsePacketBuffersInverseAndCropToNV12Textures(
            packetPlanes: frame.planes.map {
                (
                    packetBuffer: $0.encoded.outputBuffer,
                    packetByteLength: $0.encoded.descriptorCount * $0.encoded.maxPacketBytes,
                    sampleCount: $0.sampleCount,
                    paddedWidth: $0.paddedWidth,
                    paddedHeight: $0.paddedHeight,
                    levels: $0.levels,
                    descriptorCount: $0.decodeDescriptorCount,
                    descriptorBuffer: $0.decodeDescriptorBuffer
                )
            },
            width: frame.width,
            height: frame.height,
            yTexture: yTexture,
            cbCrTexture: cbCrTexture
        )
    }

    public func exportGPUFrame(_ frame: PyrowaveGPUFrame) throws -> EncodedFrame {
        let layout = try cachedLayout(width: frame.width, height: frame.height, chroma: frame.chroma)
        var packets = [(blockIndex: Int, buffer: MTLBuffer, offset: Int, size: Int)]()
        packets.reserveCapacity(frame.planes.reduce(0) { $0 + $1.encoded.descriptorCount })
        var payloadBytes = 0
        for plane in frame.planes {
            guard plane.encoded.descriptorCount == plane.blockIndexCount else {
                throw PyrowaveError.processFailed("GPU frame packet metadata does not match packet buffers")
            }
            guard plane.encoded.descriptorCount == plane.decodeDescriptorCount else {
                throw PyrowaveError.processFailed("GPU frame decode metadata does not match packet buffers")
            }
            guard plane.encoded.descriptorCount == plane.selectedQuantLevelCount else {
                throw PyrowaveError.processFailed("GPU frame quant-level metadata does not match packet buffers")
            }
            guard plane.encoded.outputBuffer.length >= plane.encoded.descriptorCount * plane.encoded.maxPacketBytes,
                  plane.encoded.sizeBuffer.length >= plane.encoded.descriptorCount * MemoryLayout<UInt32>.stride else {
                throw PyrowaveError.invalidBitstream("GPU frame packet buffers are too small")
            }
            let packetByteLength = plane.encoded.descriptorCount * plane.encoded.maxPacketBytes
            let packetBuffer = try metalBackend.sharedReadbackBuffer(
                from: plane.encoded.outputBuffer,
                byteLength: packetByteLength
            )
            let sizes = plane.encoded.sizeBuffer.contents().bindMemory(to: UInt32.self, capacity: plane.encoded.descriptorCount)
            let blockIndices = plane.blockIndicesForInspection()
            for index in 0..<plane.encoded.descriptorCount {
                let size = Int(sizes[index])
                guard size >= 0, size <= plane.encoded.maxPacketBytes else {
                    throw PyrowaveError.invalidBitstream("GPU frame packet size exceeds packet capacity")
                }
                guard size > 0 else {
                    continue
                }
                packets.append((
                    blockIndex: blockIndices[index],
                    buffer: packetBuffer,
                    offset: index * plane.encoded.maxPacketBytes,
                    size: size
                ))
                payloadBytes += size
            }
        }

        packets.sort { $0.blockIndex < $1.blockIndex }
        var previousBlockIndex: Int?
        for packet in packets {
            guard packet.blockIndex >= 0, packet.blockIndex < layout.descriptors.count else {
                throw PyrowaveError.processFailed("sparse block index \(packet.blockIndex) is outside layout block count \(layout.descriptors.count)")
            }
            guard packet.blockIndex != previousBlockIndex else {
                throw PyrowaveError.processFailed("duplicate sparse block index \(packet.blockIndex)")
            }
            previousBlockIndex = packet.blockIndex
        }

        // Compact fixed GPU packet slots directly into one contiguous pyrw
        // payload. This copies only actual packet bytes and avoids constructing
        // thousands of temporary Data values on the production export path.
        var writer = BinaryWriter(capacity: frameHeaderSize + payloadBytes)
        let sequence = try PyrowaveSequenceHeader(
            width: frame.width,
            height: frame.height,
            sequence: frame.sequenceNumber,
            totalBlocks: packets.count,
            chroma: frame.chroma,
            videoSignal: frame.videoSignal
        )
        sequence.write(to: &writer)
        for packet in packets {
            writer.append(
                unsafeBytes: packet.buffer.contents().advanced(by: packet.offset),
                count: packet.size
            )
        }
        return EncodedFrame(data: writer.data)
    }

    public func importGPUFrame(_ frame: EncodedFrame) throws -> PyrowaveGPUFrame {
        let decoded = try decodeGPUPlanes(frame, allowPartialFrame: false, appliesSparsePackets: false)
        let maxPacketBytes = PyrowaveCoefficientBlockCodec.maximumEncodedBlockBytes
        let planes = try decoded.planes.indices.map { planeIndex -> PyrowaveGPUFramePlane in
            let descriptors = decoded.packetDescriptorsByPlane[planeIndex]
            let blockIndices = decoded.blockIndicesByPlane[planeIndex]
            guard descriptors.count == blockIndices.count else {
                throw PyrowaveError.invalidBitstream("sparse packet metadata mismatch")
            }

            let outputByteCount = descriptors.count * maxPacketBytes
            let sizeByteCount = max(descriptors.count, 1) * MemoryLayout<UInt32>.stride
            guard let outputBuffer = metalBackend.device.makeBuffer(length: max(outputByteCount, 1), options: .storageModeShared),
                  let sizeBuffer = metalBackend.device.makeBuffer(length: sizeByteCount, options: .storageModeShared) else {
                throw PyrowaveError.processFailed("failed to allocate imported GPU frame packet buffers")
            }

            let outputPointer = outputBuffer.contents().bindMemory(to: UInt8.self, capacity: max(outputByteCount, 1))
            let sizePointer = sizeBuffer.contents().bindMemory(to: UInt32.self, capacity: max(descriptors.count, 1))
            var importedDescriptors = [MetalSparsePacketDecodeDescriptor]()
            importedDescriptors.reserveCapacity(descriptors.count)

            for (index, descriptor) in descriptors.enumerated() {
                let packetOffset = Int(descriptor.packetOffset)
                let payloadEnd = Int(descriptor.payloadEnd)
                guard payloadEnd >= packetOffset,
                      payloadEnd <= frame.data.count else {
                    throw PyrowaveError.invalidBitstream("sparse packet points outside encoded frame")
                }
                let packetSize = payloadEnd - packetOffset
                guard packetSize <= maxPacketBytes else {
                    throw PyrowaveError.invalidBitstream("sparse packet exceeds GPU packet slot capacity")
                }
                let slotOffset = index * maxPacketBytes
                frame.data.withUnsafeBytes { rawBuffer in
                    if let source = rawBuffer.baseAddress?.advanced(by: packetOffset), packetSize > 0 {
                        outputPointer.advanced(by: slotOffset).update(from: source.assumingMemoryBound(to: UInt8.self), count: packetSize)
                    }
                }
                sizePointer[index] = UInt32(packetSize)
                importedDescriptors.append(MetalSparsePacketDecodeDescriptor(
                    packetOffset: UInt32(slotOffset),
                    payloadEnd: UInt32(slotOffset + packetSize),
                    originX: descriptor.originX,
                    originY: descriptor.originY,
                    validWidth: descriptor.validWidth,
                    validHeight: descriptor.validHeight,
                    stride: descriptor.stride
                ))
            }

            let plane = decoded.planes[planeIndex]
            return PyrowaveGPUFramePlane(
                component: planeIndex,
                paddedWidth: plane.paddedWidth,
                paddedHeight: plane.paddedHeight,
                levels: plane.levels,
                sampleCount: plane.sampleCount,
                encoded: MetalSparsePacketEncodedPlane(
                    outputBuffer: outputBuffer,
                    sizeBuffer: sizeBuffer,
                    descriptorCount: descriptors.count,
                    maxPacketBytes: maxPacketBytes
                ),
                decodeDescriptorCount: importedDescriptors.count,
                decodeDescriptorBuffer: try metalBackend.makeStaticSharedBuffer(bytes: importedDescriptors),
                blockIndexCount: blockIndices.count,
                blockIndexBuffer: try metalBackend.makeStaticSharedBuffer(bytes: try blockIndices.map { value -> UInt32 in
                    guard value >= 0, value <= Int(UInt32.max) else {
                        throw PyrowaveError.invalidDimensions
                    }
                    return UInt32(value)
                }),
                selectedQuantLevelCount: blockIndices.count,
                selectedQuantLevelBuffer: try metalBackend.makeStaticSharedBuffer(bytes: Array(repeating: UInt32(0), count: blockIndices.count))
            )
        }
        return PyrowaveGPUFrame(
            width: decoded.sequence.width,
            height: decoded.sequence.height,
            chroma: decoded.sequence.chroma,
            videoSignal: decoded.sequence.videoSignal,
            sequenceNumber: decoded.sequence.sequence,
            planes: planes
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
        return try encodeFrame(
            width: width,
            height: height,
            chroma: chroma,
            videoSignal: videoSignal,
            sequenceNumber: sequenceNumber,
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
        sequenceNumber: UInt8,
        encodedPlanes: [EncodedPlane],
        layout: PyrowaveBlockLayout,
        configuration: CodecConfiguration
    ) throws -> EncodedFrame {
        try assembleEncodedFrame(
            width: width,
            height: height,
            chroma: chroma,
            videoSignal: videoSignal,
            sequenceNumber: sequenceNumber,
            totalLayoutBlocks: layout.descriptors.count,
            encodedPlanes: encodedPlanes,
            layout: layout,
            quantLevelsByPlane: nil,
            packetByteCostsByPlane: nil,
            defaultQuantLevel: 0
        )
    }

    private func defaultFrameByteEstimate(packetByteCostsByPlane: [[[Int]]]) -> Int {
        packetByteCostsByPlane.reduce(frameHeaderSize) { frameBytes, planeCosts in
            frameBytes + planeCosts.reduce(0) { planeBytes, blockCosts in
                planeBytes + (blockCosts.first ?? 0)
            }
        }
    }

    private func assembleEncodedFrame(
        width: Int,
        height: Int,
        chroma: ChromaSubsampling,
        videoSignal: VideoSignalMetadata,
        sequenceNumber: UInt8,
        totalLayoutBlocks: Int,
        encodedPlanes: [EncodedPlane],
        layout: PyrowaveBlockLayout,
        quantLevelsByPlane: [[Int]]?,
        packetByteCostsByPlane: [[[Int]]]?,
        defaultQuantLevel: Int
    ) throws -> EncodedFrame {
        let blocks = try sparseBlocks(
            encodedPlanes,
            layout: layout,
            sequence: sequenceNumber,
            quantLevelsByPlane: quantLevelsByPlane,
            packetByteCostsByPlane: packetByteCostsByPlane,
            defaultQuantLevel: defaultQuantLevel
        ).flatMap { $0 }
        return try assembleEncodedFrame(
            width: width,
            height: height,
            chroma: chroma,
            videoSignal: videoSignal,
            sequenceNumber: sequenceNumber,
            totalLayoutBlocks: totalLayoutBlocks,
            blocks: blocks
        )
    }

    private func assembleEncodedFrame(
        width: Int,
        height: Int,
        chroma: ChromaSubsampling,
        videoSignal: VideoSignalMetadata,
        sequenceNumber: UInt8,
        totalLayoutBlocks: Int,
        blocks: [SparseBlock]
    ) throws -> EncodedFrame {
        guard totalLayoutBlocks >= 0 else {
            throw PyrowaveError.invalidDimensions
        }
        var blockDataByIndex = Array<Data?>(repeating: nil, count: totalLayoutBlocks)
        var payloadBytes = 0
        for block in blocks {
            guard block.blockIndex >= 0, block.blockIndex < totalLayoutBlocks else {
                throw PyrowaveError.processFailed("sparse block index \(block.blockIndex) is outside layout block count \(totalLayoutBlocks)")
            }
            guard blockDataByIndex[block.blockIndex] == nil else {
                throw PyrowaveError.processFailed("duplicate sparse block index \(block.blockIndex)")
            }
            blockDataByIndex[block.blockIndex] = block.data
            payloadBytes += block.data.count
        }

        let capacity = frameHeaderSize + payloadBytes
        var writer = BinaryWriter(capacity: capacity)
        let sequence = try PyrowaveSequenceHeader(
            width: width,
            height: height,
            sequence: sequenceNumber,
            totalBlocks: blocks.count,
            chroma: chroma,
            videoSignal: videoSignal
        )
        sequence.write(to: &writer)
        for blockData in blockDataByIndex {
            if let blockData {
                writer.append(data: blockData)
            }
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
        let reconstructed = try inverseWaveletBuffers(decoded.planes)
        try metalBackend.cropPlaneToTexture(
            reconstructed[0],
            sampleCount: decoded.planes[0].sampleCount,
            paddedWidth: decoded.planes[0].paddedWidth,
            width: decoded.planes[0].visibleWidth,
            height: decoded.planes[0].visibleHeight,
            texture: yTexture
        )
        try metalBackend.cropPlaneToTexture(
            reconstructed[1],
            sampleCount: decoded.planes[1].sampleCount,
            paddedWidth: decoded.planes[1].paddedWidth,
            width: decoded.planes[1].visibleWidth,
            height: decoded.planes[1].visibleHeight,
            texture: cbTexture
        )
        try metalBackend.cropPlaneToTexture(
            reconstructed[2],
            sampleCount: decoded.planes[2].sampleCount,
            paddedWidth: decoded.planes[2].paddedWidth,
            width: decoded.planes[2].visibleWidth,
            height: decoded.planes[2].visibleHeight,
            texture: crTexture
        )
        return (yTexture, cbTexture, crTexture)
    }

    func decodeToNV12Textures(
        _ frame: EncodedFrame,
        yTexture: MTLTexture,
        cbCrTexture: MTLTexture
    ) async throws -> PyrowaveContiguousNV12DecodeMetrics {
        let packetParseStart = ContinuousClock.now
        let decoded = try decodeContiguousGPUPlanes(frame)
        let packetParseEnd = ContinuousClock.now
        guard decoded.sequence.chroma == .yuv420,
              yTexture.width == decoded.sequence.width,
              yTexture.height == decoded.sequence.height,
              cbCrTexture.width == decoded.sequence.width / 2,
              cbCrTexture.height == decoded.sequence.height / 2 else {
            throw PyrowaveError.invalidDimensions
        }

        let metalDecodeStart = ContinuousClock.now
        if shouldUseCombinedNV12Decode(width: decoded.sequence.width, height: decoded.sequence.height) {
            try await metalBackend.decodeSparsePacketsInverseAndCropToNV12Textures(
                packetData: frame.data,
                planes: decoded.planes.indices.map { index in
                    (
                        sampleCount: decoded.planes[index].sampleCount,
                        paddedWidth: decoded.planes[index].paddedWidth,
                        paddedHeight: decoded.planes[index].paddedHeight,
                        levels: decoded.planes[index].levels,
                        descriptors: decoded.packetDescriptorsByPlane[index]
                    )
                },
                width: decoded.sequence.width,
                height: decoded.sequence.height,
                yTexture: yTexture,
                cbCrTexture: cbCrTexture
            )
        } else {
            var applied = decoded.planes
            try applySparsePacketDescriptorsToBuffers(decoded.packetDescriptorsByPlane, packetData: frame.data, decodedPlanes: &applied)
            let reconstructed = try inverseWaveletBuffers(applied)
            try metalBackend.cropPlanesToNV12Textures(
                yBuffer: reconstructed[0],
                ySampleCount: applied[0].sampleCount,
                yPaddedWidth: applied[0].paddedWidth,
                cbBuffer: reconstructed[1],
                cbSampleCount: applied[1].sampleCount,
                crBuffer: reconstructed[2],
                crSampleCount: applied[2].sampleCount,
                chromaPaddedWidth: applied[1].paddedWidth,
                width: decoded.sequence.width,
                height: decoded.sequence.height,
                yTexture: yTexture,
                cbCrTexture: cbCrTexture
            )
        }
        let metalDecodeEnd = ContinuousClock.now
        return PyrowaveContiguousNV12DecodeMetrics(
            packetParseMilliseconds: packetParseStart.milliseconds(to: packetParseEnd),
            metalDecodeMilliseconds: metalDecodeStart.milliseconds(to: metalDecodeEnd)
        )
    }

    private func decodeContiguousGPUPlanes(_ frame: EncodedFrame) throws -> GPUDecodedFramePlanes {
        var headerReader = BinaryReader(frame.data)
        let sequence = try PyrowaveSequenceHeader(reader: &headerReader)
        let layout = try cachedLayout(width: sequence.width, height: sequence.height, chroma: sequence.chroma)
        let decodedPlanes = try (0..<PyrowaveBitstream.componentCount).map { component in
            try makeGPUDecodedPlane(
                component: component,
                width: sequence.width,
                height: sequence.height,
                chroma: sequence.chroma,
                layout: layout
            )
        }
        let blockTargets = try cachedSparseBlockTargets(layout: layout)
        var packetDescriptorsByPlane = decodedPlanes.map { plane in
            var descriptors = [MetalSparsePacketDecodeDescriptor]()
            descriptors.reserveCapacity(plane.descriptorsByBlockIndex.count)
            return descriptors
        }
        var seenBlocks = [UInt8](repeating: 0, count: layout.descriptors.count)
        var parsedBlockCount = 0

        try frame.data.withUnsafeBytes { bytes in
            var offset = headerReader.offset
            while offset < bytes.count {
                let blockStart = offset
                guard blockStart + 8 <= bytes.count else {
                    throw PyrowaveError.truncatedInput
                }

                let ballot = UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: blockStart, as: UInt16.self))
                let packedPayload = UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: blockStart + 2, as: UInt16.self))
                let packedBlock = UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: blockStart + 4, as: UInt32.self))
                let payloadWords = Int(packedPayload & 0x0fff)
                let packetSequence = UInt8((packedPayload >> 12) & 0x7)
                let isExtended = ((packedPayload >> 15) & 0x1) != 0
                let blockIndex = Int(packedBlock >> 8)
                let payloadEnd = blockStart + payloadWords * 4

                guard !isExtended else {
                    throw PyrowaveError.invalidBitstream("coefficient decoder received extended packet")
                }
                guard packetSequence == sequence.sequence else {
                    throw PyrowaveError.invalidBitstream("coefficient packet sequence mismatch")
                }
                guard payloadEnd >= blockStart + 8 else {
                    throw PyrowaveError.invalidBitstream("payload_words is not large enough")
                }
                guard payloadEnd <= bytes.count else {
                    throw PyrowaveError.truncatedInput
                }
                guard blockIndex >= 0,
                      blockIndex < blockTargets.count,
                      seenBlocks[blockIndex] == 0 else {
                    throw PyrowaveError.invalidBitstream("bad sparse block index")
                }
                seenBlocks[blockIndex] = 1
                guard let target = blockTargets[blockIndex] else {
                    throw PyrowaveError.invalidBitstream("sparse block index has no plane mapping")
                }

                if ballot == 0 {
                    var paddingOffset = blockStart + 8
                    while paddingOffset < payloadEnd {
                        guard bytes[paddingOffset] == 0 else {
                            throw PyrowaveError.invalidBitstream("non-zero coefficient packet padding")
                        }
                        paddingOffset += 1
                    }
                } else {
                    let decodedPlane = decodedPlanes[target.planeIndex]
                    guard blockStart <= Int(UInt32.max),
                          payloadEnd <= Int(UInt32.max),
                          target.descriptor.originX <= Int(UInt32.max),
                          target.descriptor.originY <= Int(UInt32.max),
                          target.descriptor.validWidth <= Int(UInt32.max),
                          target.descriptor.validHeight <= Int(UInt32.max),
                          decodedPlane.paddedWidth <= Int(UInt32.max) else {
                        throw PyrowaveError.invalidDimensions
                    }
                    packetDescriptorsByPlane[target.planeIndex].append(
                        MetalSparsePacketDecodeDescriptor(
                            packetOffset: UInt32(blockStart),
                            payloadEnd: UInt32(payloadEnd),
                            originX: UInt32(target.descriptor.originX),
                            originY: UInt32(target.descriptor.originY),
                            validWidth: UInt32(target.descriptor.validWidth),
                            validHeight: UInt32(target.descriptor.validHeight),
                            stride: UInt32(decodedPlane.paddedWidth)
                        )
                    )
                }

                parsedBlockCount += 1
                offset = payloadEnd
            }
            guard offset == bytes.count else {
                throw PyrowaveError.invalidBitstream("trailing bytes")
            }
        }

        guard parsedBlockCount == sequence.totalBlocks else {
            throw PyrowaveError.invalidBitstream(
                "expected \(sequence.totalBlocks) blocks, decoded \(parsedBlockCount)"
            )
        }
        return GPUDecodedFramePlanes(
            sequence: sequence,
            planes: decodedPlanes,
            packetDescriptorsByPlane: packetDescriptorsByPlane,
            blockIndicesByPlane: Array(repeating: [], count: PyrowaveBitstream.componentCount)
        )
    }

    private func decodePlanes(_ frame: EncodedFrame, allowPartialFrame: Bool) throws -> DecodedFramePlanes {
        var reader = BinaryReader(frame.data)
        let sequence = try PyrowaveSequenceHeader(reader: &reader)
        let layout = try cachedLayout(width: sequence.width, height: sequence.height, chroma: sequence.chroma)
        var decodedPlanes = try (0..<PyrowaveBitstream.componentCount).map { component in
            try makeDecodedPlane(component: component, width: sequence.width, height: sequence.height, chroma: sequence.chroma, layout: layout)
        }
        let blockTargets = try cachedSparseBlockTargets(layout: layout)
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
            guard let target = blockTargets[block.blockIndex] else {
                throw PyrowaveError.invalidBitstream("sparse block index has no plane mapping")
            }
            pendingSparseBlocks[target.planeIndex].append(PendingSparseBlock(block: block, descriptor: target.descriptor))
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

    private func decodeGPUPlanes(
        _ frame: EncodedFrame,
        allowPartialFrame: Bool,
        appliesSparsePackets: Bool = true
    ) throws -> GPUDecodedFramePlanes {
        var reader = BinaryReader(frame.data)
        let sequence = try PyrowaveSequenceHeader(reader: &reader)
        let layout = try cachedLayout(width: sequence.width, height: sequence.height, chroma: sequence.chroma)
        var decodedPlanes = try (0..<PyrowaveBitstream.componentCount).map { component in
            try makeGPUDecodedPlane(component: component, width: sequence.width, height: sequence.height, chroma: sequence.chroma, layout: layout)
        }
        let blockTargets = try cachedSparseBlockTargets(layout: layout)
        var packetDescriptorsByPlane = Array(repeating: [MetalSparsePacketDecodeDescriptor](), count: PyrowaveBitstream.componentCount)
        var blockIndicesByPlane = Array(repeating: [Int](), count: PyrowaveBitstream.componentCount)
        var seenBlocks = Set<Int>()

        while reader.offset < frame.data.count {
            let blockStart = reader.offset
            let header = try PyrowavePacketHeader(reader: &reader)
            guard !header.extended else {
                throw PyrowaveError.invalidBitstream("coefficient decoder received extended packet")
            }
            guard header.sequence == sequence.sequence else {
                throw PyrowaveError.invalidBitstream("coefficient packet sequence mismatch")
            }
            guard header.blockIndex >= 0, header.blockIndex < layout.descriptors.count, seenBlocks.insert(header.blockIndex).inserted else {
                throw PyrowaveError.invalidBitstream("bad sparse block index")
            }
            guard let target = blockTargets[header.blockIndex] else {
                throw PyrowaveError.invalidBitstream("sparse block index has no plane mapping")
            }
            let descriptorCount = packetDescriptorsByPlane[target.planeIndex].count
            try appendSparsePacketDecodeDescriptor(
                header: header,
                blockStart: blockStart,
                reader: &reader,
                target: target,
                decodedPlane: decodedPlanes[target.planeIndex],
                descriptors: &packetDescriptorsByPlane[target.planeIndex]
            )
            if packetDescriptorsByPlane[target.planeIndex].count > descriptorCount {
                blockIndicesByPlane[target.planeIndex].append(header.blockIndex)
            }
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

        if appliesSparsePackets {
            try applySparsePacketDescriptorsToBuffers(packetDescriptorsByPlane, packetData: frame.data, decodedPlanes: &decodedPlanes)
        }

        return GPUDecodedFramePlanes(
            sequence: sequence,
            planes: decodedPlanes,
            packetDescriptorsByPlane: packetDescriptorsByPlane,
            blockIndicesByPlane: blockIndicesByPlane
        )
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

    private struct LayoutCacheKey: Hashable {
        var width: Int
        var height: Int
        var chroma: UInt8
    }

    private struct PlaneDescriptorCacheKey: Hashable {
        var layoutWidth: Int
        var layoutHeight: Int
        var chroma: UInt8
        var component: Int
        var paddedWidth: Int
        var paddedHeight: Int
        var levels: Int
    }

    private struct QuantizationDescriptorCacheKey: Hashable {
        var stride: Int
        var component: Int
        var quantizationStep: Float
        var descriptorHash: Int
    }

    private struct QuantizationDescriptorCacheEntry {
        var metalDescriptors: [MetalPlaneQuantizationDescriptor]
        var descriptorBuffer: MTLBuffer
        var quantCodesByBlockIndex: [Int: UInt8]
        var quantCodesByDescriptor: [UInt8]
    }

    private struct SparsePacketEncodeDescriptorCacheKey: Hashable {
        var layoutWidth: Int
        var layoutHeight: Int
        var chroma: UInt8
        var component: Int
        var paddedWidth: Int
        var paddedHeight: Int
        var levels: Int
        var defaultQuantLevel: Int
        var descriptorHash: Int
        var quantCodeHash: Int
    }

    private struct SparsePacketEncodeDescriptorCacheEntry {
        var packetDescriptorCount: Int
        var descriptorBuffer: MTLBuffer
        var decodeDescriptorCount: Int
        var decodeDescriptorBuffer: MTLBuffer
        var blockIndexCount: Int
        var blockIndexBuffer: MTLBuffer
        var selectedQuantLevelCount: Int
        var selectedQuantLevelBuffer: MTLBuffer
    }

    private struct DecodedPlaneTemplateCacheKey: Hashable {
        var layoutWidth: Int
        var layoutHeight: Int
        var chroma: UInt8
        var component: Int
    }

    private struct DecodedPlaneTemplate {
        var visibleWidth: Int
        var visibleHeight: Int
        var paddedWidth: Int
        var paddedHeight: Int
        var levels: Int
        var sampleCount: Int
        var descriptorsByBlockIndex: [Int: PlaneBlockDescriptor]
    }

    private struct EncodedPlane {
        var component: Int
        var paddedWidth: Int
        var paddedHeight: Int
        var levels: Int
        var quantCodesByBlockIndex: [Int: UInt8]
        var qScaleCodesByBlockIndex: [Int: [UInt8]]
        var quantCodesByDescriptor: [UInt8]
        var qScaleCodesByDescriptor: [[UInt8]]
        var coefficients: [Int16]
        var coefficientBuffer: MTLBuffer?
        var coefficientCount: Int
        var qScaleBuffer: MTLBuffer?
        var qScaleDescriptorCount: Int
    }

    private struct SparseBlock {
        var blockIndex: Int
        var data: Data
    }

    private struct PendingSparseBlock {
        var block: PyrowaveCoefficientBlockCodec.DecodedBlock
        var descriptor: PlaneBlockDescriptor
    }

    private struct SparseBlockTarget {
        var planeIndex: Int
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

    private struct SparseRateControlInputs {
        var distortionsByPlane: [[[Float]]]
        var packetByteCostsByPlane: [[[Int]]]
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

    private struct PlaneQuantizationMetadata {
        var quantCodesByBlockIndex: [Int: UInt8]
        var qScaleCodesByBlockIndex: [Int: [UInt8]]
        var quantCodesByDescriptor: [UInt8]
        var qScaleCodesByDescriptor: [[UInt8]]
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
        var packetDescriptorsByPlane: [[MetalSparsePacketDecodeDescriptor]]
        var blockIndicesByPlane: [[Int]]
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

    private func encodeTexturePlanes(
        _ inputs: [(texture: MTLTexture, channel: Int, component: Int)],
        frameWidth: Int,
        frameHeight: Int,
        chroma: ChromaSubsampling,
        layout: PyrowaveBlockLayout,
        configuration: CodecConfiguration,
        readsQScaleCodes: Bool = true,
        waitsForDWTCompletion: Bool = true
    ) throws -> [EncodedPlane] {
        let geometries = try inputs.map { input in
            let geometry = planeGeometry(
                component: input.component,
                frameWidth: frameWidth,
                frameHeight: frameHeight,
                chroma: chroma,
                requestedLevels: configuration.decompositionLevels
            )
            guard input.texture.width == geometry.visibleWidth,
                  input.texture.height == geometry.visibleHeight else {
                throw PyrowaveError.invalidDimensions
            }
            return geometry
        }
        let levels = geometries.map {
            Wavelet.usableLevels(width: $0.paddedWidth, height: $0.paddedHeight, requested: $0.requestedLevels)
        }
        let descriptorsByPlane = inputs.indices.map { index in
            planeBlockDescriptors(
                component: inputs[index].component,
                paddedWidth: geometries[index].paddedWidth,
                paddedHeight: geometries[index].paddedHeight,
                levels: levels[index],
                layout: layout
            )
        }
        let transformed = try metalBackend.padTexturePlaneBuffersAndForwardWaveletBuffers(inputs.indices.map { index in
            (
                texture: inputs[index].texture,
                channel: inputs[index].channel,
                paddedWidth: geometries[index].paddedWidth,
                paddedHeight: geometries[index].paddedHeight,
                levels: levels[index]
            )
        },
            useTiledLevelZero: shouldUseTiledLevelZeroForwardDWT(width: frameWidth, height: frameHeight),
            waitsForCompletion: waitsForDWTCompletion
        )

        let quantizedPlanes = try quantizeResidentBuffers(inputs.indices.map { index in
            (
                samples: transformed[index],
                sampleCount: geometries[index].paddedWidth * geometries[index].paddedHeight,
                stride: geometries[index].paddedWidth,
                descriptors: descriptorsByPlane[index],
                component: inputs[index].component
            )
        }, configuration: configuration, readsQScaleCodes: readsQScaleCodes)

        return inputs.indices.map { index in
            let quantized = quantizedPlanes[index]
            return EncodedPlane(
                component: inputs[index].component,
                paddedWidth: geometries[index].paddedWidth,
                paddedHeight: geometries[index].paddedHeight,
                levels: levels[index],
                quantCodesByBlockIndex: quantized.metadata.quantCodesByBlockIndex,
                qScaleCodesByBlockIndex: quantized.metadata.qScaleCodesByBlockIndex,
                quantCodesByDescriptor: quantized.metadata.quantCodesByDescriptor,
                qScaleCodesByDescriptor: quantized.metadata.qScaleCodesByDescriptor,
                coefficients: [],
                coefficientBuffer: quantized.coefficientBuffer,
                coefficientCount: quantized.coefficientCount,
                qScaleBuffer: quantized.qScaleBuffer,
                qScaleDescriptorCount: quantized.qScaleDescriptorCount
            )
        }
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
            quantCodesByBlockIndex: quantized.metadata.quantCodesByBlockIndex,
            qScaleCodesByBlockIndex: quantized.metadata.qScaleCodesByBlockIndex,
            quantCodesByDescriptor: quantized.metadata.quantCodesByDescriptor,
            qScaleCodesByDescriptor: quantized.metadata.qScaleCodesByDescriptor,
            coefficients: quantized.coefficients,
            coefficientBuffer: nil,
            coefficientCount: quantized.coefficients.count,
            qScaleBuffer: nil,
            qScaleDescriptorCount: 0
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
        let transformed = try metalBackend.forwardWaveletBuffer(
            samples,
            sampleCount: sampleCount,
            width: paddedWidth,
            height: paddedHeight,
            levels: levels,
            useTiledLevelZero: shouldUseTiledLevelZeroForwardDWT(width: paddedWidth, height: paddedHeight)
        )

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
            quantCodesByBlockIndex: quantized.metadata.quantCodesByBlockIndex,
            qScaleCodesByBlockIndex: quantized.metadata.qScaleCodesByBlockIndex,
            quantCodesByDescriptor: quantized.metadata.quantCodesByDescriptor,
            qScaleCodesByDescriptor: quantized.metadata.qScaleCodesByDescriptor,
            coefficients: [],
            coefficientBuffer: quantized.coefficientBuffer,
            coefficientCount: quantized.coefficientCount,
            qScaleBuffer: quantized.qScaleBuffer,
            qScaleDescriptorCount: quantized.qScaleDescriptorCount
        )
    }

    private func metalRateControlBucketData(
        distortionsByPlane: [[[Float]]],
        packetByteCostsByPlane: [[[Int]]]
    ) throws -> MetalRateControlBucketData {
        let totalBlockCount = packetByteCostsByPlane.reduce(0) { $0 + $1.count }
        if totalBlockCount < fusedRateControlBucketBlockThreshold {
            return try separateMetalRateControlBucketData(
                distortionsByPlane: distortionsByPlane,
                packetByteCostsByPlane: packetByteCostsByPlane
            )
        }

        return try fusedMetalRateControlBucketData(
            distortionsByPlane: distortionsByPlane,
            packetByteCostsByPlane: packetByteCostsByPlane
        )
    }

    private func fusedMetalRateControlBucketData(
        distortionsByPlane: [[[Float]]],
        packetByteCostsByPlane: [[[Int]]]
    ) throws -> MetalRateControlBucketData {
        let bucketData = try metalBackend.rateControlBucketDataBatch(packetByteCostsByPlane.indices.map { index in
            (
                distortions: distortionsByPlane[index],
                packetByteCosts: packetByteCostsByPlane[index]
            )
        })
        let bucketsByPlane = bucketData.bucketIndicesByPlane
        guard bucketsByPlane.count == packetByteCostsByPlane.count else {
            throw PyrowaveError.processFailed("Metal rate-control bucket batch returned \(bucketsByPlane.count) planes for \(packetByteCostsByPlane.count) inputs")
        }
        for planeIndex in packetByteCostsByPlane.indices {
            guard bucketsByPlane[planeIndex].count == packetByteCostsByPlane[planeIndex].count else {
                throw PyrowaveError.processFailed("Metal rate-control bucket pass returned \(bucketsByPlane[planeIndex].count) blocks for \(packetByteCostsByPlane[planeIndex].count) inputs")
            }
        }
        return MetalRateControlBucketData(indicesByPlane: bucketsByPlane, cumulativeSavings: bucketData.cumulativeSavings)
    }

    private func separateMetalRateControlBucketData(
        distortionsByPlane: [[[Float]]],
        packetByteCostsByPlane: [[[Int]]]
    ) throws -> MetalRateControlBucketData {
        let bucketsByPlane = try metalBackend.rateControlBucketIndicesBatch(packetByteCostsByPlane.indices.map { index in
            (
                distortions: distortionsByPlane[index],
                packetByteCosts: packetByteCostsByPlane[index]
            )
        })
        guard bucketsByPlane.count == packetByteCostsByPlane.count else {
            throw PyrowaveError.processFailed("Metal rate-control bucket batch returned \(bucketsByPlane.count) planes for \(packetByteCostsByPlane.count) inputs")
        }

        var flatBuckets = [[Int]]()
        var flatPacketByteCosts = [[Int]]()
        for planeIndex in packetByteCostsByPlane.indices {
            guard bucketsByPlane[planeIndex].count == packetByteCostsByPlane[planeIndex].count else {
                throw PyrowaveError.processFailed("Metal rate-control bucket pass returned \(bucketsByPlane[planeIndex].count) blocks for \(packetByteCostsByPlane[planeIndex].count) inputs")
            }
            flatBuckets.append(contentsOf: bucketsByPlane[planeIndex])
            flatPacketByteCosts.append(contentsOf: packetByteCostsByPlane[planeIndex])
        }
        let cumulativeSavings = try metalBackend.rateControlCumulativeBucketSavings(
            bucketIndices: flatBuckets,
            packetByteCosts: flatPacketByteCosts
        )
        return MetalRateControlBucketData(indicesByPlane: bucketsByPlane, cumulativeSavings: cumulativeSavings)
    }

    private func metalRateControlBucketDataFromTileStats(
        _ planes: [EncodedPlane],
        layout: PyrowaveBlockLayout,
        packetByteCostsByPlane: [[[Int]]]
    ) throws -> MetalRateControlBucketData {
        guard packetByteCostsByPlane.count == planes.count else {
            throw PyrowaveError.processFailed("packet byte-cost plane count \(packetByteCostsByPlane.count) does not match plane count \(planes.count)")
        }
        guard planes.allSatisfy({ $0.coefficientBuffer != nil }) else {
            throw PyrowaveError.processFailed("resident rate-control bucket path requires Metal coefficient buffers")
        }

        let descriptorsByPlane = planes.map { planeBlockDescriptors(plane: $0, layout: layout) }
        var statsDescriptorsByPlane = [[MetalRateControlStatsDescriptor]]()
        statsDescriptorsByPlane.reserveCapacity(planes.count)

        for (planeIndex, plane) in planes.enumerated() {
            let descriptors = descriptorsByPlane[planeIndex]
            guard packetByteCostsByPlane[planeIndex].count == descriptors.count else {
                throw PyrowaveError.processFailed("packet byte-cost count \(packetByteCostsByPlane[planeIndex].count) does not match block count \(descriptors.count)")
            }

            var statsDescriptors = [MetalRateControlStatsDescriptor]()
            statsDescriptors.reserveCapacity(descriptors.count * 16)
            for (descriptorIndex, descriptor) in descriptors.enumerated() {
                let quantCode = try quantCode(for: descriptor, descriptorIndex: descriptorIndex, plane: plane)
                let qScaleCodes = try qScaleCodes(for: descriptor, descriptorIndex: descriptorIndex, plane: plane)
                let rdoDistortionScale = PyrowaveQuantization.rdoDistortionScale(
                    level: descriptor.globalLevel,
                    component: plane.component,
                    band: descriptor.band,
                    chroma: layout.chroma
                )
                for tileY in 0..<4 {
                    for tileX in 0..<4 {
                        let tileIndex = tileY * 4 + tileX
                        statsDescriptors.append(MetalRateControlStatsDescriptor(
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
            statsDescriptorsByPlane.append(statsDescriptors)
        }

        let bucketData = try metalBackend.rateControlBucketDataFromTileStatsBatch(planes.indices.map { index in
            (
                coefficientBuffer: planes[index].coefficientBuffer!,
                coefficientCount: planes[index].coefficientCount,
                statsDescriptors: statsDescriptorsByPlane[index],
                packetByteCosts: packetByteCostsByPlane[index]
            )
        })
        let bucketsByPlane = bucketData.bucketIndicesByPlane
        guard bucketsByPlane.count == packetByteCostsByPlane.count else {
            throw PyrowaveError.processFailed("Metal rate-control tile bucket batch returned \(bucketsByPlane.count) planes for \(packetByteCostsByPlane.count) inputs")
        }
        for planeIndex in packetByteCostsByPlane.indices {
            guard bucketsByPlane[planeIndex].count == packetByteCostsByPlane[planeIndex].count else {
                throw PyrowaveError.processFailed("Metal rate-control tile bucket pass returned \(bucketsByPlane[planeIndex].count) blocks for \(packetByteCostsByPlane[planeIndex].count) inputs")
            }
        }
        return MetalRateControlBucketData(indicesByPlane: bucketsByPlane, cumulativeSavings: bucketData.cumulativeSavings)
    }

    private var frameHeaderSize: Int {
        8
    }

    private var fusedRateControlBucketBlockThreshold: Int {
        30_000
    }

    private func shouldUseTiledLevelZeroForwardDWT(width: Int, height: Int) -> Bool {
        let pixels = width * height
        return pixels <= 2_500_000 || pixels >= 18_000_000
    }

    private func shouldUseCombinedNV12Decode(width: Int, height: Int) -> Bool {
        let pixels = width * height
        return pixels <= 2_500_000 || pixels >= 18_000_000
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

    private func sparseBlocks(
        _ planes: [EncodedPlane],
        layout: PyrowaveBlockLayout,
        sequence: UInt8,
        quantLevelsByPlane: [[Int]]?,
        packetByteCostsByPlane: [[[Int]]]?,
        defaultQuantLevel: Int
    ) throws -> [[SparseBlock]] {
        guard planes.allSatisfy({ $0.coefficientBuffer != nil }) else {
            return try planes.enumerated().map { index, plane in
                try sparseBlocks(
                    plane,
                    layout: layout,
                    sequence: sequence,
                    quantLevels: quantLevelsByPlane?[index],
                    packetByteCosts: packetByteCostsByPlane?[index],
                    defaultQuantLevel: defaultQuantLevel
                )
            }
        }

        var packetDescriptorsByPlane = [[MetalSparsePacketEncodeDescriptor]]()
        var packetQScaleCodesByPlane = [[[UInt8]]]()
        var blockIndicesByPlane = [[Int]]()
        packetDescriptorsByPlane.reserveCapacity(planes.count)
        packetQScaleCodesByPlane.reserveCapacity(planes.count)
        blockIndicesByPlane.reserveCapacity(planes.count)

        for (planeIndex, plane) in planes.enumerated() {
            let descriptors = planeBlockDescriptors(plane: plane, layout: layout)
            let quantLevels = quantLevelsByPlane?[planeIndex]
            let packetByteCosts = packetByteCostsByPlane?[planeIndex]
            if let quantLevels, quantLevels.count != descriptors.count {
                throw PyrowaveError.processFailed("quant level count \(quantLevels.count) does not match block count \(descriptors.count)")
            }
            if let packetByteCosts, packetByteCosts.count != descriptors.count {
                throw PyrowaveError.processFailed("packet byte-cost count \(packetByteCosts.count) does not match block count \(descriptors.count)")
            }

            var packetDescriptors = [MetalSparsePacketEncodeDescriptor]()
            var packetQScaleCodes = [[UInt8]]()
            var blockIndices = [Int]()
            packetDescriptors.reserveCapacity(descriptors.count)
            packetQScaleCodes.reserveCapacity(descriptors.count)
            blockIndices.reserveCapacity(descriptors.count)

            for (descriptorIndex, descriptor) in descriptors.enumerated() {
                let quantLevel = quantLevels?[descriptorIndex] ?? defaultQuantLevel
                guard quantLevel >= 0 else {
                    throw PyrowaveError.invalidBitstream("negative quant level")
                }
                if let packetByteCosts,
                   quantLevel < packetByteCosts[descriptorIndex].count,
                   packetByteCosts[descriptorIndex][quantLevel] == 0 {
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
                    quantCode: UInt32(try quantCode(for: descriptor, descriptorIndex: descriptorIndex, plane: plane))
                ))
                packetQScaleCodes.append(try qScaleCodes(for: descriptor, descriptorIndex: descriptorIndex, plane: plane))
                blockIndices.append(descriptor.blockIndex)
            }

            packetDescriptorsByPlane.append(packetDescriptors)
            packetQScaleCodesByPlane.append(packetQScaleCodes)
            blockIndicesByPlane.append(blockIndices)
        }

        let packetsByPlane = try metalBackend.encodeSparsePacketsBatch(planes.indices.map { index in
            (
                coefficientBuffer: planes[index].coefficientBuffer!,
                coefficientCount: planes[index].coefficientCount,
                descriptors: packetDescriptorsByPlane[index],
                qScaleCodes: packetQScaleCodesByPlane[index]
            )
        }, sequence: sequence)
        return try planes.indices.map { planeIndex in
            let packets = packetsByPlane[planeIndex]
            guard packets.count == packetDescriptorsByPlane[planeIndex].count else {
                throw PyrowaveError.processFailed("Metal sparse packet encode returned \(packets.count) packets for \(packetDescriptorsByPlane[planeIndex].count) descriptors")
            }
            return packets.indices.compactMap { index in
                packets[index].map { SparseBlock(blockIndex: blockIndicesByPlane[planeIndex][index], data: $0) }
            }
        }
    }

    private func makeGPUFramePlanes(
        _ planes: [EncodedPlane],
        layout: PyrowaveBlockLayout,
        sequence: UInt8,
        quantLevelsByPlane: [[Int]]?,
        packetByteCostsByPlane: [[[Int]]]?,
        defaultQuantLevel: Int,
        reusesPacketBuffers: Bool = false,
        packetOutputStorageMode: MTLStorageMode = .private
    ) throws -> [PyrowaveGPUFramePlane] {
        if let packetByteCostsByPlane, packetByteCostsByPlane.count != planes.count {
            throw PyrowaveError.processFailed("packet byte-cost plane count \(packetByteCostsByPlane.count) does not match plane count \(planes.count)")
        }
        let usesResidentQScaleBuffers = packetByteCostsByPlane == nil && planes.allSatisfy { $0.qScaleBuffer != nil }
        let usesCachedPacketDescriptors = usesResidentQScaleBuffers && quantLevelsByPlane == nil && defaultQuantLevel == 0
        var packetDescriptorsByPlane = [[MetalSparsePacketEncodeDescriptor]]()
        var packetDescriptorCountsByPlane = [Int]()
        var packetDescriptorBuffersByPlane = [MTLBuffer?]()
        var packetQScaleCodesByPlane = [[[UInt8]]]()
        var decodeDescriptorCountsByPlane = [Int]()
        var decodeDescriptorBuffersByPlane = [MTLBuffer]()
        var blockIndexCountsByPlane = [Int]()
        var blockIndexBuffersByPlane = [MTLBuffer]()
        var selectedQuantLevelCountsByPlane = [Int]()
        var selectedQuantLevelBuffersByPlane = [MTLBuffer]()
        packetDescriptorsByPlane.reserveCapacity(planes.count)
        packetDescriptorCountsByPlane.reserveCapacity(planes.count)
        packetDescriptorBuffersByPlane.reserveCapacity(planes.count)
        packetQScaleCodesByPlane.reserveCapacity(planes.count)
        decodeDescriptorCountsByPlane.reserveCapacity(planes.count)
        decodeDescriptorBuffersByPlane.reserveCapacity(planes.count)
        blockIndexCountsByPlane.reserveCapacity(planes.count)
        blockIndexBuffersByPlane.reserveCapacity(planes.count)
        selectedQuantLevelCountsByPlane.reserveCapacity(planes.count)
        selectedQuantLevelBuffersByPlane.reserveCapacity(planes.count)
        let maxPacketBytes = PyrowaveCoefficientBlockCodec.maximumEncodedBlockBytes

        for (planeIndex, plane) in planes.enumerated() {
            let descriptors = planeBlockDescriptors(plane: plane, layout: layout)
            let quantLevels = quantLevelsByPlane?[planeIndex]
            let packetByteCosts = packetByteCostsByPlane?[planeIndex]
            if let quantLevels, quantLevels.count != descriptors.count {
                throw PyrowaveError.processFailed("quant level count \(quantLevels.count) does not match block count \(descriptors.count)")
            }
            if let packetByteCosts, packetByteCosts.count != descriptors.count {
                throw PyrowaveError.processFailed("packet byte-cost count \(packetByteCosts.count) does not match block count \(descriptors.count)")
            }
            if usesResidentQScaleBuffers, plane.qScaleDescriptorCount != descriptors.count {
                throw PyrowaveError.processFailed("resident q-scale descriptor count \(plane.qScaleDescriptorCount) does not match block count \(descriptors.count)")
            }
            if usesCachedPacketDescriptors {
                let cached = try cachedSparsePacketEncodeDescriptors(
                    plane: plane,
                    layout: layout,
                    descriptors: descriptors,
                    defaultQuantLevel: defaultQuantLevel,
                    maxPacketBytes: maxPacketBytes
                )
                packetDescriptorsByPlane.append([])
                packetDescriptorCountsByPlane.append(cached.packetDescriptorCount)
                packetDescriptorBuffersByPlane.append(cached.descriptorBuffer)
                decodeDescriptorCountsByPlane.append(cached.decodeDescriptorCount)
                decodeDescriptorBuffersByPlane.append(cached.decodeDescriptorBuffer)
                blockIndexCountsByPlane.append(cached.blockIndexCount)
                blockIndexBuffersByPlane.append(cached.blockIndexBuffer)
                selectedQuantLevelCountsByPlane.append(cached.selectedQuantLevelCount)
                selectedQuantLevelBuffersByPlane.append(cached.selectedQuantLevelBuffer)
                continue
            }

            var packetDescriptors = [MetalSparsePacketEncodeDescriptor]()
            var packetQScaleCodes = [[UInt8]]()
            var decodeDescriptors = [MetalSparsePacketDecodeDescriptor]()
            var blockIndices = [UInt32]()
            var selectedQuantLevels = [UInt32]()
            packetDescriptors.reserveCapacity(descriptors.count)
            packetQScaleCodes.reserveCapacity(descriptors.count)
            decodeDescriptors.reserveCapacity(descriptors.count)
            blockIndices.reserveCapacity(descriptors.count)
            selectedQuantLevels.reserveCapacity(descriptors.count)

            for (descriptorIndex, descriptor) in descriptors.enumerated() {
                let quantLevel = quantLevels?[descriptorIndex] ?? defaultQuantLevel
                guard quantLevel >= 0,
                      quantLevel < (packetByteCosts?[descriptorIndex].count ?? PyrowaveBlockStats.candidateCount) else {
                    throw PyrowaveError.invalidBitstream("invalid quant level")
                }
                if let packetByteCosts, packetByteCosts[descriptorIndex][quantLevel] == 0 {
                    continue
                }
                guard quantLevel <= Int(UInt32.max),
                      descriptor.blockIndex <= Int(UInt32.max),
                      plane.paddedWidth <= Int(UInt32.max),
                      packetDescriptors.count * maxPacketBytes <= Int(UInt32.max) - maxPacketBytes else {
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
                    quantCode: UInt32(try quantCode(for: descriptor, descriptorIndex: descriptorIndex, plane: plane))
                ))
                if !usesResidentQScaleBuffers {
                    packetQScaleCodes.append(try qScaleCodes(for: descriptor, descriptorIndex: descriptorIndex, plane: plane))
                }
                blockIndices.append(UInt32(descriptor.blockIndex))
                selectedQuantLevels.append(UInt32(quantLevel))
                let packetOffset = packetDescriptors.count - 1
                decodeDescriptors.append(MetalSparsePacketDecodeDescriptor(
                    packetOffset: UInt32(packetOffset * maxPacketBytes),
                    payloadEnd: UInt32((packetOffset + 1) * maxPacketBytes),
                    originX: UInt32(descriptor.originX),
                    originY: UInt32(descriptor.originY),
                    validWidth: UInt32(descriptor.validWidth),
                    validHeight: UInt32(descriptor.validHeight),
                    stride: UInt32(plane.paddedWidth)
                ))
            }
            packetDescriptorsByPlane.append(packetDescriptors)
            packetDescriptorCountsByPlane.append(packetDescriptors.count)
            packetDescriptorBuffersByPlane.append(nil)
            if !usesResidentQScaleBuffers {
                packetQScaleCodesByPlane.append(packetQScaleCodes)
            }
            decodeDescriptorCountsByPlane.append(decodeDescriptors.count)
            decodeDescriptorBuffersByPlane.append(try metalBackend.makeStaticSharedBuffer(bytes: decodeDescriptors))
            blockIndexCountsByPlane.append(blockIndices.count)
            blockIndexBuffersByPlane.append(try metalBackend.makeStaticSharedBuffer(bytes: blockIndices))
            selectedQuantLevelCountsByPlane.append(selectedQuantLevels.count)
            selectedQuantLevelBuffersByPlane.append(try metalBackend.makeStaticSharedBuffer(bytes: selectedQuantLevels))
        }

        let encodedPacketPlanes: [MetalSparsePacketEncodedPlane]
        if usesResidentQScaleBuffers {
            encodedPacketPlanes = try metalBackend.encodeSparsePacketBuffersBatchResidentQScales(planes.indices.map { index in
                guard let coefficientBuffer = planes[index].coefficientBuffer,
                      let qScaleBuffer = planes[index].qScaleBuffer else {
                    throw PyrowaveError.processFailed("GPU frame packet emission requires resident Metal coefficient and q-scale buffers")
                }
                return (
                    coefficientBuffer: coefficientBuffer,
                    coefficientCount: planes[index].coefficientCount,
                    descriptorCount: packetDescriptorCountsByPlane[index],
                    descriptors: packetDescriptorBuffersByPlane[index] == nil ? packetDescriptorsByPlane[index] : nil,
                    descriptorBuffer: packetDescriptorBuffersByPlane[index],
                    qScaleBuffer: qScaleBuffer,
                    qScaleDescriptorCount: planes[index].qScaleDescriptorCount
                )
            }, sequence: sequence, outputStorageMode: packetOutputStorageMode, reusesOutputBuffers: reusesPacketBuffers)
        } else {
            encodedPacketPlanes = try metalBackend.encodeSparsePacketBuffersBatch(planes.indices.map { index in
                guard let coefficientBuffer = planes[index].coefficientBuffer else {
                    throw PyrowaveError.processFailed("GPU frame packet emission requires resident Metal coefficient buffers")
                }
                return (
                    coefficientBuffer: coefficientBuffer,
                    coefficientCount: planes[index].coefficientCount,
                    descriptors: packetDescriptorsByPlane[index],
                    qScaleCodes: packetQScaleCodesByPlane[index]
                )
            }, sequence: sequence, outputStorageMode: packetOutputStorageMode, reusesOutputBuffers: reusesPacketBuffers)
        }
        guard encodedPacketPlanes.count == planes.count else {
            throw PyrowaveError.processFailed("Metal sparse packet encode returned \(encodedPacketPlanes.count) planes for \(planes.count) inputs")
        }
        return planes.indices.map { index in
            PyrowaveGPUFramePlane(
                component: planes[index].component,
                paddedWidth: planes[index].paddedWidth,
                paddedHeight: planes[index].paddedHeight,
                levels: planes[index].levels,
                sampleCount: planes[index].coefficientCount,
                encoded: encodedPacketPlanes[index],
                decodeDescriptorCount: decodeDescriptorCountsByPlane[index],
                decodeDescriptorBuffer: decodeDescriptorBuffersByPlane[index],
                blockIndexCount: blockIndexCountsByPlane[index],
                blockIndexBuffer: blockIndexBuffersByPlane[index],
                selectedQuantLevelCount: selectedQuantLevelCountsByPlane[index],
                selectedQuantLevelBuffer: selectedQuantLevelBuffersByPlane[index]
            )
        }
    }

    private func cachedSparsePacketEncodeDescriptors(
        plane: EncodedPlane,
        layout: PyrowaveBlockLayout,
        descriptors: [PlaneBlockDescriptor],
        defaultQuantLevel: Int,
        maxPacketBytes: Int
    ) throws -> SparsePacketEncodeDescriptorCacheEntry {
        let key = sparsePacketEncodeDescriptorCacheKey(
            plane: plane,
            layout: layout,
            descriptors: descriptors,
            defaultQuantLevel: defaultQuantLevel
        )
        geometryCacheLock.lock()
        if let cached = sparsePacketEncodeDescriptorCache[key] {
            geometryCacheLock.unlock()
            return cached
        }
        geometryCacheLock.unlock()

        var packetDescriptors = [MetalSparsePacketEncodeDescriptor]()
        var decodeDescriptors = [MetalSparsePacketDecodeDescriptor]()
        var blockIndices = [UInt32]()
        var selectedQuantLevels = [UInt32]()
        packetDescriptors.reserveCapacity(descriptors.count)
        decodeDescriptors.reserveCapacity(descriptors.count)
        blockIndices.reserveCapacity(descriptors.count)
        selectedQuantLevels.reserveCapacity(descriptors.count)

        for (descriptorIndex, descriptor) in descriptors.enumerated() {
            guard defaultQuantLevel >= 0,
                  defaultQuantLevel <= Int(UInt32.max),
                  descriptor.blockIndex <= Int(UInt32.max),
                  plane.paddedWidth <= Int(UInt32.max),
                  packetDescriptors.count * maxPacketBytes <= Int(UInt32.max) - maxPacketBytes else {
                throw PyrowaveError.invalidDimensions
            }
            packetDescriptors.append(MetalSparsePacketEncodeDescriptor(
                originX: UInt32(descriptor.originX),
                originY: UInt32(descriptor.originY),
                validWidth: UInt32(descriptor.validWidth),
                validHeight: UInt32(descriptor.validHeight),
                stride: UInt32(plane.paddedWidth),
                blockIndex: UInt32(descriptor.blockIndex),
                quantLevel: UInt32(defaultQuantLevel),
                quantCode: UInt32(try quantCode(for: descriptor, descriptorIndex: descriptorIndex, plane: plane))
            ))
            blockIndices.append(UInt32(descriptor.blockIndex))
            selectedQuantLevels.append(UInt32(defaultQuantLevel))
            let packetOffset = packetDescriptors.count - 1
            decodeDescriptors.append(MetalSparsePacketDecodeDescriptor(
                packetOffset: UInt32(packetOffset * maxPacketBytes),
                payloadEnd: UInt32((packetOffset + 1) * maxPacketBytes),
                originX: UInt32(descriptor.originX),
                originY: UInt32(descriptor.originY),
                validWidth: UInt32(descriptor.validWidth),
                validHeight: UInt32(descriptor.validHeight),
                stride: UInt32(plane.paddedWidth)
            ))
        }

        let entry = SparsePacketEncodeDescriptorCacheEntry(
            packetDescriptorCount: packetDescriptors.count,
            descriptorBuffer: try metalBackend.makeStaticSharedBuffer(bytes: packetDescriptors),
            decodeDescriptorCount: decodeDescriptors.count,
            decodeDescriptorBuffer: try metalBackend.makeStaticSharedBuffer(bytes: decodeDescriptors),
            blockIndexCount: blockIndices.count,
            blockIndexBuffer: try metalBackend.makeStaticSharedBuffer(bytes: blockIndices),
            selectedQuantLevelCount: selectedQuantLevels.count,
            selectedQuantLevelBuffer: try metalBackend.makeStaticSharedBuffer(bytes: selectedQuantLevels)
        )
        geometryCacheLock.lock()
        sparsePacketEncodeDescriptorCache[key] = entry
        geometryCacheLock.unlock()
        return entry
    }

    private func sparsePacketEncodeDescriptorCacheKey(
        plane: EncodedPlane,
        layout: PyrowaveBlockLayout,
        descriptors: [PlaneBlockDescriptor],
        defaultQuantLevel: Int
    ) -> SparsePacketEncodeDescriptorCacheKey {
        var descriptorHash = Hasher()
        descriptorHash.combine(descriptors.count)
        for descriptor in descriptors {
            descriptorHash.combine(descriptor.blockIndex)
            descriptorHash.combine(descriptor.globalLevel)
            descriptorHash.combine(descriptor.level)
            descriptorHash.combine(descriptor.band)
            descriptorHash.combine(descriptor.originX)
            descriptorHash.combine(descriptor.originY)
            descriptorHash.combine(descriptor.validWidth)
            descriptorHash.combine(descriptor.validHeight)
        }
        var quantCodeHash = Hasher()
        quantCodeHash.combine(plane.quantCodesByDescriptor.count)
        for quantCode in plane.quantCodesByDescriptor {
            quantCodeHash.combine(quantCode)
        }
        return SparsePacketEncodeDescriptorCacheKey(
            layoutWidth: layout.width,
            layoutHeight: layout.height,
            chroma: layout.chroma.rawValue,
            component: plane.component,
            paddedWidth: plane.paddedWidth,
            paddedHeight: plane.paddedHeight,
            levels: plane.levels,
            defaultQuantLevel: defaultQuantLevel,
            descriptorHash: descriptorHash.finalize(),
            quantCodeHash: quantCodeHash.finalize()
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
                quantCode: UInt32(try quantCode(for: descriptor, descriptorIndex: index, plane: plane))
            ))
            packetQScaleCodes.append(try qScaleCodes(for: descriptor, descriptorIndex: index, plane: plane))
            blockIndices.append(descriptor.blockIndex)
        }

        let packets = try encodeSparsePackets(
            plane,
            descriptors: packetDescriptors,
            qScaleCodes: packetQScaleCodes,
            sequence: sequence,
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

    private func metalSparsePacketByteCosts(
        planes: [EncodedPlane],
        descriptorsByPlane: [[PlaneBlockDescriptor]]
    ) throws -> [[[Int]]] {
        guard planes.count == descriptorsByPlane.count else {
            throw PyrowaveError.processFailed("packet byte-cost plane count does not match descriptor plane count")
        }
        guard planes.allSatisfy({ $0.coefficientBuffer != nil }) else {
            return try planes.indices.map { index in
                try metalSparsePacketByteCosts(
                    plane: planes[index],
                    descriptors: descriptorsByPlane[index]
                )
            }
        }

        let packetCostDescriptorsByPlane = zip(planes, descriptorsByPlane).map { plane, descriptors in
            metalPacketByteCostDescriptors(descriptors: descriptors, stride: plane.paddedWidth)
        }
        let costsByPlane = try metalBackend.packetByteCostsBatch(planes.indices.map { index in
            (
                coefficientBuffer: planes[index].coefficientBuffer!,
                coefficientCount: planes[index].coefficientCount,
                descriptors: packetCostDescriptorsByPlane[index]
            )
        })
        guard costsByPlane.count == planes.count else {
            throw PyrowaveError.processFailed("Metal packet byte-cost returned \(costsByPlane.count) planes for \(planes.count) inputs")
        }
        for planeIndex in planes.indices {
            guard costsByPlane[planeIndex].count == descriptorsByPlane[planeIndex].count else {
                throw PyrowaveError.processFailed("Metal packet byte-cost returned \(costsByPlane[planeIndex].count) block costs for \(descriptorsByPlane[planeIndex].count) descriptors")
            }
        }
        return costsByPlane
    }

    private func makeSparseRateControlInputs(
        _ planes: [EncodedPlane],
        layout: PyrowaveBlockLayout,
        packetByteCostsByPlane: [[[Int]]]? = nil
    ) throws -> SparseRateControlInputs {
        if let packetByteCostsByPlane, packetByteCostsByPlane.count != planes.count {
            throw PyrowaveError.processFailed("packet byte-cost plane count \(packetByteCostsByPlane.count) does not match plane count \(planes.count)")
        }
        guard planes.allSatisfy({ $0.coefficientBuffer != nil }) else {
            let blocksByPlane = try makeRateControlBlocks(planes, layout: layout, packetByteCostsByPlane: packetByteCostsByPlane)
            return SparseRateControlInputs(
                distortionsByPlane: blocksByPlane.map { blocks in
                    blocks.map { block in
                        (0..<PyrowaveBlockStats.candidateCount).map { block.distortion(quantLevel: $0) }
                    }
                },
                packetByteCostsByPlane: blocksByPlane.map { $0.map(\.packetByteCosts) }
            )
        }

        let descriptorsByPlane = planes.map { planeBlockDescriptors(plane: $0, layout: layout) }
        var statsDescriptorsByPlane = [[MetalRateControlStatsDescriptor]]()
        statsDescriptorsByPlane.reserveCapacity(planes.count)

        for (planeIndex, plane) in planes.enumerated() {
            var statsDescriptors = [MetalRateControlStatsDescriptor]()
            statsDescriptors.reserveCapacity(descriptorsByPlane[planeIndex].count * 16)
            for (descriptorIndex, descriptor) in descriptorsByPlane[planeIndex].enumerated() {
                let quantCode = try quantCode(for: descriptor, descriptorIndex: descriptorIndex, plane: plane)
                let qScaleCodes = try qScaleCodes(for: descriptor, descriptorIndex: descriptorIndex, plane: plane)
                let rdoDistortionScale = PyrowaveQuantization.rdoDistortionScale(
                    level: descriptor.globalLevel,
                    component: plane.component,
                    band: descriptor.band,
                    chroma: layout.chroma
                )
                for tileY in 0..<4 {
                    for tileX in 0..<4 {
                        let tileIndex = tileY * 4 + tileX
                        statsDescriptors.append(MetalRateControlStatsDescriptor(
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
            statsDescriptorsByPlane.append(statsDescriptors)
        }

        let tileStatsByPlane = try metalBackend.rateControlTileStatsFlatBatch(planes.indices.map { index in
            (
                coefficientBuffer: planes[index].coefficientBuffer!,
                coefficientCount: planes[index].coefficientCount,
                descriptors: statsDescriptorsByPlane[index]
            )
        })

        let selectedPacketByteCostsByPlane: [[[Int]]]
        if let packetByteCostsByPlane {
            selectedPacketByteCostsByPlane = packetByteCostsByPlane
        } else {
            selectedPacketByteCostsByPlane = try metalSparsePacketByteCosts(
                planes: planes,
                descriptorsByPlane: descriptorsByPlane
            )
        }

        let distortionsByPlane = try planes.indices.map { planeIndex -> [[Float]] in
            let descriptors = descriptorsByPlane[planeIndex]
            let tileStats = tileStatsByPlane[planeIndex]
            let expectedTileCount = descriptors.count * 16
            guard tileStats.numPlanes.count == expectedTileCount else {
                throw PyrowaveError.processFailed("Metal rate-control returned \(tileStats.numPlanes.count) tile stats for \(expectedTileCount) descriptors")
            }
            guard tileStats.stats.count == expectedTileCount * PyrowaveBlockStats.candidateCount else {
                throw PyrowaveError.processFailed("Metal rate-control returned \(tileStats.stats.count) quant stats for \(expectedTileCount) tiles")
            }
            guard selectedPacketByteCostsByPlane[planeIndex].count == descriptors.count else {
                throw PyrowaveError.processFailed("Metal packet byte-cost returned \(selectedPacketByteCostsByPlane[planeIndex].count) block costs for \(descriptors.count) descriptors")
            }

            return descriptors.indices.map { descriptorIndex in
                let firstTile = descriptorIndex * 16
                var distortions = Array(repeating: Float(0), count: PyrowaveBlockStats.candidateCount)
                for tileOffset in 0..<16 {
                    let tileIndex = firstTile + tileOffset
                    let statsStart = tileIndex * PyrowaveBlockStats.candidateCount
                    for candidate in 0..<PyrowaveBlockStats.candidateCount {
                        let stat = tileStats.stats[statsStart + candidate]
                        distortions[candidate] += PyrowaveQuantStats.quantizedSquareError(stat.squareError)
                    }
                }
                return distortions
            }
        }

        return SparseRateControlInputs(
            distortionsByPlane: distortionsByPlane,
            packetByteCostsByPlane: selectedPacketByteCostsByPlane
        )
    }

    private func estimateFrameBytes(
        packetByteCostsByPlane: [[[Int]]],
        thresholdsByPlane: [[Int]],
        fixedHeaderBytes: Int
    ) -> Int {
        var byteCount = fixedHeaderBytes
        for (planeCosts, thresholds) in zip(packetByteCostsByPlane, thresholdsByPlane) {
            for (blockCosts, threshold) in zip(planeCosts, thresholds) {
                byteCount += blockCosts[min(max(threshold, 0), PyrowaveBlockStats.candidateCount - 1)]
            }
        }
        return byteCount
    }

    private func makeRateControlOperations(
        packetByteCostsByPlane: [[[Int]]],
        bucketIndicesByPlane: [[[Int]]]
    ) -> [PyrowaveRateController.RDOperation] {
        var buckets = Array(repeating: [PyrowaveRateController.RDOperation](), count: 128)

        for (planeIndex, planeCosts) in packetByteCostsByPlane.enumerated() {
            for (blockIndex, blockCosts) in planeCosts.enumerated() {
                guard planeIndex < bucketIndicesByPlane.count,
                      blockIndex < bucketIndicesByPlane[planeIndex].count,
                      bucketIndicesByPlane[planeIndex][blockIndex].count == PyrowaveBlockStats.candidateCount,
                      blockCosts.count == PyrowaveBlockStats.candidateCount else {
                    continue
                }

                let bucketIndices = bucketIndicesByPlane[planeIndex][blockIndex]
                for quantLevel in 1..<PyrowaveBlockStats.candidateCount {
                    let saving = blockCosts[quantLevel - 1] - blockCosts[quantLevel]
                    guard saving > 0 else {
                        continue
                    }

                    let bucket = bucketIndices[quantLevel]
                    guard bucket >= 0, bucket < buckets.count else {
                        continue
                    }
                    buckets[bucket].append(PyrowaveRateController.RDOperation(
                        bucket: bucket,
                        planeIndex: planeIndex,
                        blockIndex: blockIndex,
                        quantLevel: quantLevel,
                        saving: saving
                    ))
                }
            }
        }

        return buckets.flatMap { $0 }
    }

    private func makeRateControlBlocks(_ plane: EncodedPlane, layout: PyrowaveBlockLayout) throws -> [PyrowaveRateControlBlock] {
        let descriptors = planeBlockDescriptors(plane: plane, layout: layout)
        return try makeRateControlBlocksWithMetal(plane, descriptors: descriptors, layout: layout, backend: metalBackend)
    }

    private func makeRateControlBlocks(
        _ planes: [EncodedPlane],
        layout: PyrowaveBlockLayout,
        packetByteCostsByPlane: [[[Int]]]? = nil
    ) throws -> [[PyrowaveRateControlBlock]] {
        if let packetByteCostsByPlane, packetByteCostsByPlane.count != planes.count {
            throw PyrowaveError.processFailed("packet byte-cost plane count \(packetByteCostsByPlane.count) does not match plane count \(planes.count)")
        }
        guard planes.allSatisfy({ $0.coefficientBuffer != nil }) else {
            return try planes.map { try makeRateControlBlocks($0, layout: layout) }
        }

        let descriptorsByPlane = planes.map { planeBlockDescriptors(plane: $0, layout: layout) }
        var statsDescriptorsByPlane = [[MetalRateControlStatsDescriptor]]()
        statsDescriptorsByPlane.reserveCapacity(planes.count)

        for (planeIndex, plane) in planes.enumerated() {
            var statsDescriptors = [MetalRateControlStatsDescriptor]()
            statsDescriptors.reserveCapacity(descriptorsByPlane[planeIndex].count * 16)
            for (descriptorIndex, descriptor) in descriptorsByPlane[planeIndex].enumerated() {
                let quantCode = try quantCode(for: descriptor, descriptorIndex: descriptorIndex, plane: plane)
                let qScaleCodes = try qScaleCodes(for: descriptor, descriptorIndex: descriptorIndex, plane: plane)
                let rdoDistortionScale = PyrowaveQuantization.rdoDistortionScale(
                    level: descriptor.globalLevel,
                    component: plane.component,
                    band: descriptor.band,
                    chroma: layout.chroma
                )
                for tileY in 0..<4 {
                    for tileX in 0..<4 {
                        let tileIndex = tileY * 4 + tileX
                        statsDescriptors.append(MetalRateControlStatsDescriptor(
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
            statsDescriptorsByPlane.append(statsDescriptors)
        }

        let tileStatsByPlane = try metalBackend.rateControlTileStatsBatch(planes.indices.map { index in
            (
                coefficientBuffer: planes[index].coefficientBuffer!,
                coefficientCount: planes[index].coefficientCount,
                descriptors: statsDescriptorsByPlane[index]
            )
        })
        let selectedPacketByteCostsByPlane: [[[Int]]]
        if let packetByteCostsByPlane {
            selectedPacketByteCostsByPlane = packetByteCostsByPlane
        } else {
            selectedPacketByteCostsByPlane = try metalSparsePacketByteCosts(
                planes: planes,
                descriptorsByPlane: descriptorsByPlane
            )
        }

        return try planes.indices.map { planeIndex in
            let descriptors = descriptorsByPlane[planeIndex]
            let tileStats = tileStatsByPlane[planeIndex]
            let packetByteCosts = selectedPacketByteCostsByPlane[planeIndex]
            guard tileStats.count == statsDescriptorsByPlane[planeIndex].count else {
                throw PyrowaveError.processFailed("Metal rate-control returned \(tileStats.count) tile stats for \(statsDescriptorsByPlane[planeIndex].count) descriptors")
            }
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
    }

    private func makeRateControlBlocksWithMetal(
        _ plane: EncodedPlane,
        descriptors: [PlaneBlockDescriptor],
        layout: PyrowaveBlockLayout,
        backend: MetalPyrowaveBackend
    ) throws -> [PyrowaveRateControlBlock] {
        var metalDescriptors = [MetalRateControlStatsDescriptor]()
        metalDescriptors.reserveCapacity(descriptors.count * 16)

        for (descriptorIndex, descriptor) in descriptors.enumerated() {
            let quantCode = try quantCode(for: descriptor, descriptorIndex: descriptorIndex, plane: plane)
            let qScaleCodes = try qScaleCodes(for: descriptor, descriptorIndex: descriptorIndex, plane: plane)
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
        sequence: UInt8,
        backend: MetalPyrowaveBackend
    ) throws -> [Data?] {
        if let coefficientBuffer = plane.coefficientBuffer {
            return try backend.encodeSparsePackets(
                coefficientBuffer: coefficientBuffer,
                coefficientCount: plane.coefficientCount,
                descriptors: descriptors,
                qScaleCodes: qScaleCodes,
                sequence: sequence
            )
        }
        return try backend.encodeSparsePackets(
            coefficients: plane.coefficients,
            descriptors: descriptors,
            qScaleCodes: qScaleCodes,
            sequence: sequence
        )
    }

    private func makeDecodedPlane(component: Int, width: Int, height: Int, chroma: ChromaSubsampling, layout: PyrowaveBlockLayout) throws -> DecodedPlane {
        let template = try cachedDecodedPlaneTemplate(component: component, width: width, height: height, chroma: chroma, layout: layout)
        return DecodedPlane(
            visibleWidth: template.visibleWidth,
            visibleHeight: template.visibleHeight,
            paddedWidth: template.paddedWidth,
            paddedHeight: template.paddedHeight,
            levels: template.levels,
            samples: Array(repeating: 0, count: template.sampleCount),
            descriptorsByBlockIndex: template.descriptorsByBlockIndex
        )
    }

    private func makeGPUDecodedPlane(component: Int, width: Int, height: Int, chroma: ChromaSubsampling, layout: PyrowaveBlockLayout) throws -> GPUDecodedPlane {
        let template = try cachedDecodedPlaneTemplate(component: component, width: width, height: height, chroma: chroma, layout: layout)
        return GPUDecodedPlane(
            visibleWidth: template.visibleWidth,
            visibleHeight: template.visibleHeight,
            paddedWidth: template.paddedWidth,
            paddedHeight: template.paddedHeight,
            levels: template.levels,
            sampleCount: template.sampleCount,
            samples: gpuDecodedPlanePlaceholder,
            descriptorsByBlockIndex: template.descriptorsByBlockIndex
        )
    }

    private func cachedDecodedPlaneTemplate(
        component: Int,
        width: Int,
        height: Int,
        chroma: ChromaSubsampling,
        layout: PyrowaveBlockLayout
    ) throws -> DecodedPlaneTemplate {
        let key = DecodedPlaneTemplateCacheKey(
            layoutWidth: layout.width,
            layoutHeight: layout.height,
            chroma: chroma.rawValue,
            component: component
        )
        geometryCacheLock.lock()
        if let cached = decodedPlaneTemplateCache[key] {
            geometryCacheLock.unlock()
            return cached
        }
        geometryCacheLock.unlock()

        let geometry = planeGeometry(component: component, frameWidth: width, frameHeight: height, chroma: chroma, requestedLevels: PyrowaveBitstream.decompositionLevels)
        let levels = Wavelet.usableLevels(width: geometry.paddedWidth, height: geometry.paddedHeight, requested: geometry.requestedLevels)
        let descriptors = planeBlockDescriptors(
            component: component,
            paddedWidth: geometry.paddedWidth,
            paddedHeight: geometry.paddedHeight,
            levels: levels,
            layout: layout
        )
        let template = DecodedPlaneTemplate(
            visibleWidth: geometry.visibleWidth,
            visibleHeight: geometry.visibleHeight,
            paddedWidth: geometry.paddedWidth,
            paddedHeight: geometry.paddedHeight,
            levels: levels,
            sampleCount: geometry.paddedWidth * geometry.paddedHeight,
            descriptorsByBlockIndex: Dictionary(uniqueKeysWithValues: descriptors.map { ($0.blockIndex, $0) })
        )

        geometryCacheLock.lock()
        decodedPlaneTemplateCache[key] = template
        geometryCacheLock.unlock()
        return template
    }

    private func cachedSparseBlockTargets(layout: PyrowaveBlockLayout) throws -> [SparseBlockTarget?] {
        let key = LayoutCacheKey(width: layout.width, height: layout.height, chroma: layout.chroma.rawValue)
        geometryCacheLock.lock()
        if let cached = sparseBlockTargetCache[key] {
            geometryCacheLock.unlock()
            return cached
        }
        geometryCacheLock.unlock()

        var targets = Array<SparseBlockTarget?>(repeating: nil, count: layout.descriptors.count)
        for component in 0..<PyrowaveBitstream.componentCount {
            let template = try cachedDecodedPlaneTemplate(
                component: component,
                width: layout.width,
                height: layout.height,
                chroma: layout.chroma,
                layout: layout
            )
            for (blockIndex, descriptor) in template.descriptorsByBlockIndex {
                guard blockIndex >= 0, blockIndex < layout.descriptors.count else {
                    throw PyrowaveError.invalidBitstream("sparse block index has no plane mapping")
                }
                guard targets[blockIndex] == nil else {
                    throw PyrowaveError.invalidBitstream("duplicate sparse block plane mapping")
                }
                targets[blockIndex] = SparseBlockTarget(planeIndex: component, descriptor: descriptor)
            }
        }

        geometryCacheLock.lock()
        sparseBlockTargetCache[key] = targets
        geometryCacheLock.unlock()
        return targets
    }

    private func sparseBlockTargets(decodedPlanes: [DecodedPlane], totalBlocks: Int) throws -> [SparseBlockTarget?] {
        var targets = Array<SparseBlockTarget?>(repeating: nil, count: totalBlocks)
        for (planeIndex, plane) in decodedPlanes.enumerated() {
            for (blockIndex, descriptor) in plane.descriptorsByBlockIndex {
                guard blockIndex >= 0, blockIndex < totalBlocks else {
                    throw PyrowaveError.invalidBitstream("sparse block index has no plane mapping")
                }
                guard targets[blockIndex] == nil else {
                    throw PyrowaveError.invalidBitstream("duplicate sparse block plane mapping")
                }
                targets[blockIndex] = SparseBlockTarget(planeIndex: planeIndex, descriptor: descriptor)
            }
        }
        return targets
    }

    private func sparseBlockTargets(decodedPlanes: [GPUDecodedPlane], totalBlocks: Int) throws -> [SparseBlockTarget?] {
        var targets = Array<SparseBlockTarget?>(repeating: nil, count: totalBlocks)
        for (planeIndex, plane) in decodedPlanes.enumerated() {
            for (blockIndex, descriptor) in plane.descriptorsByBlockIndex {
                guard blockIndex >= 0, blockIndex < totalBlocks else {
                    throw PyrowaveError.invalidBitstream("sparse block index has no plane mapping")
                }
                guard targets[blockIndex] == nil else {
                    throw PyrowaveError.invalidBitstream("duplicate sparse block plane mapping")
                }
                targets[blockIndex] = SparseBlockTarget(planeIndex: planeIndex, descriptor: descriptor)
            }
        }
        return targets
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

    private func appendSparsePacketDecodeDescriptor(
        header: PyrowavePacketHeader,
        blockStart: Int,
        reader: inout BinaryReader,
        target: SparseBlockTarget,
        decodedPlane: GPUDecodedPlane,
        descriptors: inout [MetalSparsePacketDecodeDescriptor]
    ) throws {
        let payloadEnd = blockStart + Int(header.payloadWords) * 4
        guard payloadEnd >= reader.offset else {
            throw PyrowaveError.invalidBitstream("payload_words is not large enough")
        }
        guard payloadEnd <= reader.data.count else {
            throw PyrowaveError.truncatedInput
        }

        guard header.ballot != 0 else {
            if reader.offset < payloadEnd {
                guard reader.data[reader.offset..<payloadEnd].allSatisfy({ $0 == 0 }) else {
                    throw PyrowaveError.invalidBitstream("non-zero coefficient packet padding")
                }
            }
            try reader.seek(to: payloadEnd)
            return
        }

        guard blockStart <= Int(UInt32.max),
              payloadEnd <= Int(UInt32.max),
              target.descriptor.originX <= Int(UInt32.max),
              target.descriptor.originY <= Int(UInt32.max),
              target.descriptor.validWidth <= Int(UInt32.max),
              target.descriptor.validHeight <= Int(UInt32.max),
              decodedPlane.paddedWidth <= Int(UInt32.max) else {
            throw PyrowaveError.invalidDimensions
        }
        descriptors.append(MetalSparsePacketDecodeDescriptor(
            packetOffset: UInt32(blockStart),
            payloadEnd: UInt32(payloadEnd),
            originX: UInt32(target.descriptor.originX),
            originY: UInt32(target.descriptor.originY),
            validWidth: UInt32(target.descriptor.validWidth),
            validHeight: UInt32(target.descriptor.validHeight),
            stride: UInt32(decodedPlane.paddedWidth)
        ))
        try reader.seek(to: payloadEnd)
    }

    private func appendSparseEntries(
        header: PyrowavePacketHeader,
        blockStart: Int,
        reader: inout BinaryReader,
        target: SparseBlockTarget,
        decodedPlane: GPUDecodedPlane,
        entries: inout [MetalSparseCoefficientEntry]
    ) throws {
        let payloadEnd = blockStart + Int(header.payloadWords) * 4
        guard payloadEnd >= reader.offset else {
            throw PyrowaveError.invalidBitstream("payload_words is not large enough")
        }
        guard payloadEnd <= reader.data.count else {
            throw PyrowaveError.truncatedInput
        }
        guard header.ballot != 0 else {
            if reader.offset < payloadEnd {
                guard reader.data[reader.offset..<payloadEnd].allSatisfy({ $0 == 0 }) else {
                    throw PyrowaveError.invalidBitstream("non-zero coefficient packet padding")
                }
            }
            try reader.seek(to: payloadEnd)
            return
        }

        let activeBlockCount = header.ballot.nonzeroBitCount
        var codeWords = [UInt16]()
        codeWords.reserveCapacity(activeBlockCount)
        for _ in 0..<activeBlockCount {
            codeWords.append(try reader.readUInt16())
        }

        var qScales = [UInt8]()
        qScales.reserveCapacity(activeBlockCount)
        for _ in 0..<activeBlockCount {
            qScales.append(try reader.readUInt8())
        }

        var signEntryIndices = [Int]()
        signEntryIndices.reserveCapacity(activeBlockCount * PyrowaveBitstream.smallBlockSize * PyrowaveBitstream.smallBlockSize)

        var compactIndex = 0
        for smallBlock in 0..<16 where (header.ballot & (UInt16(1) << UInt16(smallBlock))) != 0 {
            let smallOriginX = (smallBlock % 4) * PyrowaveBitstream.smallBlockSize
            let smallOriginY = (smallBlock / 4) * PyrowaveBitstream.smallBlockSize
            let codeWord = codeWords[compactIndex]
            let basePlanes = Int(qScales[compactIndex] & 0x0f)
            let qScaleCode = qScales[compactIndex] >> 4

            for subblock in 0..<8 {
                let encodedPlanes = Int((codeWord >> UInt16(2 * subblock)) & 0x3) + basePlanes

                try withUnsafeTemporaryAllocation(of: Int.self, capacity: 8) { magnitudes in
                    for pixel in 0..<8 {
                        magnitudes[pixel] = 0
                    }

                    for _ in 0..<encodedPlanes {
                        guard reader.offset < payloadEnd else {
                            throw PyrowaveError.truncatedInput
                        }
                        let byte = try reader.readUInt8()
                        for pixel in 0..<8 {
                            magnitudes[pixel] <<= 1
                            magnitudes[pixel] |= Int((byte >> UInt8(pixel)) & 1)
                        }
                    }

                    for pixel in 0..<8 where magnitudes[pixel] != 0 {
                        let coord = coefficientSubblockCoordinate(subblock: subblock, pixel: pixel)
                        let x = smallOriginX + coord.x
                        let y = smallOriginY + coord.y
                        let localOffset = UInt16(y * PyrowaveBitstream.coefficientBlockSize + x)
                        let destinationOffset = try sparseDestinationOffset(
                            entryOffset: localOffset,
                            descriptor: target.descriptor,
                            paddedWidth: decodedPlane.paddedWidth,
                            paddedHeight: decodedPlane.paddedHeight
                        )
                        signEntryIndices.append(entries.count)
                        entries.append(MetalSparseCoefficientEntry(
                            destinationOffset: UInt32(destinationOffset),
                            coefficient: Int32(magnitudes[pixel]),
                            quantCode: UInt32(header.quantCode),
                            qScaleCode: UInt32(qScaleCode)
                        ))
                    }
                }
            }

            compactIndex += 1
        }

        let signStart = reader.offset
        let signByteCount = (signEntryIndices.count + 7) / 8
        guard signStart + signByteCount <= payloadEnd else {
            throw PyrowaveError.truncatedInput
        }

        for signIndex in signEntryIndices.indices {
            let signByteOffset = signIndex / 8
            let bit = UInt8(signIndex & 7)
            let signByte = reader.data[signStart + signByteOffset]
            if ((signByte >> bit) & 1) != 0 {
                entries[signEntryIndices[signIndex]].coefficient = -entries[signEntryIndices[signIndex]].coefficient
            }
        }

        let paddingStart = signStart + signByteCount
        if paddingStart < payloadEnd {
            guard reader.data[paddingStart..<payloadEnd].allSatisfy({ $0 == 0 }) else {
                throw PyrowaveError.invalidBitstream("non-zero coefficient packet padding")
            }
        }

        try reader.seek(to: payloadEnd)
    }

    private func coefficientSubblockCoordinate(subblock: Int, pixel: Int) -> (x: Int, y: Int) {
        let x = (subblock / 4) * 4 + (pixel >> 1)
        let y = (subblock % 4) * 2 + (pixel & 1)
        return (x, y)
    }

    private func applySparseEntriesToBuffers(
        _ entriesByPlane: [[MetalSparseCoefficientEntry]],
        decodedPlanes: inout [GPUDecodedPlane]
    ) throws {
        guard entriesByPlane.count == decodedPlanes.count else {
            throw PyrowaveError.invalidDimensions
        }
        let requests = decodedPlanes.indices.map { index in
            (
                sampleCount: decodedPlanes[index].sampleCount,
                entries: entriesByPlane[index]
            )
        }
        let buffers = try metalBackend.applySparseCoefficientBuffers(requests)
        guard buffers.count == decodedPlanes.count else {
            throw PyrowaveError.processFailed("Metal sparse coefficient batch returned \(buffers.count) buffers for \(decodedPlanes.count) planes")
        }
        for index in decodedPlanes.indices {
            decodedPlanes[index].samples = buffers[index]
        }
    }

    private func applySparsePacketDescriptorsToBuffers(
        _ descriptorsByPlane: [[MetalSparsePacketDecodeDescriptor]],
        packetData: Data,
        decodedPlanes: inout [GPUDecodedPlane]
    ) throws {
        guard descriptorsByPlane.count == decodedPlanes.count else {
            throw PyrowaveError.invalidDimensions
        }
        let requests = decodedPlanes.indices.map { index in
            (
                sampleCount: decodedPlanes[index].sampleCount,
                descriptors: descriptorsByPlane[index]
            )
        }
        let buffers = try metalBackend.decodeSparsePacketBuffers(packetData: packetData, planes: requests)
        guard buffers.count == decodedPlanes.count else {
            throw PyrowaveError.processFailed("Metal sparse packet decode batch returned \(buffers.count) buffers for \(decodedPlanes.count) planes")
        }
        for index in decodedPlanes.indices {
            decodedPlanes[index].samples = buffers[index]
        }
    }

    private func applySparseBlocksToBuffers(
        _ blocksByPlane: [[PendingSparseBlock]],
        decodedPlanes: inout [GPUDecodedPlane]
    ) throws {
        guard blocksByPlane.count == decodedPlanes.count else {
            throw PyrowaveError.invalidDimensions
        }
        let requests = try decodedPlanes.indices.map { index in
            (
                sampleCount: decodedPlanes[index].sampleCount,
                entries: try metalSparseCoefficientEntries(
                    blocksByPlane[index],
                    paddedWidth: decodedPlanes[index].paddedWidth,
                    paddedHeight: decodedPlanes[index].paddedHeight
                )
            )
        }
        let buffers = try metalBackend.applySparseCoefficientBuffers(requests)
        guard buffers.count == decodedPlanes.count else {
            throw PyrowaveError.processFailed("Metal sparse coefficient batch returned \(buffers.count) buffers for \(decodedPlanes.count) planes")
        }
        for index in decodedPlanes.indices {
            decodedPlanes[index].samples = buffers[index]
        }
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
            levels: plane.levels,
            useTiledLevelZero: true
        )
    }

    private func inverseWaveletBuffers(_ planes: [GPUDecodedPlane]) throws -> [MTLBuffer] {
        try metalBackend.inverseWaveletBuffers(planes.map {
            (
                buffer: $0.samples,
                sampleCount: $0.sampleCount,
                width: $0.paddedWidth,
                height: $0.paddedHeight,
                levels: $0.levels
            )
        }, useTiledLevelZero: true)
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
        let key = PlaneDescriptorCacheKey(
            layoutWidth: layout.width,
            layoutHeight: layout.height,
            chroma: layout.chroma.rawValue,
            component: component,
            paddedWidth: paddedWidth,
            paddedHeight: paddedHeight,
            levels: levels
        )
        geometryCacheLock.lock()
        if let cached = planeDescriptorCache[key] {
            geometryCacheLock.unlock()
            return cached
        }
        geometryCacheLock.unlock()

        let descriptors: [PlaneBlockDescriptor] = layout.descriptors.compactMap { global in
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

        geometryCacheLock.lock()
        planeDescriptorCache[key] = descriptors
        geometryCacheLock.unlock()
        return descriptors
    }

    private func quantCode(for descriptor: PlaneBlockDescriptor, plane: EncodedPlane) throws -> UInt8 {
        guard let quantCode = plane.quantCodesByBlockIndex[descriptor.blockIndex] else {
            throw PyrowaveError.processFailed("missing quant code for block \(descriptor.blockIndex)")
        }
        return quantCode
    }

    private func quantCode(for descriptor: PlaneBlockDescriptor, descriptorIndex: Int, plane: EncodedPlane) throws -> UInt8 {
        if descriptorIndex < plane.quantCodesByDescriptor.count {
            return plane.quantCodesByDescriptor[descriptorIndex]
        }
        return try quantCode(for: descriptor, plane: plane)
    }

    private func qScaleCodes(for descriptor: PlaneBlockDescriptor, plane: EncodedPlane) throws -> [UInt8] {
        guard let qScaleCodes = plane.qScaleCodesByBlockIndex[descriptor.blockIndex] else {
            throw PyrowaveError.processFailed("missing 8x8 quant scale codes for block \(descriptor.blockIndex)")
        }
        return qScaleCodes
    }

    private func qScaleCodes(for descriptor: PlaneBlockDescriptor, descriptorIndex: Int, plane: EncodedPlane) throws -> [UInt8] {
        if descriptorIndex < plane.qScaleCodesByDescriptor.count {
            return plane.qScaleCodesByDescriptor[descriptorIndex]
        }
        return try qScaleCodes(for: descriptor, plane: plane)
    }

    private func cachedLayout(width: Int, height: Int, chroma: ChromaSubsampling) throws -> PyrowaveBlockLayout {
        let key = LayoutCacheKey(width: width, height: height, chroma: chroma.rawValue)
        geometryCacheLock.lock()
        if let cached = layoutCache[key] {
            geometryCacheLock.unlock()
            return cached
        }
        geometryCacheLock.unlock()

        let layout = try PyrowaveBlockLayout(width: width, height: height, chroma: chroma)
        geometryCacheLock.lock()
        layoutCache[key] = layout
        geometryCacheLock.unlock()
        return layout
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
    ) throws -> (coefficients: [Int16], metadata: PlaneQuantizationMetadata) {
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
    ) throws -> (coefficients: [Int16], metadata: PlaneQuantizationMetadata) {
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
    ) throws -> (coefficientBuffer: MTLBuffer, coefficientCount: Int, metadata: PlaneQuantizationMetadata, qScaleBuffer: MTLBuffer?, qScaleDescriptorCount: Int) {
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

    private func quantizeResidentBuffers(
        _ planes: [(samples: MTLBuffer, sampleCount: Int, stride: Int, descriptors: [PlaneBlockDescriptor], component: Int)],
        configuration: CodecConfiguration,
        readsQScaleCodes: Bool = true
    ) throws -> [(coefficientBuffer: MTLBuffer, coefficientCount: Int, metadata: PlaneQuantizationMetadata, qScaleBuffer: MTLBuffer?, qScaleDescriptorCount: Int)] {
        var descriptorInputs = [(metalDescriptors: [MetalPlaneQuantizationDescriptor], descriptorBuffer: MTLBuffer, quantCodesByBlockIndex: [Int: UInt8], quantCodesByDescriptor: [UInt8])]()
        descriptorInputs.reserveCapacity(planes.count)
        for plane in planes {
            descriptorInputs.append(try makeQuantizationDescriptors(
                stride: plane.stride,
                descriptors: plane.descriptors,
                component: plane.component,
                configuration: configuration
            ))
        }

        let results = try metalBackend.quantizePlaneBufferResultsResidentDescriptors(planes.indices.map { index in
            (
                samples: planes[index].samples,
                sampleCount: planes[index].sampleCount,
                stride: planes[index].stride,
                descriptors: descriptorInputs[index].metalDescriptors,
                descriptorBuffer: descriptorInputs[index].descriptorBuffer
            )
        }, reusesOutputBuffers: true, readsQScaleCodes: readsQScaleCodes)
        guard results.count == planes.count else {
            throw PyrowaveError.processFailed("Metal batch quantization returned \(results.count) planes for \(planes.count) inputs")
        }

        return try planes.indices.map { index in
            try finishQuantizationBufferResult(
                results[index],
                descriptors: planes[index].descriptors,
                quantCodesByBlockIndex: descriptorInputs[index].quantCodesByBlockIndex,
                quantCodesByDescriptor: descriptorInputs[index].quantCodesByDescriptor,
                requiresQScaleCodes: readsQScaleCodes
            )
        }
    }

    private func quantizeWithMetal(
        _ samples: [Float],
        stride: Int,
        descriptors: [PlaneBlockDescriptor],
        component: Int,
        configuration: CodecConfiguration,
        backend: MetalPyrowaveBackend
    ) throws -> (coefficients: [Int16], metadata: PlaneQuantizationMetadata) {
        let input = try makeQuantizationDescriptors(
            stride: stride,
            descriptors: descriptors,
            component: component,
            configuration: configuration
        )
        let result = try backend.quantizePlane(samples, stride: stride, descriptors: input.metalDescriptors)
        return try finishQuantizationResult(
            result,
            descriptors: descriptors,
            quantCodesByBlockIndex: input.quantCodesByBlockIndex,
            quantCodesByDescriptor: input.quantCodesByDescriptor
        )
    }

    private func quantizeWithMetal(
        _ samples: MTLBuffer,
        sampleCount: Int,
        stride: Int,
        descriptors: [PlaneBlockDescriptor],
        component: Int,
        configuration: CodecConfiguration,
        backend: MetalPyrowaveBackend
    ) throws -> (coefficients: [Int16], metadata: PlaneQuantizationMetadata) {
        let input = try makeQuantizationDescriptors(
            stride: stride,
            descriptors: descriptors,
            component: component,
            configuration: configuration
        )
        let result = try backend.quantizePlaneBuffer(samples, sampleCount: sampleCount, stride: stride, descriptors: input.metalDescriptors)
        return try finishQuantizationResult(
            result,
            descriptors: descriptors,
            quantCodesByBlockIndex: input.quantCodesByBlockIndex,
            quantCodesByDescriptor: input.quantCodesByDescriptor
        )
    }

    private func quantizeWithMetalResident(
        _ samples: MTLBuffer,
        sampleCount: Int,
        stride: Int,
        descriptors: [PlaneBlockDescriptor],
        component: Int,
        configuration: CodecConfiguration,
        backend: MetalPyrowaveBackend
    ) throws -> (coefficientBuffer: MTLBuffer, coefficientCount: Int, metadata: PlaneQuantizationMetadata, qScaleBuffer: MTLBuffer?, qScaleDescriptorCount: Int) {
        let input = try makeQuantizationDescriptors(
            stride: stride,
            descriptors: descriptors,
            component: component,
            configuration: configuration
        )
        let result = try backend.quantizePlaneBufferResult(samples, sampleCount: sampleCount, stride: stride, descriptors: input.metalDescriptors)
        return try finishQuantizationBufferResult(
            result,
            descriptors: descriptors,
            quantCodesByBlockIndex: input.quantCodesByBlockIndex,
            quantCodesByDescriptor: input.quantCodesByDescriptor
        )
    }

    private func makeQuantizationDescriptors(
        stride: Int,
        descriptors: [PlaneBlockDescriptor],
        component: Int,
        configuration: CodecConfiguration
    ) throws -> (metalDescriptors: [MetalPlaneQuantizationDescriptor], descriptorBuffer: MTLBuffer, quantCodesByBlockIndex: [Int: UInt8], quantCodesByDescriptor: [UInt8]) {
        let cacheKey = quantizationDescriptorCacheKey(
            stride: stride,
            descriptors: descriptors,
            component: component,
            configuration: configuration
        )
        geometryCacheLock.lock()
        if let cached = quantizationDescriptorCache[cacheKey] {
            geometryCacheLock.unlock()
            return (cached.metalDescriptors, cached.descriptorBuffer, cached.quantCodesByBlockIndex, cached.quantCodesByDescriptor)
        }
        geometryCacheLock.unlock()

        var quantCodesByBlockIndex = [Int: UInt8]()
        quantCodesByBlockIndex.reserveCapacity(descriptors.count)
        var quantCodesByDescriptor = [UInt8]()
        quantCodesByDescriptor.reserveCapacity(descriptors.count)
        var metalDescriptors = [MetalPlaneQuantizationDescriptor]()
        metalDescriptors.reserveCapacity(descriptors.count)
        for descriptor in descriptors {
            let requestedStep = PyrowaveQuantization.quantizationStep(
                level: descriptor.globalLevel,
                component: component,
                band: descriptor.band,
                baseStep: configuration.quantizationStep
            )
            let quantCode = try PyrowaveQuantization.encodeBlockScale(requestedStep)
            let decodedStep = PyrowaveQuantization.decodeBlockScale(quantCode)
            quantCodesByBlockIndex[descriptor.blockIndex] = quantCode
            quantCodesByDescriptor.append(quantCode)

            metalDescriptors.append(MetalPlaneQuantizationDescriptor(
                originX: UInt32(descriptor.originX),
                originY: UInt32(descriptor.originY),
                validWidth: UInt32(descriptor.validWidth),
                validHeight: UInt32(descriptor.validHeight),
                stride: UInt32(stride),
                quantCode: UInt32(quantCode),
                baseScale: 1.0 / decodedStep
            ))
        }

        let descriptorBuffer = try metalBackend.makeStaticSharedBuffer(bytes: metalDescriptors)

        geometryCacheLock.lock()
        quantizationDescriptorCache[cacheKey] = QuantizationDescriptorCacheEntry(
            metalDescriptors: metalDescriptors,
            descriptorBuffer: descriptorBuffer,
            quantCodesByBlockIndex: quantCodesByBlockIndex,
            quantCodesByDescriptor: quantCodesByDescriptor
        )
        geometryCacheLock.unlock()
        return (metalDescriptors, descriptorBuffer, quantCodesByBlockIndex, quantCodesByDescriptor)
    }

    private func quantizationDescriptorCacheKey(
        stride: Int,
        descriptors: [PlaneBlockDescriptor],
        component: Int,
        configuration: CodecConfiguration
    ) -> QuantizationDescriptorCacheKey {
        var descriptorHash = Hasher()
        descriptorHash.combine(descriptors.count)
        for descriptor in descriptors {
            descriptorHash.combine(descriptor.blockIndex)
            descriptorHash.combine(descriptor.globalLevel)
            descriptorHash.combine(descriptor.level)
            descriptorHash.combine(descriptor.band)
            descriptorHash.combine(descriptor.originX)
            descriptorHash.combine(descriptor.originY)
            descriptorHash.combine(descriptor.validWidth)
            descriptorHash.combine(descriptor.validHeight)
        }
        return QuantizationDescriptorCacheKey(
            stride: stride,
            component: component,
            quantizationStep: configuration.quantizationStep,
            descriptorHash: descriptorHash.finalize()
        )
    }

    private func finishQuantizationResult(
        _ result: MetalPlaneQuantizationResult,
        descriptors: [PlaneBlockDescriptor],
        quantCodesByBlockIndex: [Int: UInt8],
        quantCodesByDescriptor: [UInt8]
    ) throws -> (coefficients: [Int16], metadata: PlaneQuantizationMetadata) {
        guard result.qScaleCodesByDescriptor.count == descriptors.count else {
            throw PyrowaveError.processFailed("Metal quantization returned \(result.qScaleCodesByDescriptor.count) q-scale rows for \(descriptors.count) descriptors")
        }
        guard quantCodesByDescriptor.count == descriptors.count else {
            throw PyrowaveError.processFailed("Metal quantization returned \(quantCodesByDescriptor.count) quant codes for \(descriptors.count) descriptors")
        }

        var qScaleCodes = [Int: [UInt8]]()
        qScaleCodes.reserveCapacity(descriptors.count)
        for (index, descriptor) in descriptors.enumerated() {
            qScaleCodes[descriptor.blockIndex] = result.qScaleCodesByDescriptor[index]
        }
        return (
            result.coefficients,
            PlaneQuantizationMetadata(
                quantCodesByBlockIndex: quantCodesByBlockIndex,
                qScaleCodesByBlockIndex: qScaleCodes,
                quantCodesByDescriptor: quantCodesByDescriptor,
                qScaleCodesByDescriptor: result.qScaleCodesByDescriptor
            )
        )
    }

    private func finishQuantizationBufferResult(
        _ result: MetalPlaneQuantizationBufferResult,
        descriptors: [PlaneBlockDescriptor],
        quantCodesByBlockIndex: [Int: UInt8],
        quantCodesByDescriptor: [UInt8],
        requiresQScaleCodes: Bool = true
    ) throws -> (coefficientBuffer: MTLBuffer, coefficientCount: Int, metadata: PlaneQuantizationMetadata, qScaleBuffer: MTLBuffer?, qScaleDescriptorCount: Int) {
        guard !requiresQScaleCodes || result.qScaleCodesByDescriptor.count == descriptors.count else {
            throw PyrowaveError.processFailed("Metal quantization returned \(result.qScaleCodesByDescriptor.count) q-scale rows for \(descriptors.count) descriptors")
        }
        guard result.qScaleDescriptorCount == descriptors.count else {
            throw PyrowaveError.processFailed("Metal quantization returned q-scale buffer metadata for \(result.qScaleDescriptorCount) descriptors, expected \(descriptors.count)")
        }
        guard quantCodesByDescriptor.count == descriptors.count else {
            throw PyrowaveError.processFailed("Metal quantization returned \(quantCodesByDescriptor.count) quant codes for \(descriptors.count) descriptors")
        }

        var qScaleCodes = [Int: [UInt8]]()
        qScaleCodes.reserveCapacity(descriptors.count)
        if requiresQScaleCodes {
            for (index, descriptor) in descriptors.enumerated() {
                qScaleCodes[descriptor.blockIndex] = result.qScaleCodesByDescriptor[index]
            }
        }
        return (
            result.coefficientBuffer,
            result.coefficientCount,
            PlaneQuantizationMetadata(
                quantCodesByBlockIndex: quantCodesByBlockIndex,
                qScaleCodesByBlockIndex: qScaleCodes,
                quantCodesByDescriptor: quantCodesByDescriptor,
                qScaleCodesByDescriptor: result.qScaleCodesByDescriptor
            ),
            result.qScaleBuffer,
            result.qScaleDescriptorCount
        )
    }
}
