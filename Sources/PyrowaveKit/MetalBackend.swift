import Foundation
import Metal

private struct SparseCoefficientOutputKey: Hashable {
    var planeIndex: Int
    var sampleCount: Int
}

private enum ReusableBufferPurpose: Hashable {
    case quantizeDescriptor
    case quantizeQScale
    case sparseEntries
    case sparsePacketDecodeData
    case sparsePacketDecodeDescriptor
    case rateStatsDescriptor
    case rateStatsNumPlanes
    case rateStats
    case packetCostDescriptor
    case packetCostOutput
    case packetCostSignCount
    case sparsePacketDescriptor
    case sparsePacketQScale
    case sparsePacketSize
    case dwtPrimary
    case dwtScratch
    case idwtScratch
}

private struct ReusableBufferKey: Hashable {
    var purpose: ReusableBufferPurpose
    var planeIndex: Int
}

final class MetalPyrowaveBackend: @unchecked Sendable {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary
    private let padPlanePipeline: MTLComputePipelineState
    private let padTexturePlanePipeline: MTLComputePipelineState
    private let cropPlanePipeline: MTLComputePipelineState
    private let cropTexturePlanePipeline: MTLComputePipelineState
    private let cropNV12TexturesPipeline: MTLComputePipelineState
    private let quantizePipeline: MTLComputePipelineState
    private let dequantizePipeline: MTLComputePipelineState
    private let quantizePlaneTilesPipeline: MTLComputePipelineState
    private let sparseApplyPipeline: MTLComputePipelineState
    private let sparsePacketDecodePipeline: MTLComputePipelineState
    private let rateControlStatsPipeline: MTLComputePipelineState
    private let packetByteCostsPipeline: MTLComputePipelineState
    private let packetByteCostsSmallblocksPipeline: MTLComputePipelineState
    private let packetByteCostsFinalizePipeline: MTLComputePipelineState
    private let sparsePacketEncodePipeline: MTLComputePipelineState
    private let rateControlBucketPipeline: MTLComputePipelineState
    private let rateControlTileStatsBucketPipeline: MTLComputePipelineState
    private let rateControlBucketSavingsPipeline: MTLComputePipelineState
    private let rateControlBucketSavingsPrefixPipeline: MTLComputePipelineState
    private let dwtLiftRowsPipeline: MTLComputePipelineState
    private let dwtLiftColumnsPipeline: MTLComputePipelineState
    private let dwtPackRowsPipeline: MTLComputePipelineState
    private let dwtPackColumnsPipeline: MTLComputePipelineState
    private let dwtUnpackRowsPipeline: MTLComputePipelineState
    private let dwtUnpackColumnsPipeline: MTLComputePipelineState
    private let idwtUnpackRowsScaledPipeline: MTLComputePipelineState
    private let idwtUnpackColumnsScaledPipeline: MTLComputePipelineState
    private let idwtLiftRowsPipeline: MTLComputePipelineState
    private let idwtLiftColumnsPipeline: MTLComputePipelineState
    private var sparseCoefficientOutputs: [SparseCoefficientOutputKey: MTLBuffer] = [:]
    private var reusableSharedBuffers: [ReusableBufferKey: MTLBuffer] = [:]
    private var reusablePrivateBuffers: [ReusableBufferKey: MTLBuffer] = [:]

    init(device: MTLDevice? = MTLCreateSystemDefaultDevice()) throws {
        guard let device else {
            throw PyrowaveError.externalToolUnavailable("Metal device")
        }
        guard let queue = device.makeCommandQueue() else {
            throw PyrowaveError.processFailed("failed to create Metal command queue")
        }
        self.device = device
        commandQueue = queue

        if let metallibURL = Bundle.module.url(forResource: "default", withExtension: "metallib") {
            library = try device.makeLibrary(URL: metallibURL)
        } else if let sourceURL = Bundle.module.url(forResource: "PyrowaveKernels", withExtension: "metal") {
            let source = try String(contentsOf: sourceURL, encoding: .utf8)
            library = try device.makeLibrary(source: source, options: nil)
        } else {
            throw PyrowaveError.processFailed("missing Metal kernel resource")
        }

        padPlanePipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_pad_plane", library: library))
        padTexturePlanePipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_pad_texture_plane", library: library))
        cropPlanePipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_crop_plane", library: library))
        cropTexturePlanePipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_crop_texture_plane", library: library))
        cropNV12TexturesPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_crop_nv12_textures", library: library))
        quantizePipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_quantize", library: library))
        dequantizePipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_dequantize", library: library))
        quantizePlaneTilesPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_quantize_plane_tiles", library: library))
        sparseApplyPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_apply_sparse_coefficients", library: library))
        sparsePacketDecodePipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_decode_sparse_packets", library: library))
        rateControlStatsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_rate_control_tile_stats", library: library))
        packetByteCostsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_packet_byte_costs", library: library))
        packetByteCostsSmallblocksPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_packet_byte_costs_smallblocks", library: library))
        packetByteCostsFinalizePipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_packet_byte_costs_finalize", library: library))
        sparsePacketEncodePipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_encode_sparse_packets", library: library))
        rateControlBucketPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_rate_control_bucket_indices", library: library))
        rateControlTileStatsBucketPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_rate_control_tile_stats_bucket_indices", library: library))
        rateControlBucketSavingsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_rate_control_bucket_savings", library: library))
        rateControlBucketSavingsPrefixPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_rate_control_bucket_savings_prefix", library: library))
        dwtLiftRowsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_dwt_lift_rows", library: library))
        dwtLiftColumnsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_dwt_lift_columns", library: library))
        dwtPackRowsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_dwt_pack_rows", library: library))
        dwtPackColumnsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_dwt_pack_columns", library: library))
        dwtUnpackRowsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_dwt_unpack_rows", library: library))
        dwtUnpackColumnsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_dwt_unpack_columns", library: library))
        idwtUnpackRowsScaledPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_idwt_unpack_rows_scaled", library: library))
        idwtUnpackColumnsScaledPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_idwt_unpack_columns_scaled", library: library))
        idwtLiftRowsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_idwt_lift_rows", library: library))
        idwtLiftColumnsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_idwt_lift_columns", library: library))
    }

    func makeFunction(named name: String) throws -> MTLFunction {
        try Self.makeFunction(named: name, library: library)
    }

    func padPlane(_ plane: Plane8, paddedWidth: Int, paddedHeight: Int) throws -> [Float] {
        guard paddedWidth > 0,
              paddedHeight > 0,
              paddedWidth <= Int(UInt32.max),
              paddedHeight <= Int(UInt32.max),
              plane.width <= Int(UInt32.max),
              plane.height <= Int(UInt32.max) else {
            throw PyrowaveError.invalidDimensions
        }
        let sampleCount = paddedWidth * paddedHeight
        guard sampleCount > 0, sampleCount <= Int(UInt32.max) else {
            throw PyrowaveError.invalidDimensions
        }

        guard let input = device.makeBuffer(bytes: plane.data, length: plane.data.count * MemoryLayout<UInt8>.stride, options: .storageModeShared),
              let output = device.makeBuffer(length: sampleCount * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal plane padding buffers")
        }

        var constants = PadPlaneConstants(
            sourceWidth: UInt32(plane.width),
            sourceHeight: UInt32(plane.height),
            paddedWidth: UInt32(paddedWidth),
            paddedHeight: UInt32(paddedHeight),
            channel: 0
        )
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal plane padding command encoder")
        }

        encoder.setComputePipelineState(padPlanePipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&constants, length: MemoryLayout<PadPlaneConstants>.stride, index: 2)
        let width = min(16, padPlanePipeline.maxTotalThreadsPerThreadgroup)
        let height = max(1, min(16, padPlanePipeline.maxTotalThreadsPerThreadgroup / width))
        encoder.dispatchThreads(
            MTLSize(width: paddedWidth, height: paddedHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal plane padding command failed: \(error)")
        }

        let pointer = output.contents().bindMemory(to: Float.self, capacity: sampleCount)
        return Array(UnsafeBufferPointer(start: pointer, count: sampleCount))
    }

    func padTexturePlane(_ texture: MTLTexture, channel: Int = 0, paddedWidth: Int, paddedHeight: Int) throws -> [Float] {
        let output = try padTexturePlaneBuffer(texture, channel: channel, paddedWidth: paddedWidth, paddedHeight: paddedHeight)
        let sampleCount = paddedWidth * paddedHeight
        let pointer = output.contents().bindMemory(to: Float.self, capacity: sampleCount)
        return Array(UnsafeBufferPointer(start: pointer, count: sampleCount))
    }

    func padTexturePlaneBuffer(_ texture: MTLTexture, channel: Int = 0, paddedWidth: Int, paddedHeight: Int) throws -> MTLBuffer {
        try padTexturePlaneBuffers([(texture: texture, channel: channel, paddedWidth: paddedWidth, paddedHeight: paddedHeight)])[0]
    }

    func padTexturePlaneBuffers(
        _ planes: [(texture: MTLTexture, channel: Int, paddedWidth: Int, paddedHeight: Int)]
    ) throws -> [MTLBuffer] {
        guard !planes.isEmpty else {
            return []
        }

        var outputs = [MTLBuffer]()
        outputs.reserveCapacity(planes.count)
        for plane in planes {
            let validChannel: Bool
            switch plane.texture.pixelFormat {
            case .r8Unorm:
                validChannel = plane.channel == 0
            case .rg8Unorm:
                validChannel = plane.channel == 0 || plane.channel == 1
            default:
                validChannel = false
            }
            guard validChannel else {
                throw PyrowaveError.unsupportedFormat("Metal texture padding expects r8Unorm channel 0 or rg8Unorm channel 0/1")
            }
            guard plane.texture.width > 0,
                  plane.texture.height > 0,
                  plane.paddedWidth > 0,
                  plane.paddedHeight > 0,
                  plane.texture.width <= Int(UInt32.max),
                  plane.texture.height <= Int(UInt32.max),
                  plane.paddedWidth <= Int(UInt32.max),
                  plane.paddedHeight <= Int(UInt32.max) else {
                throw PyrowaveError.invalidDimensions
            }
            let sampleCount = plane.paddedWidth * plane.paddedHeight
            guard sampleCount > 0, sampleCount <= Int(UInt32.max) else {
                throw PyrowaveError.invalidDimensions
            }

            guard let output = device.makeBuffer(length: sampleCount * MemoryLayout<Float>.stride, options: .storageModeShared) else {
                throw PyrowaveError.processFailed("failed to allocate Metal texture padding buffer")
            }
            outputs.append(output)
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal texture padding command encoder")
        }

        encoder.setComputePipelineState(padTexturePlanePipeline)
        let width = min(16, padTexturePlanePipeline.maxTotalThreadsPerThreadgroup)
        let height = max(1, min(16, padTexturePlanePipeline.maxTotalThreadsPerThreadgroup / width))
        let threadsPerThreadgroup = MTLSize(width: width, height: height, depth: 1)
        for (index, plane) in planes.enumerated() {
            var constants = PadPlaneConstants(
                sourceWidth: UInt32(plane.texture.width),
                sourceHeight: UInt32(plane.texture.height),
                paddedWidth: UInt32(plane.paddedWidth),
                paddedHeight: UInt32(plane.paddedHeight),
                channel: UInt32(plane.channel)
            )
            encoder.setTexture(plane.texture, index: 0)
            encoder.setBuffer(outputs[index], offset: 0, index: 0)
            encoder.setBytes(&constants, length: MemoryLayout<PadPlaneConstants>.stride, index: 1)
            encoder.dispatchThreads(
                MTLSize(width: plane.paddedWidth, height: plane.paddedHeight, depth: 1),
                threadsPerThreadgroup: threadsPerThreadgroup
            )
        }
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal texture padding command failed: \(error)")
        }

        return outputs
    }

    func padTexturePlaneBuffersAndForwardWaveletBuffers(
        _ planes: [(texture: MTLTexture, channel: Int, paddedWidth: Int, paddedHeight: Int, levels: Int)]
    ) throws -> [MTLBuffer] {
        guard !planes.isEmpty else {
            return []
        }

        var outputs = [MTLBuffer]()
        outputs.reserveCapacity(planes.count)
        var scratchBuffers = [MTLBuffer]()
        scratchBuffers.reserveCapacity(planes.count)

        for (planeIndex, plane) in planes.enumerated() {
            let validChannel = plane.texture.pixelFormat == .r8Unorm
                ? plane.channel == 0
                : (plane.texture.pixelFormat == .rg8Unorm && (plane.channel == 0 || plane.channel == 1))
            guard validChannel else {
                throw PyrowaveError.unsupportedFormat("Metal texture padding expects r8Unorm channel 0 or rg8Unorm channel 0/1")
            }
            guard plane.texture.width > 0,
                  plane.texture.height > 0,
                  plane.paddedWidth > 0,
                  plane.paddedHeight > 0,
                  plane.texture.width <= Int(UInt32.max),
                  plane.texture.height <= Int(UInt32.max),
                  plane.paddedWidth <= Int(UInt32.max),
                  plane.paddedHeight <= Int(UInt32.max) else {
                throw PyrowaveError.invalidDimensions
            }
            let sampleCount = plane.paddedWidth * plane.paddedHeight
            guard sampleCount > 0, sampleCount <= Int(UInt32.max) else {
                throw PyrowaveError.invalidDimensions
            }
            try validateWaveletShape(width: plane.paddedWidth, height: plane.paddedHeight, levels: plane.levels)

            let output = try reusablePrivateBuffer(
                byteLength: sampleCount * MemoryLayout<Float>.stride,
                purpose: .dwtPrimary,
                planeIndex: planeIndex
            )
            let scratch = try reusablePrivateBuffer(
                byteLength: sampleCount * MemoryLayout<Float>.stride,
                purpose: .dwtScratch,
                planeIndex: planeIndex
            )
            outputs.append(output)
            scratchBuffers.append(scratch)
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal texture padding/DWT command encoder")
        }

        encoder.setComputePipelineState(padTexturePlanePipeline)
        let width = min(16, padTexturePlanePipeline.maxTotalThreadsPerThreadgroup)
        let height = max(1, min(16, padTexturePlanePipeline.maxTotalThreadsPerThreadgroup / width))
        let threadsPerThreadgroup = MTLSize(width: width, height: height, depth: 1)
        for (index, plane) in planes.enumerated() {
            var constants = PadPlaneConstants(
                sourceWidth: UInt32(plane.texture.width),
                sourceHeight: UInt32(plane.texture.height),
                paddedWidth: UInt32(plane.paddedWidth),
                paddedHeight: UInt32(plane.paddedHeight),
                channel: UInt32(plane.channel)
            )
            encoder.setTexture(plane.texture, index: 0)
            encoder.setBuffer(outputs[index], offset: 0, index: 0)
            encoder.setBytes(&constants, length: MemoryLayout<PadPlaneConstants>.stride, index: 1)
            encoder.dispatchThreads(
                MTLSize(width: plane.paddedWidth, height: plane.paddedHeight, depth: 1),
                threadsPerThreadgroup: threadsPerThreadgroup
            )
        }
        encoder.endEncoding()

        let maxLevels = planes.map(\.levels).max() ?? 0
        for level in 0..<maxLevels {
            var active = [DWTBatchDispatch]()
            active.reserveCapacity(planes.count)
            for (planeIndex, plane) in planes.enumerated() where level < plane.levels {
                active.append(DWTBatchDispatch(
                    primary: outputs[planeIndex],
                    scratch: scratchBuffers[planeIndex],
                    activeWidth: plane.paddedWidth >> level,
                    activeHeight: plane.paddedHeight >> level,
                    stride: plane.paddedWidth
                ))
            }
            guard !active.isEmpty else {
                continue
            }

            for phase in 0...4 {
                try encodeDWTInPlaceBatch(
                    commandBuffer: commandBuffer,
                    pipeline: dwtLiftRowsPipeline,
                    dispatches: active.map {
                        (buffer: $0.primary, activeWidth: $0.activeWidth, activeHeight: $0.activeHeight, stride: $0.stride, phase: phase)
                    }
                )
            }
            try encodeDWTCopyBatch(
                commandBuffer: commandBuffer,
                pipeline: dwtPackRowsPipeline,
                dispatches: active.map {
                    (input: $0.primary, output: $0.scratch, activeWidth: $0.activeWidth, activeHeight: $0.activeHeight, stride: $0.stride, phase: 0)
                }
            )

            for phase in 0...4 {
                try encodeDWTInPlaceBatch(
                    commandBuffer: commandBuffer,
                    pipeline: dwtLiftColumnsPipeline,
                    dispatches: active.map {
                        (buffer: $0.scratch, activeWidth: $0.activeWidth, activeHeight: $0.activeHeight, stride: $0.stride, phase: phase)
                    }
                )
            }
            try encodeDWTCopyBatch(
                commandBuffer: commandBuffer,
                pipeline: dwtPackColumnsPipeline,
                dispatches: active.map {
                    (input: $0.scratch, output: $0.primary, activeWidth: $0.activeWidth, activeHeight: $0.activeHeight, stride: $0.stride, phase: 0)
                }
            )
        }

        try finish(commandBuffer: commandBuffer, context: "Metal texture padding/DWT")
        return outputs
    }

    func cropPlane(_ samples: [Float], paddedWidth: Int, width: Int, height: Int) throws -> Plane8 {
        guard paddedWidth > 0,
              width > 0,
              height > 0,
              paddedWidth <= Int(UInt32.max),
              width <= Int(UInt32.max),
              height <= Int(UInt32.max),
              samples.count >= paddedWidth * height else {
            throw PyrowaveError.invalidDimensions
        }
        let outputCount = width * height
        guard outputCount > 0, outputCount <= Int(UInt32.max) else {
            throw PyrowaveError.invalidDimensions
        }

        guard let input = device.makeBuffer(bytes: samples, length: samples.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let output = device.makeBuffer(length: outputCount * MemoryLayout<UInt8>.stride, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal plane crop buffers")
        }

        var constants = CropPlaneConstants(
            paddedWidth: UInt32(paddedWidth),
            outputWidth: UInt32(width),
            outputHeight: UInt32(height)
        )
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal plane crop command encoder")
        }

        encoder.setComputePipelineState(cropPlanePipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&constants, length: MemoryLayout<CropPlaneConstants>.stride, index: 2)
        let threadWidth = min(16, cropPlanePipeline.maxTotalThreadsPerThreadgroup)
        let threadHeight = max(1, min(16, cropPlanePipeline.maxTotalThreadsPerThreadgroup / threadWidth))
        encoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadWidth, height: threadHeight, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal plane crop command failed: \(error)")
        }

        let pointer = output.contents().bindMemory(to: UInt8.self, capacity: outputCount)
        return try Plane8(width: width, height: height, data: Array(UnsafeBufferPointer(start: pointer, count: outputCount)))
    }

    func cropPlaneToTexture(_ samples: [Float], paddedWidth: Int, width: Int, height: Int, texture: MTLTexture) throws {
        guard let input = device.makeBuffer(bytes: samples, length: samples.count * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal texture crop buffer")
        }
        try cropPlaneToTexture(input, sampleCount: samples.count, paddedWidth: paddedWidth, width: width, height: height, texture: texture)
    }

    func cropPlaneToTexture(_ buffer: MTLBuffer, sampleCount: Int, paddedWidth: Int, width: Int, height: Int, texture: MTLTexture) throws {
        guard texture.pixelFormat == .r8Unorm,
              texture.width == width,
              texture.height == height,
              paddedWidth > 0,
              width > 0,
              height > 0,
              paddedWidth <= Int(UInt32.max),
              width <= Int(UInt32.max),
              height <= Int(UInt32.max),
              sampleCount >= paddedWidth * height,
              buffer.length >= sampleCount * MemoryLayout<Float>.stride else {
            throw PyrowaveError.invalidDimensions
        }

        var constants = CropPlaneConstants(
            paddedWidth: UInt32(paddedWidth),
            outputWidth: UInt32(width),
            outputHeight: UInt32(height)
        )
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal texture crop command encoder")
        }

        encoder.setComputePipelineState(cropTexturePlanePipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setTexture(texture, index: 0)
        encoder.setBytes(&constants, length: MemoryLayout<CropPlaneConstants>.stride, index: 1)
        let threadWidth = min(16, cropTexturePlanePipeline.maxTotalThreadsPerThreadgroup)
        let threadHeight = max(1, min(16, cropTexturePlanePipeline.maxTotalThreadsPerThreadgroup / threadWidth))
        encoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadWidth, height: threadHeight, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal texture crop command failed: \(error)")
        }
    }

    func cropPlanesToNV12Textures(
        ySamples: [Float],
        yPaddedWidth: Int,
        cbSamples: [Float],
        crSamples: [Float],
        chromaPaddedWidth: Int,
        width: Int,
        height: Int,
        yTexture: MTLTexture,
        cbCrTexture: MTLTexture
    ) throws {
        guard let yInput = device.makeBuffer(bytes: ySamples, length: ySamples.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let cbInput = device.makeBuffer(bytes: cbSamples, length: cbSamples.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let crInput = device.makeBuffer(bytes: crSamples, length: crSamples.count * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal NV12 crop buffers")
        }
        try cropPlanesToNV12Textures(
            yBuffer: yInput,
            ySampleCount: ySamples.count,
            yPaddedWidth: yPaddedWidth,
            cbBuffer: cbInput,
            cbSampleCount: cbSamples.count,
            crBuffer: crInput,
            crSampleCount: crSamples.count,
            chromaPaddedWidth: chromaPaddedWidth,
            width: width,
            height: height,
            yTexture: yTexture,
            cbCrTexture: cbCrTexture
        )
    }

    func cropPlanesToNV12Textures(
        yBuffer: MTLBuffer,
        ySampleCount: Int,
        yPaddedWidth: Int,
        cbBuffer: MTLBuffer,
        cbSampleCount: Int,
        crBuffer: MTLBuffer,
        crSampleCount: Int,
        chromaPaddedWidth: Int,
        width: Int,
        height: Int,
        yTexture: MTLTexture,
        cbCrTexture: MTLTexture
    ) throws {
        let chromaWidth = width / 2
        let chromaHeight = height / 2
        guard yTexture.pixelFormat == .r8Unorm,
              cbCrTexture.pixelFormat == .rg8Unorm,
              yTexture.width == width,
              yTexture.height == height,
              cbCrTexture.width == chromaWidth,
              cbCrTexture.height == chromaHeight,
              yPaddedWidth > 0,
              chromaPaddedWidth > 0,
              width > 0,
              height > 0,
              yPaddedWidth <= Int(UInt32.max),
              chromaPaddedWidth <= Int(UInt32.max),
              width <= Int(UInt32.max),
              height <= Int(UInt32.max),
              ySampleCount >= yPaddedWidth * height,
              cbSampleCount >= chromaPaddedWidth * chromaHeight,
              crSampleCount >= chromaPaddedWidth * chromaHeight,
              yBuffer.length >= ySampleCount * MemoryLayout<Float>.stride,
              cbBuffer.length >= cbSampleCount * MemoryLayout<Float>.stride,
              crBuffer.length >= crSampleCount * MemoryLayout<Float>.stride else {
            throw PyrowaveError.invalidDimensions
        }

        var constants = CropNV12Constants(
            yPaddedWidth: UInt32(yPaddedWidth),
            chromaPaddedWidth: UInt32(chromaPaddedWidth),
            outputWidth: UInt32(width),
            outputHeight: UInt32(height)
        )
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal NV12 crop command encoder")
        }

        encoder.setComputePipelineState(cropNV12TexturesPipeline)
        encoder.setBuffer(yBuffer, offset: 0, index: 0)
        encoder.setBuffer(cbBuffer, offset: 0, index: 1)
        encoder.setBuffer(crBuffer, offset: 0, index: 2)
        encoder.setTexture(yTexture, index: 0)
        encoder.setTexture(cbCrTexture, index: 1)
        encoder.setBytes(&constants, length: MemoryLayout<CropNV12Constants>.stride, index: 3)
        let threadWidth = min(16, cropNV12TexturesPipeline.maxTotalThreadsPerThreadgroup)
        let threadHeight = max(1, min(16, cropNV12TexturesPipeline.maxTotalThreadsPerThreadgroup / threadWidth))
        encoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadWidth, height: threadHeight, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal NV12 texture crop command failed: \(error)")
        }
    }

    func quantize(_ samples: [Float], quantizationStep: Float) throws -> [Int16] {
        guard !samples.isEmpty else { return [] }
        guard quantizationStep > 0 else { throw PyrowaveError.invalidDimensions }

        let inputLength = samples.count * MemoryLayout<Float>.stride
        let outputLength = samples.count * MemoryLayout<Int16>.stride
        guard let input = device.makeBuffer(bytes: samples, length: inputLength, options: .storageModeShared),
              let output = device.makeBuffer(length: outputLength, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal quantization buffers")
        }

        var constants = QuantizationConstants(count: UInt32(samples.count), quantizationStep: quantizationStep)
        try dispatch(
            pipeline: quantizePipeline,
            count: samples.count,
            buffers: [(input, 0), (output, 1)],
            constants: &constants
        )

        let pointer = output.contents().bindMemory(to: Int16.self, capacity: samples.count)
        return Array(UnsafeBufferPointer(start: pointer, count: samples.count))
    }

    func dequantize(_ coefficients: [Int16], quantizationStep: Float) throws -> [Float] {
        guard !coefficients.isEmpty else { return [] }
        guard quantizationStep > 0 else { throw PyrowaveError.invalidDimensions }

        let inputLength = coefficients.count * MemoryLayout<Int16>.stride
        let outputLength = coefficients.count * MemoryLayout<Float>.stride
        guard let input = device.makeBuffer(bytes: coefficients, length: inputLength, options: .storageModeShared),
              let output = device.makeBuffer(length: outputLength, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal dequantization buffers")
        }

        var constants = QuantizationConstants(count: UInt32(coefficients.count), quantizationStep: quantizationStep)
        try dispatch(
            pipeline: dequantizePipeline,
            count: coefficients.count,
            buffers: [(input, 0), (output, 1)],
            constants: &constants
        )

        let pointer = output.contents().bindMemory(to: Float.self, capacity: coefficients.count)
        return Array(UnsafeBufferPointer(start: pointer, count: coefficients.count))
    }

    func quantizePlane(
        _ samples: [Float],
        stride: Int,
        descriptors: [MetalPlaneQuantizationDescriptor]
    ) throws -> MetalPlaneQuantizationResult {
        guard stride > 0, !samples.isEmpty else {
            throw PyrowaveError.invalidDimensions
        }
        guard !descriptors.isEmpty else {
            return MetalPlaneQuantizationResult(coefficients: Array(repeating: 0, count: samples.count), qScaleCodesByDescriptor: [])
        }

        guard let input = device.makeBuffer(bytes: samples, length: samples.count * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal plane quantization input buffer")
        }
        return try quantizePlaneBuffer(input, sampleCount: samples.count, stride: stride, descriptors: descriptors)
    }

    func quantizePlaneBuffer(
        _ samples: MTLBuffer,
        sampleCount: Int,
        stride: Int,
        descriptors: [MetalPlaneQuantizationDescriptor]
    ) throws -> MetalPlaneQuantizationResult {
        let result = try quantizePlaneBufferResult(samples, sampleCount: sampleCount, stride: stride, descriptors: descriptors)
        guard result.coefficientCount > 0 else {
            return MetalPlaneQuantizationResult(coefficients: [], qScaleCodesByDescriptor: result.qScaleCodesByDescriptor)
        }
        let coefficientsPointer = result.coefficientBuffer.contents().bindMemory(to: Int16.self, capacity: result.coefficientCount)
        let coefficientValues = Array(UnsafeBufferPointer(start: coefficientsPointer, count: result.coefficientCount))
        return MetalPlaneQuantizationResult(coefficients: coefficientValues, qScaleCodesByDescriptor: result.qScaleCodesByDescriptor)
    }

    func quantizePlaneBufferResult(
        _ samples: MTLBuffer,
        sampleCount: Int,
        stride: Int,
        descriptors: [MetalPlaneQuantizationDescriptor]
    ) throws -> MetalPlaneQuantizationBufferResult {
        try quantizePlaneBufferResults([(
            samples: samples,
            sampleCount: sampleCount,
            stride: stride,
            descriptors: descriptors
        )])[0]
    }

    func quantizePlaneBufferResults(
        _ planes: [(samples: MTLBuffer, sampleCount: Int, stride: Int, descriptors: [MetalPlaneQuantizationDescriptor])]
    ) throws -> [MetalPlaneQuantizationBufferResult] {
        guard !planes.isEmpty else {
            return []
        }

        var results = [MetalPlaneQuantizationBufferResult]()
        results.reserveCapacity(planes.count)
        var work = [(
            resultIndex: Int,
            samples: MTLBuffer,
            descriptorBuffer: MTLBuffer,
            qScaleBuffer: MTLBuffer,
            descriptorCount: Int
        )]()
        work.reserveCapacity(planes.count)

        for plane in planes {
            guard plane.stride > 0,
                  plane.sampleCount > 0,
                  plane.samples.length >= plane.sampleCount * MemoryLayout<Float>.stride,
                  plane.descriptors.count <= Int(UInt32.max) else {
                throw PyrowaveError.invalidDimensions
            }

            let coefficientByteLength = plane.sampleCount * MemoryLayout<Int16>.stride
            guard let output = device.makeBuffer(length: coefficientByteLength, options: .storageModeShared) else {
                throw PyrowaveError.processFailed("failed to allocate Metal plane quantization output buffer")
            }

            let resultIndex = results.count
            results.append(MetalPlaneQuantizationBufferResult(
                coefficientBuffer: output,
                coefficientCount: plane.sampleCount,
                qScaleCodesByDescriptor: []
            ))

            guard !plane.descriptors.isEmpty else {
                continue
            }

            let qScaleByteLength = plane.descriptors.count * 16 * MemoryLayout<UInt8>.stride
            let descriptorBuffer = try reusableSharedBuffer(bytes: plane.descriptors, purpose: .quantizeDescriptor, planeIndex: resultIndex)
            let qScaleBuffer = try reusableSharedBuffer(byteLength: qScaleByteLength, purpose: .quantizeQScale, planeIndex: resultIndex)

            work.append((
                resultIndex: resultIndex,
                samples: plane.samples,
                descriptorBuffer: descriptorBuffer,
                qScaleBuffer: qScaleBuffer,
                descriptorCount: plane.descriptors.count
            ))
        }

        guard !work.isEmpty else {
            return results
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal plane quantization command encoder")
        }

        encoder.setComputePipelineState(quantizePlaneTilesPipeline)
        let width = min(quantizePlaneTilesPipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadsPerThreadgroup = MTLSize(width: width, height: 1, depth: 1)
        for item in work {
            var constants = PlaneQuantizationConstants(descriptorCount: UInt32(item.descriptorCount))
            encoder.setBuffer(item.samples, offset: 0, index: 0)
            encoder.setBuffer(results[item.resultIndex].coefficientBuffer, offset: 0, index: 1)
            encoder.setBuffer(item.descriptorBuffer, offset: 0, index: 2)
            encoder.setBuffer(item.qScaleBuffer, offset: 0, index: 3)
            encoder.setBytes(&constants, length: MemoryLayout<PlaneQuantizationConstants>.stride, index: 4)
            encoder.dispatchThreads(
                MTLSize(width: item.descriptorCount * 16, height: 1, depth: 1),
                threadsPerThreadgroup: threadsPerThreadgroup
            )
        }
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal plane quantization command failed: \(error)")
        }

        for item in work {
            let qScaleCount = item.descriptorCount * 16
            let qScalePointer = item.qScaleBuffer.contents().bindMemory(to: UInt8.self, capacity: qScaleCount)
            let flatQScales = Array(UnsafeBufferPointer(start: qScalePointer, count: qScaleCount))
            let perDescriptor = Swift.stride(from: 0, to: flatQScales.count, by: 16).map {
                Array(flatQScales[$0..<$0 + 16])
            }
            results[item.resultIndex].qScaleCodesByDescriptor = perDescriptor
        }
        return results
    }

    func applySparseCoefficients(sampleCount: Int, entries: [MetalSparseCoefficientEntry]) throws -> [Float] {
        let output = try applySparseCoefficientBuffer(sampleCount: sampleCount, entries: entries)
        guard sampleCount > 0 else {
            return []
        }
        let pointer = output.contents().bindMemory(to: Float.self, capacity: sampleCount)
        return Array(UnsafeBufferPointer(start: pointer, count: sampleCount))
    }

    func applySparseCoefficientBuffer(sampleCount: Int, entries: [MetalSparseCoefficientEntry]) throws -> MTLBuffer {
        try applySparseCoefficientBuffers([(sampleCount: sampleCount, entries: entries)])[0]
    }

    func applySparseCoefficientBuffers(_ planes: [(sampleCount: Int, entries: [MetalSparseCoefficientEntry])]) throws -> [MTLBuffer] {
        guard !planes.isEmpty else {
            return []
        }
        guard planes.allSatisfy({ $0.sampleCount >= 0 }),
              planes.allSatisfy({ $0.sampleCount <= Int(UInt32.max) && $0.entries.count <= Int(UInt32.max) }) else {
            throw PyrowaveError.invalidDimensions
        }

        var outputs = [MTLBuffer]()
        outputs.reserveCapacity(planes.count)
        var zeroFills = [(buffer: MTLBuffer, byteLength: Int)]()
        zeroFills.reserveCapacity(planes.count)
        var dispatches = [(planeIndex: Int, entryBuffer: MTLBuffer, entryCount: Int, sampleCount: Int)]()
        dispatches.reserveCapacity(planes.count)

        for (planeIndex, plane) in planes.enumerated() {
            if plane.sampleCount == 0 {
                guard plane.entries.isEmpty else {
                    throw PyrowaveError.invalidDimensions
                }
                let output = try sparseCoefficientOutput(planeIndex: planeIndex, sampleCount: 0)
                output.contents().storeBytes(of: Float(0), as: Float.self)
                outputs.append(output)
                continue
            }

            let byteLength = plane.sampleCount * MemoryLayout<Float>.stride
            let output = try sparseCoefficientOutput(planeIndex: planeIndex, sampleCount: plane.sampleCount)
            outputs.append(output)
            zeroFills.append((buffer: output, byteLength: byteLength))

            guard !plane.entries.isEmpty else {
                continue
            }
            let entryBuffer = try reusableSharedBuffer(bytes: plane.entries, purpose: .sparseEntries, planeIndex: planeIndex)
            dispatches.append((
                planeIndex: planeIndex,
                entryBuffer: entryBuffer,
                entryCount: plane.entries.count,
                sampleCount: plane.sampleCount
            ))
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw PyrowaveError.processFailed("failed to create Metal sparse coefficient command buffer")
        }

        if !zeroFills.isEmpty {
            guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
                throw PyrowaveError.processFailed("failed to create Metal sparse coefficient fill encoder")
            }
            for fill in zeroFills {
                blitEncoder.fill(buffer: fill.buffer, range: 0..<fill.byteLength, value: 0)
            }
            blitEncoder.endEncoding()
        }

        if !dispatches.isEmpty {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw PyrowaveError.processFailed("failed to create Metal sparse coefficient command encoder")
            }

            encoder.setComputePipelineState(sparseApplyPipeline)
            let width = min(sparseApplyPipeline.maxTotalThreadsPerThreadgroup, 256)
            let threadsPerThreadgroup = MTLSize(width: width, height: 1, depth: 1)
            for dispatch in dispatches {
                var constants = SparseApplyConstants(
                    entryCount: UInt32(dispatch.entryCount),
                    sampleCount: UInt32(dispatch.sampleCount)
                )
                encoder.setBuffer(outputs[dispatch.planeIndex], offset: 0, index: 0)
                encoder.setBuffer(dispatch.entryBuffer, offset: 0, index: 1)
                encoder.setBytes(&constants, length: MemoryLayout<SparseApplyConstants>.stride, index: 2)
                encoder.dispatchThreads(
                    MTLSize(width: dispatch.entryCount, height: 1, depth: 1),
                    threadsPerThreadgroup: threadsPerThreadgroup
                )
            }
            encoder.endEncoding()
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal sparse coefficient command failed: \(error)")
        }

        return outputs
    }

    func decodeSparsePacketBuffers(
        packetData: Data,
        planes: [(sampleCount: Int, descriptors: [MetalSparsePacketDecodeDescriptor])]
    ) throws -> [MTLBuffer] {
        guard !planes.isEmpty else {
            return []
        }
        guard !packetData.isEmpty,
              packetData.count <= Int(UInt32.max),
              planes.allSatisfy({ $0.sampleCount >= 0 }),
              planes.allSatisfy({ $0.sampleCount <= Int(UInt32.max) && $0.descriptors.count <= Int(UInt32.max) }) else {
            throw PyrowaveError.invalidDimensions
        }

        let packetBuffer = try reusableSharedBuffer(data: packetData, purpose: .sparsePacketDecodeData, planeIndex: 0)
        var outputs = [MTLBuffer]()
        outputs.reserveCapacity(planes.count)
        var zeroFills = [(buffer: MTLBuffer, byteLength: Int)]()
        zeroFills.reserveCapacity(planes.count)
        var dispatches = [(planeIndex: Int, descriptorBuffer: MTLBuffer, descriptorCount: Int, sampleCount: Int)]()
        dispatches.reserveCapacity(planes.count)

        for (planeIndex, plane) in planes.enumerated() {
            if plane.sampleCount == 0 {
                guard plane.descriptors.isEmpty else {
                    throw PyrowaveError.invalidDimensions
                }
                let output = try sparseCoefficientOutput(planeIndex: planeIndex, sampleCount: 0)
                output.contents().storeBytes(of: Float(0), as: Float.self)
                outputs.append(output)
                continue
            }

            let byteLength = plane.sampleCount * MemoryLayout<Float>.stride
            let output = try sparseCoefficientOutput(planeIndex: planeIndex, sampleCount: plane.sampleCount)
            outputs.append(output)
            zeroFills.append((buffer: output, byteLength: byteLength))

            guard !plane.descriptors.isEmpty else {
                continue
            }
            let descriptorBuffer = try reusableSharedBuffer(bytes: plane.descriptors, purpose: .sparsePacketDecodeDescriptor, planeIndex: planeIndex)
            dispatches.append((
                planeIndex: planeIndex,
                descriptorBuffer: descriptorBuffer,
                descriptorCount: plane.descriptors.count,
                sampleCount: plane.sampleCount
            ))
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw PyrowaveError.processFailed("failed to create Metal sparse packet decode command buffer")
        }

        if !zeroFills.isEmpty {
            guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
                throw PyrowaveError.processFailed("failed to create Metal sparse packet decode fill encoder")
            }
            for fill in zeroFills {
                blitEncoder.fill(buffer: fill.buffer, range: 0..<fill.byteLength, value: 0)
            }
            blitEncoder.endEncoding()
        }

        if !dispatches.isEmpty {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw PyrowaveError.processFailed("failed to create Metal sparse packet decode command encoder")
            }

            encoder.setComputePipelineState(sparsePacketDecodePipeline)
            let width = min(sparsePacketDecodePipeline.maxTotalThreadsPerThreadgroup, 256)
            let threadsPerThreadgroup = MTLSize(width: width, height: 1, depth: 1)
            for dispatch in dispatches {
                var constants = SparsePacketDecodeConstants(
                    descriptorCount: UInt32(dispatch.descriptorCount),
                    sampleCount: UInt32(dispatch.sampleCount)
                )
                encoder.setBuffer(outputs[dispatch.planeIndex], offset: 0, index: 0)
                encoder.setBuffer(packetBuffer, offset: 0, index: 1)
                encoder.setBuffer(dispatch.descriptorBuffer, offset: 0, index: 2)
                encoder.setBytes(&constants, length: MemoryLayout<SparsePacketDecodeConstants>.stride, index: 3)
                encoder.dispatchThreads(
                    MTLSize(width: dispatch.descriptorCount * 16, height: 1, depth: 1),
                    threadsPerThreadgroup: threadsPerThreadgroup
                )
            }
            encoder.endEncoding()
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal sparse packet decode command failed: \(error)")
        }

        return outputs
    }

    private func sparseCoefficientOutput(planeIndex: Int, sampleCount: Int) throws -> MTLBuffer {
        let key = SparseCoefficientOutputKey(planeIndex: planeIndex, sampleCount: sampleCount)
        let byteLength = max(sampleCount, 1) * MemoryLayout<Float>.stride
        if let output = sparseCoefficientOutputs[key], output.length >= byteLength {
            return output
        }
        guard let output = device.makeBuffer(length: byteLength, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal sparse coefficient output")
        }
        sparseCoefficientOutputs[key] = output
        return output
    }

    private func reusableSharedBuffer<T>(
        bytes values: [T],
        purpose: ReusableBufferPurpose,
        planeIndex: Int
    ) throws -> MTLBuffer {
        let byteLength = values.count * MemoryLayout<T>.stride
        let buffer = try reusableSharedBuffer(byteLength: byteLength, purpose: purpose, planeIndex: planeIndex)
        values.withUnsafeBytes { rawBuffer in
            if let baseAddress = rawBuffer.baseAddress, byteLength > 0 {
                buffer.contents().copyMemory(from: baseAddress, byteCount: byteLength)
            }
        }
        return buffer
    }

    private func reusableSharedBuffer(
        data: Data,
        purpose: ReusableBufferPurpose,
        planeIndex: Int
    ) throws -> MTLBuffer {
        let buffer = try reusableSharedBuffer(byteLength: data.count, purpose: purpose, planeIndex: planeIndex)
        data.withUnsafeBytes { rawBuffer in
            if let baseAddress = rawBuffer.baseAddress, !data.isEmpty {
                buffer.contents().copyMemory(from: baseAddress, byteCount: data.count)
            }
        }
        return buffer
    }

    private func reusableSharedBuffer(
        byteLength: Int,
        purpose: ReusableBufferPurpose,
        planeIndex: Int
    ) throws -> MTLBuffer {
        let key = ReusableBufferKey(purpose: purpose, planeIndex: planeIndex)
        let length = max(byteLength, 1)
        if let buffer = reusableSharedBuffers[key], buffer.length >= length {
            return buffer
        }
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate reusable Metal buffer")
        }
        reusableSharedBuffers[key] = buffer
        return buffer
    }

    private func reusablePrivateBuffer(
        byteLength: Int,
        purpose: ReusableBufferPurpose,
        planeIndex: Int
    ) throws -> MTLBuffer {
        let key = ReusableBufferKey(purpose: purpose, planeIndex: planeIndex)
        let length = max(byteLength, 1)
        if let buffer = reusablePrivateBuffers[key], buffer.length >= length {
            return buffer
        }
        guard let buffer = device.makeBuffer(length: length, options: .storageModePrivate) else {
            throw PyrowaveError.processFailed("failed to allocate reusable private Metal buffer")
        }
        reusablePrivateBuffers[key] = buffer
        return buffer
    }

    func rateControlTileStats(
        coefficients: [Int16],
        descriptors: [MetalRateControlStatsDescriptor]
    ) throws -> [MetalRateControlTileStats] {
        guard !coefficients.isEmpty else {
            throw PyrowaveError.invalidDimensions
        }
        guard let coefficientBuffer = device.makeBuffer(bytes: coefficients, length: coefficients.count * MemoryLayout<Int16>.stride, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal rate-control coefficient buffer")
        }
        return try rateControlTileStats(coefficientBuffer: coefficientBuffer, coefficientCount: coefficients.count, descriptors: descriptors)
    }

    func rateControlTileStats(
        coefficientBuffer: MTLBuffer,
        coefficientCount: Int,
        descriptors: [MetalRateControlStatsDescriptor]
    ) throws -> [MetalRateControlTileStats] {
        try rateControlTileStatsBatch([(
            coefficientBuffer: coefficientBuffer,
            coefficientCount: coefficientCount,
            descriptors: descriptors
        )])[0]
    }

    func rateControlTileStatsBatch(
        _ planes: [(coefficientBuffer: MTLBuffer, coefficientCount: Int, descriptors: [MetalRateControlStatsDescriptor])]
    ) throws -> [[MetalRateControlTileStats]] {
        let flatResults = try rateControlTileStatsFlatBatch(planes)
        return flatResults.map { result in
            result.numPlanes.indices.map { index in
                let start = index * PyrowaveBlockStats.candidateCount
                let end = start + PyrowaveBlockStats.candidateCount
                return MetalRateControlTileStats(numPlanes: result.numPlanes[index], stats: Array(result.stats[start..<end]))
            }
        }
    }

    func rateControlTileStatsFlatBatch(
        _ planes: [(coefficientBuffer: MTLBuffer, coefficientCount: Int, descriptors: [MetalRateControlStatsDescriptor])]
    ) throws -> [MetalRateControlTileStatsFlat] {
        guard !planes.isEmpty else {
            return []
        }

        var work = [(
            planeIndex: Int,
            coefficientBuffer: MTLBuffer,
            descriptorBuffer: MTLBuffer,
            numPlanesBuffer: MTLBuffer,
            statsBuffer: MTLBuffer,
            descriptorCount: Int
        )]()
        work.reserveCapacity(planes.count)
        var emptyResults = Array(repeating: MetalRateControlTileStatsFlat(numPlanes: [], stats: []), count: planes.count)

        for (planeIndex, plane) in planes.enumerated() {
            guard plane.coefficientCount > 0,
                  plane.coefficientBuffer.length >= plane.coefficientCount * MemoryLayout<Int16>.stride,
                  plane.descriptors.count <= Int(UInt32.max) else {
                throw PyrowaveError.invalidDimensions
            }
            guard !plane.descriptors.isEmpty else {
                continue
            }

            let numPlanesByteLength = plane.descriptors.count * MemoryLayout<UInt32>.stride
            let statsByteLength = plane.descriptors.count * PyrowaveBlockStats.candidateCount * MemoryLayout<MetalRateControlQuantStats>.stride
            let descriptorBuffer = try reusableSharedBuffer(bytes: plane.descriptors, purpose: .rateStatsDescriptor, planeIndex: planeIndex)
            let numPlanesBuffer = try reusableSharedBuffer(byteLength: numPlanesByteLength, purpose: .rateStatsNumPlanes, planeIndex: planeIndex)
            let statsBuffer = try reusableSharedBuffer(byteLength: statsByteLength, purpose: .rateStats, planeIndex: planeIndex)
            work.append((
                planeIndex: planeIndex,
                coefficientBuffer: plane.coefficientBuffer,
                descriptorBuffer: descriptorBuffer,
                numPlanesBuffer: numPlanesBuffer,
                statsBuffer: statsBuffer,
                descriptorCount: plane.descriptors.count
            ))
        }

        guard !work.isEmpty else {
            return emptyResults
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal rate-control command encoder")
        }

        encoder.setComputePipelineState(rateControlStatsPipeline)
        let width = min(rateControlStatsPipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadsPerThreadgroup = MTLSize(width: width, height: 1, depth: 1)
        for item in work {
            var constants = RateControlStatsConstants(descriptorCount: UInt32(item.descriptorCount))
            encoder.setBuffer(item.coefficientBuffer, offset: 0, index: 0)
            encoder.setBuffer(item.descriptorBuffer, offset: 0, index: 1)
            encoder.setBuffer(item.numPlanesBuffer, offset: 0, index: 2)
            encoder.setBuffer(item.statsBuffer, offset: 0, index: 3)
            encoder.setBytes(&constants, length: MemoryLayout<RateControlStatsConstants>.stride, index: 4)
            encoder.dispatchThreads(
                MTLSize(width: item.descriptorCount, height: 1, depth: 1),
                threadsPerThreadgroup: threadsPerThreadgroup
            )
        }
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal rate-control command failed: \(error)")
        }

        for item in work {
            let statsCount = item.descriptorCount * PyrowaveBlockStats.candidateCount
            let numPlanesPointer = item.numPlanesBuffer.contents().bindMemory(to: UInt32.self, capacity: item.descriptorCount)
            let statsPointer = item.statsBuffer.contents().bindMemory(to: MetalRateControlQuantStats.self, capacity: statsCount)
            emptyResults[item.planeIndex] = MetalRateControlTileStatsFlat(
                numPlanes: Array(UnsafeBufferPointer(start: numPlanesPointer, count: item.descriptorCount)),
                stats: Array(UnsafeBufferPointer(start: statsPointer, count: statsCount))
            )
        }
        return emptyResults
    }

    func packetByteCosts(
        coefficients: [Int16],
        descriptors: [MetalPacketByteCostDescriptor]
    ) throws -> [[Int]] {
        guard !coefficients.isEmpty else {
            throw PyrowaveError.invalidDimensions
        }
        guard let coefficientBuffer = device.makeBuffer(bytes: coefficients, length: coefficients.count * MemoryLayout<Int16>.stride, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal packet byte-cost coefficient buffer")
        }
        return try packetByteCosts(coefficientBuffer: coefficientBuffer, coefficientCount: coefficients.count, descriptors: descriptors)
    }

    func packetByteCosts(
        coefficientBuffer: MTLBuffer,
        coefficientCount: Int,
        descriptors: [MetalPacketByteCostDescriptor]
    ) throws -> [[Int]] {
        try packetByteCostsBatch([(
            coefficientBuffer: coefficientBuffer,
            coefficientCount: coefficientCount,
            descriptors: descriptors
        )])[0]
    }

    func packetByteCostsBatch(
        _ planes: [(coefficientBuffer: MTLBuffer, coefficientCount: Int, descriptors: [MetalPacketByteCostDescriptor])]
    ) throws -> [[[Int]]] {
        guard !planes.isEmpty else {
            return []
        }

        var work = [(
            planeIndex: Int,
            coefficientBuffer: MTLBuffer,
            descriptorBuffer: MTLBuffer,
            byteCostBuffer: MTLBuffer,
            signCountBuffer: MTLBuffer,
            descriptorCount: Int
        )]()
        work.reserveCapacity(planes.count)
        var emptyResults = Array(repeating: [[Int]](), count: planes.count)

        for (planeIndex, plane) in planes.enumerated() {
            guard plane.coefficientCount > 0,
                  plane.coefficientBuffer.length >= plane.coefficientCount * MemoryLayout<Int16>.stride,
                  plane.descriptors.count <= Int(UInt32.max) else {
                throw PyrowaveError.invalidDimensions
            }
            guard !plane.descriptors.isEmpty else {
                continue
            }

            let byteCostByteLength = plane.descriptors.count * PyrowaveBlockStats.candidateCount * MemoryLayout<UInt32>.stride
            let descriptorBuffer = try reusableSharedBuffer(bytes: plane.descriptors, purpose: .packetCostDescriptor, planeIndex: planeIndex)
            let byteCostBuffer = try reusableSharedBuffer(byteLength: byteCostByteLength, purpose: .packetCostOutput, planeIndex: planeIndex)
            let signCountBuffer = try reusableSharedBuffer(byteLength: byteCostByteLength, purpose: .packetCostSignCount, planeIndex: planeIndex)
            work.append((
                planeIndex: planeIndex,
                coefficientBuffer: plane.coefficientBuffer,
                descriptorBuffer: descriptorBuffer,
                byteCostBuffer: byteCostBuffer,
                signCountBuffer: signCountBuffer,
                descriptorCount: plane.descriptors.count
            ))
        }

        guard !work.isEmpty else {
            return emptyResults
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw PyrowaveError.processFailed("failed to create Metal packet byte-cost command encoder")
        }

        guard let fillEncoder = commandBuffer.makeBlitCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal packet byte-cost fill encoder")
        }
        for item in work {
            let byteLength = item.descriptorCount * PyrowaveBlockStats.candidateCount * MemoryLayout<UInt32>.stride
            fillEncoder.fill(buffer: item.byteCostBuffer, range: 0..<byteLength, value: 0)
            fillEncoder.fill(buffer: item.signCountBuffer, range: 0..<byteLength, value: 0)
        }
        fillEncoder.endEncoding()

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal packet byte-cost command encoder")
        }
        encoder.setComputePipelineState(packetByteCostsSmallblocksPipeline)
        let width = min(packetByteCostsSmallblocksPipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadsPerThreadgroup = MTLSize(width: width, height: 1, depth: 1)
        for item in work {
            var constants = PacketByteCostConstants(descriptorCount: UInt32(item.descriptorCount))
            encoder.setBuffer(item.coefficientBuffer, offset: 0, index: 0)
            encoder.setBuffer(item.descriptorBuffer, offset: 0, index: 1)
            encoder.setBuffer(item.byteCostBuffer, offset: 0, index: 2)
            encoder.setBuffer(item.signCountBuffer, offset: 0, index: 3)
            encoder.setBytes(&constants, length: MemoryLayout<PacketByteCostConstants>.stride, index: 4)
            encoder.dispatchThreads(
                MTLSize(width: item.descriptorCount * 16, height: 1, depth: 1),
                threadsPerThreadgroup: threadsPerThreadgroup
            )
        }
        encoder.endEncoding()

        guard let finalizeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal packet byte-cost finalize encoder")
        }
        finalizeEncoder.setComputePipelineState(packetByteCostsFinalizePipeline)
        let finalizeWidth = min(packetByteCostsFinalizePipeline.maxTotalThreadsPerThreadgroup, 256)
        let finalizeThreadsPerThreadgroup = MTLSize(width: finalizeWidth, height: 1, depth: 1)
        for item in work {
            var constants = PacketByteCostConstants(descriptorCount: UInt32(item.descriptorCount))
            finalizeEncoder.setBuffer(item.byteCostBuffer, offset: 0, index: 0)
            finalizeEncoder.setBuffer(item.signCountBuffer, offset: 0, index: 1)
            finalizeEncoder.setBytes(&constants, length: MemoryLayout<PacketByteCostConstants>.stride, index: 2)
            finalizeEncoder.dispatchThreads(
                MTLSize(width: item.descriptorCount * PyrowaveBlockStats.candidateCount, height: 1, depth: 1),
                threadsPerThreadgroup: finalizeThreadsPerThreadgroup
            )
        }
        finalizeEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal packet byte-cost command failed: \(error)")
        }

        for item in work {
            let byteCostCount = item.descriptorCount * PyrowaveBlockStats.candidateCount
            let pointer = item.byteCostBuffer.contents().bindMemory(to: UInt32.self, capacity: byteCostCount)
            emptyResults[item.planeIndex] = readCandidateRows(pointer, rowCount: item.descriptorCount)
        }
        return emptyResults
    }

    func encodeSparsePackets(
        coefficients: [Int16],
        descriptors: [MetalSparsePacketEncodeDescriptor],
        qScaleCodes: [[UInt8]]
    ) throws -> [Data?] {
        guard !coefficients.isEmpty else {
            throw PyrowaveError.invalidDimensions
        }
        guard let coefficientBuffer = device.makeBuffer(bytes: coefficients, length: coefficients.count * MemoryLayout<Int16>.stride, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal sparse packet coefficient buffer")
        }
        return try encodeSparsePackets(coefficientBuffer: coefficientBuffer, coefficientCount: coefficients.count, descriptors: descriptors, qScaleCodes: qScaleCodes)
    }

    func encodeSparsePackets(
        coefficientBuffer: MTLBuffer,
        coefficientCount: Int,
        descriptors: [MetalSparsePacketEncodeDescriptor],
        qScaleCodes: [[UInt8]]
    ) throws -> [Data?] {
        try encodeSparsePacketsBatch([(
            coefficientBuffer: coefficientBuffer,
            coefficientCount: coefficientCount,
            descriptors: descriptors,
            qScaleCodes: qScaleCodes
        )])[0]
    }

    func encodeSparsePacketsBatch(
        _ planes: [(coefficientBuffer: MTLBuffer, coefficientCount: Int, descriptors: [MetalSparsePacketEncodeDescriptor], qScaleCodes: [[UInt8]])]
    ) throws -> [[Data?]] {
        guard !planes.isEmpty else {
            return []
        }

        let maxPacketBytes = PyrowaveCoefficientBlockCodec.maximumEncodedBlockBytes
        var work = [(
            planeIndex: Int,
            coefficientBuffer: MTLBuffer,
            descriptorBuffer: MTLBuffer,
            qScaleBuffer: MTLBuffer,
            outputBuffer: MTLBuffer,
            sizeBuffer: MTLBuffer,
            descriptorCount: Int
        )]()
        work.reserveCapacity(planes.count)
        var results = Array(repeating: [Data?](), count: planes.count)

        for (planeIndex, plane) in planes.enumerated() {
            guard plane.coefficientCount > 0,
                  plane.coefficientBuffer.length >= plane.coefficientCount * MemoryLayout<Int16>.stride,
                  plane.descriptors.count == plane.qScaleCodes.count,
                  plane.descriptors.count <= Int(UInt32.max) else {
                throw PyrowaveError.invalidDimensions
            }
            guard !plane.descriptors.isEmpty else {
                continue
            }

            var flatQScaleCodes = [UInt8]()
            flatQScaleCodes.reserveCapacity(plane.qScaleCodes.count * 16)
            for codes in plane.qScaleCodes {
                guard codes.count == 16 else {
                    throw PyrowaveError.invalidBitstream("expected sixteen 8x8 quant scale codes")
                }
                flatQScaleCodes.append(contentsOf: codes)
            }

            let outputByteCount = plane.descriptors.count * maxPacketBytes
            let outputSizeByteCount = plane.descriptors.count * MemoryLayout<UInt32>.stride
            let descriptorBuffer = try reusableSharedBuffer(bytes: plane.descriptors, purpose: .sparsePacketDescriptor, planeIndex: planeIndex)
            let qScaleBuffer = try reusableSharedBuffer(bytes: flatQScaleCodes, purpose: .sparsePacketQScale, planeIndex: planeIndex)
            let sizeBuffer = try reusableSharedBuffer(byteLength: outputSizeByteCount, purpose: .sparsePacketSize, planeIndex: planeIndex)
            guard let outputBuffer = device.makeBuffer(length: outputByteCount, options: .storageModeShared) else {
                throw PyrowaveError.processFailed("failed to allocate Metal sparse packet encode output")
            }
            work.append((
                planeIndex: planeIndex,
                coefficientBuffer: plane.coefficientBuffer,
                descriptorBuffer: descriptorBuffer,
                qScaleBuffer: qScaleBuffer,
                outputBuffer: outputBuffer,
                sizeBuffer: sizeBuffer,
                descriptorCount: plane.descriptors.count
            ))
        }

        guard !work.isEmpty else {
            return results
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal sparse packet encode command encoder")
        }

        encoder.setComputePipelineState(sparsePacketEncodePipeline)
        let width = min(sparsePacketEncodePipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadsPerThreadgroup = MTLSize(width: width, height: 1, depth: 1)
        for item in work {
            var constants = SparsePacketEncodeConstants(
                descriptorCount: UInt32(item.descriptorCount),
                maxPacketBytes: UInt32(maxPacketBytes)
            )
            encoder.setBuffer(item.coefficientBuffer, offset: 0, index: 0)
            encoder.setBuffer(item.descriptorBuffer, offset: 0, index: 1)
            encoder.setBuffer(item.qScaleBuffer, offset: 0, index: 2)
            encoder.setBuffer(item.outputBuffer, offset: 0, index: 3)
            encoder.setBuffer(item.sizeBuffer, offset: 0, index: 4)
            encoder.setBytes(&constants, length: MemoryLayout<SparsePacketEncodeConstants>.stride, index: 5)
            encoder.dispatchThreads(
                MTLSize(width: item.descriptorCount, height: 1, depth: 1),
                threadsPerThreadgroup: threadsPerThreadgroup
            )
        }
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal sparse packet encode command failed: \(error)")
        }

        for item in work {
            let outputByteCount = item.descriptorCount * maxPacketBytes
            let bytesPointer = item.outputBuffer.contents().bindMemory(to: UInt8.self, capacity: outputByteCount)
            let sizePointer = item.sizeBuffer.contents().bindMemory(to: UInt32.self, capacity: item.descriptorCount)
            let retainedOutputBuffer = item.outputBuffer
            results[item.planeIndex] = (0..<item.descriptorCount).map { index in
                let size = Int(sizePointer[index])
                guard size > 0 else {
                    return nil
                }
                guard size <= maxPacketBytes else {
                    return nil
                }
                let start = index * maxPacketBytes
                let packetPointer = UnsafeMutableRawPointer(bytesPointer.advanced(by: start))
                return Data(bytesNoCopy: packetPointer, count: size, deallocator: .custom { _, _ in
                    _ = retainedOutputBuffer
                })
            }
        }
        return results
    }

    func rateControlBucketIndices(
        distortions: [[Float]],
        packetByteCosts: [[Int]]
    ) throws -> [[Int]] {
        try rateControlBucketIndicesBatch([(distortions: distortions, packetByteCosts: packetByteCosts)])[0]
    }

    func rateControlBucketIndicesBatch(
        _ planes: [(distortions: [[Float]], packetByteCosts: [[Int]])]
    ) throws -> [[[Int]]] {
        guard !planes.isEmpty else {
            return []
        }

        var results = Array(repeating: [[Int]](), count: planes.count)
        var work = [(
            planeIndex: Int,
            distortionBuffer: MTLBuffer,
            packetByteCostBuffer: MTLBuffer,
            bucketIndexBuffer: MTLBuffer,
            blockCount: Int
        )]()
        work.reserveCapacity(planes.count)

        for (planeIndex, plane) in planes.enumerated() {
            guard plane.distortions.count == plane.packetByteCosts.count else {
                throw PyrowaveError.processFailed("rate-control distortion and packet-cost counts differ")
            }
            guard !plane.distortions.isEmpty else {
                continue
            }
            guard plane.distortions.count <= Int(UInt32.max) else {
                throw PyrowaveError.invalidDimensions
            }

            var flatDistortions = [Float]()
            var flatPacketByteCosts = [UInt32]()
            flatDistortions.reserveCapacity(plane.distortions.count * PyrowaveBlockStats.candidateCount)
            flatPacketByteCosts.reserveCapacity(plane.packetByteCosts.count * PyrowaveBlockStats.candidateCount)
            for index in plane.distortions.indices {
                guard plane.distortions[index].count == PyrowaveBlockStats.candidateCount,
                      plane.packetByteCosts[index].count == PyrowaveBlockStats.candidateCount else {
                    throw PyrowaveError.processFailed("rate-control bucket input must contain \(PyrowaveBlockStats.candidateCount) candidates per block")
                }
                flatDistortions.append(contentsOf: plane.distortions[index])
                for cost in plane.packetByteCosts[index] {
                    guard cost >= 0, cost <= Int(UInt32.max) else {
                        throw PyrowaveError.invalidDimensions
                    }
                    flatPacketByteCosts.append(UInt32(cost))
                }
            }

            let bucketIndexByteLength = flatDistortions.count * MemoryLayout<UInt32>.stride
            guard let distortionBuffer = device.makeBuffer(
                bytes: flatDistortions,
                length: flatDistortions.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ),
                  let packetByteCostBuffer = device.makeBuffer(
                    bytes: flatPacketByteCosts,
                    length: flatPacketByteCosts.count * MemoryLayout<UInt32>.stride,
                    options: .storageModeShared
                  ),
                  let bucketIndexBuffer = device.makeBuffer(
                    length: bucketIndexByteLength,
                    options: .storageModeShared
                  ) else {
                throw PyrowaveError.processFailed("failed to allocate Metal rate-control bucket buffers")
            }

            work.append((
                planeIndex: planeIndex,
                distortionBuffer: distortionBuffer,
                packetByteCostBuffer: packetByteCostBuffer,
                bucketIndexBuffer: bucketIndexBuffer,
                blockCount: plane.distortions.count
            ))
        }

        guard !work.isEmpty else {
            return results
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal rate-control bucket command encoder")
        }

        encoder.setComputePipelineState(rateControlBucketPipeline)
        let width = min(rateControlBucketPipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadsPerThreadgroup = MTLSize(width: width, height: 1, depth: 1)
        for item in work {
            var constants = RateControlBucketConstants(blockCount: UInt32(item.blockCount))
            encoder.setBuffer(item.distortionBuffer, offset: 0, index: 0)
            encoder.setBuffer(item.packetByteCostBuffer, offset: 0, index: 1)
            encoder.setBuffer(item.bucketIndexBuffer, offset: 0, index: 2)
            encoder.setBytes(&constants, length: MemoryLayout<RateControlBucketConstants>.stride, index: 3)
            encoder.dispatchThreads(
                MTLSize(width: item.blockCount, height: 1, depth: 1),
                threadsPerThreadgroup: threadsPerThreadgroup
            )
        }
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal rate-control bucket command failed: \(error)")
        }

        for item in work {
            let bucketIndexCount = item.blockCount * PyrowaveBlockStats.candidateCount
            let pointer = item.bucketIndexBuffer.contents().bindMemory(to: UInt32.self, capacity: bucketIndexCount)
            let flatBuckets = Array(UnsafeBufferPointer(start: pointer, count: bucketIndexCount))
            results[item.planeIndex] = Swift.stride(from: 0, to: flatBuckets.count, by: PyrowaveBlockStats.candidateCount).map {
                flatBuckets[$0..<$0 + PyrowaveBlockStats.candidateCount].map(Int.init)
            }
        }
        return results
    }

    func rateControlBucketDataFromTileStatsBatch(
        _ planes: [(
            coefficientBuffer: MTLBuffer,
            coefficientCount: Int,
            statsDescriptors: [MetalRateControlStatsDescriptor],
            packetByteCosts: [[Int]]
        )]
    ) throws -> MetalRateControlBucketBatchResult {
        guard !planes.isEmpty else {
            return MetalRateControlBucketBatchResult(bucketIndicesByPlane: [], cumulativeSavings: Array(repeating: 0, count: 128))
        }

        var results = Array(repeating: [[Int]](), count: planes.count)
        var work = [(
            planeIndex: Int,
            coefficientBuffer: MTLBuffer,
            statsDescriptorBuffer: MTLBuffer,
            numPlanesBuffer: MTLBuffer,
            statsBuffer: MTLBuffer,
            packetByteCostBuffer: MTLBuffer,
            bucketIndexBuffer: MTLBuffer,
            statsDescriptorCount: Int,
            blockCount: Int
        )]()
        work.reserveCapacity(planes.count)

        for (planeIndex, plane) in planes.enumerated() {
            guard plane.coefficientCount > 0,
                  plane.coefficientBuffer.length >= plane.coefficientCount * MemoryLayout<Int16>.stride,
                  plane.packetByteCosts.count <= Int(UInt32.max),
                  plane.statsDescriptors.count == plane.packetByteCosts.count * 16 else {
                throw PyrowaveError.invalidDimensions
            }
            guard !plane.packetByteCosts.isEmpty else {
                continue
            }

            var flatPacketByteCosts = [UInt32]()
            flatPacketByteCosts.reserveCapacity(plane.packetByteCosts.count * PyrowaveBlockStats.candidateCount)
            for blockCosts in plane.packetByteCosts {
                guard blockCosts.count == PyrowaveBlockStats.candidateCount else {
                    throw PyrowaveError.processFailed("rate-control bucket input must contain \(PyrowaveBlockStats.candidateCount) candidates per block")
                }
                for cost in blockCosts {
                    guard cost >= 0, cost <= Int(UInt32.max) else {
                        throw PyrowaveError.invalidDimensions
                    }
                    flatPacketByteCosts.append(UInt32(cost))
                }
            }

            let statsDescriptorCount = plane.statsDescriptors.count
            let numPlanesByteLength = statsDescriptorCount * MemoryLayout<UInt32>.stride
            let statsByteLength = statsDescriptorCount * PyrowaveBlockStats.candidateCount * MemoryLayout<MetalRateControlQuantStats>.stride
            let bucketIndexByteLength = flatPacketByteCosts.count * MemoryLayout<UInt32>.stride
            let statsDescriptorBuffer = try reusableSharedBuffer(bytes: plane.statsDescriptors, purpose: .rateStatsDescriptor, planeIndex: planeIndex)
            let numPlanesBuffer = try reusableSharedBuffer(byteLength: numPlanesByteLength, purpose: .rateStatsNumPlanes, planeIndex: planeIndex)
            let statsBuffer = try reusableSharedBuffer(byteLength: statsByteLength, purpose: .rateStats, planeIndex: planeIndex)
            guard let packetByteCostBuffer = device.makeBuffer(
                bytes: flatPacketByteCosts,
                length: flatPacketByteCosts.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared
            ),
                  let bucketIndexBuffer = device.makeBuffer(
                    length: bucketIndexByteLength,
                    options: .storageModeShared
                  ) else {
                throw PyrowaveError.processFailed("failed to allocate Metal rate-control tile bucket buffers")
            }

            work.append((
                planeIndex: planeIndex,
                coefficientBuffer: plane.coefficientBuffer,
                statsDescriptorBuffer: statsDescriptorBuffer,
                numPlanesBuffer: numPlanesBuffer,
                statsBuffer: statsBuffer,
                packetByteCostBuffer: packetByteCostBuffer,
                bucketIndexBuffer: bucketIndexBuffer,
                statsDescriptorCount: statsDescriptorCount,
                blockCount: plane.packetByteCosts.count
            ))
        }

        guard !work.isEmpty else {
            return MetalRateControlBucketBatchResult(bucketIndicesByPlane: results, cumulativeSavings: Array(repeating: 0, count: 128))
        }

        let bucketSavings = Array(repeating: UInt32(0), count: 128)
        let cumulativeSavings = Array(repeating: UInt32(0), count: 128)
        guard let bucketSavingsBuffer = device.makeBuffer(bytes: bucketSavings, length: bucketSavings.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let cumulativeSavingsBuffer = device.makeBuffer(bytes: cumulativeSavings, length: cumulativeSavings.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw PyrowaveError.processFailed("failed to allocate Metal fused rate-control tile bucket buffers")
        }

        guard let statsEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal rate-control stats command encoder")
        }
        statsEncoder.setComputePipelineState(rateControlStatsPipeline)
        let statsWidth = min(rateControlStatsPipeline.maxTotalThreadsPerThreadgroup, 256)
        let statsThreads = MTLSize(width: statsWidth, height: 1, depth: 1)
        for item in work {
            var constants = RateControlStatsConstants(descriptorCount: UInt32(item.statsDescriptorCount))
            statsEncoder.setBuffer(item.coefficientBuffer, offset: 0, index: 0)
            statsEncoder.setBuffer(item.statsDescriptorBuffer, offset: 0, index: 1)
            statsEncoder.setBuffer(item.numPlanesBuffer, offset: 0, index: 2)
            statsEncoder.setBuffer(item.statsBuffer, offset: 0, index: 3)
            statsEncoder.setBytes(&constants, length: MemoryLayout<RateControlStatsConstants>.stride, index: 4)
            statsEncoder.dispatchThreads(
                MTLSize(width: item.statsDescriptorCount, height: 1, depth: 1),
                threadsPerThreadgroup: statsThreads
            )
        }
        statsEncoder.endEncoding()

        guard let bucketEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal rate-control tile bucket command encoder")
        }
        bucketEncoder.setComputePipelineState(rateControlTileStatsBucketPipeline)
        let bucketWidth = min(rateControlTileStatsBucketPipeline.maxTotalThreadsPerThreadgroup, 256)
        let bucketThreads = MTLSize(width: bucketWidth, height: 1, depth: 1)
        for item in work {
            var constants = RateControlBucketConstants(blockCount: UInt32(item.blockCount))
            bucketEncoder.setBuffer(item.statsBuffer, offset: 0, index: 0)
            bucketEncoder.setBuffer(item.packetByteCostBuffer, offset: 0, index: 1)
            bucketEncoder.setBuffer(item.bucketIndexBuffer, offset: 0, index: 2)
            bucketEncoder.setBytes(&constants, length: MemoryLayout<RateControlBucketConstants>.stride, index: 3)
            bucketEncoder.dispatchThreads(
                MTLSize(width: item.blockCount, height: 1, depth: 1),
                threadsPerThreadgroup: bucketThreads
            )
        }
        bucketEncoder.endEncoding()

        guard let savingsEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal rate-control savings command encoder")
        }
        savingsEncoder.setComputePipelineState(rateControlBucketSavingsPipeline)
        let savingsWidth = min(rateControlBucketSavingsPipeline.maxTotalThreadsPerThreadgroup, 256)
        let savingsThreads = MTLSize(width: savingsWidth, height: 1, depth: 1)
        for item in work {
            var constants = RateControlBucketSavingsConstants(blockCount: UInt32(item.blockCount))
            savingsEncoder.setBuffer(item.bucketIndexBuffer, offset: 0, index: 0)
            savingsEncoder.setBuffer(item.packetByteCostBuffer, offset: 0, index: 1)
            savingsEncoder.setBuffer(bucketSavingsBuffer, offset: 0, index: 2)
            savingsEncoder.setBytes(&constants, length: MemoryLayout<RateControlBucketSavingsConstants>.stride, index: 3)
            savingsEncoder.dispatchThreads(
                MTLSize(width: item.blockCount, height: 1, depth: 1),
                threadsPerThreadgroup: savingsThreads
            )
        }
        savingsEncoder.endEncoding()

        guard let prefixEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal rate-control savings prefix command encoder")
        }
        prefixEncoder.setComputePipelineState(rateControlBucketSavingsPrefixPipeline)
        prefixEncoder.setBuffer(bucketSavingsBuffer, offset: 0, index: 0)
        prefixEncoder.setBuffer(cumulativeSavingsBuffer, offset: 0, index: 1)
        let prefixWidth = min(rateControlBucketSavingsPrefixPipeline.maxTotalThreadsPerThreadgroup, 128)
        prefixEncoder.dispatchThreads(
            MTLSize(width: 128, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: prefixWidth, height: 1, depth: 1)
        )
        prefixEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal fused rate-control tile bucket command failed: \(error)")
        }

        for item in work {
            let bucketIndexCount = item.blockCount * PyrowaveBlockStats.candidateCount
            let pointer = item.bucketIndexBuffer.contents().bindMemory(to: UInt32.self, capacity: bucketIndexCount)
            results[item.planeIndex] = readCandidateRows(pointer, rowCount: item.blockCount)
        }

        let savingsPointer = cumulativeSavingsBuffer.contents().bindMemory(to: UInt32.self, capacity: cumulativeSavings.count)
        let metalSavings = Array(UnsafeBufferPointer(start: savingsPointer, count: cumulativeSavings.count)).map(Int.init)
        return MetalRateControlBucketBatchResult(bucketIndicesByPlane: results, cumulativeSavings: metalSavings)
    }

    func rateControlBucketDataBatch(
        _ planes: [(distortions: [[Float]], packetByteCosts: [[Int]])]
    ) throws -> MetalRateControlBucketBatchResult {
        guard !planes.isEmpty else {
            return MetalRateControlBucketBatchResult(bucketIndicesByPlane: [], cumulativeSavings: Array(repeating: 0, count: 128))
        }

        var results = Array(repeating: [[Int]](), count: planes.count)
        var work = [(
            planeIndex: Int,
            distortionBuffer: MTLBuffer,
            packetByteCostBuffer: MTLBuffer,
            bucketIndexBuffer: MTLBuffer,
            blockCount: Int
        )]()
        work.reserveCapacity(planes.count)

        for (planeIndex, plane) in planes.enumerated() {
            guard plane.distortions.count == plane.packetByteCosts.count else {
                throw PyrowaveError.processFailed("rate-control distortion and packet-cost counts differ")
            }
            guard !plane.distortions.isEmpty else {
                continue
            }
            guard plane.distortions.count <= Int(UInt32.max) else {
                throw PyrowaveError.invalidDimensions
            }

            var flatDistortions = [Float]()
            var flatPacketByteCosts = [UInt32]()
            flatDistortions.reserveCapacity(plane.distortions.count * PyrowaveBlockStats.candidateCount)
            flatPacketByteCosts.reserveCapacity(plane.packetByteCosts.count * PyrowaveBlockStats.candidateCount)
            for index in plane.distortions.indices {
                guard plane.distortions[index].count == PyrowaveBlockStats.candidateCount,
                      plane.packetByteCosts[index].count == PyrowaveBlockStats.candidateCount else {
                    throw PyrowaveError.processFailed("rate-control bucket input must contain \(PyrowaveBlockStats.candidateCount) candidates per block")
                }
                flatDistortions.append(contentsOf: plane.distortions[index])
                for cost in plane.packetByteCosts[index] {
                    guard cost >= 0, cost <= Int(UInt32.max) else {
                        throw PyrowaveError.invalidDimensions
                    }
                    flatPacketByteCosts.append(UInt32(cost))
                }
            }

            let bucketIndexByteLength = flatDistortions.count * MemoryLayout<UInt32>.stride
            guard let distortionBuffer = device.makeBuffer(
                bytes: flatDistortions,
                length: flatDistortions.count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ),
                  let packetByteCostBuffer = device.makeBuffer(
                    bytes: flatPacketByteCosts,
                    length: flatPacketByteCosts.count * MemoryLayout<UInt32>.stride,
                    options: .storageModeShared
                  ),
                  let bucketIndexBuffer = device.makeBuffer(
                    length: bucketIndexByteLength,
                    options: .storageModeShared
                  ) else {
                throw PyrowaveError.processFailed("failed to allocate Metal rate-control bucket buffers")
            }

            work.append((
                planeIndex: planeIndex,
                distortionBuffer: distortionBuffer,
                packetByteCostBuffer: packetByteCostBuffer,
                bucketIndexBuffer: bucketIndexBuffer,
                blockCount: plane.distortions.count
            ))
        }

        guard !work.isEmpty else {
            return MetalRateControlBucketBatchResult(bucketIndicesByPlane: results, cumulativeSavings: Array(repeating: 0, count: 128))
        }

        let bucketSavings = Array(repeating: UInt32(0), count: 128)
        let cumulativeSavings = Array(repeating: UInt32(0), count: 128)
        guard let bucketSavingsBuffer = device.makeBuffer(bytes: bucketSavings, length: bucketSavings.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let cumulativeSavingsBuffer = device.makeBuffer(bytes: cumulativeSavings, length: cumulativeSavings.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw PyrowaveError.processFailed("failed to allocate Metal fused rate-control bucket buffers")
        }

        guard let bucketEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal rate-control bucket command encoder")
        }
        bucketEncoder.setComputePipelineState(rateControlBucketPipeline)
        let bucketWidth = min(rateControlBucketPipeline.maxTotalThreadsPerThreadgroup, 256)
        let bucketThreads = MTLSize(width: bucketWidth, height: 1, depth: 1)
        for item in work {
            var constants = RateControlBucketConstants(blockCount: UInt32(item.blockCount))
            bucketEncoder.setBuffer(item.distortionBuffer, offset: 0, index: 0)
            bucketEncoder.setBuffer(item.packetByteCostBuffer, offset: 0, index: 1)
            bucketEncoder.setBuffer(item.bucketIndexBuffer, offset: 0, index: 2)
            bucketEncoder.setBytes(&constants, length: MemoryLayout<RateControlBucketConstants>.stride, index: 3)
            bucketEncoder.dispatchThreads(
                MTLSize(width: item.blockCount, height: 1, depth: 1),
                threadsPerThreadgroup: bucketThreads
            )
        }
        bucketEncoder.endEncoding()

        guard let savingsEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal rate-control savings command encoder")
        }
        savingsEncoder.setComputePipelineState(rateControlBucketSavingsPipeline)
        let savingsWidth = min(rateControlBucketSavingsPipeline.maxTotalThreadsPerThreadgroup, 256)
        let savingsThreads = MTLSize(width: savingsWidth, height: 1, depth: 1)
        for item in work {
            var constants = RateControlBucketSavingsConstants(blockCount: UInt32(item.blockCount))
            savingsEncoder.setBuffer(item.bucketIndexBuffer, offset: 0, index: 0)
            savingsEncoder.setBuffer(item.packetByteCostBuffer, offset: 0, index: 1)
            savingsEncoder.setBuffer(bucketSavingsBuffer, offset: 0, index: 2)
            savingsEncoder.setBytes(&constants, length: MemoryLayout<RateControlBucketSavingsConstants>.stride, index: 3)
            savingsEncoder.dispatchThreads(
                MTLSize(width: item.blockCount, height: 1, depth: 1),
                threadsPerThreadgroup: savingsThreads
            )
        }
        savingsEncoder.endEncoding()

        guard let prefixEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal rate-control savings prefix command encoder")
        }
        prefixEncoder.setComputePipelineState(rateControlBucketSavingsPrefixPipeline)
        prefixEncoder.setBuffer(bucketSavingsBuffer, offset: 0, index: 0)
        prefixEncoder.setBuffer(cumulativeSavingsBuffer, offset: 0, index: 1)
        let prefixWidth = min(rateControlBucketSavingsPrefixPipeline.maxTotalThreadsPerThreadgroup, 128)
        prefixEncoder.dispatchThreads(
            MTLSize(width: 128, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: prefixWidth, height: 1, depth: 1)
        )
        prefixEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal fused rate-control bucket command failed: \(error)")
        }

        for item in work {
            let bucketIndexCount = item.blockCount * PyrowaveBlockStats.candidateCount
            let pointer = item.bucketIndexBuffer.contents().bindMemory(to: UInt32.self, capacity: bucketIndexCount)
            results[item.planeIndex] = readCandidateRows(pointer, rowCount: item.blockCount)
        }

        let savingsPointer = cumulativeSavingsBuffer.contents().bindMemory(to: UInt32.self, capacity: cumulativeSavings.count)
        let metalSavings = Array(UnsafeBufferPointer(start: savingsPointer, count: cumulativeSavings.count)).map(Int.init)
        return MetalRateControlBucketBatchResult(bucketIndicesByPlane: results, cumulativeSavings: metalSavings)
    }

    func rateControlCumulativeBucketSavings(
        bucketIndices: [[Int]],
        packetByteCosts: [[Int]]
    ) throws -> [Int] {
        guard bucketIndices.count == packetByteCosts.count else {
            throw PyrowaveError.processFailed("rate-control bucket and packet-cost counts differ")
        }
        guard !bucketIndices.isEmpty else {
            return Array(repeating: 0, count: 128)
        }
        guard bucketIndices.count <= Int(UInt32.max) else {
            throw PyrowaveError.invalidDimensions
        }

        var flatBuckets = [UInt32]()
        var flatPacketByteCosts = [UInt32]()
        flatBuckets.reserveCapacity(bucketIndices.count * PyrowaveBlockStats.candidateCount)
        flatPacketByteCosts.reserveCapacity(packetByteCosts.count * PyrowaveBlockStats.candidateCount)
        for index in bucketIndices.indices {
            guard bucketIndices[index].count == PyrowaveBlockStats.candidateCount,
                  packetByteCosts[index].count == PyrowaveBlockStats.candidateCount else {
                throw PyrowaveError.processFailed("rate-control savings input must contain \(PyrowaveBlockStats.candidateCount) candidates per block")
            }
            for bucket in bucketIndices[index] {
                guard bucket >= 0, bucket < 128 else {
                    throw PyrowaveError.invalidDimensions
                }
                flatBuckets.append(UInt32(bucket))
            }
            for cost in packetByteCosts[index] {
                guard cost >= 0, cost <= Int(UInt32.max) else {
                    throw PyrowaveError.invalidDimensions
                }
                flatPacketByteCosts.append(UInt32(cost))
            }
        }

        let bucketSavings = Array(repeating: UInt32(0), count: 128)
        let cumulativeSavings = Array(repeating: UInt32(0), count: 128)
        guard let bucketBuffer = device.makeBuffer(bytes: flatBuckets, length: flatBuckets.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let packetByteCostBuffer = device.makeBuffer(bytes: flatPacketByteCosts, length: flatPacketByteCosts.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let bucketSavingsBuffer = device.makeBuffer(bytes: bucketSavings, length: bucketSavings.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let cumulativeSavingsBuffer = device.makeBuffer(bytes: cumulativeSavings, length: cumulativeSavings.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw PyrowaveError.processFailed("failed to allocate Metal rate-control savings buffers")
        }

        var constants = RateControlBucketSavingsConstants(blockCount: UInt32(bucketIndices.count))
        guard let savingsEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal rate-control savings command encoder")
        }
        savingsEncoder.setComputePipelineState(rateControlBucketSavingsPipeline)
        savingsEncoder.setBuffer(bucketBuffer, offset: 0, index: 0)
        savingsEncoder.setBuffer(packetByteCostBuffer, offset: 0, index: 1)
        savingsEncoder.setBuffer(bucketSavingsBuffer, offset: 0, index: 2)
        savingsEncoder.setBytes(&constants, length: MemoryLayout<RateControlBucketSavingsConstants>.stride, index: 3)
        let savingsWidth = min(rateControlBucketSavingsPipeline.maxTotalThreadsPerThreadgroup, 256)
        savingsEncoder.dispatchThreads(
            MTLSize(width: bucketIndices.count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: savingsWidth, height: 1, depth: 1)
        )
        savingsEncoder.endEncoding()

        guard let prefixEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal rate-control savings prefix command encoder")
        }
        prefixEncoder.setComputePipelineState(rateControlBucketSavingsPrefixPipeline)
        prefixEncoder.setBuffer(bucketSavingsBuffer, offset: 0, index: 0)
        prefixEncoder.setBuffer(cumulativeSavingsBuffer, offset: 0, index: 1)
        let prefixWidth = min(rateControlBucketSavingsPrefixPipeline.maxTotalThreadsPerThreadgroup, 128)
        prefixEncoder.dispatchThreads(
            MTLSize(width: 128, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: prefixWidth, height: 1, depth: 1)
        )
        prefixEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal rate-control savings command failed: \(error)")
        }

        let pointer = cumulativeSavingsBuffer.contents().bindMemory(to: UInt32.self, capacity: cumulativeSavings.count)
        return Array(UnsafeBufferPointer(start: pointer, count: cumulativeSavings.count)).map(Int.init)
    }

    func forwardWavelet(_ samples: [Float], width: Int, height: Int, levels: Int) throws -> [Float] {
        try validateWaveletInput(samples, width: width, height: height, levels: levels)
        guard !samples.isEmpty else { return [] }

        let byteLength = samples.count * MemoryLayout<Float>.stride
        guard let primary = device.makeBuffer(bytes: samples, length: byteLength, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal DWT input buffer")
        }
        let output = try forwardWaveletBuffer(primary, sampleCount: samples.count, width: width, height: height, levels: levels)
        let pointer = output.contents().bindMemory(to: Float.self, capacity: samples.count)
        return Array(UnsafeBufferPointer(start: pointer, count: samples.count))
    }

    func forwardWaveletBuffer(_ buffer: MTLBuffer, sampleCount: Int, width: Int, height: Int, levels: Int) throws -> MTLBuffer {
        return try forwardWaveletBuffers([(buffer: buffer, sampleCount: sampleCount, width: width, height: height, levels: levels)])[0]
    }

    func forwardWaveletBuffers(_ planes: [(buffer: MTLBuffer, sampleCount: Int, width: Int, height: Int, levels: Int)]) throws -> [MTLBuffer] {
        guard !planes.isEmpty else {
            return []
        }
        for plane in planes {
            guard plane.sampleCount == plane.width * plane.height,
                  plane.buffer.length >= plane.sampleCount * MemoryLayout<Float>.stride else {
                throw PyrowaveError.invalidDimensions
            }
            try validateWaveletShape(width: plane.width, height: plane.height, levels: plane.levels)
        }

        var scratchBuffers = [MTLBuffer]()
        scratchBuffers.reserveCapacity(planes.count)
        for (planeIndex, plane) in planes.enumerated() {
            guard plane.sampleCount > 0 else {
                scratchBuffers.append(plane.buffer)
                continue
            }
            let scratch = try reusablePrivateBuffer(
                byteLength: plane.sampleCount * MemoryLayout<Float>.stride,
                purpose: .dwtScratch,
                planeIndex: planeIndex
            )
            scratchBuffers.append(scratch)
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw PyrowaveError.processFailed("failed to create Metal DWT command buffer")
        }

        let maxLevels = planes.map(\.levels).max() ?? 0
        for level in 0..<maxLevels {
            var active = [DWTBatchDispatch]()
            active.reserveCapacity(planes.count)
            for (planeIndex, plane) in planes.enumerated() where plane.sampleCount > 0 && level < plane.levels {
                active.append(DWTBatchDispatch(
                    primary: plane.buffer,
                    scratch: scratchBuffers[planeIndex],
                    activeWidth: plane.width >> level,
                    activeHeight: plane.height >> level,
                    stride: plane.width
                ))
            }
            guard !active.isEmpty else {
                continue
            }

            for phase in 0...4 {
                try encodeDWTInPlaceBatch(
                    commandBuffer: commandBuffer,
                    pipeline: dwtLiftRowsPipeline,
                    dispatches: active.map {
                        (buffer: $0.primary, activeWidth: $0.activeWidth, activeHeight: $0.activeHeight, stride: $0.stride, phase: phase)
                    }
                )
            }
            try encodeDWTCopyBatch(
                commandBuffer: commandBuffer,
                pipeline: dwtPackRowsPipeline,
                dispatches: active.map {
                    (input: $0.primary, output: $0.scratch, activeWidth: $0.activeWidth, activeHeight: $0.activeHeight, stride: $0.stride, phase: 0)
                }
            )

            for phase in 0...4 {
                try encodeDWTInPlaceBatch(
                    commandBuffer: commandBuffer,
                    pipeline: dwtLiftColumnsPipeline,
                    dispatches: active.map {
                        (buffer: $0.scratch, activeWidth: $0.activeWidth, activeHeight: $0.activeHeight, stride: $0.stride, phase: phase)
                    }
                )
            }
            try encodeDWTCopyBatch(
                commandBuffer: commandBuffer,
                pipeline: dwtPackColumnsPipeline,
                dispatches: active.map {
                    (input: $0.scratch, output: $0.primary, activeWidth: $0.activeWidth, activeHeight: $0.activeHeight, stride: $0.stride, phase: 0)
                }
            )
        }

        try finish(commandBuffer: commandBuffer, context: "Metal DWT")
        return planes.map(\.buffer)
    }

    func inverseWavelet(_ coefficients: [Float], width: Int, height: Int, levels: Int) throws -> [Float] {
        try validateWaveletInput(coefficients, width: width, height: height, levels: levels)
        guard !coefficients.isEmpty else { return [] }

        let byteLength = coefficients.count * MemoryLayout<Float>.stride
        guard let primary = device.makeBuffer(bytes: coefficients, length: byteLength, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal iDWT input buffer")
        }
        let output = try inverseWaveletBuffer(primary, sampleCount: coefficients.count, width: width, height: height, levels: levels)
        let pointer = output.contents().bindMemory(to: Float.self, capacity: coefficients.count)
        return Array(UnsafeBufferPointer(start: pointer, count: coefficients.count))
    }

    func inverseWaveletBuffer(_ buffer: MTLBuffer, sampleCount: Int, width: Int, height: Int, levels: Int) throws -> MTLBuffer {
        guard sampleCount == width * height,
              buffer.length >= sampleCount * MemoryLayout<Float>.stride else {
            throw PyrowaveError.invalidDimensions
        }
        try validateWaveletShape(width: width, height: height, levels: levels)
        guard sampleCount > 0 else { return buffer }

        return try inverseWaveletBuffers([(buffer: buffer, sampleCount: sampleCount, width: width, height: height, levels: levels)])[0]
    }

    func inverseWaveletBuffers(_ planes: [(buffer: MTLBuffer, sampleCount: Int, width: Int, height: Int, levels: Int)]) throws -> [MTLBuffer] {
        guard !planes.isEmpty else {
            return []
        }
        for plane in planes {
            guard plane.sampleCount == plane.width * plane.height,
                  plane.buffer.length >= plane.sampleCount * MemoryLayout<Float>.stride else {
                throw PyrowaveError.invalidDimensions
            }
            try validateWaveletShape(width: plane.width, height: plane.height, levels: plane.levels)
        }

        var scratchBuffers = [MTLBuffer]()
        scratchBuffers.reserveCapacity(planes.count)
        for (planeIndex, plane) in planes.enumerated() {
            guard plane.sampleCount > 0 else {
                scratchBuffers.append(plane.buffer)
                continue
            }
            let scratch = try reusablePrivateBuffer(
                byteLength: plane.sampleCount * MemoryLayout<Float>.stride,
                purpose: .idwtScratch,
                planeIndex: planeIndex
            )
            scratchBuffers.append(scratch)
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw PyrowaveError.processFailed("failed to create Metal iDWT command buffer")
        }

        let maxLevels = planes.map(\.levels).max() ?? 0
        guard maxLevels > 0 else {
            try finish(commandBuffer: commandBuffer, context: "Metal iDWT")
            return planes.map(\.buffer)
        }
        let useScaledUnpack = shouldFoldInverseWaveletScaleIntoUnpack(planes)
        for level in stride(from: maxLevels - 1, through: 0, by: -1) {
            var active = [DWTBatchDispatch]()
            active.reserveCapacity(planes.count)
            for (planeIndex, plane) in planes.enumerated() where plane.sampleCount > 0 && level < plane.levels {
                active.append(DWTBatchDispatch(
                    primary: plane.buffer,
                    scratch: scratchBuffers[planeIndex],
                    activeWidth: plane.width >> level,
                    activeHeight: plane.height >> level,
                    stride: plane.width
                ))
            }
            guard !active.isEmpty else {
                continue
            }

            try encodeDWTCopyBatch(
                commandBuffer: commandBuffer,
                pipeline: useScaledUnpack ? idwtUnpackColumnsScaledPipeline : dwtUnpackColumnsPipeline,
                dispatches: active.map {
                    (input: $0.primary, output: $0.scratch, activeWidth: $0.activeWidth, activeHeight: $0.activeHeight, stride: $0.stride, phase: 0)
                }
            )
            for phase in (useScaledUnpack ? 1 : 0)...4 {
                try encodeDWTInPlaceBatch(
                    commandBuffer: commandBuffer,
                    pipeline: idwtLiftColumnsPipeline,
                    dispatches: active.map {
                        (buffer: $0.scratch, activeWidth: $0.activeWidth, activeHeight: $0.activeHeight, stride: $0.stride, phase: phase)
                    }
                )
            }

            try encodeDWTCopyBatch(
                commandBuffer: commandBuffer,
                pipeline: useScaledUnpack ? idwtUnpackRowsScaledPipeline : dwtUnpackRowsPipeline,
                dispatches: active.map {
                    (input: $0.scratch, output: $0.primary, activeWidth: $0.activeWidth, activeHeight: $0.activeHeight, stride: $0.stride, phase: 0)
                }
            )
            for phase in (useScaledUnpack ? 1 : 0)...4 {
                try encodeDWTInPlaceBatch(
                    commandBuffer: commandBuffer,
                    pipeline: idwtLiftRowsPipeline,
                    dispatches: active.map {
                        (buffer: $0.primary, activeWidth: $0.activeWidth, activeHeight: $0.activeHeight, stride: $0.stride, phase: phase)
                    }
                )
            }
        }

        try finish(commandBuffer: commandBuffer, context: "Metal iDWT")
        return planes.map(\.buffer)
    }

    private func shouldFoldInverseWaveletScaleIntoUnpack(
        _ planes: [(buffer: MTLBuffer, sampleCount: Int, width: Int, height: Int, levels: Int)]
    ) -> Bool {
        let largestPlaneSamples = planes.map(\.sampleCount).max() ?? 0
        return largestPlaneSamples >= 6144 * 3456
    }

    private func readCandidateRows(_ pointer: UnsafePointer<UInt32>, rowCount: Int) -> [[Int]] {
        var rows = [[Int]]()
        rows.reserveCapacity(rowCount)
        for row in 0..<rowCount {
            let base = row * PyrowaveBlockStats.candidateCount
            var candidates = [Int]()
            candidates.reserveCapacity(PyrowaveBlockStats.candidateCount)
            for candidate in 0..<PyrowaveBlockStats.candidateCount {
                candidates.append(Int(pointer[base + candidate]))
            }
            rows.append(candidates)
        }
        return rows
    }

    private static func makeFunction(named name: String, library: MTLLibrary) throws -> MTLFunction {
        guard let function = library.makeFunction(name: name) else {
            throw PyrowaveError.processFailed("missing Metal function \(name)")
        }
        return function
    }

    private func dispatch(
        pipeline: MTLComputePipelineState,
        count: Int,
        buffers: [(MTLBuffer, Int)],
        constants: inout QuantizationConstants
    ) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal compute command encoder")
        }

        encoder.setComputePipelineState(pipeline)
        for (buffer, index) in buffers {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        encoder.setBytes(&constants, length: MemoryLayout<QuantizationConstants>.stride, index: 2)

        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadsPerThreadgroup = MTLSize(width: width, height: 1, depth: 1)
        let threads = MTLSize(width: count, height: 1, depth: 1)
        encoder.dispatchThreads(threads, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal command failed: \(error)")
        }
    }

    private func dispatchDWTInPlace(
        pipeline: MTLComputePipelineState,
        buffer: MTLBuffer,
        activeWidth: Int,
        activeHeight: Int,
        stride: Int,
        phase: Int
    ) throws {
        var constants = DWTConstants(
            activeWidth: UInt32(activeWidth),
            activeHeight: UInt32(activeHeight),
            stride: UInt32(stride),
            phase: UInt32(phase)
        )
        try dispatchDWT(pipeline: pipeline, activeWidth: activeWidth, activeHeight: activeHeight) { encoder in
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&constants, length: MemoryLayout<DWTConstants>.stride, index: 1)
        }
    }

    private func encodeDWTInPlace(
        commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        buffer: MTLBuffer,
        activeWidth: Int,
        activeHeight: Int,
        stride: Int,
        phase: Int
    ) throws {
        var constants = DWTConstants(
            activeWidth: UInt32(activeWidth),
            activeHeight: UInt32(activeHeight),
            stride: UInt32(stride),
            phase: UInt32(phase)
        )
        try encodeDWT(commandBuffer: commandBuffer, pipeline: pipeline, activeWidth: activeWidth, activeHeight: activeHeight) { encoder in
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&constants, length: MemoryLayout<DWTConstants>.stride, index: 1)
        }
    }

    private func encodeDWTInPlaceBatch(
        commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        dispatches: [(buffer: MTLBuffer, activeWidth: Int, activeHeight: Int, stride: Int, phase: Int)]
    ) throws {
        guard !dispatches.isEmpty else {
            return
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal DWT command encoder")
        }

        encoder.setComputePipelineState(pipeline)
        let width = min(16, pipeline.maxTotalThreadsPerThreadgroup)
        let height = max(1, min(16, pipeline.maxTotalThreadsPerThreadgroup / width))
        let threadsPerThreadgroup = MTLSize(width: width, height: height, depth: 1)
        for dispatch in dispatches {
            var constants = DWTConstants(
                activeWidth: UInt32(dispatch.activeWidth),
                activeHeight: UInt32(dispatch.activeHeight),
                stride: UInt32(dispatch.stride),
                phase: UInt32(dispatch.phase)
            )
            encoder.setBuffer(dispatch.buffer, offset: 0, index: 0)
            encoder.setBytes(&constants, length: MemoryLayout<DWTConstants>.stride, index: 1)
            encoder.dispatchThreads(
                MTLSize(width: dispatch.activeWidth, height: dispatch.activeHeight, depth: 1),
                threadsPerThreadgroup: threadsPerThreadgroup
            )
        }
        encoder.endEncoding()
    }

    private func dispatchDWTCopy(
        pipeline: MTLComputePipelineState,
        input: MTLBuffer,
        output: MTLBuffer,
        activeWidth: Int,
        activeHeight: Int,
        stride: Int,
        phase: Int
    ) throws {
        var constants = DWTConstants(
            activeWidth: UInt32(activeWidth),
            activeHeight: UInt32(activeHeight),
            stride: UInt32(stride),
            phase: UInt32(phase)
        )
        try dispatchDWT(pipeline: pipeline, activeWidth: activeWidth, activeHeight: activeHeight) { encoder in
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(&constants, length: MemoryLayout<DWTConstants>.stride, index: 2)
        }
    }

    private func encodeDWTCopy(
        commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        input: MTLBuffer,
        output: MTLBuffer,
        activeWidth: Int,
        activeHeight: Int,
        stride: Int,
        phase: Int
    ) throws {
        var constants = DWTConstants(
            activeWidth: UInt32(activeWidth),
            activeHeight: UInt32(activeHeight),
            stride: UInt32(stride),
            phase: UInt32(phase)
        )
        try encodeDWT(commandBuffer: commandBuffer, pipeline: pipeline, activeWidth: activeWidth, activeHeight: activeHeight) { encoder in
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(&constants, length: MemoryLayout<DWTConstants>.stride, index: 2)
        }
    }

    private func encodeDWTCopyBatch(
        commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        dispatches: [(input: MTLBuffer, output: MTLBuffer, activeWidth: Int, activeHeight: Int, stride: Int, phase: Int)]
    ) throws {
        guard !dispatches.isEmpty else {
            return
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal DWT command encoder")
        }

        encoder.setComputePipelineState(pipeline)
        let width = min(16, pipeline.maxTotalThreadsPerThreadgroup)
        let height = max(1, min(16, pipeline.maxTotalThreadsPerThreadgroup / width))
        let threadsPerThreadgroup = MTLSize(width: width, height: height, depth: 1)
        for dispatch in dispatches {
            var constants = DWTConstants(
                activeWidth: UInt32(dispatch.activeWidth),
                activeHeight: UInt32(dispatch.activeHeight),
                stride: UInt32(dispatch.stride),
                phase: UInt32(dispatch.phase)
            )
            encoder.setBuffer(dispatch.input, offset: 0, index: 0)
            encoder.setBuffer(dispatch.output, offset: 0, index: 1)
            encoder.setBytes(&constants, length: MemoryLayout<DWTConstants>.stride, index: 2)
            encoder.dispatchThreads(
                MTLSize(width: dispatch.activeWidth, height: dispatch.activeHeight, depth: 1),
                threadsPerThreadgroup: threadsPerThreadgroup
            )
        }
        encoder.endEncoding()
    }

    private func dispatchDWT(
        pipeline: MTLComputePipelineState,
        activeWidth: Int,
        activeHeight: Int,
        encode: (MTLComputeCommandEncoder) -> Void
    ) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal DWT command encoder")
        }

        encoder.setComputePipelineState(pipeline)
        encode(encoder)
        let width = min(16, pipeline.maxTotalThreadsPerThreadgroup)
        let height = max(1, min(16, pipeline.maxTotalThreadsPerThreadgroup / width))
        encoder.dispatchThreads(
            MTLSize(width: activeWidth, height: activeHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal DWT command failed: \(error)")
        }
    }

    private func encodeDWT(
        commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        activeWidth: Int,
        activeHeight: Int,
        encode: (MTLComputeCommandEncoder) -> Void
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal DWT command encoder")
        }

        encoder.setComputePipelineState(pipeline)
        encode(encoder)
        let width = min(16, pipeline.maxTotalThreadsPerThreadgroup)
        let height = max(1, min(16, pipeline.maxTotalThreadsPerThreadgroup / width))
        encoder.dispatchThreads(
            MTLSize(width: activeWidth, height: activeHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1)
        )
        encoder.endEncoding()
    }

    private func finish(commandBuffer: MTLCommandBuffer, context: String) throws {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("\(context) command failed: \(error)")
        }
    }

    private func validateWaveletInput(_ samples: [Float], width: Int, height: Int, levels: Int) throws {
        guard width > 0, height > 0, levels >= 0, samples.count == width * height else {
            throw PyrowaveError.invalidDimensions
        }
        try validateWaveletShape(width: width, height: height, levels: levels)
    }

    private func validateWaveletShape(width: Int, height: Int, levels: Int) throws {
        guard width > 0, height > 0, levels >= 0 else {
            throw PyrowaveError.invalidDimensions
        }
        var activeWidth = width
        var activeHeight = height
        for _ in 0..<levels {
            guard activeWidth >= 2, activeHeight >= 2, activeWidth % 2 == 0, activeHeight % 2 == 0 else {
                throw PyrowaveError.invalidDimensions
            }
            activeWidth /= 2
            activeHeight /= 2
        }
    }
}

private struct QuantizationConstants {
    var count: UInt32
    var quantizationStep: Float
}

struct MetalPlaneQuantizationDescriptor {
    var originX: UInt32
    var originY: UInt32
    var validWidth: UInt32
    var validHeight: UInt32
    var stride: UInt32
    var quantCode: UInt32
    var baseScale: Float
}

struct MetalPlaneQuantizationResult {
    var coefficients: [Int16]
    var qScaleCodesByDescriptor: [[UInt8]]
}

struct MetalPlaneQuantizationBufferResult {
    var coefficientBuffer: MTLBuffer
    var coefficientCount: Int
    var qScaleCodesByDescriptor: [[UInt8]]
}

struct MetalSparseCoefficientEntry {
    var destinationOffset: UInt32
    var coefficient: Int32
    var quantCode: UInt32
    var qScaleCode: UInt32
}

struct MetalSparsePacketDecodeDescriptor {
    var packetOffset: UInt32
    var payloadEnd: UInt32
    var originX: UInt32
    var originY: UInt32
    var validWidth: UInt32
    var validHeight: UInt32
    var stride: UInt32
}

struct MetalRateControlStatsDescriptor {
    var originX: UInt32
    var originY: UInt32
    var validWidth: UInt32
    var validHeight: UInt32
    var stride: UInt32
    var quantCode: UInt32
    var qScaleCode: UInt32
    var distortionScale: Float
}

struct MetalRateControlQuantStats {
    var squareError: Float
    var encodeCostBits: UInt32
}

struct MetalRateControlTileStatsFlat {
    var numPlanes: [UInt32]
    var stats: [MetalRateControlQuantStats]
}

struct MetalRateControlTileStats {
    var numPlanes: UInt32
    var stats: [MetalRateControlQuantStats]
}

struct MetalRateControlBucketBatchResult {
    var bucketIndicesByPlane: [[[Int]]]
    var cumulativeSavings: [Int]
}

struct MetalPacketByteCostDescriptor {
    var originX: UInt32
    var originY: UInt32
    var validWidth: UInt32
    var validHeight: UInt32
    var stride: UInt32
}

struct MetalSparsePacketEncodeDescriptor {
    var originX: UInt32
    var originY: UInt32
    var validWidth: UInt32
    var validHeight: UInt32
    var stride: UInt32
    var blockIndex: UInt32
    var quantLevel: UInt32
    var sequence: UInt32
    var quantCode: UInt32
}

private struct PadPlaneConstants {
    var sourceWidth: UInt32
    var sourceHeight: UInt32
    var paddedWidth: UInt32
    var paddedHeight: UInt32
    var channel: UInt32
}

private struct CropPlaneConstants {
    var paddedWidth: UInt32
    var outputWidth: UInt32
    var outputHeight: UInt32
}

private struct CropNV12Constants {
    var yPaddedWidth: UInt32
    var chromaPaddedWidth: UInt32
    var outputWidth: UInt32
    var outputHeight: UInt32
}

private struct PlaneQuantizationConstants {
    var descriptorCount: UInt32
}

private struct SparseApplyConstants {
    var entryCount: UInt32
    var sampleCount: UInt32
}

private struct SparsePacketDecodeConstants {
    var descriptorCount: UInt32
    var sampleCount: UInt32
}

private struct RateControlStatsConstants {
    var descriptorCount: UInt32
}

private struct PacketByteCostConstants {
    var descriptorCount: UInt32
}

private struct SparsePacketEncodeConstants {
    var descriptorCount: UInt32
    var maxPacketBytes: UInt32
}

private struct DWTBatchDispatch {
    var primary: MTLBuffer
    var scratch: MTLBuffer
    var activeWidth: Int
    var activeHeight: Int
    var stride: Int
}

private struct RateControlBucketConstants {
    var blockCount: UInt32
}

private struct RateControlBucketSavingsConstants {
    var blockCount: UInt32
}

private struct DWTConstants {
    var activeWidth: UInt32
    var activeHeight: UInt32
    var stride: UInt32
    var phase: UInt32
}
