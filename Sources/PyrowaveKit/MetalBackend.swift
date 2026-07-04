import Foundation
import Metal

public final class MetalPyrowaveBackend: @unchecked Sendable {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary
    private let padPlanePipeline: MTLComputePipelineState
    private let padTexturePlanePipeline: MTLComputePipelineState
    private let cropPlanePipeline: MTLComputePipelineState
    private let cropTexturePlanePipeline: MTLComputePipelineState
    private let cropNV12TexturesPipeline: MTLComputePipelineState
    private let quantizePipeline: MTLComputePipelineState
    private let dequantizePipeline: MTLComputePipelineState
    private let quantizePlaneTilesPipeline: MTLComputePipelineState
    private let sparseApplyPipeline: MTLComputePipelineState
    private let rateControlStatsPipeline: MTLComputePipelineState
    private let packetByteCostsPipeline: MTLComputePipelineState
    private let sparsePacketEncodePipeline: MTLComputePipelineState
    private let rateControlBucketPipeline: MTLComputePipelineState
    private let rateControlBucketSavingsPipeline: MTLComputePipelineState
    private let rateControlBucketSavingsPrefixPipeline: MTLComputePipelineState
    private let dwtLiftRowsPipeline: MTLComputePipelineState
    private let dwtLiftColumnsPipeline: MTLComputePipelineState
    private let dwtPackRowsPipeline: MTLComputePipelineState
    private let dwtPackColumnsPipeline: MTLComputePipelineState
    private let dwtUnpackRowsPipeline: MTLComputePipelineState
    private let dwtUnpackColumnsPipeline: MTLComputePipelineState
    private let idwtLiftRowsPipeline: MTLComputePipelineState
    private let idwtLiftColumnsPipeline: MTLComputePipelineState

    public init(device: MTLDevice? = MTLCreateSystemDefaultDevice()) throws {
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
        rateControlStatsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_rate_control_tile_stats", library: library))
        packetByteCostsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_packet_byte_costs", library: library))
        sparsePacketEncodePipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_encode_sparse_packets", library: library))
        rateControlBucketPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_rate_control_bucket_indices", library: library))
        rateControlBucketSavingsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_rate_control_bucket_savings", library: library))
        rateControlBucketSavingsPrefixPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_rate_control_bucket_savings_prefix", library: library))
        dwtLiftRowsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_dwt_lift_rows", library: library))
        dwtLiftColumnsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_dwt_lift_columns", library: library))
        dwtPackRowsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_dwt_pack_rows", library: library))
        dwtPackColumnsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_dwt_pack_columns", library: library))
        dwtUnpackRowsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_dwt_unpack_rows", library: library))
        dwtUnpackColumnsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_dwt_unpack_columns", library: library))
        idwtLiftRowsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_idwt_lift_rows", library: library))
        idwtLiftColumnsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_idwt_lift_columns", library: library))
    }

    public func makeFunction(named name: String) throws -> MTLFunction {
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
        let validChannel: Bool
        switch texture.pixelFormat {
        case .r8Unorm:
            validChannel = channel == 0
        case .rg8Unorm:
            validChannel = channel == 0 || channel == 1
        default:
            validChannel = false
        }
        guard validChannel else {
            throw PyrowaveError.unsupportedFormat("Metal texture padding expects r8Unorm channel 0 or rg8Unorm channel 0/1")
        }
        guard texture.width > 0,
              texture.height > 0,
              paddedWidth > 0,
              paddedHeight > 0,
              texture.width <= Int(UInt32.max),
              texture.height <= Int(UInt32.max),
              paddedWidth <= Int(UInt32.max),
              paddedHeight <= Int(UInt32.max) else {
            throw PyrowaveError.invalidDimensions
        }
        let sampleCount = paddedWidth * paddedHeight
        guard sampleCount > 0, sampleCount <= Int(UInt32.max) else {
            throw PyrowaveError.invalidDimensions
        }

        guard let output = device.makeBuffer(length: sampleCount * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal texture padding buffer")
        }

        var constants = PadPlaneConstants(
            sourceWidth: UInt32(texture.width),
            sourceHeight: UInt32(texture.height),
            paddedWidth: UInt32(paddedWidth),
            paddedHeight: UInt32(paddedHeight),
            channel: UInt32(channel)
        )
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal texture padding command encoder")
        }

        encoder.setComputePipelineState(padTexturePlanePipeline)
        encoder.setTexture(texture, index: 0)
        encoder.setBuffer(output, offset: 0, index: 0)
        encoder.setBytes(&constants, length: MemoryLayout<PadPlaneConstants>.stride, index: 1)
        let width = min(16, padTexturePlanePipeline.maxTotalThreadsPerThreadgroup)
        let height = max(1, min(16, padTexturePlanePipeline.maxTotalThreadsPerThreadgroup / width))
        encoder.dispatchThreads(
            MTLSize(width: paddedWidth, height: paddedHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal texture padding command failed: \(error)")
        }

        return output
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

    public func quantize(_ samples: [Float], quantizationStep: Float) throws -> [Int16] {
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

    public func dequantize(_ coefficients: [Int16], quantizationStep: Float) throws -> [Float] {
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
        guard stride > 0, sampleCount > 0, samples.length >= sampleCount * MemoryLayout<Float>.stride else {
            throw PyrowaveError.invalidDimensions
        }
        guard !descriptors.isEmpty else {
            return MetalPlaneQuantizationResult(coefficients: Array(repeating: 0, count: sampleCount), qScaleCodesByDescriptor: [])
        }

        let coefficientByteLength = sampleCount * MemoryLayout<Int16>.stride
        let descriptorByteLength = descriptors.count * MemoryLayout<MetalPlaneQuantizationDescriptor>.stride
        let qScaleByteLength = descriptors.count * 16 * MemoryLayout<UInt8>.stride
        let zeroCoefficients = Array(repeating: Int16(0), count: sampleCount)
        let qScaleCodes = Array(repeating: PyrowaveQuantization.identityQScaleCode, count: descriptors.count * 16)
        guard let output = device.makeBuffer(bytes: zeroCoefficients, length: coefficientByteLength, options: .storageModeShared),
              let descriptorBuffer = device.makeBuffer(bytes: descriptors, length: descriptorByteLength, options: .storageModeShared),
              let qScaleBuffer = device.makeBuffer(bytes: qScaleCodes, length: qScaleByteLength, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal plane quantization buffers")
        }

        var constants = PlaneQuantizationConstants(descriptorCount: UInt32(descriptors.count))
        try dispatchPlaneQuantization(
            count: descriptors.count * 16,
            buffers: [
                (samples, 0),
                (output, 1),
                (descriptorBuffer, 2),
                (qScaleBuffer, 3)
            ],
            constants: &constants
        )

        let coefficientsPointer = output.contents().bindMemory(to: Int16.self, capacity: sampleCount)
        let coefficientValues = Array(UnsafeBufferPointer(start: coefficientsPointer, count: sampleCount))
        let qScalePointer = qScaleBuffer.contents().bindMemory(to: UInt8.self, capacity: descriptors.count * 16)
        let flatQScales = Array(UnsafeBufferPointer(start: qScalePointer, count: descriptors.count * 16))
        let perDescriptor = Swift.stride(from: 0, to: flatQScales.count, by: 16).map {
            Array(flatQScales[$0..<$0 + 16])
        }
        return MetalPlaneQuantizationResult(coefficients: coefficientValues, qScaleCodesByDescriptor: perDescriptor)
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
        guard sampleCount >= 0 else {
            throw PyrowaveError.invalidDimensions
        }
        guard sampleCount <= Int(UInt32.max), entries.count <= Int(UInt32.max) else {
            throw PyrowaveError.invalidDimensions
        }
        if sampleCount == 0 {
            guard entries.isEmpty else {
                throw PyrowaveError.invalidDimensions
            }
            guard let output = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared) else {
                throw PyrowaveError.processFailed("failed to allocate Metal sparse coefficient output")
            }
            output.contents().storeBytes(of: Float(0), as: Float.self)
            return output
        }
        let byteLength = sampleCount * MemoryLayout<Float>.stride
        guard let output = device.makeBuffer(length: byteLength, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal sparse coefficient output")
        }
        memset(output.contents(), 0, byteLength)
        guard !entries.isEmpty else {
            return output
        }
        guard let entryBuffer = device.makeBuffer(bytes: entries, length: entries.count * MemoryLayout<MetalSparseCoefficientEntry>.stride, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal sparse coefficient buffers")
        }

        var constants = SparseApplyConstants(entryCount: UInt32(entries.count), sampleCount: UInt32(sampleCount))
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal sparse coefficient command encoder")
        }

        encoder.setComputePipelineState(sparseApplyPipeline)
        encoder.setBuffer(output, offset: 0, index: 0)
        encoder.setBuffer(entryBuffer, offset: 0, index: 1)
        encoder.setBytes(&constants, length: MemoryLayout<SparseApplyConstants>.stride, index: 2)
        let width = min(sparseApplyPipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: entries.count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal sparse coefficient command failed: \(error)")
        }

        return output
    }

    func rateControlTileStats(
        coefficients: [Int16],
        descriptors: [MetalRateControlStatsDescriptor]
    ) throws -> [MetalRateControlTileStats] {
        guard !coefficients.isEmpty else {
            throw PyrowaveError.invalidDimensions
        }
        guard !descriptors.isEmpty else {
            return []
        }
        guard descriptors.count <= Int(UInt32.max) else {
            throw PyrowaveError.invalidDimensions
        }

        let numPlanes = Array(repeating: UInt32(0), count: descriptors.count)
        let stats = Array(
            repeating: MetalRateControlQuantStats(squareError: 0, encodeCostBits: 0),
            count: descriptors.count * PyrowaveBlockStats.candidateCount
        )
        guard let coefficientBuffer = device.makeBuffer(bytes: coefficients, length: coefficients.count * MemoryLayout<Int16>.stride, options: .storageModeShared),
              let descriptorBuffer = device.makeBuffer(bytes: descriptors, length: descriptors.count * MemoryLayout<MetalRateControlStatsDescriptor>.stride, options: .storageModeShared),
              let numPlanesBuffer = device.makeBuffer(bytes: numPlanes, length: numPlanes.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let statsBuffer = device.makeBuffer(bytes: stats, length: stats.count * MemoryLayout<MetalRateControlQuantStats>.stride, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal rate-control buffers")
        }

        var constants = RateControlStatsConstants(descriptorCount: UInt32(descriptors.count))
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal rate-control command encoder")
        }

        encoder.setComputePipelineState(rateControlStatsPipeline)
        encoder.setBuffer(coefficientBuffer, offset: 0, index: 0)
        encoder.setBuffer(descriptorBuffer, offset: 0, index: 1)
        encoder.setBuffer(numPlanesBuffer, offset: 0, index: 2)
        encoder.setBuffer(statsBuffer, offset: 0, index: 3)
        encoder.setBytes(&constants, length: MemoryLayout<RateControlStatsConstants>.stride, index: 4)
        let width = min(rateControlStatsPipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: descriptors.count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal rate-control command failed: \(error)")
        }

        let numPlanesPointer = numPlanesBuffer.contents().bindMemory(to: UInt32.self, capacity: descriptors.count)
        let statsPointer = statsBuffer.contents().bindMemory(to: MetalRateControlQuantStats.self, capacity: stats.count)
        let metalNumPlanes = Array(UnsafeBufferPointer(start: numPlanesPointer, count: descriptors.count))
        let metalStats = Array(UnsafeBufferPointer(start: statsPointer, count: stats.count))

        return metalNumPlanes.indices.map { index in
            let start = index * PyrowaveBlockStats.candidateCount
            let end = start + PyrowaveBlockStats.candidateCount
            return MetalRateControlTileStats(numPlanes: metalNumPlanes[index], stats: Array(metalStats[start..<end]))
        }
    }

    func packetByteCosts(
        coefficients: [Int16],
        descriptors: [MetalPacketByteCostDescriptor]
    ) throws -> [[Int]] {
        guard !coefficients.isEmpty else {
            throw PyrowaveError.invalidDimensions
        }
        guard !descriptors.isEmpty else {
            return []
        }
        guard descriptors.count <= Int(UInt32.max) else {
            throw PyrowaveError.invalidDimensions
        }

        let byteCosts = Array(repeating: UInt32(0), count: descriptors.count * PyrowaveBlockStats.candidateCount)
        guard let coefficientBuffer = device.makeBuffer(bytes: coefficients, length: coefficients.count * MemoryLayout<Int16>.stride, options: .storageModeShared),
              let descriptorBuffer = device.makeBuffer(bytes: descriptors, length: descriptors.count * MemoryLayout<MetalPacketByteCostDescriptor>.stride, options: .storageModeShared),
              let byteCostBuffer = device.makeBuffer(bytes: byteCosts, length: byteCosts.count * MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal packet byte-cost buffers")
        }

        var constants = PacketByteCostConstants(descriptorCount: UInt32(descriptors.count))
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal packet byte-cost command encoder")
        }

        encoder.setComputePipelineState(packetByteCostsPipeline)
        encoder.setBuffer(coefficientBuffer, offset: 0, index: 0)
        encoder.setBuffer(descriptorBuffer, offset: 0, index: 1)
        encoder.setBuffer(byteCostBuffer, offset: 0, index: 2)
        encoder.setBytes(&constants, length: MemoryLayout<PacketByteCostConstants>.stride, index: 3)
        let width = min(packetByteCostsPipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: descriptors.count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal packet byte-cost command failed: \(error)")
        }

        let pointer = byteCostBuffer.contents().bindMemory(to: UInt32.self, capacity: byteCosts.count)
        let flatCosts = Array(UnsafeBufferPointer(start: pointer, count: byteCosts.count))
        return Swift.stride(from: 0, to: flatCosts.count, by: PyrowaveBlockStats.candidateCount).map {
            flatCosts[$0..<$0 + PyrowaveBlockStats.candidateCount].map(Int.init)
        }
    }

    func encodeSparsePackets(
        coefficients: [Int16],
        descriptors: [MetalSparsePacketEncodeDescriptor],
        qScaleCodes: [[UInt8]]
    ) throws -> [Data?] {
        guard !coefficients.isEmpty else {
            throw PyrowaveError.invalidDimensions
        }
        guard descriptors.count == qScaleCodes.count else {
            throw PyrowaveError.processFailed("sparse packet descriptor and q-scale counts differ")
        }
        guard !descriptors.isEmpty else {
            return []
        }
        guard descriptors.count <= Int(UInt32.max) else {
            throw PyrowaveError.invalidDimensions
        }

        var flatQScaleCodes = [UInt8]()
        flatQScaleCodes.reserveCapacity(qScaleCodes.count * 16)
        for codes in qScaleCodes {
            guard codes.count == 16 else {
                throw PyrowaveError.invalidBitstream("expected sixteen 8x8 quant scale codes")
            }
            flatQScaleCodes.append(contentsOf: codes)
        }

        let maxPacketBytes = PyrowaveCoefficientBlockCodec.maximumEncodedBlockBytes
        let outputBytes = Array(repeating: UInt8(0), count: descriptors.count * maxPacketBytes)
        let outputSizes = Array(repeating: UInt32(0), count: descriptors.count)
        guard let coefficientBuffer = device.makeBuffer(bytes: coefficients, length: coefficients.count * MemoryLayout<Int16>.stride, options: .storageModeShared),
              let descriptorBuffer = device.makeBuffer(bytes: descriptors, length: descriptors.count * MemoryLayout<MetalSparsePacketEncodeDescriptor>.stride, options: .storageModeShared),
              let qScaleBuffer = device.makeBuffer(bytes: flatQScaleCodes, length: flatQScaleCodes.count * MemoryLayout<UInt8>.stride, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(bytes: outputBytes, length: outputBytes.count * MemoryLayout<UInt8>.stride, options: .storageModeShared),
              let sizeBuffer = device.makeBuffer(bytes: outputSizes, length: outputSizes.count * MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal sparse packet encode buffers")
        }

        var constants = SparsePacketEncodeConstants(
            descriptorCount: UInt32(descriptors.count),
            maxPacketBytes: UInt32(maxPacketBytes)
        )
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal sparse packet encode command encoder")
        }

        encoder.setComputePipelineState(sparsePacketEncodePipeline)
        encoder.setBuffer(coefficientBuffer, offset: 0, index: 0)
        encoder.setBuffer(descriptorBuffer, offset: 0, index: 1)
        encoder.setBuffer(qScaleBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBuffer(sizeBuffer, offset: 0, index: 4)
        encoder.setBytes(&constants, length: MemoryLayout<SparsePacketEncodeConstants>.stride, index: 5)
        let width = min(sparsePacketEncodePipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: descriptors.count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal sparse packet encode command failed: \(error)")
        }

        let bytesPointer = outputBuffer.contents().bindMemory(to: UInt8.self, capacity: outputBytes.count)
        let sizePointer = sizeBuffer.contents().bindMemory(to: UInt32.self, capacity: outputSizes.count)
        let encodedBytes = Array(UnsafeBufferPointer(start: bytesPointer, count: outputBytes.count))
        let encodedSizes = Array(UnsafeBufferPointer(start: sizePointer, count: outputSizes.count))
        return encodedSizes.indices.map { index in
            let size = Int(encodedSizes[index])
            guard size > 0 else {
                return nil
            }
            guard size <= maxPacketBytes else {
                return nil
            }
            let start = index * maxPacketBytes
            return Data(encodedBytes[start..<start + size])
        }
    }

    func rateControlBucketIndices(
        distortions: [[Float]],
        packetByteCosts: [[Int]]
    ) throws -> [[Int]] {
        guard distortions.count == packetByteCosts.count else {
            throw PyrowaveError.processFailed("rate-control distortion and packet-cost counts differ")
        }
        guard !distortions.isEmpty else {
            return []
        }
        guard distortions.count <= Int(UInt32.max) else {
            throw PyrowaveError.invalidDimensions
        }

        var flatDistortions = [Float]()
        var flatPacketByteCosts = [UInt32]()
        flatDistortions.reserveCapacity(distortions.count * PyrowaveBlockStats.candidateCount)
        flatPacketByteCosts.reserveCapacity(packetByteCosts.count * PyrowaveBlockStats.candidateCount)
        for index in distortions.indices {
            guard distortions[index].count == PyrowaveBlockStats.candidateCount,
                  packetByteCosts[index].count == PyrowaveBlockStats.candidateCount else {
                throw PyrowaveError.processFailed("rate-control bucket input must contain \(PyrowaveBlockStats.candidateCount) candidates per block")
            }
            flatDistortions.append(contentsOf: distortions[index])
            for cost in packetByteCosts[index] {
                guard cost >= 0, cost <= Int(UInt32.max) else {
                    throw PyrowaveError.invalidDimensions
                }
                flatPacketByteCosts.append(UInt32(cost))
            }
        }

        let bucketIndices = Array(repeating: UInt32(0), count: flatDistortions.count)
        guard let distortionBuffer = device.makeBuffer(bytes: flatDistortions, length: flatDistortions.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let packetByteCostBuffer = device.makeBuffer(bytes: flatPacketByteCosts, length: flatPacketByteCosts.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let bucketIndexBuffer = device.makeBuffer(bytes: bucketIndices, length: bucketIndices.count * MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal rate-control bucket buffers")
        }

        var constants = RateControlBucketConstants(blockCount: UInt32(distortions.count))
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal rate-control bucket command encoder")
        }

        encoder.setComputePipelineState(rateControlBucketPipeline)
        encoder.setBuffer(distortionBuffer, offset: 0, index: 0)
        encoder.setBuffer(packetByteCostBuffer, offset: 0, index: 1)
        encoder.setBuffer(bucketIndexBuffer, offset: 0, index: 2)
        encoder.setBytes(&constants, length: MemoryLayout<RateControlBucketConstants>.stride, index: 3)
        let width = min(rateControlBucketPipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: distortions.count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal rate-control bucket command failed: \(error)")
        }

        let pointer = bucketIndexBuffer.contents().bindMemory(to: UInt32.self, capacity: bucketIndices.count)
        let flatBuckets = Array(UnsafeBufferPointer(start: pointer, count: bucketIndices.count))
        return Swift.stride(from: 0, to: flatBuckets.count, by: PyrowaveBlockStats.candidateCount).map {
            flatBuckets[$0..<$0 + PyrowaveBlockStats.candidateCount].map(Int.init)
        }
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

    public func forwardWavelet(_ samples: [Float], width: Int, height: Int, levels: Int) throws -> [Float] {
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
        guard sampleCount == width * height,
              buffer.length >= sampleCount * MemoryLayout<Float>.stride else {
            throw PyrowaveError.invalidDimensions
        }
        try validateWaveletShape(width: width, height: height, levels: levels)
        guard sampleCount > 0 else { return buffer }

        let byteLength = sampleCount * MemoryLayout<Float>.stride
        guard let scratch = device.makeBuffer(length: byteLength, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal DWT scratch buffer")
        }
        let primary = buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw PyrowaveError.processFailed("failed to create Metal DWT command buffer")
        }

        var activeWidth = width
        var activeHeight = height
        for _ in 0..<levels {
            for phase in 0...4 {
                try encodeDWTInPlace(
                    commandBuffer: commandBuffer,
                    pipeline: dwtLiftRowsPipeline,
                    buffer: primary,
                    activeWidth: activeWidth,
                    activeHeight: activeHeight,
                    stride: width,
                    phase: phase
                )
            }
            try encodeDWTCopy(
                commandBuffer: commandBuffer,
                pipeline: dwtPackRowsPipeline,
                input: primary,
                output: scratch,
                activeWidth: activeWidth,
                activeHeight: activeHeight,
                stride: width,
                phase: 0
            )

            for phase in 0...4 {
                try encodeDWTInPlace(
                    commandBuffer: commandBuffer,
                    pipeline: dwtLiftColumnsPipeline,
                    buffer: scratch,
                    activeWidth: activeWidth,
                    activeHeight: activeHeight,
                    stride: width,
                    phase: phase
                )
            }
            try encodeDWTCopy(
                commandBuffer: commandBuffer,
                pipeline: dwtPackColumnsPipeline,
                input: scratch,
                output: primary,
                activeWidth: activeWidth,
                activeHeight: activeHeight,
                stride: width,
                phase: 0
            )

            activeWidth /= 2
            activeHeight /= 2
        }

        try finish(commandBuffer: commandBuffer, context: "Metal DWT")
        return primary
    }

    public func inverseWavelet(_ coefficients: [Float], width: Int, height: Int, levels: Int) throws -> [Float] {
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

        let byteLength = sampleCount * MemoryLayout<Float>.stride
        guard let scratch = device.makeBuffer(length: byteLength, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal iDWT scratch buffer")
        }
        let primary = buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw PyrowaveError.processFailed("failed to create Metal iDWT command buffer")
        }

        var sizes = [(Int, Int)]()
        var activeWidth = width
        var activeHeight = height
        for _ in 0..<levels {
            sizes.append((activeWidth, activeHeight))
            activeWidth /= 2
            activeHeight /= 2
        }

        for (levelWidth, levelHeight) in sizes.reversed() {
            try encodeDWTCopy(
                commandBuffer: commandBuffer,
                pipeline: dwtUnpackColumnsPipeline,
                input: primary,
                output: scratch,
                activeWidth: levelWidth,
                activeHeight: levelHeight,
                stride: width,
                phase: 0
            )
            for phase in 0...4 {
                try encodeDWTInPlace(
                    commandBuffer: commandBuffer,
                    pipeline: idwtLiftColumnsPipeline,
                    buffer: scratch,
                    activeWidth: levelWidth,
                    activeHeight: levelHeight,
                    stride: width,
                    phase: phase
                )
            }

            try encodeDWTCopy(
                commandBuffer: commandBuffer,
                pipeline: dwtUnpackRowsPipeline,
                input: scratch,
                output: primary,
                activeWidth: levelWidth,
                activeHeight: levelHeight,
                stride: width,
                phase: 0
            )
            for phase in 0...4 {
                try encodeDWTInPlace(
                    commandBuffer: commandBuffer,
                    pipeline: idwtLiftRowsPipeline,
                    buffer: primary,
                    activeWidth: levelWidth,
                    activeHeight: levelHeight,
                    stride: width,
                    phase: phase
                )
            }
        }

        try finish(commandBuffer: commandBuffer, context: "Metal iDWT")
        return primary
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

    private func dispatchPlaneQuantization(
        count: Int,
        buffers: [(MTLBuffer, Int)],
        constants: inout PlaneQuantizationConstants
    ) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PyrowaveError.processFailed("failed to create Metal plane quantization command encoder")
        }

        encoder.setComputePipelineState(quantizePlaneTilesPipeline)
        for (buffer, index) in buffers {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        encoder.setBytes(&constants, length: MemoryLayout<PlaneQuantizationConstants>.stride, index: 4)

        let width = min(quantizePlaneTilesPipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw PyrowaveError.processFailed("Metal plane quantization command failed: \(error)")
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

struct MetalSparseCoefficientEntry {
    var destinationOffset: UInt32
    var coefficient: Int32
    var quantCode: UInt32
    var qScaleCode: UInt32
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

struct MetalRateControlTileStats {
    var numPlanes: UInt32
    var stats: [MetalRateControlQuantStats]
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
