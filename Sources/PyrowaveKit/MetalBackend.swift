import Foundation

#if canImport(Metal)
import Metal

public final class MetalPyrowaveBackend: @unchecked Sendable {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary
    private let padPlanePipeline: MTLComputePipelineState
    private let quantizePipeline: MTLComputePipelineState
    private let dequantizePipeline: MTLComputePipelineState
    private let quantizePlaneTilesPipeline: MTLComputePipelineState
    private let sparseApplyPipeline: MTLComputePipelineState
    private let rateControlStatsPipeline: MTLComputePipelineState
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
        quantizePipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_quantize", library: library))
        dequantizePipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_dequantize", library: library))
        quantizePlaneTilesPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_quantize_plane_tiles", library: library))
        sparseApplyPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_apply_sparse_coefficients", library: library))
        rateControlStatsPipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_rate_control_tile_stats", library: library))
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
            paddedHeight: UInt32(paddedHeight)
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

        let coefficientByteLength = samples.count * MemoryLayout<Int16>.stride
        let descriptorByteLength = descriptors.count * MemoryLayout<MetalPlaneQuantizationDescriptor>.stride
        let qScaleByteLength = descriptors.count * 16 * MemoryLayout<UInt8>.stride
        let zeroCoefficients = Array(repeating: Int16(0), count: samples.count)
        let qScaleCodes = Array(repeating: PyrowaveQuantization.identityQScaleCode, count: descriptors.count * 16)
        guard let input = device.makeBuffer(bytes: samples, length: samples.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let output = device.makeBuffer(bytes: zeroCoefficients, length: coefficientByteLength, options: .storageModeShared),
              let descriptorBuffer = device.makeBuffer(bytes: descriptors, length: descriptorByteLength, options: .storageModeShared),
              let qScaleBuffer = device.makeBuffer(bytes: qScaleCodes, length: qScaleByteLength, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal plane quantization buffers")
        }

        var constants = PlaneQuantizationConstants(descriptorCount: UInt32(descriptors.count))
        try dispatchPlaneQuantization(
            count: descriptors.count * 16,
            buffers: [
                (input, 0),
                (output, 1),
                (descriptorBuffer, 2),
                (qScaleBuffer, 3)
            ],
            constants: &constants
        )

        let coefficientsPointer = output.contents().bindMemory(to: Int16.self, capacity: samples.count)
        let coefficientValues = Array(UnsafeBufferPointer(start: coefficientsPointer, count: samples.count))
        let qScalePointer = qScaleBuffer.contents().bindMemory(to: UInt8.self, capacity: descriptors.count * 16)
        let flatQScales = Array(UnsafeBufferPointer(start: qScalePointer, count: descriptors.count * 16))
        let perDescriptor = Swift.stride(from: 0, to: flatQScales.count, by: 16).map {
            Array(flatQScales[$0..<$0 + 16])
        }
        return MetalPlaneQuantizationResult(coefficients: coefficientValues, qScaleCodesByDescriptor: perDescriptor)
    }

    func applySparseCoefficients(sampleCount: Int, entries: [MetalSparseCoefficientEntry]) throws -> [Float] {
        guard sampleCount >= 0 else {
            throw PyrowaveError.invalidDimensions
        }
        guard sampleCount <= Int(UInt32.max), entries.count <= Int(UInt32.max) else {
            throw PyrowaveError.invalidDimensions
        }
        guard !entries.isEmpty else {
            return Array(repeating: 0, count: sampleCount)
        }

        let samples = Array(repeating: Float(0), count: sampleCount)
        guard let output = device.makeBuffer(bytes: samples, length: sampleCount * MemoryLayout<Float>.stride, options: .storageModeShared),
              let entryBuffer = device.makeBuffer(bytes: entries, length: entries.count * MemoryLayout<MetalSparseCoefficientEntry>.stride, options: .storageModeShared) else {
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

        let pointer = output.contents().bindMemory(to: Float.self, capacity: sampleCount)
        return Array(UnsafeBufferPointer(start: pointer, count: sampleCount))
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

    public func forwardWavelet(_ samples: [Float], width: Int, height: Int, levels: Int) throws -> [Float] {
        try validateWaveletInput(samples, width: width, height: height, levels: levels)
        guard !samples.isEmpty else { return [] }

        let byteLength = samples.count * MemoryLayout<Float>.stride
        guard let primary = device.makeBuffer(bytes: samples, length: byteLength, options: .storageModeShared),
              let scratch = device.makeBuffer(length: byteLength, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal DWT buffers")
        }
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
        let pointer = primary.contents().bindMemory(to: Float.self, capacity: samples.count)
        return Array(UnsafeBufferPointer(start: pointer, count: samples.count))
    }

    public func inverseWavelet(_ coefficients: [Float], width: Int, height: Int, levels: Int) throws -> [Float] {
        try validateWaveletInput(coefficients, width: width, height: height, levels: levels)
        guard !coefficients.isEmpty else { return [] }

        let byteLength = coefficients.count * MemoryLayout<Float>.stride
        guard let primary = device.makeBuffer(bytes: coefficients, length: byteLength, options: .storageModeShared),
              let scratch = device.makeBuffer(length: byteLength, options: .storageModeShared) else {
            throw PyrowaveError.processFailed("failed to allocate Metal iDWT buffers")
        }
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
        let pointer = primary.contents().bindMemory(to: Float.self, capacity: coefficients.count)
        return Array(UnsafeBufferPointer(start: pointer, count: coefficients.count))
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

private struct PadPlaneConstants {
    var sourceWidth: UInt32
    var sourceHeight: UInt32
    var paddedWidth: UInt32
    var paddedHeight: UInt32
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

private struct DWTConstants {
    var activeWidth: UInt32
    var activeHeight: UInt32
    var stride: UInt32
    var phase: UInt32
}
#else
public final class MetalPyrowaveBackend: @unchecked Sendable {
    public init() throws {
        throw PyrowaveError.externalToolUnavailable("Metal")
    }

    func padPlane(_ plane: Plane8, paddedWidth: Int, paddedHeight: Int) throws -> [Float] {
        throw PyrowaveError.externalToolUnavailable("Metal")
    }

    public func quantize(_ samples: [Float], quantizationStep: Float) throws -> [Int16] {
        throw PyrowaveError.externalToolUnavailable("Metal")
    }

    public func dequantize(_ coefficients: [Int16], quantizationStep: Float) throws -> [Float] {
        throw PyrowaveError.externalToolUnavailable("Metal")
    }

    func quantizePlane(
        _ samples: [Float],
        stride: Int,
        descriptors: [MetalPlaneQuantizationDescriptor]
    ) throws -> MetalPlaneQuantizationResult {
        throw PyrowaveError.externalToolUnavailable("Metal")
    }

    func applySparseCoefficients(sampleCount: Int, entries: [MetalSparseCoefficientEntry]) throws -> [Float] {
        throw PyrowaveError.externalToolUnavailable("Metal")
    }

    func rateControlTileStats(
        coefficients: [Int16],
        descriptors: [MetalRateControlStatsDescriptor]
    ) throws -> [MetalRateControlTileStats] {
        throw PyrowaveError.externalToolUnavailable("Metal")
    }

    public func forwardWavelet(_ samples: [Float], width: Int, height: Int, levels: Int) throws -> [Float] {
        throw PyrowaveError.externalToolUnavailable("Metal")
    }

    public func inverseWavelet(_ coefficients: [Float], width: Int, height: Int, levels: Int) throws -> [Float] {
        throw PyrowaveError.externalToolUnavailable("Metal")
    }
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
#endif
