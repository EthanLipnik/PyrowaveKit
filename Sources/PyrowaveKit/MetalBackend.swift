import Foundation

#if canImport(Metal)
import Metal

public final class MetalPyrowaveBackend: @unchecked Sendable {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary
    private let quantizePipeline: MTLComputePipelineState
    private let dequantizePipeline: MTLComputePipelineState
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

        quantizePipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_quantize", library: library))
        dequantizePipeline = try device.makeComputePipelineState(function: try Self.makeFunction(named: "pyrowave_dequantize", library: library))
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

    public func quantize(_ samples: [Float], quantizationStep: Float) throws -> [Int16] {
        throw PyrowaveError.externalToolUnavailable("Metal")
    }

    public func dequantize(_ coefficients: [Int16], quantizationStep: Float) throws -> [Float] {
        throw PyrowaveError.externalToolUnavailable("Metal")
    }

    public func forwardWavelet(_ samples: [Float], width: Int, height: Int, levels: Int) throws -> [Float] {
        throw PyrowaveError.externalToolUnavailable("Metal")
    }

    public func inverseWavelet(_ coefficients: [Float], width: Int, height: Int, levels: Int) throws -> [Float] {
        throw PyrowaveError.externalToolUnavailable("Metal")
    }
}
#endif
