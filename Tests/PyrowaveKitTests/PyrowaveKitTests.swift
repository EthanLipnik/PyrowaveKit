import Foundation
import Testing
@testable import PyrowaveKit

@Test func roundTripSynthetic420() throws {
    let frame = try TestFrames.synthetic420(width: 160, height: 96)
    let codec = PyrowaveCodec()
    let encoded = try codec.encode(frame, configuration: CodecConfiguration(quantizationStep: 1.0 / 2048.0))
    let decoded = try codec.decode(encoded)
    let metrics = try Metrics.compare(frame, decoded)

    #expect(decoded.width == frame.width)
    #expect(decoded.height == frame.height)
    #expect(decoded.chroma == .yuv420)
    #expect(metrics.y.psnr > 46.0)
    #expect(metrics.weightedPSNR > 44.0)
}

@Test func yuv4mpegReadWrite() throws {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("sample.y4m")
    let frame = try TestFrames.synthetic420(width: 64, height: 64)

    var writer = try YUV4MPEGWriter(url: url, width: frame.width, height: frame.height, chroma: frame.chroma)
    try writer.writeFrame(frame)

    var reader = try YUV4MPEGReader(url: url)
    let decoded = try #require(try reader.readFrame())
    #expect(decoded == frame)
}

@Test func metalBackendCompilesKernelsWhenDeviceExists() throws {
    #if canImport(Metal)
    do {
        let backend = try MetalPyrowaveBackend()
        _ = try backend.makeFunction(named: "pyrowave_quantize")
        _ = try backend.makeFunction(named: "pyrowave_dequantize")
    } catch PyrowaveError.externalToolUnavailable {
        return
    }
    #endif
}

@Test func metalQuantizationMatchesCPUReferenceWhenDeviceExists() throws {
    #if canImport(Metal)
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let samples: [Float] = [-2.0, -1.25, -0.5, -0.0004, 0.0, 0.0004, 0.5, 1.25, 2.0]
    let step: Float = 1.0 / 1024.0
    let metal = try backend.quantize(samples, quantizationStep: step)
    let cpu = samples.map { sample -> Int16 in
        let quantized = Int((sample / step).rounded())
        return Int16(max(Int(Int16.min), min(Int(Int16.max), quantized)))
    }
    #expect(metal == cpu)

    let dequantized = try backend.dequantize(metal, quantizationStep: step)
    #expect(dequantized == cpu.map { Float($0) * step })
    #endif
}

@Test func metalCodecMatchesCPUReferenceWhenDeviceExists() throws {
    #if canImport(Metal)
    do {
        _ = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let frame = try TestFrames.synthetic420(width: 160, height: 96)
    let configuration = CodecConfiguration(quantizationStep: 1.0 / 2048.0)
    let cpu = try PyrowaveCodec(useMetalAcceleration: false).encode(frame, configuration: configuration)
    let metal = try PyrowaveCodec(useMetalAcceleration: true).encode(frame, configuration: configuration)
    #expect(metal.data == cpu.data)

    let decodedCPU = try PyrowaveCodec(useMetalAcceleration: false).decode(cpu)
    let decodedMetal = try PyrowaveCodec(useMetalAcceleration: true).decode(metal)
    #expect(decodedMetal == decodedCPU)
    #endif
}

@Test func metalWaveletMatchesCPUReferenceWhenDeviceExists() throws {
    #if canImport(Metal)
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let width = 128
    let height = 128
    let levels = 3
    var samples = Array(repeating: Float(0), count: width * height)
    for y in 0..<height {
        for x in 0..<width {
            samples[y * width + x] = Float((x * 13 + y * 7) % 251) / 251.0 - 0.5
        }
    }

    var cpuForward = samples
    Wavelet.forward2D(&cpuForward, width: width, height: height, levels: levels)
    let metalForward = try backend.forwardWavelet(samples, width: width, height: height, levels: levels)

    let forwardError = zip(cpuForward, metalForward).map { abs($0 - $1) }.max() ?? 0
    #expect(forwardError < 0.0001)

    var cpuInverse = cpuForward
    Wavelet.inverse2D(&cpuInverse, width: width, height: height, levels: levels)
    let metalInverse = try backend.inverseWavelet(metalForward, width: width, height: height, levels: levels)

    let inverseError = zip(cpuInverse, metalInverse).map { abs($0 - $1) }.max() ?? 0
    #expect(inverseError < 0.0001)
    #endif
}

@Test func sparseRateControlCapsEncodedFrameSize() throws {
    let frame = try TestFrames.synthetic420(width: 256, height: 144)
    let codec = PyrowaveCodec(useMetalAcceleration: false)
    let uncapped = try codec.encode(frame, configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0))
    let cap = max(4096, uncapped.data.count / 3)
    let capped = try codec.encode(
        frame,
        configuration: CodecConfiguration(
            quantizationStep: 1.0 / 1024.0,
            maximumEncodedBytes: cap
        )
    )

    #expect(capped.data.count <= cap)
    #expect(capped.data.count < uncapped.data.count)

    let decoded = try codec.decode(capped)
    let metrics = try Metrics.compare(frame, decoded)
    #expect(metrics.weightedPSNR > 20.0)
}
