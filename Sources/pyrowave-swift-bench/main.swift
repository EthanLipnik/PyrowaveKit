import Foundation
import PyrowaveKit

struct BenchmarkReport: Codable {
    var generatedAt: String
    var width: Int
    var height: Int
    var frames: Int
    var frameRateNumerator: Int
    var frameRateDenominator: Int
    var bitrate: Int
    var pyrowaveFrameBudgetBytes: Int?
    var pyrowave: CodecBenchmarkResult
    var hevc: CodecBenchmarkResult
}

struct Arguments {
    var input: URL?
    var frames = 1
    var outputDirectory = URL(fileURLWithPath: ".pyrowave-results", isDirectory: true)
    var width = 6144
    var height = 3456
    var bitrate = 80_000_000
    var quantizationStep: Float = 1.0 / 1024.0
    var maximumPyrowaveBytes: Int?
    var matchHEVCFrameBudget = true

    init() throws {
        var iterator = CommandLine.arguments.dropFirst().makeIterator()
        while let argument = iterator.next() {
            switch argument {
            case "--input":
                guard let value = iterator.next() else { throw PyrowaveError.invalidDimensions }
                input = URL(fileURLWithPath: value)
            case "--frames":
                guard let value = iterator.next(), let parsed = Int(value) else { throw PyrowaveError.invalidDimensions }
                frames = parsed
            case "--output-dir":
                guard let value = iterator.next() else { throw PyrowaveError.invalidDimensions }
                outputDirectory = URL(fileURLWithPath: value, isDirectory: true)
            case "--bitrate":
                guard let value = iterator.next(), let parsed = Int(value) else { throw PyrowaveError.invalidDimensions }
                bitrate = parsed
            case "--width":
                guard let value = iterator.next(), let parsed = Int(value) else { throw PyrowaveError.invalidDimensions }
                width = parsed
            case "--height":
                guard let value = iterator.next(), let parsed = Int(value) else { throw PyrowaveError.invalidDimensions }
                height = parsed
            case "--size":
                guard let value = iterator.next() else { throw PyrowaveError.invalidDimensions }
                let parts = value.lowercased().split(separator: "x")
                guard parts.count == 2, let parsedWidth = Int(parts[0]), let parsedHeight = Int(parts[1]) else {
                    throw PyrowaveError.invalidDimensions
                }
                width = parsedWidth
                height = parsedHeight
            case "--preset":
                guard let value = iterator.next() else { throw PyrowaveError.invalidDimensions }
                switch value.lowercased() {
                case "6k":
                    width = 6144
                    height = 3456
                case "4k":
                    width = 3840
                    height = 2160
                case "1080p":
                    width = 1920
                    height = 1080
                case "720p":
                    width = 1280
                    height = 720
                default:
                    throw PyrowaveError.unsupportedFormat("unknown preset \(value)")
                }
            case "--quantization-step":
                guard let value = iterator.next(), let parsed = Float(value) else { throw PyrowaveError.invalidDimensions }
                quantizationStep = parsed
            case "--max-pyrowave-bytes":
                guard let value = iterator.next(), let parsed = Int(value) else { throw PyrowaveError.invalidDimensions }
                maximumPyrowaveBytes = parsed
                matchHEVCFrameBudget = false
            case "--unbounded-pyrowave":
                maximumPyrowaveBytes = nil
                matchHEVCFrameBudget = false
            case "--help", "-h":
                print("Usage: pyrowave-swift-bench [--input file.y4m] [--frames N] [--preset 6k|4k|1080p|720p] [--size WxH] [--output-dir DIR] [--bitrate BPS] [--quantization-step Q] [--max-pyrowave-bytes N|--unbounded-pyrowave]")
                exit(0)
            default:
                throw PyrowaveError.unsupportedFormat("unknown argument \(argument)")
            }
        }
    }
}

struct LoadedFrames {
    var frames: [YUVFrame]
    var frameRateNumerator: Int
    var frameRateDenominator: Int
    var bitDepth: Int
}

func loadFrames(arguments: Arguments) throws -> LoadedFrames {
    if let input = arguments.input {
        var reader = try YUV4MPEGReader(url: input)
        var frames = [YUVFrame]()
        while frames.count < arguments.frames, let frame = try reader.readFrame() {
            frames.append(frame)
        }
        guard !frames.isEmpty else { throw PyrowaveError.truncatedInput }
        return LoadedFrames(
            frames: frames,
            frameRateNumerator: reader.frameRateNumerator,
            frameRateDenominator: reader.frameRateDenominator,
            bitDepth: reader.bitDepth
        )
    }

    let frames = try (0..<arguments.frames).map { try TestFrames.synthetic420(width: arguments.width, height: arguments.height, frameIndex: $0) }
    return LoadedFrames(frames: frames, frameRateNumerator: 60, frameRateDenominator: 1, bitDepth: 8)
}

func runPyrowave(loaded: LoadedFrames, configuration: CodecConfiguration, outputDirectory: URL) throws -> CodecBenchmarkResult {
    let codec = try PyrowaveCodec()
    let frames = loaded.frames
    var encodedFrames = [EncodedFrame]()
    encodedFrames.reserveCapacity(frames.count)

    var stopwatch = Stopwatch()
    for frame in frames {
        encodedFrames.append(try codec.encode(frame, configuration: configuration))
    }
    let encodeSeconds = stopwatch.lapSeconds()

    var decodedFrames = [YUVFrame]()
    decodedFrames.reserveCapacity(frames.count)
    for frame in encodedFrames {
        decodedFrames.append(try codec.decode(frame))
    }
    let decodeSeconds = stopwatch.lapSeconds()

    let encodedBytes = encodedFrames.reduce(0) { $0 + $1.data.count }
    if let firstFrame = frames.first {
        var stream = try PyrowaveStreamWriter(
            url: outputDirectory.appendingPathComponent("pyrowave-sample.pwks"),
            header: PyrowaveStreamHeader(
                frame: firstFrame,
                frameRateNumerator: loaded.frameRateNumerator,
                frameRateDenominator: loaded.frameRateDenominator,
                bitDepth: loaded.bitDepth
            )
        )
        for frame in encodedFrames {
            try stream.writeFrame(frame)
        }
    }

    let metric = try Metrics.compare(frames, decodedFrames)
    return CodecBenchmarkResult(
        codec: "pyrowavekit-swift-metal-hybrid",
        encodedBytes: encodedBytes,
        encodeSeconds: encodeSeconds,
        decodeSeconds: decodeSeconds,
        metrics: metric,
        note: "Hard-cutover v2 stream with Metal DWT, quantize/dequantize, sparse 32x32 blocks, and optional frame-size cap"
    )
}

do {
    let arguments = try Arguments()
    try FileManager.default.createDirectory(at: arguments.outputDirectory, withIntermediateDirectories: true)
    let loaded = try loadFrames(arguments: arguments)
    let frames = loaded.frames
    let pyrowaveBudget: Int?
    if let maximumPyrowaveBytes = arguments.maximumPyrowaveBytes {
        pyrowaveBudget = maximumPyrowaveBytes
    } else if arguments.matchHEVCFrameBudget {
        pyrowaveBudget = try HEVCComparison.matchedFrameByteBudget(
            bitrate: arguments.bitrate,
            frameRateNumerator: loaded.frameRateNumerator,
            frameRateDenominator: loaded.frameRateDenominator
        )
    } else {
        pyrowaveBudget = nil
    }
    let configuration = CodecConfiguration(
        quantizationStep: arguments.quantizationStep,
        maximumEncodedBytes: pyrowaveBudget
    )
    let pyrowave = try runPyrowave(loaded: loaded, configuration: configuration, outputDirectory: arguments.outputDirectory)
    let hevc = try HEVCComparison.runAVKitHEVCComparison(
        referenceFrames: frames,
        workingDirectory: arguments.outputDirectory,
        bitrate: arguments.bitrate,
        frameRateNumerator: loaded.frameRateNumerator,
        frameRateDenominator: loaded.frameRateDenominator
    )

    let report = BenchmarkReport(
        generatedAt: ISO8601DateFormatter().string(from: Date()),
        width: frames[0].width,
        height: frames[0].height,
        frames: frames.count,
        frameRateNumerator: loaded.frameRateNumerator,
        frameRateDenominator: loaded.frameRateDenominator,
        bitrate: arguments.bitrate,
        pyrowaveFrameBudgetBytes: pyrowaveBudget,
        pyrowave: pyrowave,
        hevc: hevc
    )

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let reportData = try encoder.encode(report)
    let reportURL = arguments.outputDirectory.appendingPathComponent("benchmark-report.json")
    try reportData.write(to: reportURL)
    print("Wrote \(reportURL.path)")
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
