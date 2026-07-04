import Foundation
import PyrowaveKit

struct LoadedFrames {
    var frames: [YUVFrame]
    var frameRateNumerator: Int
    var frameRateDenominator: Int
    var bitDepth: Int
}

func loadFrames(arguments: PyrowaveBenchmarkArguments) throws -> LoadedFrames {
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
            url: outputDirectory.appendingPathComponent(PyrowaveBenchmarkArtifactNames.pyrowaveStream),
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
    try YUV4MPEGWriter.write(
        frames: decodedFrames,
        to: outputDirectory.appendingPathComponent(PyrowaveBenchmarkArtifactNames.pyrowaveDecodedY4M),
        frameRateNumerator: loaded.frameRateNumerator,
        frameRateDenominator: loaded.frameRateDenominator
    )

    let metric = try Metrics.compare(frames, decodedFrames)
    return CodecBenchmarkResult(
        codec: "pyrowavekit-swift-metal-hybrid",
        frameCount: frames.count,
        encodedBytes: encodedBytes,
        encodeSeconds: encodeSeconds,
        decodeSeconds: decodeSeconds,
        metrics: metric,
        note: "Hard-cutover v2 stream with Metal plane pad/crop, DWT/iDWT, block quantization, sparse packet byte-cost prefiltering, sparse decode apply, rate-control stats, 32x32 sparse packets, and optional frame-size cap"
    )
}

do {
    let arguments = try PyrowaveBenchmarkArguments()
    if arguments.shouldShowHelp {
        print(PyrowaveBenchmarkArguments.usage)
        exit(0)
    }
    try FileManager.default.createDirectory(at: arguments.outputDirectory, withIntermediateDirectories: true)
    let loaded = try loadFrames(arguments: arguments)
    let frames = loaded.frames
    try YUV4MPEGWriter.write(
        frames: frames,
        to: arguments.outputDirectory.appendingPathComponent(PyrowaveBenchmarkArtifactNames.referenceY4M),
        frameRateNumerator: loaded.frameRateNumerator,
        frameRateDenominator: loaded.frameRateDenominator
    )
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

    let report = PyrowaveBenchmarkReport(
        generatedAt: ISO8601DateFormatter().string(from: Date()),
        width: frames[0].width,
        height: frames[0].height,
        frames: frames.count,
        frameRateNumerator: loaded.frameRateNumerator,
        frameRateDenominator: loaded.frameRateDenominator,
        bitrate: arguments.bitrate,
        pyrowaveFrameBudgetBytes: pyrowaveBudget,
        artifacts: PyrowaveBenchmarkArtifacts(),
        pyrowave: pyrowave,
        hevc: hevc,
        comparison: CodecBenchmarkComparison(pyrowave: pyrowave, hevc: hevc)
    )

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let reportData = try encoder.encode(report)
    let reportURL = arguments.outputDirectory.appendingPathComponent(PyrowaveBenchmarkArtifactNames.report)
    try reportData.write(to: reportURL)
    print("Wrote \(reportURL.path)")
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
