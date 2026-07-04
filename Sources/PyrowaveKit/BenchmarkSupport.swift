import Foundation

public enum PyrowaveBenchmarkArtifactNames {
    public static let referenceY4M = "reference.y4m"
    public static let pyrowaveStream = "pyrowave-sample.pwks"
    public static let pyrowaveDecodedY4M = "pyrowave-decoded.y4m"
    public static let hevcMovie = "hevc-avkit.mov"
    public static let hevcDecodedY4M = "hevc-decoded.y4m"
    public static let report = "benchmark-report.json"
}

public struct PyrowaveBenchmarkFrames: Equatable, Sendable {
    public var frames: [YUVFrame]
    public var frameRateNumerator: Int
    public var frameRateDenominator: Int
    public var bitDepth: Int

    public init(
        frames: [YUVFrame],
        frameRateNumerator: Int,
        frameRateDenominator: Int,
        bitDepth: Int
    ) throws {
        guard !frames.isEmpty,
              frameRateNumerator > 0,
              frameRateDenominator > 0,
              [8, 10, 12, 14, 16].contains(bitDepth) else {
            throw PyrowaveError.invalidDimensions
        }
        self.frames = frames
        self.frameRateNumerator = frameRateNumerator
        self.frameRateDenominator = frameRateDenominator
        self.bitDepth = bitDepth
    }
}

public struct PyrowaveBenchmarkArtifacts: Codable, Equatable, Sendable {
    public var referenceY4M: String
    public var pyrowaveStream: String
    public var pyrowaveDecodedY4M: String
    public var hevcMovie: String
    public var hevcDecodedY4M: String

    public init(
        referenceY4M: String = PyrowaveBenchmarkArtifactNames.referenceY4M,
        pyrowaveStream: String = PyrowaveBenchmarkArtifactNames.pyrowaveStream,
        pyrowaveDecodedY4M: String = PyrowaveBenchmarkArtifactNames.pyrowaveDecodedY4M,
        hevcMovie: String = PyrowaveBenchmarkArtifactNames.hevcMovie,
        hevcDecodedY4M: String = PyrowaveBenchmarkArtifactNames.hevcDecodedY4M
    ) {
        self.referenceY4M = referenceY4M
        self.pyrowaveStream = pyrowaveStream
        self.pyrowaveDecodedY4M = pyrowaveDecodedY4M
        self.hevcMovie = hevcMovie
        self.hevcDecodedY4M = hevcDecodedY4M
    }
}

public struct PyrowaveBenchmarkReport: Codable, Equatable, Sendable {
    public var generatedAt: String
    public var width: Int
    public var height: Int
    public var frames: Int
    public var frameRateNumerator: Int
    public var frameRateDenominator: Int
    public var bitrate: Int
    public var pyrowaveFrameBudgetBytes: Int?
    public var artifacts: PyrowaveBenchmarkArtifacts
    public var pyrowave: CodecBenchmarkResult
    public var hevc: CodecBenchmarkResult
    public var comparison: CodecBenchmarkComparison

    public init(
        generatedAt: String,
        width: Int,
        height: Int,
        frames: Int,
        frameRateNumerator: Int,
        frameRateDenominator: Int,
        bitrate: Int,
        pyrowaveFrameBudgetBytes: Int?,
        artifacts: PyrowaveBenchmarkArtifacts = PyrowaveBenchmarkArtifacts(),
        pyrowave: CodecBenchmarkResult,
        hevc: CodecBenchmarkResult,
        comparison: CodecBenchmarkComparison
    ) {
        self.generatedAt = generatedAt
        self.width = width
        self.height = height
        self.frames = frames
        self.frameRateNumerator = frameRateNumerator
        self.frameRateDenominator = frameRateDenominator
        self.bitrate = bitrate
        self.pyrowaveFrameBudgetBytes = pyrowaveFrameBudgetBytes
        self.artifacts = artifacts
        self.pyrowave = pyrowave
        self.hevc = hevc
        self.comparison = comparison
    }
}

public struct PyrowaveBenchmarkArguments: Equatable, Sendable {
    public static let defaultOutputDirectory = URL(fileURLWithPath: ".pyrowave-results", isDirectory: true)
    public static let defaultWidth = 6144
    public static let defaultHeight = 3456
    public static let defaultFrames = 60
    public static let defaultBitrate = 80_000_000
    public static let defaultQuantizationStep: Float = 1.0 / 1024.0
    public static let usage = "Usage: pyrowave-swift-bench [--input file.y4m] [--frames N] [--preset 6k|4k|1080p|720p] [--size WxH] [--output-dir DIR] [--bitrate BPS] [--quantization-step Q] [--max-pyrowave-bytes N|--unbounded-pyrowave]"

    public var input: URL?
    public var frames: Int
    public var outputDirectory: URL
    public var width: Int
    public var height: Int
    public var bitrate: Int
    public var quantizationStep: Float
    public var maximumPyrowaveBytes: Int?
    public var matchHEVCFrameBudget: Bool
    public var shouldShowHelp: Bool

    public init(_ arguments: [String] = Array(CommandLine.arguments.dropFirst())) throws {
        input = nil
        frames = Self.defaultFrames
        outputDirectory = Self.defaultOutputDirectory
        width = Self.defaultWidth
        height = Self.defaultHeight
        bitrate = Self.defaultBitrate
        quantizationStep = Self.defaultQuantizationStep
        maximumPyrowaveBytes = nil
        matchHEVCFrameBudget = true
        shouldShowHelp = false

        var iterator = arguments.makeIterator()
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
                guard parts.count == 2,
                      let parsedWidth = Int(parts[0]),
                      let parsedHeight = Int(parts[1]) else {
                    throw PyrowaveError.invalidDimensions
                }
                width = parsedWidth
                height = parsedHeight
            case "--preset":
                guard let value = iterator.next() else { throw PyrowaveError.invalidDimensions }
                try applyPreset(value)
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
                shouldShowHelp = true
            default:
                throw PyrowaveError.unsupportedFormat("unknown argument \(argument)")
            }
        }
        guard shouldShowHelp || (frames > 0 && width > 0 && height > 0 && bitrate > 0 && quantizationStep > 0) else {
            throw PyrowaveError.invalidDimensions
        }
    }

    private mutating func applyPreset(_ value: String) throws {
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
    }
}

public enum PyrowaveBenchmarkRunner {
    public static let pyrowaveCodecName = "pyrowavekit-swift-metal"
    public static let timedBenchmarkScopeNote = "Timed encode/decode excludes input loading, pixel-buffer preparation, artifact writes, report serialization, and quality metric generation."
    public static let pyrowaveImplementationNote = "Hard-cutover v2 stream with Metal plane and texture pad, crop, DWT/iDWT, block quantization, sparse packet byte-cost prefiltering, sparse packet emission, sparse decode apply, rate-control stats, bucket resolve, and savings prefix, 32x32 sparse packets, and optional frame-size cap. \(timedBenchmarkScopeNote)"

    public static func loadFrames(arguments: PyrowaveBenchmarkArguments) throws -> PyrowaveBenchmarkFrames {
        if let input = arguments.input {
            var reader = try YUV4MPEGReader(url: input)
            var frames = [YUVFrame]()
            while frames.count < arguments.frames, let frame = try reader.readFrame() {
                frames.append(frame)
            }
            guard frames.count == arguments.frames else {
                throw PyrowaveError.truncatedInput
            }
            return try PyrowaveBenchmarkFrames(
                frames: frames,
                frameRateNumerator: reader.frameRateNumerator,
                frameRateDenominator: reader.frameRateDenominator,
                bitDepth: reader.bitDepth
            )
        }

        let frames = try (0..<arguments.frames).map {
            try TestFrames.synthetic420(width: arguments.width, height: arguments.height, frameIndex: $0)
        }
        return try PyrowaveBenchmarkFrames(frames: frames, frameRateNumerator: 60, frameRateDenominator: 1, bitDepth: 8)
    }

    public static func runPyrowave(
        loaded: PyrowaveBenchmarkFrames,
        configuration: CodecConfiguration,
        outputDirectory: URL,
        useMetalAcceleration: Bool = true
    ) throws -> CodecBenchmarkResult {
        let codec = try PyrowaveCodec(useMetalAcceleration: useMetalAcceleration)
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
        let streamURL = outputDirectory.appendingPathComponent(PyrowaveBenchmarkArtifactNames.pyrowaveStream)
        var stream = try PyrowaveStreamWriter(
            url: streamURL,
            header: PyrowaveStreamHeader(
                frame: frames[0],
                frameRateNumerator: loaded.frameRateNumerator,
                frameRateDenominator: loaded.frameRateDenominator,
                bitDepth: loaded.bitDepth
            )
        )
        for frame in encodedFrames {
            try stream.writeFrame(frame)
        }
        try YUV4MPEGWriter.write(
            frames: decodedFrames,
            to: outputDirectory.appendingPathComponent(PyrowaveBenchmarkArtifactNames.pyrowaveDecodedY4M),
            frameRateNumerator: loaded.frameRateNumerator,
            frameRateDenominator: loaded.frameRateDenominator
        )

        let metric = try Metrics.compare(frames, decodedFrames)
        return CodecBenchmarkResult(
            codec: pyrowaveCodecName,
            frameCount: frames.count,
            encodedBytes: encodedBytes,
            encodeSeconds: encodeSeconds,
            decodeSeconds: decodeSeconds,
            metrics: metric,
            note: pyrowaveImplementationNote
        )
    }
}
