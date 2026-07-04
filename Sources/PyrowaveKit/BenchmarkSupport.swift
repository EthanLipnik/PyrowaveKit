import Foundation
import CoreVideo
import Metal

public enum PyrowaveBenchmarkArtifactNames {
    public static let referenceY4M = "reference.y4m"
    public static let pyrowaveStream = "pyrowave-sample.pwks"
    public static let pyrowaveDecodedY4M = "pyrowave-decoded.y4m"
    public static let hevcMovie = "hevc-avkit.mov"
    public static let hevcDecodedY4M = "hevc-decoded.y4m"
    public static let report = "benchmark-report.json"
}

struct PyrowaveBenchmarkFrames: Equatable, Sendable {
    var frames: [YUVFrame]
    var frameRateNumerator: Int
    var frameRateDenominator: Int
    var bitDepth: Int

    init(
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
    public static let usage = "Usage: pyrowave-swift-bench [--input file.y4m] [--frames N] [--preset 6k|4k|1080p|720p] [--size WxH] [--output-dir DIR] [--bitrate BPS] [--quantization-step Q] [--max-pyrowave-bytes N|--unbounded-pyrowave] [--require-pyrowave-faster-than-hevc|--require-pyrowave-encode-speedup X|--require-pyrowave-decode-speedup X]"

    public var input: URL?
    public var frames: Int
    public var outputDirectory: URL
    public var width: Int
    public var height: Int
    public var bitrate: Int
    public var quantizationStep: Float
    public var maximumPyrowaveBytes: Int?
    public var matchHEVCFrameBudget: Bool
    public var requiredPyrowaveEncodeSpeedup: Double?
    public var requiredPyrowaveDecodeSpeedup: Double?
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
        requiredPyrowaveEncodeSpeedup = nil
        requiredPyrowaveDecodeSpeedup = nil
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
            case "--require-pyrowave-faster-than-hevc":
                requiredPyrowaveEncodeSpeedup = max(requiredPyrowaveEncodeSpeedup ?? 1.0, 1.0)
                requiredPyrowaveDecodeSpeedup = max(requiredPyrowaveDecodeSpeedup ?? 1.0, 1.0)
            case "--require-pyrowave-encode-speedup":
                requiredPyrowaveEncodeSpeedup = try Self.parsePositiveDouble(iterator.next())
            case "--require-pyrowave-decode-speedup":
                requiredPyrowaveDecodeSpeedup = try Self.parsePositiveDouble(iterator.next())
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

    public func validate(report: PyrowaveBenchmarkReport) throws {
        if let requiredPyrowaveEncodeSpeedup {
            try Self.validate(
                report.comparison.pyrowaveEncodeSpeedupOverHEVC,
                minimum: requiredPyrowaveEncodeSpeedup,
                label: "encode"
            )
        }
        if let requiredPyrowaveDecodeSpeedup {
            try Self.validate(
                report.comparison.pyrowaveDecodeSpeedupOverHEVC,
                minimum: requiredPyrowaveDecodeSpeedup,
                label: "decode"
            )
        }
    }

    private static func parsePositiveDouble(_ value: String?) throws -> Double {
        guard let value,
              let parsed = Double(value),
              parsed.isFinite,
              parsed > 0 else {
            throw PyrowaveError.invalidDimensions
        }
        return parsed
    }

    private static func validate(_ measured: Double?, minimum: Double, label: String) throws {
        guard let measured, measured.isFinite else {
            throw PyrowaveError.processFailed("Pyrowave \(label) speedup over HEVC is unavailable")
        }
        guard measured >= minimum else {
            throw PyrowaveError.processFailed(
                "Pyrowave \(label) speedup over HEVC \(measured) is below required \(minimum)"
            )
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

enum PyrowaveBenchmarkRunner {
    static let pyrowaveCodecName = "pyrowavekit-swift-metal"
    static let timedBenchmarkScopeNote = "Timed encode/decode excludes input loading, pixel-buffer preparation, reusable output allocation, artifact writes, report serialization, and quality metric generation; encode starts from reusable CoreVideo-backed Metal texture views and decode writes into reusable CVPixelBuffer-backed Metal texture views."
    static let pyrowaveImplementationNote = "Hard-cutover v2 stream with Metal plane, texture, and NV12 texture-channel pad, crop, DWT/iDWT, block quantization, sparse packet byte-cost prefiltering, sparse packet emission, sparse decode apply, rate-control stats, bucket resolve, and savings prefix, 32x32 sparse packets, and optional frame-size cap. \(timedBenchmarkScopeNote)"

    private struct DecodeTarget {
        var pixelBuffer: CVPixelBuffer
        var yTextureReference: CVMetalTexture
        var cbCrTextureReference: CVMetalTexture
        var yTexture: MTLTexture
        var cbCrTexture: MTLTexture
    }

    static func loadFrames(arguments: PyrowaveBenchmarkArguments) throws -> PyrowaveBenchmarkFrames {
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

    static func runPyrowave(
        loaded: PyrowaveBenchmarkFrames,
        configuration: CodecConfiguration,
        outputDirectory: URL
    ) throws -> CodecBenchmarkResult {
        let codec = try PyrowaveCodec()
        let frames = loaded.frames
        let inputPixelBuffers = try frames.map {
            try $0.makeCVPixelBuffer(pixelFormat: YUVFrame.cvPixelFormat(for: $0.videoSignal))
        }
        var encodedFrames = [EncodedFrame]()
        encodedFrames.reserveCapacity(frames.count)

        var stopwatch = Stopwatch()
        for pixelBuffer in inputPixelBuffers {
            encodedFrames.append(try codec.encode(pixelBuffer, configuration: configuration))
        }
        let encodeSeconds = stopwatch.lapSeconds()

        let decodeTargets = try makeDecodeTargets(
            codec: codec,
            width: frames[0].width,
            height: frames[0].height,
            pixelFormat: YUVFrame.cvPixelFormat(for: frames[0].videoSignal),
            count: frames.count
        )
        for index in encodedFrames.indices {
            try codec.decodeToNV12Textures(
                encodedFrames[index],
                yTexture: decodeTargets[index].yTexture,
                cbCrTexture: decodeTargets[index].cbCrTexture
            )
        }
        let decodeSeconds = stopwatch.lapSeconds()
        let decodedFrames = try decodeTargets.map {
            try YUVFrame(cvPixelBuffer: $0.pixelBuffer, videoSignal: frames[0].videoSignal)
        }

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

    private static func makeDecodeTargets(
        codec: PyrowaveCodec,
        width: Int,
        height: Int,
        pixelFormat: OSType,
        count: Int
    ) throws -> [DecodeTarget] {
        guard count > 0, width > 0, height > 0 else {
            throw PyrowaveError.invalidDimensions
        }
        return try (0..<count).map { _ in
            try makeDecodeTarget(codec: codec, width: width, height: height, pixelFormat: pixelFormat)
        }
    }

    private static func makeDecodeTarget(
        codec: PyrowaveCodec,
        width: Int,
        height: Int,
        pixelFormat: OSType
    ) throws -> DecodeTarget {
        var pixelBuffer: CVPixelBuffer?
        let attributes = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferMetalCompatibilityKey as String: true
        ] as CFDictionary
        let status = CVPixelBufferCreate(
            nil,
            width,
            height,
            pixelFormat,
            attributes,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer else {
            throw PyrowaveError.processFailed("failed to allocate benchmark decode CVPixelBuffer")
        }

        let y = try makeMetalTexture(
            codec: codec,
            pixelBuffer: pixelBuffer,
            pixelFormat: .r8Unorm,
            width: width,
            height: height,
            planeIndex: 0
        )
        let cbCr = try makeMetalTexture(
            codec: codec,
            pixelBuffer: pixelBuffer,
            pixelFormat: .rg8Unorm,
            width: width / 2,
            height: height / 2,
            planeIndex: 1
        )
        return DecodeTarget(
            pixelBuffer: pixelBuffer,
            yTextureReference: y.reference,
            cbCrTextureReference: cbCr.reference,
            yTexture: y.texture,
            cbCrTexture: cbCr.texture
        )
    }

    private static func makeMetalTexture(
        codec: PyrowaveCodec,
        pixelBuffer: CVPixelBuffer,
        pixelFormat: MTLPixelFormat,
        width: Int,
        height: Int,
        planeIndex: Int
    ) throws -> (texture: MTLTexture, reference: CVMetalTexture) {
        var textureReference: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            nil,
            codec.coreVideoTextureCache,
            pixelBuffer,
            nil,
            pixelFormat,
            width,
            height,
            planeIndex,
            &textureReference
        )
        guard status == kCVReturnSuccess,
              let textureReference,
              let texture = CVMetalTextureGetTexture(textureReference) else {
            throw PyrowaveError.processFailed("failed to create benchmark decode Metal texture for plane \(planeIndex)")
        }
        return (texture, textureReference)
    }
}

public enum PyrowaveBenchmarkCLI {
    public static func run(_ rawArguments: [String] = Array(CommandLine.arguments.dropFirst())) throws -> URL? {
        let arguments = try PyrowaveBenchmarkArguments(rawArguments)
        if arguments.shouldShowHelp {
            print(PyrowaveBenchmarkArguments.usage)
            return nil
        }

        try FileManager.default.createDirectory(at: arguments.outputDirectory, withIntermediateDirectories: true)
        let loaded = try PyrowaveBenchmarkRunner.loadFrames(arguments: arguments)
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
        let pyrowave = try PyrowaveBenchmarkRunner.runPyrowave(
            loaded: loaded,
            configuration: configuration,
            outputDirectory: arguments.outputDirectory
        )
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
        try arguments.validate(report: report)
        return reportURL
    }
}
