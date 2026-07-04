import Foundation
import AVFoundation
import AVKit

public struct CodecBenchmarkResult: Codable, Equatable, Sendable {
    public var codec: String
    public var frameCount: Int
    public var encodedBytes: Int
    public var encodedBytesPerFrame: Double
    public var encodeSeconds: Double
    public var encodeMillisecondsPerFrame: Double
    public var decodeSeconds: Double
    public var decodeMillisecondsPerFrame: Double
    public var metrics: FrameMetrics?
    public var note: String?

    public init(
        codec: String,
        frameCount: Int,
        encodedBytes: Int,
        encodeSeconds: Double,
        decodeSeconds: Double,
        metrics: FrameMetrics?,
        note: String?
    ) {
        let normalizedFrameCount = max(0, frameCount)
        self.codec = codec
        self.frameCount = normalizedFrameCount
        self.encodedBytes = encodedBytes
        self.encodedBytesPerFrame = normalizedFrameCount > 0 ? Double(encodedBytes) / Double(normalizedFrameCount) : 0
        self.encodeSeconds = encodeSeconds
        self.encodeMillisecondsPerFrame = normalizedFrameCount > 0 ? encodeSeconds * 1000.0 / Double(normalizedFrameCount) : 0
        self.decodeSeconds = decodeSeconds
        self.decodeMillisecondsPerFrame = normalizedFrameCount > 0 ? decodeSeconds * 1000.0 / Double(normalizedFrameCount) : 0
        self.metrics = metrics
        self.note = note
    }
}

public struct CodecBenchmarkComparison: Codable, Equatable, Sendable {
    public var pyrowaveToHEVCByteRatio: Double?
    public var pyrowaveEncodeSpeedupOverHEVC: Double?
    public var pyrowaveDecodeSpeedupOverHEVC: Double?
    public var weightedPSNRDelta: Double?
    public var note: String

    public init(pyrowave: CodecBenchmarkResult, hevc: CodecBenchmarkResult) {
        pyrowaveToHEVCByteRatio = Self.ratio(Double(pyrowave.encodedBytes), Double(hevc.encodedBytes))
        pyrowaveEncodeSpeedupOverHEVC = Self.ratio(hevc.encodeSeconds, pyrowave.encodeSeconds)
        pyrowaveDecodeSpeedupOverHEVC = Self.ratio(hevc.decodeSeconds, pyrowave.decodeSeconds)
        if let pyrowavePSNR = pyrowave.metrics?.weightedPSNR,
           let hevcPSNR = hevc.metrics?.weightedPSNR {
            weightedPSNRDelta = pyrowavePSNR - hevcPSNR
        } else {
            weightedPSNRDelta = nil
        }
        note = "Ratios above 1.0 mean Pyrowave used more bytes or was faster than HEVC; weightedPSNRDelta is Pyrowave minus HEVC in dB."
    }

    private static func ratio(_ numerator: Double, _ denominator: Double) -> Double? {
        guard numerator.isFinite, denominator.isFinite, denominator > 0 else {
            return nil
        }
        return numerator / denominator
    }
}

public enum HEVCComparison {
    private static let maximumQualityReferenceFrameRate = 60
    public static let avKitTimingNote = "Timed HEVC encode/decode stops at AVFoundation pixel-buffer append/read; planar conversion, PSNR, Y4M artifacts, and report writing are excluded."

    public static func runAVKitHEVCComparison(
        referenceFrames: [YUVFrame],
        workingDirectory: URL,
        bitrate: Int,
        frameRateNumerator: Int = 60,
        frameRateDenominator: Int = 1
    ) throws -> CodecBenchmarkResult {
        guard let firstFrame = referenceFrames.first else {
            throw PyrowaveError.truncatedInput
        }
        guard bitrate > 0 else {
            throw PyrowaveError.invalidDimensions
        }
        for frame in referenceFrames {
            guard frame.width == firstFrame.width,
                  frame.height == firstFrame.height,
                  frame.chroma == firstFrame.chroma else {
                throw PyrowaveError.invalidDimensions
            }
            guard frame.chroma == .yuv420 else {
                throw PyrowaveError.unsupportedFormat("HEVC comparison expects yuv420 frames")
            }
        }

        let frameDuration = try Self.frameDuration(numerator: frameRateNumerator, denominator: frameRateDenominator)
        try FileManager.default.createDirectory(at: workingDirectory, withIntermediateDirectories: true)
        let hevcURL = workingDirectory.appendingPathComponent(PyrowaveBenchmarkArtifactNames.hevcMovie)
        let decodedURL = workingDirectory.appendingPathComponent(PyrowaveBenchmarkArtifactNames.hevcDecodedY4M)
        if FileManager.default.fileExists(atPath: hevcURL.path) {
            try FileManager.default.removeItem(at: hevcURL)
        }
        if FileManager.default.fileExists(atPath: decodedURL.path) {
            try FileManager.default.removeItem(at: decodedURL)
        }

        let inputPixelBuffers = try makePixelBuffers(
            referenceFrames,
            pixelFormat: YUVFrame.cvPixelFormat(for: firstFrame.videoSignal)
        )
        let encodeSeconds = try writeHEVCMovie(
            inputPixelBuffers,
            frameSize: (width: firstFrame.width, height: firstFrame.height),
            to: hevcURL,
            bitrate: bitrate,
            frameDuration: frameDuration
        )

        let decoded = try readHEVCMoviePixelBuffers(
            hevcURL,
            expectedFrames: referenceFrames.count,
            videoSignal: firstFrame.videoSignal
        )
        let decodeSeconds = decoded.decodeSeconds
        let decodedFrames = try decoded.pixelBuffers.map {
            try YUVFrame(cvPixelBuffer: $0, videoSignal: firstFrame.videoSignal)
        }
        try YUV4MPEGWriter.write(
            frames: decodedFrames,
            to: decodedURL,
            frameRateNumerator: frameRateNumerator,
            frameRateDenominator: frameRateDenominator
        )
        let encodedBytes = (try FileManager.default.attributesOfItem(atPath: hevcURL.path)[.size] as? NSNumber)?.intValue ?? 0
        let comparedCount = min(referenceFrames.count, decodedFrames.count)
        let metrics = try Metrics.compare(
            Array(referenceFrames.prefix(comparedCount)),
            Array(decodedFrames.prefix(comparedCount))
        )

        return CodecBenchmarkResult(
            codec: "hevc_avkit",
            frameCount: referenceFrames.count,
            encodedBytes: encodedBytes,
            encodeSeconds: encodeSeconds,
            decodeSeconds: decodeSeconds,
            metrics: metrics,
            note: decodedFrames.count == referenceFrames.count ? avKitTimingNote : "\(avKitTimingNote) Decoded \(decodedFrames.count) of \(referenceFrames.count) frames."
        )
    }

    public static func matchedFrameByteBudget(
        bitrate: Int,
        frameRateNumerator: Int,
        frameRateDenominator: Int
    ) throws -> Int {
        guard bitrate > 0 else {
            throw PyrowaveError.invalidDimensions
        }
        let referenceFrameRate = try qualityReferenceFrameRate(
            numerator: frameRateNumerator,
            denominator: frameRateDenominator
        )
        let numerator = Int64(referenceFrameRate.numerator)
        let denominator = Int64(referenceFrameRate.denominator)
        let bitsPerFrameDenominator = Int64(8) * numerator
        let multiplied = Int64(bitrate).multipliedReportingOverflow(by: denominator)
        guard !multiplied.overflow, multiplied.partialValue > 0, bitsPerFrameDenominator > 0 else {
            throw PyrowaveError.invalidDimensions
        }
        let bitsPerFrameNumerator = multiplied.partialValue
        let roundedNumerator = bitsPerFrameNumerator.addingReportingOverflow(bitsPerFrameDenominator - 1)
        guard !roundedNumerator.overflow else {
            throw PyrowaveError.invalidDimensions
        }
        return max(1, Int(roundedNumerator.partialValue / bitsPerFrameDenominator))
    }

    public static func qualityReferenceFrameRate(numerator: Int, denominator: Int) throws -> (numerator: Int, denominator: Int) {
        guard numerator > 0, denominator > 0 else {
            throw PyrowaveError.invalidDimensions
        }
        if numerator > maximumQualityReferenceFrameRate * denominator {
            return (maximumQualityReferenceFrameRate, 1)
        }
        return (numerator, denominator)
    }

    static func frameDuration(numerator: Int, denominator: Int) throws -> CMTime {
        guard numerator > 0, denominator > 0 else {
            throw PyrowaveError.invalidDimensions
        }
        return CMTime(value: CMTimeValue(denominator), timescale: CMTimeScale(numerator))
    }

    private static func writeHEVCMovie(
        _ pixelBuffers: [CVPixelBuffer],
        frameSize: (width: Int, height: Int),
        to url: URL,
        bitrate: Int,
        frameDuration: CMTime
    ) throws -> Double {
        guard !pixelBuffers.isEmpty else {
            throw PyrowaveError.truncatedInput
        }
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let compressionProperties: [String: Any] = [
            AVVideoAverageBitRateKey: bitrate,
            AVVideoExpectedSourceFrameRateKey: max(1, Int(round(Double(frameDuration.timescale) / Double(frameDuration.value))))
        ]
        let outputSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: frameSize.width,
            AVVideoHeightKey: frameSize.height,
            AVVideoCompressionPropertiesKey: compressionProperties
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
        input.expectsMediaDataInRealTime = false
        guard writer.canAdd(input) else {
            throw PyrowaveError.processFailed("AVAssetWriter rejected HEVC input")
        }
        writer.add(input)

        let pixelAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: CVPixelBufferGetPixelFormatType(pixelBuffers[0]),
            kCVPixelBufferWidthKey as String: frameSize.width,
            kCVPixelBufferHeightKey as String: frameSize.height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: pixelAttributes
        )

        var stopwatch = Stopwatch()
        guard writer.startWriting() else {
            throw PyrowaveError.processFailed(writer.error?.localizedDescription ?? "failed to start AVAssetWriter")
        }
        writer.startSession(atSourceTime: .zero)

        for (index, pixelBuffer) in pixelBuffers.enumerated() {
            while !input.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.001)
            }
            let presentationTime = CMTimeMultiply(frameDuration, multiplier: Int32(index))
            guard adaptor.append(pixelBuffer, withPresentationTime: presentationTime) else {
                throw PyrowaveError.processFailed(writer.error?.localizedDescription ?? "failed to append HEVC frame")
            }
        }

        input.markAsFinished()
        let semaphore = DispatchSemaphore(value: 0)
        writer.finishWriting {
            semaphore.signal()
        }
        semaphore.wait()
        let encodeSeconds = stopwatch.lapSeconds()
        guard writer.status == .completed else {
            throw PyrowaveError.processFailed(writer.error?.localizedDescription ?? "failed to finish HEVC movie")
        }
        return encodeSeconds
    }

    private static func readHEVCMoviePixelBuffers(
        _ url: URL,
        expectedFrames: Int,
        videoSignal: VideoSignalMetadata
    ) throws -> (pixelBuffers: [CVPixelBuffer], decodeSeconds: Double) {
        let asset = AVURLAsset(url: url)
        let track = try firstVideoTrack(in: asset)
        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: YUVFrame.cvPixelFormat(for: videoSignal)
        ]
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw PyrowaveError.processFailed("AVAssetReader rejected HEVC output")
        }
        reader.add(output)
        var stopwatch = Stopwatch()
        guard reader.startReading() else {
            throw PyrowaveError.processFailed(reader.error?.localizedDescription ?? "failed to start AVAssetReader")
        }

        var pixelBuffers = [CVPixelBuffer]()
        pixelBuffers.reserveCapacity(expectedFrames)
        while let sampleBuffer = output.copyNextSampleBuffer() {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                throw PyrowaveError.truncatedInput
            }
            pixelBuffers.append(pixelBuffer)
        }
        let decodeSeconds = stopwatch.lapSeconds()
        guard reader.status == .completed else {
            throw PyrowaveError.processFailed(reader.error?.localizedDescription ?? "failed to read HEVC movie")
        }
        return (pixelBuffers, decodeSeconds)
    }

    private static func firstVideoTrack(in asset: AVURLAsset) throws -> AVAssetTrack {
        let result = AsyncResultBox<[AVAssetTrack]>()
        let semaphore = DispatchSemaphore(value: 0)
        Task {
            do {
                result.set(.success(try await asset.loadTracks(withMediaType: .video)))
            } catch {
                result.set(.failure(error))
            }
            semaphore.signal()
        }
        semaphore.wait()

        switch result.get() {
        case .success(let tracks):
            guard let track = tracks.first else {
                throw PyrowaveError.truncatedInput
            }
            return track
        case .failure(let error):
            throw PyrowaveError.processFailed(error.localizedDescription)
        case nil:
            throw PyrowaveError.processFailed("AVAsset track load did not complete")
        }
    }

    private static func makePixelBuffers(_ frames: [YUVFrame], pixelFormat: OSType) throws -> [CVPixelBuffer] {
        var pixelBuffers = [CVPixelBuffer]()
        pixelBuffers.reserveCapacity(frames.count)
        for frame in frames {
            pixelBuffers.append(try frame.makeCVPixelBuffer(pixelFormat: pixelFormat))
        }
        return pixelBuffers
    }

    private static func requireFirstFrame(_ frames: [YUVFrame]) throws -> YUVFrame {
        guard let first = frames.first else {
            throw PyrowaveError.truncatedInput
        }
        return first
    }

    private final class AsyncResultBox<Value>: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Result<Value, Error>?

        func set(_ value: Result<Value, Error>) {
            lock.lock()
            self.value = value
            lock.unlock()
        }

        func get() -> Result<Value, Error>? {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
    }
}
