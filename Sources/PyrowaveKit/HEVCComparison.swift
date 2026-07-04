import Foundation

#if canImport(AVFoundation)
import AVFoundation
#endif

#if canImport(AVKit)
import AVKit
#endif

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

public enum HEVCComparison {
    private static let maximumQualityReferenceFrameRate = 60

    public static func runAVKitHEVCComparison(
        referenceFrames: [YUVFrame],
        workingDirectory: URL,
        bitrate: Int,
        frameRateNumerator: Int = 60,
        frameRateDenominator: Int = 1
    ) throws -> CodecBenchmarkResult {
        #if canImport(AVFoundation)
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
        let hevcURL = workingDirectory.appendingPathComponent("hevc-avkit.mov")
        let decodedURL = workingDirectory.appendingPathComponent("hevc-decoded.y4m")
        if FileManager.default.fileExists(atPath: hevcURL.path) {
            try FileManager.default.removeItem(at: hevcURL)
        }
        if FileManager.default.fileExists(atPath: decodedURL.path) {
            try FileManager.default.removeItem(at: decodedURL)
        }

        var stopwatch = Stopwatch()
        try writeHEVCMovie(
            referenceFrames,
            to: hevcURL,
            bitrate: bitrate,
            frameDuration: frameDuration
        )
        let encodeSeconds = stopwatch.lapSeconds()

        let decodedFrames = try readHEVCMovie(
            hevcURL,
            expectedFrames: referenceFrames.count,
            videoSignal: firstFrame.videoSignal
        )
        let decodeSeconds = stopwatch.lapSeconds()
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
            note: decodedFrames.count == referenceFrames.count ? nil : "decoded \(decodedFrames.count) of \(referenceFrames.count) frames"
        )
        #else
        return CodecBenchmarkResult(
            codec: "hevc_avkit",
            frameCount: 0,
            encodedBytes: 0,
            encodeSeconds: 0,
            decodeSeconds: 0,
            metrics: nil,
            note: "AVFoundation is unavailable on this platform"
        )
        #endif
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

    #if canImport(AVFoundation)
    private static func writeHEVCMovie(
        _ frames: [YUVFrame],
        to url: URL,
        bitrate: Int,
        frameDuration: CMTime
    ) throws {
        let firstFrame = try requireFirstFrame(frames)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let compressionProperties: [String: Any] = [
            AVVideoAverageBitRateKey: bitrate,
            AVVideoExpectedSourceFrameRateKey: max(1, Int(round(Double(frameDuration.timescale) / Double(frameDuration.value))))
        ]
        let outputSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: firstFrame.width,
            AVVideoHeightKey: firstFrame.height,
            AVVideoCompressionPropertiesKey: compressionProperties
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
        input.expectsMediaDataInRealTime = false
        guard writer.canAdd(input) else {
            throw PyrowaveError.processFailed("AVAssetWriter rejected HEVC input")
        }
        writer.add(input)

        let pixelAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: pixelFormat(for: firstFrame.videoSignal),
            kCVPixelBufferWidthKey as String: firstFrame.width,
            kCVPixelBufferHeightKey as String: firstFrame.height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: pixelAttributes
        )

        guard writer.startWriting() else {
            throw PyrowaveError.processFailed(writer.error?.localizedDescription ?? "failed to start AVAssetWriter")
        }
        writer.startSession(atSourceTime: .zero)

        for (index, frame) in frames.enumerated() {
            while !input.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.001)
            }
            let pixelBuffer = try makePixelBuffer(frame, adaptor: adaptor)
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
        guard writer.status == .completed else {
            throw PyrowaveError.processFailed(writer.error?.localizedDescription ?? "failed to finish HEVC movie")
        }
    }

    private static func readHEVCMovie(
        _ url: URL,
        expectedFrames: Int,
        videoSignal: VideoSignalMetadata
    ) throws -> [YUVFrame] {
        let asset = AVURLAsset(url: url)
        let track = try firstVideoTrack(in: asset)
        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: pixelFormat(for: videoSignal)
        ]
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw PyrowaveError.processFailed("AVAssetReader rejected HEVC output")
        }
        reader.add(output)
        guard reader.startReading() else {
            throw PyrowaveError.processFailed(reader.error?.localizedDescription ?? "failed to start AVAssetReader")
        }

        var frames = [YUVFrame]()
        frames.reserveCapacity(expectedFrames)
        while let sampleBuffer = output.copyNextSampleBuffer() {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                throw PyrowaveError.truncatedInput
            }
            frames.append(try makeFrame(pixelBuffer, videoSignal: videoSignal))
        }
        guard reader.status == .completed else {
            throw PyrowaveError.processFailed(reader.error?.localizedDescription ?? "failed to read HEVC movie")
        }
        return frames
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

    private static func makePixelBuffer(
        _ frame: YUVFrame,
        adaptor: AVAssetWriterInputPixelBufferAdaptor
    ) throws -> CVPixelBuffer {
        guard let pool = adaptor.pixelBufferPool else {
            throw PyrowaveError.processFailed("missing AVAssetWriter pixel buffer pool")
        }
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixelBuffer)
        guard status == kCVReturnSuccess, let pixelBuffer else {
            throw PyrowaveError.processFailed("failed to allocate CVPixelBuffer")
        }
        try fill(pixelBuffer, with: frame)
        return pixelBuffer
    }

    private static func fill(_ pixelBuffer: CVPixelBuffer, with frame: YUVFrame) throws {
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard CVPixelBufferGetPlaneCount(pixelBuffer) >= 2,
              CVPixelBufferGetWidthOfPlane(pixelBuffer, 0) == frame.width,
              CVPixelBufferGetHeightOfPlane(pixelBuffer, 0) == frame.height,
              CVPixelBufferGetWidthOfPlane(pixelBuffer, 1) == frame.width / 2,
              CVPixelBufferGetHeightOfPlane(pixelBuffer, 1) == frame.height / 2,
              let yBase = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0),
              let uvBase = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1) else {
            throw PyrowaveError.invalidDimensions
        }

        let yStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let uvStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1)
        let yDestination = yBase.assumingMemoryBound(to: UInt8.self)
        frame.y.data.withUnsafeBufferPointer { source in
            for row in 0..<frame.height {
                let sourceStart = row * frame.width
                yDestination.advanced(by: row * yStride).update(from: source.baseAddress!.advanced(by: sourceStart), count: frame.width)
            }
        }

        let uvDestination = uvBase.assumingMemoryBound(to: UInt8.self)
        for row in 0..<(frame.height / 2) {
            let cbStart = row * (frame.width / 2)
            let destinationRow = uvDestination.advanced(by: row * uvStride)
            for column in 0..<(frame.width / 2) {
                destinationRow[column * 2] = frame.cb.data[cbStart + column]
                destinationRow[column * 2 + 1] = frame.cr.data[cbStart + column]
            }
        }
    }

    private static func makeFrame(_ pixelBuffer: CVPixelBuffer, videoSignal: VideoSignalMetadata) throws -> YUVFrame {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        guard width > 0, height > 0,
              CVPixelBufferGetPlaneCount(pixelBuffer) >= 2,
              CVPixelBufferGetWidthOfPlane(pixelBuffer, 1) == width / 2,
              CVPixelBufferGetHeightOfPlane(pixelBuffer, 1) == height / 2,
              let yBase = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0),
              let uvBase = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1) else {
            throw PyrowaveError.invalidDimensions
        }

        let yStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let uvStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1)
        let ySource = yBase.assumingMemoryBound(to: UInt8.self)
        let uvSource = uvBase.assumingMemoryBound(to: UInt8.self)
        var y = [UInt8]()
        var cb = [UInt8]()
        var cr = [UInt8]()
        y.reserveCapacity(width * height)
        cb.reserveCapacity((width / 2) * (height / 2))
        cr.reserveCapacity((width / 2) * (height / 2))

        for row in 0..<height {
            let source = ySource.advanced(by: row * yStride)
            y.append(contentsOf: UnsafeBufferPointer(start: source, count: width))
        }
        for row in 0..<(height / 2) {
            let source = uvSource.advanced(by: row * uvStride)
            for column in 0..<(width / 2) {
                cb.append(source[column * 2])
                cr.append(source[column * 2 + 1])
            }
        }

        return try YUVFrame(
            width: width,
            height: height,
            chroma: .yuv420,
            y: Plane8(width: width, height: height, data: y),
            cb: Plane8(width: width / 2, height: height / 2, data: cb),
            cr: Plane8(width: width / 2, height: height / 2, data: cr),
            videoSignal: videoSignal
        )
    }

    private static func pixelFormat(for videoSignal: VideoSignalMetadata) -> OSType {
        videoSignal.yCbCrRange == .limited ? kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange : kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
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
    #endif
}
