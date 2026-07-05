import Foundation
import CoreMedia
import CoreVideo
import VideoToolbox

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
    public var pyrowaveBytesPerFrame: Double
    public var hevcBytesPerFrame: Double
    public var pyrowaveToHEVCBytesPerFrameRatio: Double?
    public var pyrowaveEncodeSpeedupOverHEVC: Double?
    public var pyrowaveDecodeSpeedupOverHEVC: Double?
    public var weightedPSNRDelta: Double?
    public var note: String

    public init(pyrowave: CodecBenchmarkResult, hevc: CodecBenchmarkResult) {
        pyrowaveToHEVCByteRatio = Self.ratio(Double(pyrowave.encodedBytes), Double(hevc.encodedBytes))
        pyrowaveBytesPerFrame = pyrowave.encodedBytesPerFrame
        hevcBytesPerFrame = hevc.encodedBytesPerFrame
        pyrowaveToHEVCBytesPerFrameRatio = Self.ratio(pyrowave.encodedBytesPerFrame, hevc.encodedBytesPerFrame)
        pyrowaveEncodeSpeedupOverHEVC = Self.ratio(hevc.encodeSeconds, pyrowave.encodeSeconds)
        pyrowaveDecodeSpeedupOverHEVC = Self.ratio(hevc.decodeSeconds, pyrowave.decodeSeconds)
        if let pyrowavePSNR = pyrowave.metrics?.weightedPSNR,
           let hevcPSNR = hevc.metrics?.weightedPSNR {
            weightedPSNRDelta = pyrowavePSNR - hevcPSNR
        } else {
            weightedPSNRDelta = nil
        }
        note = "Ratios above 1.0 mean Pyrowave used more bytes per frame or was faster than HEVC; weightedPSNRDelta is Pyrowave minus HEVC in dB."
    }

    private static func ratio(_ numerator: Double, _ denominator: Double) -> Double? {
        guard numerator.isFinite, denominator.isFinite, denominator > 0 else {
            return nil
        }
        return numerator / denominator
    }
}

enum HEVCComparison {
    static let defaultQuality = 0.8
    static let maximumQuality = 0.8
    static let mirageTimingNote = "Timed HEVC encode/decode uses a Mirage-style direct VideoToolbox realtime path: hardware HEVC requested, low-latency rate control requested with Mirage fallback tiers, RealTime true, frame reordering disabled, MaxFrameDelayCount 0, 30-second keyframe cadence, speed-over-quality enabled, AverageBitRate plus DataRateLimits, Quality capped at 0.8 with Mirage QP bounds. planar conversion, PSNR, Y4M artifacts, and report writing are excluded."

    static func runMirageHEVCComparison(
        referenceFrames: [YUVFrame],
        workingDirectory: URL,
        bitrate: Int,
        quality: Double = defaultQuality,
        frameRateNumerator: Int = 60,
        frameRateDenominator: Int = 1
    ) throws -> CodecBenchmarkResult {
        guard let firstFrame = referenceFrames.first else {
            throw PyrowaveError.truncatedInput
        }
        guard bitrate > 0 else {
            throw PyrowaveError.invalidDimensions
        }
        guard quality >= 0, quality <= Self.maximumQuality else {
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
        let hevcURL = workingDirectory.appendingPathComponent(PyrowaveBenchmarkArtifactNames.hevcStream)
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
        let encoded = try encodeMirageHEVC(
            inputPixelBuffers,
            frameSize: (width: firstFrame.width, height: firstFrame.height),
            to: hevcURL,
            bitrate: bitrate,
            quality: quality,
            frameDuration: frameDuration
        )

        let decoded = try decodeMirageHEVC(
            encoded.frames,
            expectedFrames: referenceFrames.count,
            videoSignal: firstFrame.videoSignal
        )
        let decodedFrames = try decoded.pixelBuffers.map {
            try YUVFrame(cvPixelBuffer: $0, videoSignal: firstFrame.videoSignal)
        }
        try YUV4MPEGWriter.write(
            frames: decodedFrames,
            to: decodedURL,
            frameRateNumerator: frameRateNumerator,
            frameRateDenominator: frameRateDenominator
        )
        let comparedCount = min(referenceFrames.count, decodedFrames.count)
        let metrics = try Metrics.compare(
            Array(referenceFrames.prefix(comparedCount)),
            Array(decodedFrames.prefix(comparedCount))
        )

        let note = decodedFrames.count == referenceFrames.count
            ? encoded.note
            : "\(encoded.note) Decoded \(decodedFrames.count) of \(referenceFrames.count) frames."
        return CodecBenchmarkResult(
            codec: "hevc_videotoolbox_mirage",
            frameCount: referenceFrames.count,
            encodedBytes: encoded.encodedBytes,
            encodeSeconds: encoded.encodeSeconds,
            decodeSeconds: decoded.decodeSeconds,
            metrics: metrics,
            note: note
        )
    }

    static func frameDuration(numerator: Int, denominator: Int) throws -> CMTime {
        guard numerator > 0, denominator > 0 else {
            throw PyrowaveError.invalidDimensions
        }
        return CMTime(value: CMTimeValue(denominator), timescale: CMTimeScale(numerator))
    }

    private static func encodeMirageHEVC(
        _ pixelBuffers: [CVPixelBuffer],
        frameSize: (width: Int, height: Int),
        to url: URL,
        bitrate: Int,
        quality: Double,
        frameDuration: CMTime
    ) throws -> (frames: [MirageHEVCEncodedFrame], encodedBytes: Int, encodeSeconds: Double, note: String) {
        guard !pixelBuffers.isEmpty else {
            throw PyrowaveError.truncatedInput
        }
        let targetFrameRate = max(1, Int(round(Double(frameDuration.timescale) / Double(frameDuration.value))))
        let sessionResult = try makeMirageCompressionSession(
            frameSize: frameSize,
            pixelFormat: CVPixelBufferGetPixelFormatType(pixelBuffers[0])
        )
        let session = sessionResult.session
        defer { VTCompressionSessionInvalidate(session) }
        configureMirageCompressionSession(
            session,
            bitrate: bitrate,
            quality: Float(quality),
            targetFrameRate: targetFrameRate
        )

        let collector = HEVCEncodeCollector()
        let group = DispatchGroup()
        var stopwatch = Stopwatch()
        for (index, pixelBuffer) in pixelBuffers.enumerated() {
            group.enter()
            var properties: [CFString: Any] = [:]
            if index == 0 {
                properties[kVTEncodeFrameOptionKey_ForceKeyFrame] = true
            }
            let presentationTime = CMTimeMultiply(frameDuration, multiplier: Int32(index))
            let status = VTCompressionSessionEncodeFrame(
                session,
                imageBuffer: pixelBuffer,
                presentationTimeStamp: presentationTime,
                duration: frameDuration,
                frameProperties: properties.isEmpty ? nil : properties as CFDictionary,
                infoFlagsOut: nil
            ) { status, infoFlags, sampleBuffer in
                defer { group.leave() }
                guard status == noErr,
                      !infoFlags.contains(.frameDropped),
                      let sampleBuffer else {
                    collector.recordError("HEVC encode callback failed: status \(status), flags \(infoFlags.rawValue)")
                    return
                }
                do {
                    collector.append(try makeMirageEncodedFrame(sampleBuffer))
                } catch {
                    collector.recordError(error.localizedDescription)
                }
            }
            guard status == noErr else {
                group.leave()
                throw PyrowaveError.processFailed("VTCompressionSessionEncodeFrame failed: \(status)")
            }
        }

        let completeStatus = VTCompressionSessionCompleteFrames(session, untilPresentationTimeStamp: .invalid)
        guard completeStatus == noErr else {
            throw PyrowaveError.processFailed("VTCompressionSessionCompleteFrames failed: \(completeStatus)")
        }
        let timeoutSeconds = max(30, pixelBuffers.count)
        guard group.wait(timeout: .now() + .seconds(timeoutSeconds)) == .success else {
            throw PyrowaveError.processFailed("Timed out waiting for Mirage-style HEVC encode callbacks")
        }
        let encodeSeconds = stopwatch.lapSeconds()
        try collector.throwIfNeeded()
        let frames = collector.frames.sorted {
            CMTimeCompare($0.presentationTime, $1.presentationTime) < 0
        }
        guard frames.count == pixelBuffers.count else {
            throw PyrowaveError.processFailed("HEVC encoded \(frames.count) of \(pixelBuffers.count) frames")
        }

        try writeMirageWireStream(frames, to: url)
        let encodedBytes = frames.reduce(0) { $0 + $1.wireData.count }
        let note = sessionResult.tier == .lowLatencyHardwareRequired
            ? mirageTimingNote
            : "\(mirageTimingNote) Encoder session used Mirage fallback tier \(sessionResult.tier.label)."
        return (frames, encodedBytes, encodeSeconds, note)
    }

    private static func decodeMirageHEVC(
        _ frames: [MirageHEVCEncodedFrame],
        expectedFrames: Int,
        videoSignal: VideoSignalMetadata
    ) throws -> (pixelBuffers: [CVPixelBuffer], decodeSeconds: Double) {
        guard let first = frames.first,
              let formatDescription = CMSampleBufferGetFormatDescription(first.sampleBuffer) else {
            throw PyrowaveError.truncatedInput
        }
        let collector = HEVCDecodeCollector()
        var callbackRecord = VTDecompressionOutputCallbackRecord(
            decompressionOutputCallback: { refcon, _, status, _, imageBuffer, presentationTimeStamp, _ in
                guard let refcon else { return }
                let collector = Unmanaged<HEVCDecodeCollector>.fromOpaque(refcon).takeUnretainedValue()
                collector.record(status: status, imageBuffer: imageBuffer, presentationTimeStamp: presentationTimeStamp)
            },
            decompressionOutputRefCon: Unmanaged.passUnretained(collector).toOpaque()
        )
        let destinationAttributes: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: YUVFrame.cvPixelFormat(for: videoSignal),
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
        ]
        var session: VTDecompressionSession?
        let status = VTDecompressionSessionCreate(
            allocator: kCFAllocatorDefault,
            formatDescription: formatDescription,
            decoderSpecification: [
                kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder: true,
            ] as CFDictionary,
            imageBufferAttributes: destinationAttributes as CFDictionary,
            outputCallback: &callbackRecord,
            decompressionSessionOut: &session
        )
        guard status == noErr, let session else {
            throw PyrowaveError.processFailed("VTDecompressionSessionCreate failed: \(status)")
        }
        defer { VTDecompressionSessionInvalidate(session) }
        _ = VTSessionSetProperty(session, key: kVTDecompressionPropertyKey_RealTime, value: kCFBooleanTrue)
        _ = VTSessionSetProperty(session, key: kVTDecompressionPropertyKey_MaximizePowerEfficiency, value: kCFBooleanFalse)

        var stopwatch = Stopwatch()
        for frame in frames {
            collector.enter()
            var infoFlags = VTDecodeInfoFlags()
            let decodeStatus = VTDecompressionSessionDecodeFrame(
                session,
                sampleBuffer: frame.sampleBuffer,
                flags: [._EnableAsynchronousDecompression],
                frameRefcon: nil,
                infoFlagsOut: &infoFlags
            )
            guard decodeStatus == noErr else {
                collector.leaveWithoutCallback()
                throw PyrowaveError.processFailed("VTDecompressionSessionDecodeFrame failed: \(decodeStatus)")
            }
        }
        let waitStatus = VTDecompressionSessionWaitForAsynchronousFrames(session)
        guard waitStatus == noErr else {
            throw PyrowaveError.processFailed("VTDecompressionSessionWaitForAsynchronousFrames failed: \(waitStatus)")
        }
        let timeoutSeconds = max(30, expectedFrames)
        guard collector.wait(timeout: .now() + .seconds(timeoutSeconds)) else {
            throw PyrowaveError.processFailed("Timed out waiting for Mirage-style HEVC decode callbacks")
        }
        let decodeSeconds = stopwatch.lapSeconds()
        try collector.throwIfNeeded()
        let pixelBuffers = collector.outputs
            .sorted { CMTimeCompare($0.presentationTimeStamp, $1.presentationTimeStamp) < 0 }
            .map(\.pixelBuffer)
        guard pixelBuffers.count == expectedFrames else {
            throw PyrowaveError.processFailed("HEVC decoded \(pixelBuffers.count) of \(expectedFrames) frames")
        }
        return (pixelBuffers, decodeSeconds)
    }

    private static func makeMirageCompressionSession(
        frameSize: (width: Int, height: Int),
        pixelFormat: OSType
    ) throws -> (session: VTCompressionSession, tier: MirageEncoderSpecTier) {
        let imageBufferAttributes: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: pixelFormat,
            kCVPixelBufferWidthKey: frameSize.width,
            kCVPixelBufferHeightKey: frameSize.height,
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
        ]
        var lastStatus: OSStatus = noErr
        for tier in MirageEncoderSpecTier.allCases {
            var session: VTCompressionSession?
            let status = VTCompressionSessionCreate(
                allocator: kCFAllocatorDefault,
                width: Int32(frameSize.width),
                height: Int32(frameSize.height),
                codecType: kCMVideoCodecType_HEVC,
                encoderSpecification: tier.encoderSpecification as CFDictionary,
                imageBufferAttributes: imageBufferAttributes as CFDictionary,
                compressedDataAllocator: nil,
                outputCallback: nil,
                refcon: nil,
                compressionSessionOut: &session
            )
            if status == noErr, let session {
                return (session, tier)
            }
            lastStatus = status
        }
        throw PyrowaveError.processFailed("VTCompressionSessionCreate failed: \(lastStatus)")
    }

    private static func configureMirageCompressionSession(
        _ session: VTCompressionSession,
        bitrate: Int,
        quality: Float,
        targetFrameRate: Int
    ) {
        let clampedQuality = max(0.02, min(Float(maximumQuality), quality))
        let qp = qualitySettings(for: clampedQuality)
        _ = setCompressionProperty(session, key: kVTCompressionPropertyKey_RealTime, value: kCFBooleanTrue)
        _ = setCompressionProperty(session, key: kVTCompressionPropertyKey_AllowFrameReordering, value: kCFBooleanFalse)
        _ = setCompressionProperty(session, key: kVTCompressionPropertyKey_MaxFrameDelayCount, value: NSNumber(value: 0))
        _ = setCompressionProperty(session, key: kVTCompressionPropertyKey_ExpectedFrameRate, value: NSNumber(value: targetFrameRate))
        _ = setCompressionProperty(session, key: kVTCompressionPropertyKey_MaximizePowerEfficiency, value: kCFBooleanFalse)
        _ = setCompressionProperty(session, key: kVTCompressionPropertyKey_ReferenceBufferCount, value: NSNumber(value: 1))
        _ = setCompressionProperty(session, key: kVTCompressionPropertyKey_AllowOpenGOP, value: kCFBooleanFalse)
        _ = setCompressionProperty(session, key: kVTCompressionPropertyKey_AllowTemporalCompression, value: kCFBooleanTrue)
        let keyFrameInterval = max(1, targetFrameRate * 30)
        _ = setCompressionProperty(session, key: kVTCompressionPropertyKey_MaxKeyFrameInterval, value: NSNumber(value: keyFrameInterval))
        _ = setCompressionProperty(session, key: kVTCompressionPropertyKey_MaxKeyFrameIntervalDuration, value: NSNumber(value: 30.0))
        _ = setCompressionProperty(session, key: kVTCompressionPropertyKey_ProfileLevel, value: kVTProfileLevel_HEVC_Main_AutoLevel)
        _ = setCompressionProperty(session, key: kVTCompressionPropertyKey_PrioritizeEncodingSpeedOverQuality, value: kCFBooleanTrue)
        _ = setCompressionProperty(session, key: kVTCompressionPropertyKey_Quality, value: NSNumber(value: qp.quality))
        if let minQP = qp.minQP {
            _ = setCompressionProperty(session, key: kVTCompressionPropertyKey_MinAllowedFrameQP, value: NSNumber(value: minQP))
        }
        if let maxQP = qp.maxQP {
            _ = setCompressionProperty(session, key: kVTCompressionPropertyKey_MaxAllowedFrameQP, value: NSNumber(value: maxQP))
        }
        _ = clearCompressionProperty(session, key: kVTCompressionPropertyKey_ConstantBitRate)
        _ = setCompressionProperty(session, key: kVTCompressionPropertyKey_AverageBitRate, value: NSNumber(value: bitrate))
        let rateLimit = dataRateLimit(targetBitrateBps: bitrate, targetFrameRate: targetFrameRate)
        let rateLimits: [NSNumber] = [
            NSNumber(value: rateLimit.bytes),
            NSNumber(value: rateLimit.windowSeconds),
        ]
        _ = setCompressionProperty(session, key: kVTCompressionPropertyKey_DataRateLimits, value: rateLimits as CFArray)
        VTCompressionSessionPrepareToEncodeFrames(session)
    }

    private static func qualitySettings(for quality: Float) -> (quality: Float, minQP: Int?, maxQP: Int?) {
        let useQP = quality < 0.98
        guard useQP else {
            return (quality, nil, nil)
        }
        let rawMin = 8.0 + (1.0 - Double(quality)) * 43.0
        let minQP = max(8, min(50, Int(rawMin.rounded())))
        let maxQP = min(51, minQP + 10)
        return (quality, minQP, maxQP)
    }

    private static func dataRateLimit(
        targetBitrateBps: Int,
        targetFrameRate: Int
    ) -> (bytes: Int, windowSeconds: Double) {
        let clampedFrameRate = max(1, targetFrameRate)
        let windowSeconds = min(0.05, 2.0 / Double(clampedFrameRate))
        let bytesPerSecond = max(1.0, Double(targetBitrateBps) / 8.0)
        let bytes = max(1, Int((bytesPerSecond * windowSeconds).rounded()))
        return (bytes, windowSeconds)
    }

    @discardableResult
    private static func setCompressionProperty(
        _ session: VTCompressionSession,
        key: CFString,
        value: CFTypeRef
    ) -> Bool {
        VTSessionSetProperty(session, key: key, value: value) == noErr
    }

    @discardableResult
    private static func clearCompressionProperty(_ session: VTCompressionSession, key: CFString) -> Bool {
        VTSessionSetProperty(session, key: key, value: nil) == noErr
    }

    private static func makeMirageEncodedFrame(_ sampleBuffer: CMSampleBuffer) throws -> MirageHEVCEncodedFrame {
        guard let dataBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else {
            throw PyrowaveError.truncatedInput
        }
        let rawFrameData = try extractData(from: dataBuffer)
        let keyframe = isKeyframe(sampleBuffer)
        let wireData: Data
        if keyframe,
           let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
           let parameterSets = extractHEVCParameterSets(from: formatDescription),
           !parameterSets.isEmpty {
            var framed = Data(capacity: 4 + parameterSets.count + rawFrameData.count)
            var parameterSetLength = UInt32(parameterSets.count).bigEndian
            withUnsafeBytes(of: &parameterSetLength) { framed.append(contentsOf: $0) }
            framed.append(parameterSets)
            framed.append(rawFrameData)
            wireData = framed
        } else {
            wireData = rawFrameData
        }
        return MirageHEVCEncodedFrame(
            sampleBuffer: sampleBuffer,
            presentationTime: CMSampleBufferGetPresentationTimeStamp(sampleBuffer),
            wireData: wireData
        )
    }

    private static func extractData(from blockBuffer: CMBlockBuffer) throws -> Data {
        let length = CMBlockBufferGetDataLength(blockBuffer)
        var data = Data(count: length)
        let status = data.withUnsafeMutableBytes { bytes in
            CMBlockBufferCopyDataBytes(
                blockBuffer,
                atOffset: 0,
                dataLength: length,
                destination: bytes.baseAddress!
            )
        }
        guard status == noErr else {
            throw PyrowaveError.processFailed("CMBlockBufferCopyDataBytes failed: \(status)")
        }
        return data
    }

    private static func extractHEVCParameterSets(from formatDescription: CMFormatDescription) -> Data? {
        var parameterSetCount = 0
        var nalUnitHeaderLength: Int32 = 0
        var status = CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
            formatDescription,
            parameterSetIndex: 0,
            parameterSetPointerOut: nil,
            parameterSetSizeOut: nil,
            parameterSetCountOut: &parameterSetCount,
            nalUnitHeaderLengthOut: &nalUnitHeaderLength
        )
        guard status == noErr, parameterSetCount > 0 else {
            return nil
        }

        var data = Data()
        for index in 0 ..< parameterSetCount {
            var pointer: UnsafePointer<UInt8>?
            var size = 0
            status = CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
                formatDescription,
                parameterSetIndex: index,
                parameterSetPointerOut: &pointer,
                parameterSetSizeOut: &size,
                parameterSetCountOut: nil,
                nalUnitHeaderLengthOut: nil
            )
            guard status == noErr, let pointer, size > 0 else {
                return nil
            }
            data.append(contentsOf: [0, 0, 0, 1])
            data.append(pointer, count: size)
        }
        return data
    }

    private static func isKeyframe(_ sampleBuffer: CMSampleBuffer) -> Bool {
        guard let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false),
              CFArrayGetCount(attachments) > 0,
              let attachment = unsafeBitCast(CFArrayGetValueAtIndex(attachments, 0), to: CFDictionary?.self) else {
            return true
        }
        let notSync = CFDictionaryGetValue(attachment, Unmanaged.passUnretained(kCMSampleAttachmentKey_NotSync).toOpaque())
        guard let notSync else {
            return true
        }
        return !CFBooleanGetValue(unsafeBitCast(notSync, to: CFBoolean.self))
    }

    private static func writeMirageWireStream(_ frames: [MirageHEVCEncodedFrame], to url: URL) throws {
        var data = Data()
        for frame in frames {
            data.append(frame.wireData)
        }
        try data.write(to: url)
    }

    private static func makePixelBuffers(_ frames: [YUVFrame], pixelFormat: OSType) throws -> [CVPixelBuffer] {
        var pixelBuffers = [CVPixelBuffer]()
        pixelBuffers.reserveCapacity(frames.count)
        for frame in frames {
            pixelBuffers.append(try frame.makeCVPixelBuffer(pixelFormat: pixelFormat))
        }
        return pixelBuffers
    }

    private enum MirageEncoderSpecTier: CaseIterable {
        case lowLatencyHardwareRequired
        case hardwareRequired
        case hardwarePreferred

        var label: String {
            switch self {
            case .lowLatencyHardwareRequired:
                "hw-required+lowLatency"
            case .hardwareRequired:
                "hw-required"
            case .hardwarePreferred:
                "hw-preferred"
            }
        }

        var encoderSpecification: [CFString: Any] {
            switch self {
            case .lowLatencyHardwareRequired:
                [
                    kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder: true,
                    kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder: true,
                    kVTVideoEncoderSpecification_EnableLowLatencyRateControl: true,
                ]
            case .hardwareRequired:
                [
                    kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder: true,
                    kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder: true,
                ]
            case .hardwarePreferred:
                [
                    kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder: true,
                ]
            }
        }
    }

    private struct MirageHEVCEncodedFrame: @unchecked Sendable {
        var sampleBuffer: CMSampleBuffer
        var presentationTime: CMTime
        var wireData: Data
    }

    private final class HEVCEncodeCollector: @unchecked Sendable {
        private let lock = NSLock()
        private var storage: [MirageHEVCEncodedFrame] = []
        private var errors: [String] = []

        var frames: [MirageHEVCEncodedFrame] {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }

        func append(_ frame: MirageHEVCEncodedFrame) {
            lock.lock()
            storage.append(frame)
            lock.unlock()
        }

        func recordError(_ error: String) {
            lock.lock()
            errors.append(error)
            lock.unlock()
        }

        func throwIfNeeded() throws {
            lock.lock()
            let errors = self.errors
            lock.unlock()
            if let first = errors.first {
                throw PyrowaveError.processFailed(first)
            }
        }
    }

    private struct HEVCDecodedOutput {
        var pixelBuffer: CVPixelBuffer
        var presentationTimeStamp: CMTime
    }

    private final class HEVCDecodeCollector: @unchecked Sendable {
        private let lock = NSLock()
        private let group = DispatchGroup()
        private var storage: [HEVCDecodedOutput] = []
        private var errors: [String] = []

        var outputs: [HEVCDecodedOutput] {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }

        func enter() {
            group.enter()
        }

        func leaveWithoutCallback() {
            group.leave()
        }

        func record(status: OSStatus, imageBuffer: CVImageBuffer?, presentationTimeStamp: CMTime) {
            defer { group.leave() }
            lock.lock()
            if status == noErr, let imageBuffer {
                storage.append(HEVCDecodedOutput(pixelBuffer: imageBuffer, presentationTimeStamp: presentationTimeStamp))
            } else {
                errors.append("HEVC decode callback failed: status \(status)")
            }
            lock.unlock()
        }

        func wait(timeout: DispatchTime) -> Bool {
            group.wait(timeout: timeout) == .success
        }

        func throwIfNeeded() throws {
            lock.lock()
            let errors = self.errors
            lock.unlock()
            if let first = errors.first {
                throw PyrowaveError.processFailed(first)
            }
        }
    }
}
