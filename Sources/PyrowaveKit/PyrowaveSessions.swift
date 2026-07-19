import CoreGraphics
import CoreVideo
import Foundation

public struct PyrowaveSessionDescriptor: Equatable, Sendable {
    public let width: Int
    public let height: Int
    public let pixelFormat: OSType
    public let videoSignal: VideoSignalMetadata

    public init(
        width: Int,
        height: Int,
        pixelFormat: OSType = kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        videoSignal: VideoSignalMetadata = .default
    ) throws {
        guard width > 0, height > 0, width.isMultiple(of: 2), height.isMultiple(of: 2) else {
            throw PyrowaveSessionError.invalidDescriptor
        }
        guard Self.supportedPixelFormats.contains(pixelFormat) else {
            throw PyrowaveSessionError.unsupportedPixelFormat(pixelFormat)
        }
        self.width = width
        self.height = height
        self.pixelFormat = pixelFormat
        self.videoSignal = videoSignal
    }

    static let supportedPixelFormats: Set<OSType> = [
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
    ]
}

public enum PyrowaveSessionError: Error, Equatable, Sendable, CustomStringConvertible {
    case invalidDescriptor
    case unsupportedPixelFormat(OSType)
    case inputGeometryMismatch(expectedWidth: Int, expectedHeight: Int, actualWidth: Int, actualHeight: Int)
    case outputPoolFailure(CVReturn)
    case decoderBusy
    case recoverableFrameFailure(PyrowaveError)
    case sessionFailure(PyrowaveError)

    public var isRecoverable: Bool {
        switch self {
        case .inputGeometryMismatch, .decoderBusy, .recoverableFrameFailure:
            return true
        case .invalidDescriptor, .unsupportedPixelFormat, .outputPoolFailure, .sessionFailure:
            return false
        }
    }

    public var description: String {
        switch self {
        case .invalidDescriptor:
            return "Invalid Pyrowave session descriptor."
        case .unsupportedPixelFormat(let format):
            return "Unsupported Pyrowave session pixel format: \(format)"
        case let .inputGeometryMismatch(expectedWidth, expectedHeight, actualWidth, actualHeight):
            return "Pyrowave input geometry \(actualWidth)x\(actualHeight) does not match session geometry \(expectedWidth)x\(expectedHeight)."
        case .outputPoolFailure(let status):
            return "Pyrowave output pixel-buffer pool failed with status \(status)."
        case .decoderBusy:
            return "Pyrowave decoder is already processing a frame."
        case .recoverableFrameFailure(let error):
            return "Recoverable Pyrowave frame failure: \(error)"
        case .sessionFailure(let error):
            return "Pyrowave session failure: \(error)"
        }
    }
}

public struct PyrowaveEncoderStageMetrics: Equatable, Sendable {
    public let encodeMilliseconds: Double
    public let exportMilliseconds: Double
    public let totalMilliseconds: Double
    public let encodedBytes: Int
    public let payloadBytesCopied: Int
    public let packetSlotCapacityBytes: Int
}

public struct PyrowaveDecoderStageMetrics: Equatable, Sendable {
    public let headerParseMilliseconds: Double
    public let outputAcquisitionMilliseconds: Double
    public let textureCreationMilliseconds: Double
    public let packetParseMilliseconds: Double
    public let metalDecodeMilliseconds: Double
    public let contiguousDecodeMilliseconds: Double
    public let totalMilliseconds: Double
    public let encodedBytes: Int
}

public struct PyrowaveEncodedFrameResult: Equatable, Sendable {
    public let frame: EncodedFrame
    public let quality: PyrowaveQuality
    public let configuration: CodecConfiguration
    public let metrics: PyrowaveEncoderStageMetrics
}

/// Owns a pooled CoreVideo output until all consumers release this result.
public final class PyrowaveDecodedFrameResult: @unchecked Sendable {
    public let pixelBuffer: CVPixelBuffer
    public let videoSignal: VideoSignalMetadata
    public let metrics: PyrowaveDecoderStageMetrics

    init(
        pixelBuffer: CVPixelBuffer,
        videoSignal: VideoSignalMetadata,
        metrics: PyrowaveDecoderStageMetrics
    ) {
        self.pixelBuffer = pixelBuffer
        self.videoSignal = videoSignal
        self.metrics = metrics
    }
}

/// Explicitly transfers immutable pixel-buffer ownership into an encoder
/// session across an actor boundary. The caller must not mutate the buffer
/// while the encode is in progress.
public final class PyrowavePixelBufferInput: @unchecked Sendable {
    public let pixelBuffer: CVPixelBuffer

    public init(_ pixelBuffer: CVPixelBuffer) {
        self.pixelBuffer = pixelBuffer
    }
}

/// A serial production encoder. Actor isolation prevents overlapping use of
/// codec scratch buffers and makes one-active-encode ownership explicit.
public actor PyrowaveEncoderSession {
    public let descriptor: PyrowaveSessionDescriptor
    private let codec: PyrowaveCodec

    public init(descriptor: PyrowaveSessionDescriptor) throws {
        self.descriptor = descriptor
        do {
            codec = try PyrowaveCodec()
        } catch let error as PyrowaveError {
            throw PyrowaveSessionError.sessionFailure(error)
        }
    }

    public func encode(
        _ pixelBuffer: CVPixelBuffer,
        quality: PyrowaveQuality = .highest
    ) throws -> PyrowaveEncodedFrameResult {
        try validateGeometry(pixelBuffer)
        let totalStart = ContinuousClock.now
        let configuration = quality.codecConfiguration

        do {
            let encodeStart = ContinuousClock.now
            let gpuFrame = try codec.encodeGPUFrameForSession(
                pixelBuffer,
                configuration: configuration,
                videoSignal: descriptor.videoSignal
            )
            let encodeEnd = ContinuousClock.now
            let frame = try codec.exportGPUFrame(gpuFrame)
            let exportEnd = ContinuousClock.now
            return PyrowaveEncodedFrameResult(
                frame: frame,
                quality: quality,
                configuration: configuration,
                metrics: PyrowaveEncoderStageMetrics(
                    encodeMilliseconds: encodeStart.milliseconds(to: encodeEnd),
                    exportMilliseconds: encodeEnd.milliseconds(to: exportEnd),
                    totalMilliseconds: totalStart.milliseconds(to: exportEnd),
                    encodedBytes: frame.data.count,
                    payloadBytesCopied: max(frame.data.count - 8, 0),
                    packetSlotCapacityBytes: gpuFrame.estimatedPacketCapacityBytes
                )
            )
        } catch let error as PyrowaveError {
            throw PyrowaveSessionError.recoverableFrameFailure(error)
        }
    }

    public func encode(
        _ input: PyrowavePixelBufferInput,
        quality: PyrowaveQuality = .highest
    ) throws -> PyrowaveEncodedFrameResult {
        try encode(input.pixelBuffer, quality: quality)
    }

    private func validateGeometry(_ pixelBuffer: CVPixelBuffer) throws {
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        guard width == descriptor.width, height == descriptor.height else {
            throw PyrowaveSessionError.inputGeometryMismatch(
                expectedWidth: descriptor.width,
                expectedHeight: descriptor.height,
                actualWidth: width,
                actualHeight: height
            )
        }
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        guard PyrowaveSessionDescriptor.supportedPixelFormats.contains(format) else {
            throw PyrowaveSessionError.unsupportedPixelFormat(format)
        }
    }
}

/// A serial production decoder with a geometry-specific CoreVideo output pool.
public actor PyrowaveDecoderSession {
    public let descriptor: PyrowaveSessionDescriptor
    private let codec: PyrowaveCodec
    private let outputPool: CVPixelBufferPool
    private var isDecodingFrame = false

    public init(
        descriptor: PyrowaveSessionDescriptor,
        minimumOutputBufferCount: Int = 3
    ) throws {
        guard minimumOutputBufferCount > 0 else {
            throw PyrowaveSessionError.invalidDescriptor
        }
        self.descriptor = descriptor
        do {
            codec = try PyrowaveCodec()
        } catch let error as PyrowaveError {
            throw PyrowaveSessionError.sessionFailure(error)
        }

        let poolAttributes = [
            kCVPixelBufferPoolMinimumBufferCountKey as String: minimumOutputBufferCount
        ] as CFDictionary
        let pixelBufferAttributes = [
            kCVPixelBufferWidthKey as String: descriptor.width,
            kCVPixelBufferHeightKey as String: descriptor.height,
            kCVPixelBufferPixelFormatTypeKey as String: descriptor.pixelFormat,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ] as CFDictionary
        var pool: CVPixelBufferPool?
        let status = CVPixelBufferPoolCreate(nil, poolAttributes, pixelBufferAttributes, &pool)
        guard status == kCVReturnSuccess, let pool else {
            throw PyrowaveSessionError.outputPoolFailure(status)
        }
        outputPool = pool
    }

    public func decode(_ frame: EncodedFrame) async throws -> PyrowaveDecodedFrameResult {
        guard !isDecodingFrame else {
            throw PyrowaveSessionError.decoderBusy
        }
        isDecodingFrame = true
        defer { isDecodingFrame = false }
        let totalStart = ContinuousClock.now
        do {
            var reader = BinaryReader(frame.data)
            let sequence = try PyrowaveSequenceHeader(reader: &reader)
            guard sequence.width == descriptor.width, sequence.height == descriptor.height else {
                throw PyrowaveSessionError.inputGeometryMismatch(
                    expectedWidth: descriptor.width,
                    expectedHeight: descriptor.height,
                    actualWidth: sequence.width,
                    actualHeight: sequence.height
                )
            }
            let headerEnd = ContinuousClock.now
            var pixelBuffer: CVPixelBuffer?
            let poolStatus = CVPixelBufferPoolCreatePixelBuffer(nil, outputPool, &pixelBuffer)
            guard poolStatus == kCVReturnSuccess, let pixelBuffer else {
                throw PyrowaveSessionError.outputPoolFailure(poolStatus)
            }
            let acquisitionEnd = ContinuousClock.now
            let output = PyrowavePixelBufferInput(pixelBuffer)
            let contiguousMetrics = try await codec.decodeContiguousFrame(frame, to: output)
            Self.applyVideoSignalAttachments(descriptor.videoSignal, to: pixelBuffer)
            let decodeEnd = ContinuousClock.now
            return PyrowaveDecodedFrameResult(
                pixelBuffer: pixelBuffer,
                videoSignal: descriptor.videoSignal,
                metrics: PyrowaveDecoderStageMetrics(
                    headerParseMilliseconds: totalStart.milliseconds(to: headerEnd),
                    outputAcquisitionMilliseconds: headerEnd.milliseconds(to: acquisitionEnd),
                    textureCreationMilliseconds: contiguousMetrics.textureCreationMilliseconds,
                    packetParseMilliseconds: contiguousMetrics.packetParseMilliseconds,
                    metalDecodeMilliseconds: contiguousMetrics.metalDecodeMilliseconds,
                    contiguousDecodeMilliseconds: acquisitionEnd.milliseconds(to: decodeEnd),
                    totalMilliseconds: totalStart.milliseconds(to: decodeEnd),
                    encodedBytes: frame.data.count
                )
            )
        } catch let error as PyrowaveSessionError {
            throw error
        } catch let error as PyrowaveError {
            switch error {
            case .truncatedInput, .invalidBitstream, .invalidDimensions, .unsupportedFormat:
                throw PyrowaveSessionError.recoverableFrameFailure(error)
            case .externalToolUnavailable, .processFailed:
                throw PyrowaveSessionError.sessionFailure(error)
            }
        }
    }

    private static func applyVideoSignalAttachments(
        _ videoSignal: VideoSignalMetadata,
        to pixelBuffer: CVPixelBuffer
    ) {
        let colorPrimaries: CFString = switch videoSignal.colorPrimaries {
        case .bt709:
            kCVImageBufferColorPrimaries_ITU_R_709_2
        case .bt2020:
            kCVImageBufferColorPrimaries_ITU_R_2020
        }
        let transferFunction: CFString = switch videoSignal.transferFunction {
        case .bt709:
            kCVImageBufferTransferFunction_ITU_R_709_2
        case .pq:
            kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ
        case .sRGB:
            kCVImageBufferTransferFunction_sRGB
        }
        let yCbCrMatrix: CFString = switch videoSignal.yCbCrTransform {
        case .bt709:
            kCVImageBufferYCbCrMatrix_ITU_R_709_2
        case .bt2020:
            kCVImageBufferYCbCrMatrix_ITU_R_2020
        }
        let chromaLocation: CFString = switch videoSignal.chromaSiting {
        case .center:
            kCVImageBufferChromaLocation_Center
        case .left:
            kCVImageBufferChromaLocation_Left
        }
        CVBufferSetAttachment(pixelBuffer, kCVImageBufferColorPrimariesKey, colorPrimaries, .shouldPropagate)
        CVBufferSetAttachment(pixelBuffer, kCVImageBufferTransferFunctionKey, transferFunction, .shouldPropagate)
        CVBufferSetAttachment(pixelBuffer, kCVImageBufferYCbCrMatrixKey, yCbCrMatrix, .shouldPropagate)
        CVBufferSetAttachment(pixelBuffer, kCVImageBufferChromaLocationTopFieldKey, chromaLocation, .shouldPropagate)
        CVBufferSetAttachment(pixelBuffer, kCVImageBufferChromaLocationBottomFieldKey, chromaLocation, .shouldPropagate)

        let colorSpaceName: CFString = switch (videoSignal.colorPrimaries, videoSignal.transferFunction) {
        case (.bt709, .sRGB):
            CGColorSpace.sRGB
        case (.bt2020, .pq):
            CGColorSpace.itur_2020
        case (.bt709, .bt709), (.bt2020, .bt709), (.bt709, .pq), (.bt2020, .sRGB):
            CGColorSpace.sRGB
        }
        if let colorSpace = CGColorSpace(name: colorSpaceName) {
            CVBufferSetAttachment(pixelBuffer, kCVImageBufferCGColorSpaceKey, colorSpace, .shouldPropagate)
        }
    }
}

extension ContinuousClock.Instant {
    func milliseconds(to end: ContinuousClock.Instant) -> Double {
        let components = duration(to: end).components
        return Double(components.seconds) * 1_000
            + Double(components.attoseconds) / 1_000_000_000_000_000
    }
}
