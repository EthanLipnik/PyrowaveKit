import Foundation

public enum PyrowaveError: Error, Equatable, CustomStringConvertible {
    case invalidDimensions
    case unsupportedFormat(String)
    case truncatedInput
    case invalidBitstream(String)
    case externalToolUnavailable(String)
    case processFailed(String)

    public var description: String {
        switch self {
        case .invalidDimensions:
            return "Invalid image dimensions."
        case .unsupportedFormat(let format):
            return "Unsupported format: \(format)"
        case .truncatedInput:
            return "Input ended before a full frame could be read."
        case .invalidBitstream(let reason):
            return "Invalid PyrowaveKit bitstream: \(reason)"
        case .externalToolUnavailable(let name):
            return "External tool unavailable: \(name)"
        case .processFailed(let reason):
            return "Process failed: \(reason)"
        }
    }
}

public enum ChromaSubsampling: UInt8, Codable, Sendable {
    case yuv420 = 0
    case yuv444 = 1

    public var chromaDivisor: Int {
        switch self {
        case .yuv420:
            return 2
        case .yuv444:
            return 1
        }
    }
}

public enum ColorPrimaries: UInt8, Codable, Sendable {
    case bt709 = 0
    case bt2020 = 1
}

public enum TransferFunction: UInt8, Codable, Sendable {
    case bt709 = 0
    case pq = 1
}

public enum YCbCrTransform: UInt8, Codable, Sendable {
    case bt709 = 0
    case bt2020 = 1
}

public enum YCbCrRange: UInt8, Codable, Sendable {
    case full = 0
    case limited = 1
}

public enum ChromaSiting: UInt8, Codable, Sendable {
    case center = 0
    case left = 1
}

public struct VideoSignalMetadata: Codable, Equatable, Sendable {
    public var colorPrimaries: ColorPrimaries
    public var transferFunction: TransferFunction
    public var yCbCrTransform: YCbCrTransform
    public var yCbCrRange: YCbCrRange
    public var chromaSiting: ChromaSiting

    public static let `default` = VideoSignalMetadata()

    public init(
        colorPrimaries: ColorPrimaries = .bt709,
        transferFunction: TransferFunction = .bt709,
        yCbCrTransform: YCbCrTransform = .bt709,
        yCbCrRange: YCbCrRange = .full,
        chromaSiting: ChromaSiting = .center
    ) {
        self.colorPrimaries = colorPrimaries
        self.transferFunction = transferFunction
        self.yCbCrTransform = yCbCrTransform
        self.yCbCrRange = yCbCrRange
        self.chromaSiting = chromaSiting
    }
}

public struct CodecConfiguration: Codable, Equatable, Sendable {
    public var decompositionLevels: Int
    public var quantizationStep: Float
    public var maximumEncodedBytes: Int?

    public init(
        decompositionLevels: Int = 5,
        quantizationStep: Float = 1.0 / 1024.0,
        maximumEncodedBytes: Int? = nil
    ) {
        self.decompositionLevels = decompositionLevels
        self.quantizationStep = quantizationStep
        self.maximumEncodedBytes = maximumEncodedBytes
    }
}

public struct Plane8: Equatable, Sendable {
    public var width: Int
    public var height: Int
    public var data: [UInt8]

    public init(width: Int, height: Int, data: [UInt8]) throws {
        guard width > 0, height > 0, data.count == width * height else {
            throw PyrowaveError.invalidDimensions
        }
        self.width = width
        self.height = height
        self.data = data
    }
}

public struct YUVFrame: Equatable, Sendable {
    public var width: Int
    public var height: Int
    public var chroma: ChromaSubsampling
    public var y: Plane8
    public var cb: Plane8
    public var cr: Plane8
    public var videoSignal: VideoSignalMetadata

    public init(
        width: Int,
        height: Int,
        chroma: ChromaSubsampling,
        y: Plane8,
        cb: Plane8,
        cr: Plane8,
        videoSignal: VideoSignalMetadata = .default
    ) throws {
        guard width > 0, height > 0, y.width == width, y.height == height else {
            throw PyrowaveError.invalidDimensions
        }

        let div = chroma.chromaDivisor
        guard width % div == 0, height % div == 0,
              cb.width == width / div, cb.height == height / div,
              cr.width == width / div, cr.height == height / div else {
            throw PyrowaveError.invalidDimensions
        }

        self.width = width
        self.height = height
        self.chroma = chroma
        self.y = y
        self.cb = cb
        self.cr = cr
        self.videoSignal = videoSignal
    }
}

public struct EncodedFrame: Equatable, Sendable {
    public var data: Data

    public init(data: Data) {
        self.data = data
    }
}

public struct ComponentMetrics: Codable, Equatable, Sendable {
    public var mse: Double
    public var psnr: Double
}

public struct FrameMetrics: Codable, Equatable, Sendable {
    public var y: ComponentMetrics
    public var cb: ComponentMetrics
    public var cr: ComponentMetrics
    public var weightedPSNR: Double
}
