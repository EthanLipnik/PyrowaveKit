import Foundation

public enum PyrowaveError: Error, Equatable, Sendable, CustomStringConvertible {
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
    case sRGB = 2
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

    public init(
        decompositionLevels: Int = 5,
        quantizationStep: Float = 1.0 / 1024.0
    ) {
        self.decompositionLevels = decompositionLevels
        self.quantizationStep = quantizationStep
    }
}

/// A codec-native visual quality value. Higher values preserve more detail.
///
/// Pyrowave deliberately does not interpret this value as a byte-rate target.
public struct PyrowaveQuality: Codable, Equatable, Sendable {
    public static let highest = PyrowaveQuality(normalized: 1)

    public let normalized: Float

    public init(normalized: Float) {
        if normalized.isFinite {
            self.normalized = min(max(normalized, 0), 1)
        } else {
            self.normalized = 1
        }
    }

    public init(configuration: CodecConfiguration) {
        let scaledStep = max(configuration.quantizationStep * 2048, .leastNonzeroMagnitude)
        self.init(normalized: 1 - log2(scaledStep) / 5)
    }

    public var codecConfiguration: CodecConfiguration {
        let quantizationStep = (1.0 / 2048.0) * pow(2.0, 5.0 * (1.0 - normalized))
        return CodecConfiguration(quantizationStep: quantizationStep)
    }
}

struct Plane8: Equatable, Sendable {
    var width: Int
    var height: Int
    var data: [UInt8]

    init(width: Int, height: Int, data: [UInt8]) throws {
        guard width > 0, height > 0, data.count == width * height else {
            throw PyrowaveError.invalidDimensions
        }
        self.width = width
        self.height = height
        self.data = data
    }
}

struct YUVFrame: Equatable, Sendable {
    var width: Int
    var height: Int
    var chroma: ChromaSubsampling
    var y: Plane8
    var cb: Plane8
    var cr: Plane8
    var videoSignal: VideoSignalMetadata

    init(
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

    init(
        width: Int,
        height: Int,
        nv12Y: [UInt8],
        nv12CbCr: [UInt8],
        yRowStride: Int? = nil,
        cbCrRowStride: Int? = nil,
        videoSignal: VideoSignalMetadata = .default
    ) throws {
        guard width > 0, height > 0, width % 2 == 0, height % 2 == 0 else {
            throw PyrowaveError.invalidDimensions
        }

        let yStride = yRowStride ?? width
        let cbCrStride = cbCrRowStride ?? width
        let chromaWidth = width / 2
        let chromaHeight = height / 2
        guard yStride >= width,
              cbCrStride >= width,
              nv12Y.count >= yStride * (height - 1) + width,
              nv12CbCr.count >= cbCrStride * (chromaHeight - 1) + width else {
            throw PyrowaveError.invalidDimensions
        }

        var y = [UInt8]()
        y.reserveCapacity(width * height)
        for row in 0..<height {
            let rowStart = row * yStride
            y.append(contentsOf: nv12Y[rowStart..<(rowStart + width)])
        }

        var cb = [UInt8]()
        var cr = [UInt8]()
        cb.reserveCapacity(chromaWidth * chromaHeight)
        cr.reserveCapacity(chromaWidth * chromaHeight)
        for row in 0..<chromaHeight {
            let rowStart = row * cbCrStride
            for column in 0..<chromaWidth {
                let offset = rowStart + column * 2
                cb.append(nv12CbCr[offset])
                cr.append(nv12CbCr[offset + 1])
            }
        }

        try self.init(
            width: width,
            height: height,
            chroma: .yuv420,
            y: Plane8(width: width, height: height, data: y),
            cb: Plane8(width: chromaWidth, height: chromaHeight, data: cb),
            cr: Plane8(width: chromaWidth, height: chromaHeight, data: cr),
            videoSignal: videoSignal
        )
    }

    func nv12Planes(
        yRowStride: Int? = nil,
        cbCrRowStride: Int? = nil
    ) throws -> (y: [UInt8], cbCr: [UInt8]) {
        guard chroma == .yuv420 else {
            throw PyrowaveError.unsupportedFormat("NV12 export expects yuv420 frames")
        }

        let yStride = yRowStride ?? width
        let cbCrStride = cbCrRowStride ?? width
        let chromaWidth = width / 2
        let chromaHeight = height / 2
        guard yStride >= width, cbCrStride >= width else {
            throw PyrowaveError.invalidDimensions
        }

        var yPlane = Array(repeating: UInt8(0), count: yStride * height)
        for row in 0..<height {
            let sourceStart = row * width
            let destinationStart = row * yStride
            yPlane[destinationStart..<(destinationStart + width)] = y.data[sourceStart..<(sourceStart + width)]
        }

        var cbCrPlane = Array(repeating: UInt8(0), count: cbCrStride * chromaHeight)
        for row in 0..<chromaHeight {
            let sourceStart = row * chromaWidth
            let destinationStart = row * cbCrStride
            for column in 0..<chromaWidth {
                let destinationOffset = destinationStart + column * 2
                cbCrPlane[destinationOffset] = cb.data[sourceStart + column]
                cbCrPlane[destinationOffset + 1] = cr.data[sourceStart + column]
            }
        }

        return (yPlane, cbCrPlane)
    }
}

public struct EncodedFrame: Equatable, Sendable {
    public var data: Data

    public init(data: Data) {
        self.data = data
    }
}

public struct PyrowaveEncodedFrameDescriptor: Equatable, Sendable {
    public let width: Int
    public let height: Int
    public let chroma: ChromaSubsampling
    public let videoSignal: VideoSignalMetadata

    public init(
        width: Int,
        height: Int,
        chroma: ChromaSubsampling,
        videoSignal: VideoSignalMetadata
    ) {
        self.width = width
        self.height = height
        self.chroma = chroma
        self.videoSignal = videoSignal
    }
}

public struct EncodedPacket: Equatable, Sendable {
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
