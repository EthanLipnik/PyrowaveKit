import Foundation

struct YUV4MPEGReader {
    private var handle: FileHandle
    private let bytesPerSample: Int
    private let sampleMax: Int
    private let videoSignal: VideoSignalMetadata
    let width: Int
    let height: Int
    let chroma: ChromaSubsampling
    let bitDepth: Int
    let frameRateNumerator: Int
    let frameRateDenominator: Int
    let headerParameters: String

    init(url: URL) throws {
        handle = try FileHandle(forReadingFrom: url)
        guard let line = try handle.readLine(), line.hasPrefix("YUV4MPEG2 ") else {
            throw PyrowaveError.unsupportedFormat("missing YUV4MPEG2 header")
        }

        headerParameters = String(line.dropFirst("YUV4MPEG2 ".count))
        width = try Self.parseRequiredInt(prefix: "W", parameters: headerParameters)
        height = try Self.parseRequiredInt(prefix: "H", parameters: headerParameters)
        let chromaToken = Self.parseChromaToken(parameters: headerParameters)
        let frameRate = try Self.parseFrameRate(parameters: headerParameters)
        frameRateNumerator = frameRate.numerator
        frameRateDenominator = frameRate.denominator

        if chromaToken?.hasPrefix("C444") == true {
            chroma = .yuv444
        } else if chromaToken?.hasPrefix("C420") == true || chromaToken == nil {
            chroma = .yuv420
        } else {
            throw PyrowaveError.unsupportedFormat(headerParameters)
        }

        bitDepth = try Self.parseBitDepth(chromaToken: chromaToken)
        bytesPerSample = bitDepth > 8 ? 2 : 1
        sampleMax = bitDepth == 16 ? 0xffff : (1 << bitDepth) - 1
        let range: YCbCrRange = headerParameters.contains("XCOLORRANGE=FULL") ? .full : .limited
        videoSignal = VideoSignalMetadata(yCbCrRange: range)
    }

    mutating func readFrame() throws -> YUVFrame? {
        guard let line = try handle.readLine() else {
            return nil
        }
        guard line.hasPrefix("FRAME") else {
            throw PyrowaveError.unsupportedFormat("expected FRAME marker")
        }

        let chromaWidth = width / chroma.chromaDivisor
        let chromaHeight = height / chroma.chromaDivisor
        let ySampleCount = width * height
        let cSampleCount = chromaWidth * chromaHeight

        guard let yData = try handle.readExactly(byteCount: ySampleCount * bytesPerSample),
              let cbData = try handle.readExactly(byteCount: cSampleCount * bytesPerSample),
              let crData = try handle.readExactly(byteCount: cSampleCount * bytesPerSample) else {
            throw PyrowaveError.truncatedInput
        }

        return try YUVFrame(
            width: width,
            height: height,
            chroma: chroma,
            y: Plane8(width: width, height: height, data: Self.decodeSamples(yData, bitDepth: bitDepth, sampleMax: sampleMax)),
            cb: Plane8(width: chromaWidth, height: chromaHeight, data: Self.decodeSamples(cbData, bitDepth: bitDepth, sampleMax: sampleMax)),
            cr: Plane8(width: chromaWidth, height: chromaHeight, data: Self.decodeSamples(crData, bitDepth: bitDepth, sampleMax: sampleMax)),
            videoSignal: videoSignal
        )
    }

    private static func parseChromaToken(parameters: String) -> String? {
        parameters.split(separator: " ").first { $0.hasPrefix("C") }.map(String.init)
    }

    private static func parseBitDepth(chromaToken: String?) throws -> Int {
        guard let chromaToken else {
            return 8
        }

        for depth in [10, 12, 14, 16] where chromaToken.hasSuffix("p\(depth)") {
            return depth
        }
        return 8
    }

    private static func parseFrameRate(parameters: String) throws -> (numerator: Int, denominator: Int) {
        for token in parameters.split(separator: " ") where token.hasPrefix("F") {
            let values = token.dropFirst().split(separator: ":", maxSplits: 1)
            guard values.count == 2,
                  let numerator = Int(values[0]),
                  let denominator = Int(values[1]),
                  numerator > 0,
                  denominator > 0 else {
                throw PyrowaveError.unsupportedFormat("bad frame rate \(token)")
            }
            return (numerator, denominator)
        }
        return (60, 1)
    }

    private static func parseRequiredInt(prefix: String, parameters: String) throws -> Int {
        for token in parameters.split(separator: " ") {
            if token.hasPrefix(prefix), let value = Int(token.dropFirst()) {
                return value
            }
        }
        throw PyrowaveError.unsupportedFormat("missing \(prefix) parameter")
    }

    private static func decodeSamples(_ data: Data, bitDepth: Int, sampleMax: Int) -> [UInt8] {
        guard bitDepth > 8 else {
            return [UInt8](data)
        }

        var samples = [UInt8]()
        samples.reserveCapacity(data.count / 2)
        var offset = 0
        while offset + 1 < data.count {
            let raw = Int(data[offset]) | (Int(data[offset + 1]) << 8)
            let clamped = min(raw, sampleMax)
            samples.append(UInt8((clamped * 255 + sampleMax / 2) / sampleMax))
            offset += 2
        }
        return samples
    }
}

struct YUV4MPEGWriter {
    private var handle: FileHandle
    let width: Int
    let height: Int
    let chroma: ChromaSubsampling

    init(url: URL, width: Int, height: Int, chroma: ChromaSubsampling, frameRate: String = "60:1") throws {
        self.width = width
        self.height = height
        self.chroma = chroma
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
        let chromaTag = chroma == .yuv444 ? "C444" : "C420jpeg"
        try handle.write(contentsOf: Data("YUV4MPEG2 W\(width) H\(height) F\(frameRate) Ip A1:1 \(chromaTag) XCOLORRANGE=FULL\n".utf8))
    }

    mutating func writeFrame(_ frame: YUVFrame) throws {
        guard frame.width == width, frame.height == height, frame.chroma == chroma else {
            throw PyrowaveError.invalidDimensions
        }
        try handle.write(contentsOf: Data("FRAME\n".utf8))
        try handle.write(contentsOf: Data(frame.y.data))
        try handle.write(contentsOf: Data(frame.cb.data))
        try handle.write(contentsOf: Data(frame.cr.data))
    }

    static func write(
        frames: [YUVFrame],
        to url: URL,
        frameRateNumerator: Int = 60,
        frameRateDenominator: Int = 1
    ) throws {
        guard let firstFrame = frames.first,
              frameRateNumerator > 0,
              frameRateDenominator > 0 else {
            throw PyrowaveError.invalidDimensions
        }

        var writer = try YUV4MPEGWriter(
            url: url,
            width: firstFrame.width,
            height: firstFrame.height,
            chroma: firstFrame.chroma,
            frameRate: "\(frameRateNumerator):\(frameRateDenominator)"
        )
        for frame in frames {
            try writer.writeFrame(frame)
        }
    }
}

private extension FileHandle {
    func readLine() throws -> String? {
        var bytes = [UInt8]()
        while true {
            let data = try read(upToCount: 1)
            guard let byte = data?.first else {
                return bytes.isEmpty ? nil : String(decoding: bytes, as: UTF8.self)
            }
            if byte == 0x0a {
                return String(decoding: bytes, as: UTF8.self)
            }
            bytes.append(byte)
        }
    }

    func readExactly(byteCount: Int) throws -> Data? {
        guard let data = try read(upToCount: byteCount), data.count == byteCount else {
            return nil
        }
        return data
    }
}
