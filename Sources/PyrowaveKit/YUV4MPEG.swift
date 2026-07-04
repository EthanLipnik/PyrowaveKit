import Foundation

public struct YUV4MPEGReader {
    private var handle: FileHandle
    public let width: Int
    public let height: Int
    public let chroma: ChromaSubsampling
    public let headerParameters: String

    public init(url: URL) throws {
        handle = try FileHandle(forReadingFrom: url)
        guard let line = try handle.readLine(), line.hasPrefix("YUV4MPEG2 ") else {
            throw PyrowaveError.unsupportedFormat("missing YUV4MPEG2 header")
        }

        headerParameters = String(line.dropFirst("YUV4MPEG2 ".count))
        width = try Self.parseRequiredInt(prefix: "W", parameters: headerParameters)
        height = try Self.parseRequiredInt(prefix: "H", parameters: headerParameters)

        if headerParameters.contains("C444") {
            chroma = .yuv444
        } else if headerParameters.contains("C420") || !headerParameters.contains("C") {
            chroma = .yuv420
        } else {
            throw PyrowaveError.unsupportedFormat(headerParameters)
        }

        if headerParameters.contains("p10") || headerParameters.contains("p12") ||
            headerParameters.contains("p14") || headerParameters.contains("p16") {
            throw PyrowaveError.unsupportedFormat("only 8-bit YUV4MPEG is supported by the Swift harness")
        }
    }

    public mutating func readFrame() throws -> YUVFrame? {
        guard let line = try handle.readLine() else {
            return nil
        }
        guard line.hasPrefix("FRAME") else {
            throw PyrowaveError.unsupportedFormat("expected FRAME marker")
        }

        let chromaWidth = width / chroma.chromaDivisor
        let chromaHeight = height / chroma.chromaDivisor
        let ySize = width * height
        let cSize = chromaWidth * chromaHeight

        guard let yData = try handle.readExactly(byteCount: ySize),
              let cbData = try handle.readExactly(byteCount: cSize),
              let crData = try handle.readExactly(byteCount: cSize) else {
            throw PyrowaveError.truncatedInput
        }

        return try YUVFrame(
            width: width,
            height: height,
            chroma: chroma,
            y: Plane8(width: width, height: height, data: [UInt8](yData)),
            cb: Plane8(width: chromaWidth, height: chromaHeight, data: [UInt8](cbData)),
            cr: Plane8(width: chromaWidth, height: chromaHeight, data: [UInt8](crData))
        )
    }

    private static func parseRequiredInt(prefix: String, parameters: String) throws -> Int {
        for token in parameters.split(separator: " ") {
            if token.hasPrefix(prefix), let value = Int(token.dropFirst()) {
                return value
            }
        }
        throw PyrowaveError.unsupportedFormat("missing \(prefix) parameter")
    }
}

public struct YUV4MPEGWriter {
    private var handle: FileHandle
    public let width: Int
    public let height: Int
    public let chroma: ChromaSubsampling

    public init(url: URL, width: Int, height: Int, chroma: ChromaSubsampling, frameRate: String = "60:1") throws {
        self.width = width
        self.height = height
        self.chroma = chroma
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
        let chromaTag = chroma == .yuv444 ? "C444" : "C420jpeg"
        try handle.write(contentsOf: Data("YUV4MPEG2 W\(width) H\(height) F\(frameRate) Ip A1:1 \(chromaTag) XCOLORRANGE=FULL\n".utf8))
    }

    public mutating func writeFrame(_ frame: YUVFrame) throws {
        guard frame.width == width, frame.height == height, frame.chroma == chroma else {
            throw PyrowaveError.invalidDimensions
        }
        try handle.write(contentsOf: Data("FRAME\n".utf8))
        try handle.write(contentsOf: Data(frame.y.data))
        try handle.write(contentsOf: Data(frame.cb.data))
        try handle.write(contentsOf: Data(frame.cr.data))
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
