import Foundation

public struct PyrowaveStreamHeader: Codable, Equatable, Sendable {
    public var width: Int
    public var height: Int
    public var chroma: ChromaSubsampling
    public var videoSignal: VideoSignalMetadata
    public var frameRateNumerator: Int
    public var frameRateDenominator: Int
    public var bitDepth: Int

    public init(
        width: Int,
        height: Int,
        chroma: ChromaSubsampling,
        videoSignal: VideoSignalMetadata = .default,
        frameRateNumerator: Int = 60,
        frameRateDenominator: Int = 1,
        bitDepth: Int = 8
    ) throws {
        guard width > 0, height > 0, frameRateNumerator > 0, frameRateDenominator > 0,
              [8, 10, 12, 14, 16].contains(bitDepth) else {
            throw PyrowaveError.invalidDimensions
        }

        self.width = width
        self.height = height
        self.chroma = chroma
        self.videoSignal = videoSignal
        self.frameRateNumerator = frameRateNumerator
        self.frameRateDenominator = frameRateDenominator
        self.bitDepth = bitDepth
    }

    public init(frame: YUVFrame, frameRateNumerator: Int = 60, frameRateDenominator: Int = 1, bitDepth: Int = 8) throws {
        try self.init(
            width: frame.width,
            height: frame.height,
            chroma: frame.chroma,
            videoSignal: frame.videoSignal,
            frameRateNumerator: frameRateNumerator,
            frameRateDenominator: frameRateDenominator,
            bitDepth: bitDepth
        )
    }

    fileprivate init(words: [UInt32]) throws {
        guard words.count == 8 else {
            throw PyrowaveError.truncatedInput
        }
        let formatCode = Int(words[2])
        let decodedChroma: ChromaSubsampling
        let decodedBitDepth: Int
        switch formatCode {
        case 0:
            decodedChroma = .yuv420
        case 1:
            decodedChroma = .yuv444
        case 2:
            decodedChroma = .yuv420
        case 3:
            decodedChroma = .yuv444
        default:
            throw PyrowaveError.unsupportedFormat("Pyrowave stream format code \(formatCode)")
        }

        decodedBitDepth = Int(words[3])
        guard [8, 10, 12, 14, 16].contains(decodedBitDepth),
              (formatCode < 2) == (decodedBitDepth == 8) else {
            throw PyrowaveError.invalidBitstream("bad stream bit depth")
        }
        guard let range = YCbCrRange(rawValue: UInt8(clamping: words[4])),
              let siting = ChromaSiting(rawValue: UInt8(clamping: words[7])) else {
            throw PyrowaveError.invalidBitstream("bad stream video metadata")
        }

        try self.init(
            width: Int(words[0]),
            height: Int(words[1]),
            chroma: decodedChroma,
            videoSignal: VideoSignalMetadata(yCbCrRange: range, chromaSiting: siting),
            frameRateNumerator: Int(words[5]),
            frameRateDenominator: Int(words[6]),
            bitDepth: decodedBitDepth
        )
    }

    fileprivate var words: [UInt32] {
        [
            UInt32(width),
            UInt32(height),
            UInt32(formatCode),
            UInt32(bitDepth),
            UInt32(videoSignal.yCbCrRange.rawValue),
            UInt32(frameRateNumerator),
            UInt32(frameRateDenominator),
            UInt32(videoSignal.chromaSiting.rawValue)
        ]
    }

    private var formatCode: Int {
        switch (chroma, bitDepth > 8) {
        case (.yuv420, false):
            return 0
        case (.yuv444, false):
            return 1
        case (.yuv420, true):
            return 2
        case (.yuv444, true):
            return 3
        }
    }
}

public struct PyrowaveStreamWriter {
    private var handle: FileHandle
    public let header: PyrowaveStreamHeader

    public init(url: URL, header: PyrowaveStreamHeader) throws {
        self.header = header
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
        try handle.write(contentsOf: Data("PYROWAVE".utf8))

        var writer = BinaryWriter()
        for word in header.words {
            writer.append(word)
        }
        try handle.write(contentsOf: writer.data)
    }

    public mutating func writeFrame(_ frame: EncodedFrame) throws {
        guard frame.data.count <= Int(UInt32.max) else {
            throw PyrowaveError.invalidBitstream("encoded frame exceeds stream packet size limit")
        }
        try header.validate(frame: frame)

        var writer = BinaryWriter()
        writer.append(UInt32(frame.data.count))
        writer.append(data: frame.data)
        try handle.write(contentsOf: writer.data)
    }
}

public struct PyrowaveStreamReader {
    private var handle: FileHandle
    public let header: PyrowaveStreamHeader

    public init(url: URL) throws {
        handle = try FileHandle(forReadingFrom: url)
        guard let magic = try handle.readStreamExactly(byteCount: 8),
              String(decoding: magic, as: UTF8.self) == "PYROWAVE" else {
            throw PyrowaveError.unsupportedFormat("missing PYROWAVE stream magic")
        }
        guard let parameterData = try handle.readStreamExactly(byteCount: 8 * MemoryLayout<UInt32>.stride) else {
            throw PyrowaveError.truncatedInput
        }

        var reader = BinaryReader(parameterData)
        var words = [UInt32]()
        words.reserveCapacity(8)
        for _ in 0..<8 {
            words.append(try reader.readUInt32())
        }
        header = try PyrowaveStreamHeader(words: words)
    }

    public mutating func readFrame() throws -> EncodedFrame? {
        guard let lengthData = try handle.readStreamExactly(byteCount: MemoryLayout<UInt32>.stride, allowEOF: true) else {
            return nil
        }

        var reader = BinaryReader(lengthData)
        let byteCount = Int(try reader.readUInt32())
        guard let frameData = try handle.readStreamExactly(byteCount: byteCount) else {
            throw PyrowaveError.truncatedInput
        }
        let frame = EncodedFrame(data: frameData)
        try header.validate(frame: frame)
        return frame
    }
}

private extension PyrowaveStreamHeader {
    func validate(frame: EncodedFrame) throws {
        var reader = BinaryReader(frame.data)
        let sequence = try PyrowaveSequenceHeader(reader: &reader)
        guard sequence.width == width,
              sequence.height == height,
              sequence.chroma == chroma else {
            throw PyrowaveError.invalidBitstream("encoded frame does not match stream header")
        }
    }
}

private extension FileHandle {
    func readStreamExactly(byteCount: Int, allowEOF: Bool = false) throws -> Data? {
        guard byteCount > 0 else {
            return Data()
        }
        guard let data = try read(upToCount: byteCount) else {
            return allowEOF ? nil : Data()
        }
        if data.isEmpty, allowEOF {
            return nil
        }
        guard data.count == byteCount else {
            return nil
        }
        return data
    }
}
