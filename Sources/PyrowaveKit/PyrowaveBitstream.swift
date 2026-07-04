import Foundation

enum PyrowaveBitstream {
    static let sequenceCountMask = 0x7
    static let decompositionLevels = 5
    static let alignment = 1 << decompositionLevels
    static let minimumImageSize = 4 << decompositionLevels
    static let componentCount = 3
    static let bandCount = 4
    static let coefficientBlockSize = 32
    static let smallBlockSize = 8
}

struct PyrowavePacketHeader: Equatable, Sendable {
    var ballot: UInt16
    var payloadWords: UInt16
    var sequence: UInt8
    var extended: Bool
    var quantCode: UInt8
    var blockIndex: Int

    init(ballot: UInt16, payloadWords: UInt16, sequence: UInt8, extended: Bool, quantCode: UInt8, blockIndex: Int) throws {
        guard payloadWords < (1 << 12),
              sequence <= PyrowaveBitstream.sequenceCountMask,
              blockIndex >= 0,
              blockIndex < (1 << 24) else {
            throw PyrowaveError.invalidBitstream("packet header field out of range")
        }

        self.ballot = ballot
        self.payloadWords = payloadWords
        self.sequence = sequence
        self.extended = extended
        self.quantCode = quantCode
        self.blockIndex = blockIndex
    }

    init(reader: inout BinaryReader) throws {
        ballot = try reader.readUInt16()
        let packedPayload = try reader.readUInt16()
        payloadWords = packedPayload & 0x0fff
        sequence = UInt8((packedPayload >> 12) & 0x7)
        extended = ((packedPayload >> 15) & 0x1) != 0

        let packedBlock = try reader.readUInt32()
        quantCode = UInt8(packedBlock & 0xff)
        blockIndex = Int(packedBlock >> 8)
    }

    func write(to writer: inout BinaryWriter) {
        writer.append(ballot)
        var packedPayload = payloadWords & 0x0fff
        packedPayload |= UInt16(sequence & UInt8(PyrowaveBitstream.sequenceCountMask)) << 12
        if extended {
            packedPayload |= 1 << 15
        }
        writer.append(packedPayload)
        writer.append((UInt32(blockIndex) << 8) | UInt32(quantCode))
    }
}

struct PyrowaveSequenceHeader: Equatable, Sendable {
    var width: Int
    var height: Int
    var sequence: UInt8
    var totalBlocks: Int
    var chroma: ChromaSubsampling

    init(width: Int, height: Int, sequence: UInt8, totalBlocks: Int, chroma: ChromaSubsampling) throws {
        guard width > 0, width <= (1 << 14),
              height > 0, height <= (1 << 14),
              sequence <= PyrowaveBitstream.sequenceCountMask,
              totalBlocks >= 0, totalBlocks < (1 << 24) else {
            throw PyrowaveError.invalidBitstream("sequence header field out of range")
        }

        self.width = width
        self.height = height
        self.sequence = sequence
        self.totalBlocks = totalBlocks
        self.chroma = chroma
    }

    init(reader: inout BinaryReader) throws {
        let first = try reader.readUInt32()
        width = Int(first & 0x3fff) + 1
        height = Int((first >> 14) & 0x3fff) + 1
        sequence = UInt8((first >> 28) & 0x7)
        guard ((first >> 31) & 0x1) != 0 else {
            throw PyrowaveError.invalidBitstream("sequence header missing extended bit")
        }

        let second = try reader.readUInt32()
        totalBlocks = Int(second & 0x00ff_ffff)
        let code = (second >> 24) & 0x3
        guard code == 0 else {
            throw PyrowaveError.invalidBitstream("unsupported extended header code \(code)")
        }

        let chromaCode = UInt8((second >> 26) & 0x1)
        guard let chroma = ChromaSubsampling(rawValue: chromaCode) else {
            throw PyrowaveError.invalidBitstream("bad chroma code")
        }
        self.chroma = chroma
    }

    func write(to writer: inout BinaryWriter) {
        var first = UInt32(width - 1) & 0x3fff
        first |= (UInt32(height - 1) & 0x3fff) << 14
        first |= UInt32(sequence & UInt8(PyrowaveBitstream.sequenceCountMask)) << 28
        first |= 1 << 31
        writer.append(first)

        var second = UInt32(totalBlocks) & 0x00ff_ffff
        second |= UInt32(chroma.rawValue) << 26
        writer.append(second)
    }
}

struct PyrowaveBlockDescriptor: Equatable, Sendable {
    var blockIndex: Int
    var component: Int
    var level: Int
    var band: Int
    var blockX: Int
    var blockY: Int
    var originX: Int
    var originY: Int
    var validWidth: Int
    var validHeight: Int
}

struct PyrowaveBlockLayout: Sendable {
    let width: Int
    let height: Int
    let chroma: ChromaSubsampling
    let descriptors: [PyrowaveBlockDescriptor]

    init(width: Int, height: Int, chroma: ChromaSubsampling) throws {
        guard width > 0, height > 0 else {
            throw PyrowaveError.invalidDimensions
        }

        self.width = width
        self.height = height
        self.chroma = chroma

        let alignedWidth = max(Self.align(width, to: PyrowaveBitstream.alignment), PyrowaveBitstream.minimumImageSize)
        let alignedHeight = max(Self.align(height, to: PyrowaveBitstream.alignment), PyrowaveBitstream.minimumImageSize)
        var descriptors = [PyrowaveBlockDescriptor]()

        for level in stride(from: PyrowaveBitstream.decompositionLevels - 1, through: 0, by: -1) {
            let subbandWidth = alignedWidth >> (level + 1)
            let subbandHeight = alignedHeight >> (level + 1)
            let blockColumns = (subbandWidth + PyrowaveBitstream.coefficientBlockSize - 1) / PyrowaveBitstream.coefficientBlockSize
            let blockRows = (subbandHeight + PyrowaveBitstream.coefficientBlockSize - 1) / PyrowaveBitstream.coefficientBlockSize

            for component in 0..<PyrowaveBitstream.componentCount {
                if level == 0, component != 0, chroma == .yuv420 {
                    continue
                }

                let firstBand = level == PyrowaveBitstream.decompositionLevels - 1 ? 0 : 1
                for band in firstBand..<PyrowaveBitstream.bandCount {
                    let bandOrigin = Self.bandOrigin(level: level, band: band, subbandWidth: subbandWidth, subbandHeight: subbandHeight)
                    for blockY in 0..<blockRows {
                        for blockX in 0..<blockColumns {
                            let originX = bandOrigin.x + blockX * PyrowaveBitstream.coefficientBlockSize
                            let originY = bandOrigin.y + blockY * PyrowaveBitstream.coefficientBlockSize
                            descriptors.append(PyrowaveBlockDescriptor(
                                blockIndex: descriptors.count,
                                component: component,
                                level: level,
                                band: band,
                                blockX: blockX,
                                blockY: blockY,
                                originX: originX,
                                originY: originY,
                                validWidth: min(PyrowaveBitstream.coefficientBlockSize, subbandWidth - blockX * PyrowaveBitstream.coefficientBlockSize),
                                validHeight: min(PyrowaveBitstream.coefficientBlockSize, subbandHeight - blockY * PyrowaveBitstream.coefficientBlockSize)
                            ))
                        }
                    }
                }
            }
        }

        self.descriptors = descriptors
    }

    static func align(_ value: Int, to alignment: Int) -> Int {
        (value + alignment - 1) & ~(alignment - 1)
    }

    static func bandOrigin(level: Int, band: Int, subbandWidth: Int, subbandHeight: Int) -> (x: Int, y: Int) {
        if level == PyrowaveBitstream.decompositionLevels - 1, band == 0 {
            return (0, 0)
        }

        switch band {
        case 1:
            return (subbandWidth, 0)
        case 2:
            return (0, subbandHeight)
        case 3:
            return (subbandWidth, subbandHeight)
        default:
            return (0, 0)
        }
    }
}
