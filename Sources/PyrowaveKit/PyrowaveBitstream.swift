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

enum PyrowaveQuantization {
    static let maxScaleExp = 4
    static let identityQScaleCode: UInt8 = 6

    static func decodeBlockScale(_ quantCode: UInt8) -> Float {
        let exponent = maxScaleExp - Int(quantCode >> 3)
        let mantissa = Int(quantCode & 0x7)
        return (1.0 / (8.0 * 1024.0 * 1024.0)) * Float((8 + mantissa) * (1 << (20 + exponent)))
    }

    static func encodeBlockScale(_ scale: Float) throws -> UInt8 {
        guard scale > 0, scale.isFinite else {
            throw PyrowaveError.invalidBitstream("bad quantization scale")
        }

        let bits = scale.bitPattern
        var exponent = Int((bits >> 23) & 0xff) - 127 - maxScaleExp
        let mantissa = Int((bits >> 20) & 0x7)
        exponent = -exponent
        guard exponent >= 0, exponent <= 20 else {
            throw PyrowaveError.invalidBitstream("quantization scale out of representable range")
        }
        return UInt8((exponent << 3) | mantissa)
    }

    static func decode8x8Scale(_ code: UInt8) -> Float {
        Float(code) / 8.0 + 0.25
    }

    static func encode8x8Scale(_ scale: Float) -> UInt8 {
        UInt8(clamping: Int(ceil((scale - 0.25) * 8.0)))
    }

    static func dequantize(coefficient: Int16, quantCode: UInt8, qScaleCode: UInt8) -> Float {
        var value = Float(coefficient)
        if value > 0 {
            value += 0.5
        } else if value < 0 {
            value -= 0.5
        }
        return value * decodeBlockScale(quantCode) * decode8x8Scale(qScaleCode)
    }
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

struct PyrowaveCoefficientBlockCodec {
    private static let subblockCount = 8
    private static let pixelsPerSubblock = 8

    struct DecodedBlock {
        var blockIndex: Int
        var quantCode: UInt8
        var qScaleCodes: [UInt8]
        var coefficients: [(offset: UInt16, value: Int16)]
    }

    static func encodeBlock(
        blockIndex: Int,
        coefficients: [Int16],
        stride: Int,
        originX: Int,
        originY: Int,
        validWidth: Int,
        validHeight: Int,
        threshold: Int,
        sequence: UInt8 = 0,
        quantCode: UInt8 = 0,
        qScaleCode: UInt8 = PyrowaveQuantization.identityQScaleCode
    ) throws -> Data? {
        var blockValues = Array(repeating: Int16(0), count: PyrowaveBitstream.coefficientBlockSize * PyrowaveBitstream.coefficientBlockSize)
        var ballot = UInt16(0)

        for y in 0..<validHeight {
            for x in 0..<validWidth {
                let value = coefficients[(originY + y) * stride + originX + x]
                if abs(Int(value)) > threshold {
                    blockValues[y * PyrowaveBitstream.coefficientBlockSize + x] = value
                    let smallBlock = (y / PyrowaveBitstream.smallBlockSize) * 4 + (x / PyrowaveBitstream.smallBlockSize)
                    ballot |= UInt16(1) << UInt16(smallBlock)
                }
            }
        }

        guard ballot != 0 else {
            return nil
        }

        var codeWords = [UInt16]()
        var qScales = [UInt8]()
        var magnitudePayload = [UInt8]()
        var signPositions = [(offset: UInt16, negative: Bool)]()

        for smallBlock in 0..<16 where (ballot & (UInt16(1) << UInt16(smallBlock))) != 0 {
            let smallOriginX = (smallBlock % 4) * PyrowaveBitstream.smallBlockSize
            let smallOriginY = (smallBlock / 4) * PyrowaveBitstream.smallBlockSize
            var bitWidths = Array(repeating: 0, count: subblockCount)
            var magnitudes = Array(repeating: 0, count: PyrowaveBitstream.smallBlockSize * PyrowaveBitstream.smallBlockSize)
            var negatives = Array(repeating: false, count: PyrowaveBitstream.smallBlockSize * PyrowaveBitstream.smallBlockSize)

            for subblock in 0..<subblockCount {
                var maxMagnitude = 0
                for pixel in 0..<pixelsPerSubblock {
                    let coord = coordinateIn8x8(subblock: subblock, pixel: pixel)
                    let x = smallOriginX + coord.x
                    let y = smallOriginY + coord.y
                    let value = blockValues[y * PyrowaveBitstream.coefficientBlockSize + x]
                    let magnitude = abs(Int(value))
                    magnitudes[subblock * pixelsPerSubblock + pixel] = magnitude
                    negatives[subblock * pixelsPerSubblock + pixel] = value < 0
                    maxMagnitude = max(maxMagnitude, magnitude)
                }
                bitWidths[subblock] = bitWidth(maxMagnitude)
            }

            let basePlanes = max(0, (bitWidths.max() ?? 0) - 3)
            var codeWord = UInt16(0)
            for subblock in 0..<subblockCount {
                let encodedPlanes = max(bitWidths[subblock], basePlanes)
                let twoBitCode = min(3, max(0, encodedPlanes - basePlanes))
                codeWord |= UInt16(twoBitCode) << UInt16(2 * subblock)

                for plane in 0..<encodedPlanes {
                    let bit = encodedPlanes - plane - 1
                    var byte = UInt8(0)
                    for pixel in 0..<pixelsPerSubblock {
                        let magnitude = magnitudes[subblock * pixelsPerSubblock + pixel]
                        if ((magnitude >> bit) & 1) != 0 {
                            byte |= UInt8(1) << UInt8(pixel)
                        }
                    }
                    magnitudePayload.append(byte)
                }
            }

            codeWords.append(codeWord)
            qScales.append((qScaleCode << 4) | UInt8(basePlanes & 0x0f))

            for subblock in 0..<subblockCount {
                for pixel in 0..<pixelsPerSubblock {
                    let magnitude = magnitudes[subblock * pixelsPerSubblock + pixel]
                    if magnitude != 0 {
                        let coord = coordinateIn8x8(subblock: subblock, pixel: pixel)
                        let x = smallOriginX + coord.x
                        let y = smallOriginY + coord.y
                        signPositions.append((offset: UInt16(y * PyrowaveBitstream.coefficientBlockSize + x), negative: negatives[subblock * pixelsPerSubblock + pixel]))
                    }
                }
            }
        }

        var signPayload = Array(repeating: UInt8(0), count: (signPositions.count + 7) / 8)
        for (index, sign) in signPositions.enumerated() where sign.negative {
            signPayload[index / 8] |= UInt8(1) << UInt8(index & 7)
        }

        let unpaddedSize = 8 + codeWords.count * 2 + qScales.count + magnitudePayload.count + signPayload.count
        let payloadWords = UInt16((unpaddedSize + 3) / 4)
        let header = try PyrowavePacketHeader(
            ballot: ballot,
            payloadWords: payloadWords,
            sequence: sequence,
            extended: false,
            quantCode: quantCode,
            blockIndex: blockIndex
        )

        var writer = BinaryWriter()
        header.write(to: &writer)
        for codeWord in codeWords {
            writer.append(codeWord)
        }
        for qScale in qScales {
            writer.append(qScale)
        }
        writer.append(bytes: magnitudePayload)
        writer.append(bytes: signPayload)
        while writer.data.count % 4 != 0 {
            writer.append(UInt8(0))
        }
        return writer.data
    }

    static func decodeBlock(reader: inout BinaryReader) throws -> DecodedBlock {
        let blockStart = reader.offset
        let header = try PyrowavePacketHeader(reader: &reader)
        guard !header.extended else {
            throw PyrowaveError.invalidBitstream("coefficient decoder received extended packet")
        }
        guard header.ballot != 0 else {
            return DecodedBlock(blockIndex: header.blockIndex, quantCode: header.quantCode, qScaleCodes: [], coefficients: [])
        }

        let payloadEnd = blockStart + Int(header.payloadWords) * 4
        guard payloadEnd <= reader.data.count else {
            throw PyrowaveError.truncatedInput
        }

        let activeBlockCount = header.ballot.nonzeroBitCount
        var codeWords = [UInt16]()
        codeWords.reserveCapacity(activeBlockCount)
        for _ in 0..<activeBlockCount {
            codeWords.append(try reader.readUInt16())
        }

        var qScales = [UInt8]()
        var qScaleCodes = [UInt8]()
        qScales.reserveCapacity(activeBlockCount)
        qScaleCodes.reserveCapacity(activeBlockCount)
        for _ in 0..<activeBlockCount {
            let qScale = try reader.readUInt8()
            qScales.append(qScale)
            qScaleCodes.append(qScale >> 4)
        }

        var coefficients = Array(repeating: Int16(0), count: PyrowaveBitstream.coefficientBlockSize * PyrowaveBitstream.coefficientBlockSize)
        var signOffsets = [UInt16]()
        signOffsets.reserveCapacity(PyrowaveBitstream.coefficientBlockSize * PyrowaveBitstream.coefficientBlockSize)

        var compactIndex = 0
        for smallBlock in 0..<16 where (header.ballot & (UInt16(1) << UInt16(smallBlock))) != 0 {
            let smallOriginX = (smallBlock % 4) * PyrowaveBitstream.smallBlockSize
            let smallOriginY = (smallBlock / 4) * PyrowaveBitstream.smallBlockSize
            let codeWord = codeWords[compactIndex]
            let basePlanes = Int(qScales[compactIndex] & 0x0f)

            for subblock in 0..<subblockCount {
                let encodedPlanes = Int((codeWord >> UInt16(2 * subblock)) & 0x3) + basePlanes
                var magnitudes = Array(repeating: 0, count: pixelsPerSubblock)

                for _ in 0..<encodedPlanes {
                    guard reader.offset < payloadEnd else {
                        throw PyrowaveError.truncatedInput
                    }
                    let byte = try reader.readUInt8()
                    for pixel in 0..<pixelsPerSubblock {
                        magnitudes[pixel] <<= 1
                        magnitudes[pixel] |= Int((byte >> UInt8(pixel)) & 1)
                    }
                }

                for pixel in 0..<pixelsPerSubblock where magnitudes[pixel] != 0 {
                    let coord = coordinateIn8x8(subblock: subblock, pixel: pixel)
                    let x = smallOriginX + coord.x
                    let y = smallOriginY + coord.y
                    let offset = UInt16(y * PyrowaveBitstream.coefficientBlockSize + x)
                    coefficients[Int(offset)] = Int16(magnitudes[pixel])
                    signOffsets.append(offset)
                }
            }

            compactIndex += 1
        }

        let signStart = reader.offset
        let signByteCount = (signOffsets.count + 7) / 8
        guard signStart + signByteCount <= payloadEnd else {
            throw PyrowaveError.truncatedInput
        }

        for signIndex in signOffsets.indices {
            let signByteOffset = signIndex / 8
            let bit = UInt8(signIndex & 7)
            let signByte = reader.data[signStart + signByteOffset]
            if ((signByte >> bit) & 1) != 0 {
                let offset = Int(signOffsets[signIndex])
                coefficients[offset] = -coefficients[offset]
            }
        }

        try reader.seek(to: payloadEnd)
        let entries = coefficients.enumerated().compactMap { index, value -> (offset: UInt16, value: Int16)? in
            value == 0 ? nil : (UInt16(index), value)
        }
        return DecodedBlock(blockIndex: header.blockIndex, quantCode: header.quantCode, qScaleCodes: qScaleCodes, coefficients: entries)
    }

    private static func coordinateIn8x8(subblock: Int, pixel: Int) -> (x: Int, y: Int) {
        let x = (subblock / 4) * 4 + (pixel >> 1)
        let y = (subblock % 4) * 2 + (pixel & 1)
        return (x, y)
    }

    private static func bitWidth(_ value: Int) -> Int {
        guard value > 0 else {
            return 0
        }
        return Int.bitWidth - value.leadingZeroBitCount
    }
}
