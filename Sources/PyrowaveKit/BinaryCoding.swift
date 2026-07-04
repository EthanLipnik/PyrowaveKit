import Foundation

struct BinaryWriter {
    private(set) var data = Data()

    mutating func append(_ value: UInt8) {
        data.append(value)
    }

    mutating func append(bytes: [UInt8]) {
        data.append(contentsOf: bytes)
    }

    mutating func append(data payload: Data) {
        data.append(payload)
    }

    mutating func append(_ value: UInt16) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }

    mutating func append(_ value: UInt32) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }

    mutating func append(_ value: Int16) {
        append(UInt16(bitPattern: value))
    }

    mutating func append(_ value: Float) {
        append(value.bitPattern)
    }

    mutating func append(contentsOf values: [Int16]) {
        data.reserveCapacity(data.count + values.count * 2)
        for value in values {
            append(value)
        }
    }
}

struct BinaryReader {
    let data: Data
    private(set) var offset = 0

    init(_ data: Data) {
        self.data = data
    }

    mutating func readUInt8() throws -> UInt8 {
        guard offset < data.count else { throw PyrowaveError.truncatedInput }
        defer { offset += 1 }
        return data[offset]
    }

    mutating func readUInt16() throws -> UInt16 {
        guard offset + 2 <= data.count else { throw PyrowaveError.truncatedInput }
        let value = data[offset..<offset + 2].enumerated().reduce(UInt16(0)) { result, element in
            result | (UInt16(element.element) << UInt16(element.offset * 8))
        }
        offset += 2
        return value
    }

    mutating func readUInt32() throws -> UInt32 {
        guard offset + 4 <= data.count else { throw PyrowaveError.truncatedInput }
        let value = data[offset..<offset + 4].enumerated().reduce(UInt32(0)) { result, element in
            result | (UInt32(element.element) << UInt32(element.offset * 8))
        }
        offset += 4
        return value
    }

    mutating func readInt16() throws -> Int16 {
        Int16(bitPattern: try readUInt16())
    }

    mutating func readFloat() throws -> Float {
        Float(bitPattern: try readUInt32())
    }

    mutating func readInt16Array(count: Int) throws -> [Int16] {
        guard count >= 0 else { throw PyrowaveError.invalidBitstream("negative array count") }
        var values = [Int16]()
        values.reserveCapacity(count)
        for _ in 0..<count {
            values.append(try readInt16())
        }
        return values
    }
}
