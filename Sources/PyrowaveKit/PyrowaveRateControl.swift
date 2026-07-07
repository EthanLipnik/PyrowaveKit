import Foundation

struct PyrowaveQuantStats: Equatable {
    var squareErrorFP16: UInt16
    var encodeCostBits: UInt16

    init(squareError: Float, encodeCostBits: Int) {
        let clampedError = min(max(squareError, 0), 60_000)
        self.squareErrorFP16 = PyrowaveHalfFloat.bitPattern(clampedError)
        self.encodeCostBits = UInt16(clamping: encodeCostBits)
    }

    var squareError: Float {
        PyrowaveHalfFloat.float(squareErrorFP16)
    }

    static func quantizedSquareError(_ squareError: Float) -> Float {
        PyrowaveHalfFloat.float(PyrowaveHalfFloat.bitPattern(min(max(squareError, 0), 60_000)))
    }
}

private enum PyrowaveHalfFloat {
    static func bitPattern(_ value: Float) -> UInt16 {
        #if arch(x86_64)
        positiveFiniteBitPattern(value)
        #else
        Float16(value).bitPattern
        #endif
    }

    static func float(_ bitPattern: UInt16) -> Float {
        #if arch(x86_64)
        positiveFiniteFloat(bitPattern)
        #else
        Float(Float16(bitPattern: bitPattern))
        #endif
    }

    #if arch(x86_64)
    private static func positiveFiniteBitPattern(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let exponent = Int((bits >> 23) & 0xff) - 127 + 15
        let mantissa = bits & 0x7f_ffff

        if exponent <= 0 {
            if exponent < -10 {
                return 0
            }

            let significand = mantissa | 0x80_0000
            let shift = UInt32(14 - exponent)
            var halfMantissa = significand >> shift
            let remainderMask = (UInt32(1) << shift) - 1
            let remainder = significand & remainderMask
            let halfway = UInt32(1) << (shift - 1)
            if remainder > halfway || (remainder == halfway && (halfMantissa & 1) == 1) {
                halfMantissa += 1
            }
            return UInt16(halfMantissa)
        }

        var half = UInt32(exponent << 10) | (mantissa >> 13)
        let remainder = mantissa & 0x1fff
        if remainder > 0x1000 || (remainder == 0x1000 && (half & 1) == 1) {
            half += 1
        }

        return UInt16(min(half, 0x7bff))
    }

    private static func positiveFiniteFloat(_ bitPattern: UInt16) -> Float {
        let exponent = UInt32((bitPattern >> 10) & 0x1f)
        var mantissa = UInt32(bitPattern & 0x03ff)

        if exponent == 0 {
            if mantissa == 0 {
                return 0
            }

            var adjustedExponent = -14
            while (mantissa & 0x0400) == 0 {
                mantissa <<= 1
                adjustedExponent -= 1
            }
            mantissa &= 0x03ff

            let floatExponent = UInt32(adjustedExponent + 127) << 23
            return Float(bitPattern: floatExponent | (mantissa << 13))
        }

        let floatExponent = (exponent - 15 + 127) << 23
        return Float(bitPattern: floatExponent | (mantissa << 13))
    }
    #endif
}

struct PyrowaveBlockStats: Equatable {
    static let candidateCount = 15
    static let packedByteCount = 64

    var numPlanes: UInt32
    var stats: [PyrowaveQuantStats]

    init(numPlanes: Int, stats: [PyrowaveQuantStats]) {
        precondition(stats.count == Self.candidateCount)
        self.numPlanes = UInt32(clamping: numPlanes)
        self.stats = stats
    }

    func packedData() -> Data {
        var writer = BinaryWriter()
        writer.append(numPlanes)
        for stat in stats {
            writer.append(stat.squareErrorFP16)
            writer.append(stat.encodeCostBits)
        }
        return writer.data
    }
}

struct PyrowaveRateControlBlock: Equatable {
    static let candidateCount = PyrowaveBlockStats.candidateCount

    var blockIndex: Int
    var eightByEightStats: [PyrowaveBlockStats]
    var packetByteCosts: [Int]

    func distortion(quantLevel: Int) -> Float {
        let index = min(max(quantLevel, 0), Self.candidateCount - 1)
        return eightByEightStats.reduce(Float(0)) { partial, block in
            partial + block.stats[index].squareError
        }
    }

    func packetByteCost(quantLevel: Int) -> Int {
        packetByteCosts[min(max(quantLevel, 0), Self.candidateCount - 1)]
    }
}

enum PyrowaveRateController {
    private static let bucketCount = 128
    private static let bucketClusterWidth = 16

    struct RDOperation: Equatable {
        var bucket: Int
        var planeIndex: Int
        var blockIndex: Int
        var quantLevel: Int
        var saving: Int
    }

    static func makeBlock(
        blockIndex: Int,
        coefficients: [Int16],
        stride: Int,
        originX: Int,
        originY: Int,
        validWidth: Int,
        validHeight: Int,
        quantCode: UInt8,
        qScaleCode: UInt8 = PyrowaveQuantization.identityQScaleCode,
        qScaleCodes: [UInt8]? = nil,
        rdoDistortionScale: Float = 1.0
    ) throws -> PyrowaveRateControlBlock {
        if let qScaleCodes, qScaleCodes.count != 16 {
            throw PyrowaveError.invalidBitstream("8x8 quant scale table must contain 16 entries")
        }

        var stats = [PyrowaveBlockStats]()
        stats.reserveCapacity(16)

        for tileY in 0..<4 {
            for tileX in 0..<4 {
                let tileIndex = tileY * 4 + tileX
                stats.append(make8x8Stats(
                    coefficients: coefficients,
                    stride: stride,
                    originX: originX + tileX * 8,
                    originY: originY + tileY * 8,
                    validWidth: max(0, min(8, validWidth - tileX * 8)),
                    validHeight: max(0, min(8, validHeight - tileY * 8)),
                    quantCode: quantCode,
                    qScaleCode: qScaleCodes?[tileIndex] ?? qScaleCode,
                    rdoDistortionScale: rdoDistortionScale
                ))
            }
        }

        let packetByteCosts = try makePacketByteCosts(
            blockIndex: blockIndex,
            coefficients: coefficients,
            stride: stride,
            originX: originX,
            originY: originY,
            validWidth: validWidth,
            validHeight: validHeight,
            quantCode: quantCode,
            qScaleCode: qScaleCode,
            qScaleCodes: qScaleCodes
        )

        return PyrowaveRateControlBlock(
            blockIndex: blockIndex,
            eightByEightStats: stats,
            packetByteCosts: packetByteCosts
        )
    }

    static func makePacketByteCosts(
        blockIndex: Int,
        coefficients: [Int16],
        stride: Int,
        originX: Int,
        originY: Int,
        validWidth: Int,
        validHeight: Int,
        quantCode: UInt8,
        qScaleCode: UInt8 = PyrowaveQuantization.identityQScaleCode,
        qScaleCodes: [UInt8]? = nil
    ) throws -> [Int] {
        var packetByteCosts = [Int]()
        packetByteCosts.reserveCapacity(PyrowaveBlockStats.candidateCount)
        for quantLevel in 0..<PyrowaveBlockStats.candidateCount {
            let packet = try PyrowaveCoefficientBlockCodec.encodeBlock(
                blockIndex: blockIndex,
                coefficients: coefficients,
                stride: stride,
                originX: originX,
                originY: originY,
                validWidth: validWidth,
                validHeight: validHeight,
                threshold: 0,
                quantLevel: quantLevel,
                quantCode: quantCode,
                qScaleCode: qScaleCode,
                qScaleCodes: qScaleCodes
            )
            packetByteCosts.append(packet?.count ?? 0)
        }
        return packetByteCosts
    }

    static func cumulativeBucketSavings(
        blocksByPlane: [[PyrowaveRateControlBlock]],
        bucketIndicesByPlane: [[[Int]]]? = nil
    ) -> [Int] {
        var savings = Array(repeating: 0, count: bucketCount)
        for (planeIndex, blocks) in blocksByPlane.enumerated() {
            for (blockIndex, block) in blocks.enumerated() {
                let bucketIndices: [Int]
                if let bucketIndicesByPlane,
                   planeIndex < bucketIndicesByPlane.count,
                   blockIndex < bucketIndicesByPlane[planeIndex].count,
                   bucketIndicesByPlane[planeIndex][blockIndex].count == PyrowaveRateControlBlock.candidateCount {
                    bucketIndices = bucketIndicesByPlane[planeIndex][blockIndex]
                } else {
                    bucketIndices = inclusiveBucketIndices(for: block)
                }

                for quantLevel in 1..<PyrowaveRateControlBlock.candidateCount {
                    let saving = block.packetByteCost(quantLevel: quantLevel - 1) - block.packetByteCost(quantLevel: quantLevel)
                    guard saving > 0 else {
                        continue
                    }
                    savings[bucketIndices[quantLevel]] += saving
                }
            }
        }

        var running = 0
        return savings.map {
            running += $0
            return running
        }
    }

    static func estimateFrameBytes(
        blocksByPlane: [[PyrowaveRateControlBlock]],
        thresholdsByPlane: [[Int]],
        fixedHeaderBytes: Int
    ) -> Int {
        var byteCount = fixedHeaderBytes
        for (blocks, thresholds) in zip(blocksByPlane, thresholdsByPlane) {
            for (block, threshold) in zip(blocks, thresholds) {
                byteCount += block.packetByteCost(quantLevel: threshold)
            }
        }
        return byteCount
    }

    static func makeRDOperations(
        blocksByPlane: [[PyrowaveRateControlBlock]],
        bucketIndicesByPlane: [[[Int]]]? = nil
    ) -> [RDOperation] {
        var buckets = Array(repeating: [RDOperation](), count: bucketCount)

        for (planeIndex, blocks) in blocksByPlane.enumerated() {
            for (blockIndex, block) in blocks.enumerated() {
                let providedBucketIndices: [Int]?
                if let bucketIndicesByPlane,
                   planeIndex < bucketIndicesByPlane.count,
                   blockIndex < bucketIndicesByPlane[planeIndex].count,
                   bucketIndicesByPlane[planeIndex][blockIndex].count == PyrowaveRateControlBlock.candidateCount {
                    providedBucketIndices = bucketIndicesByPlane[planeIndex][blockIndex]
                } else {
                    providedBucketIndices = nil
                }
                let bucketIndices = providedBucketIndices ?? inclusiveBucketIndices(for: block)
                for quantLevel in 1..<PyrowaveRateControlBlock.candidateCount {
                    let saving = block.packetByteCost(quantLevel: quantLevel - 1) - block.packetByteCost(quantLevel: quantLevel)
                    guard saving > 0 else {
                        continue
                    }

                    let bucket = bucketIndices[quantLevel]
                    buckets[bucket].append(RDOperation(
                        bucket: bucket,
                        planeIndex: planeIndex,
                        blockIndex: blockIndex,
                        quantLevel: quantLevel,
                        saving: saving
                    ))
                }
            }
        }

        return buckets.flatMap { bucket in
            bucket.sorted {
                if $0.planeIndex != $1.planeIndex {
                    return $0.planeIndex < $1.planeIndex
                }
                if $0.blockIndex != $1.blockIndex {
                    return $0.blockIndex < $1.blockIndex
                }
                return $0.quantLevel < $1.quantLevel
            }
        }
    }

    static func distortionBucketIndex(distortion: Float, cost: Int, baseDistortion: Float, baseCost: Int) -> Int {
        guard cost != baseCost else {
            return 0
        }

        let distortionDelta = max(distortion - baseDistortion, 0)
        let costSaving = max(Float(baseCost - cost), Float.leastNonzeroMagnitude)
        let index = 60.0 + 2.0 * log2(distortionDelta / costSaving)
        guard index.isFinite else {
            return 0
        }
        return min(bucketCount - 1, max(0, Int((index + 0.5).rounded(.down))))
    }

    static func inclusiveBucketIndices(for block: PyrowaveRateControlBlock) -> [Int] {
        let baseDistortion = block.distortion(quantLevel: 0)
        let baseCost = block.packetByteCost(quantLevel: 0)
        var raw = (0..<PyrowaveRateControlBlock.candidateCount).map { quantLevel in
            quantLevel == 0 ? 0 : distortionBucketIndex(
                distortion: block.distortion(quantLevel: quantLevel),
                cost: block.packetByteCost(quantLevel: quantLevel),
                baseDistortion: baseDistortion,
                baseCost: baseCost
            )
        }

        for quantLevel in raw.indices {
            raw[quantLevel] = min(raw[quantLevel], bucketCount - bucketClusterWidth + quantLevel)
        }

        var clustered = raw
        for quantLevel in clustered.indices {
            var bucket = raw[quantLevel]
            for previous in 0..<quantLevel {
                bucket = max(bucket, raw[previous] + quantLevel - previous)
            }
            clustered[quantLevel] = min(bucketCount - 1, bucket)
        }
        return clustered
    }

    private static func make8x8Stats(
        coefficients: [Int16],
        stride: Int,
        originX: Int,
        originY: Int,
        validWidth: Int,
        validHeight: Int,
        quantCode: UInt8,
        qScaleCode: UInt8,
        rdoDistortionScale: Float
    ) -> PyrowaveBlockStats {
        guard validWidth > 0, validHeight > 0 else {
            return PyrowaveBlockStats(
                numPlanes: 0,
                stats: Array(repeating: PyrowaveQuantStats(squareError: 0, encodeCostBits: 0), count: PyrowaveBlockStats.candidateCount)
            )
        }

        var maximumMagnitude = 0
        for y in 0..<validHeight {
            for x in 0..<validWidth {
                let magnitude = abs(Int(coefficients[(originY + y) * stride + originX + x]))
                maximumMagnitude = max(maximumMagnitude, magnitude)
            }
        }

        let numPlanes = maximumMagnitude == 0 ? 0 : min(14, significantBitCount(maximumMagnitude))
        var stats = [PyrowaveQuantStats]()
        stats.reserveCapacity(PyrowaveBlockStats.candidateCount)
        let coefficientToSampleScale = PyrowaveQuantization.decodeBlockScale(quantCode) * PyrowaveQuantization.decode8x8Scale(qScaleCode)
        let distortionWeight = coefficientToSampleScale * coefficientToSampleScale * rdoDistortionScale

        for quantLevel in 0..<PyrowaveBlockStats.candidateCount {
            var squareError = Float(0)
            var encodeCostBits = 0
            var retainedValues = 0

            for y in 0..<validHeight {
                for x in 0..<validWidth {
                    let value = Int(coefficients[(originY + y) * stride + originX + x])
                    let magnitude = abs(value)
                    let retainedMagnitude = magnitude >> quantLevel
                    if retainedMagnitude != 0 {
                        retainedValues += 1
                        encodeCostBits += significantBitCount(retainedMagnitude)
                        if quantLevel != 0 {
                            let reconstructedMagnitude = (Float(retainedMagnitude) + 0.5) * Float(1 << quantLevel)
                            let delta = Float(magnitude) - reconstructedMagnitude
                            squareError += delta * delta * distortionWeight
                        }
                    } else {
                        squareError += Float(magnitude * magnitude) * distortionWeight
                    }
                }
            }

            if retainedValues != 0 {
                encodeCostBits += retainedValues
            }
            stats.append(PyrowaveQuantStats(squareError: squareError, encodeCostBits: encodeCostBits))
        }

        return PyrowaveBlockStats(numPlanes: numPlanes, stats: stats)
    }

    private static func significantBitCount(_ value: Int) -> Int {
        value == 0 ? 0 : Int.bitWidth - value.leadingZeroBitCount
    }
}
