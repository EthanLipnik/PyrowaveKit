import Foundation

struct PyrowaveQuantStats: Equatable {
    var squareErrorFP16: UInt16
    var encodeCostBits: UInt16

    init(squareError: Float, encodeCostBits: Int) {
        let clampedError = min(max(squareError, 0), 60_000)
        self.squareErrorFP16 = Float16(clampedError).bitPattern
        self.encodeCostBits = UInt16(clamping: encodeCostBits)
    }

    var squareError: Float {
        Float(Float16(bitPattern: squareErrorFP16))
    }
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

    func distortion(threshold: Int) -> Float {
        let index = min(max(threshold, 0), Self.candidateCount - 1)
        return eightByEightStats.reduce(Float(0)) { partial, block in
            partial + block.stats[index].squareError
        }
    }

    func packetByteCost(threshold: Int) -> Int {
        packetByteCosts[min(max(threshold, 0), Self.candidateCount - 1)]
    }
}

enum PyrowaveRateController {
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
        qScaleCodes: [UInt8]? = nil
    ) throws -> PyrowaveRateControlBlock {
        var stats = [PyrowaveBlockStats]()
        stats.reserveCapacity(16)

        for tileY in 0..<4 {
            for tileX in 0..<4 {
                stats.append(make8x8Stats(
                    coefficients: coefficients,
                    stride: stride,
                    originX: originX + tileX * 8,
                    originY: originY + tileY * 8,
                    validWidth: max(0, min(8, validWidth - tileX * 8)),
                    validHeight: max(0, min(8, validHeight - tileY * 8))
                ))
            }
        }

        var packetByteCosts = [Int]()
        packetByteCosts.reserveCapacity(PyrowaveBlockStats.candidateCount)
        for threshold in 0..<PyrowaveBlockStats.candidateCount {
            let packet = try PyrowaveCoefficientBlockCodec.encodeBlock(
                blockIndex: blockIndex,
                coefficients: coefficients,
                stride: stride,
                originX: originX,
                originY: originY,
                validWidth: validWidth,
                validHeight: validHeight,
                threshold: threshold,
                quantCode: quantCode,
                qScaleCode: qScaleCode,
                qScaleCodes: qScaleCodes
            )
            packetByteCosts.append(packet?.count ?? 0)
        }

        return PyrowaveRateControlBlock(
            blockIndex: blockIndex,
            eightByEightStats: stats,
            packetByteCosts: packetByteCosts
        )
    }

    static func selectThresholds(
        blocksByPlane: [[PyrowaveRateControlBlock]],
        fixedHeaderBytes: Int,
        maximumEncodedBytes: Int
    ) -> [[Int]]? {
        var thresholds = blocksByPlane.map { Array(repeating: 0, count: $0.count) }
        var currentBytes = estimateFrameBytes(
            blocksByPlane: blocksByPlane,
            thresholdsByPlane: thresholds,
            fixedHeaderBytes: fixedHeaderBytes
        )

        if currentBytes <= maximumEncodedBytes {
            return thresholds
        }

        while currentBytes > maximumEncodedBytes {
            var best: (plane: Int, block: Int, nextThreshold: Int, saving: Int, score: Float)?

            for (planeIndex, blocks) in blocksByPlane.enumerated() {
                for (blockIndex, block) in blocks.enumerated() {
                    let currentThreshold = thresholds[planeIndex][blockIndex]
                    guard currentThreshold + 1 < PyrowaveRateControlBlock.candidateCount else {
                        continue
                    }

                    let currentCost = block.packetByteCost(threshold: currentThreshold)
                    var nextThreshold = currentThreshold + 1
                    while nextThreshold < PyrowaveRateControlBlock.candidateCount &&
                        block.packetByteCost(threshold: nextThreshold) == currentCost {
                        nextThreshold += 1
                    }
                    guard nextThreshold < PyrowaveRateControlBlock.candidateCount else {
                        continue
                    }

                    let nextCost = block.packetByteCost(threshold: nextThreshold)
                    let saving = currentCost - nextCost
                    guard saving > 0 else {
                        continue
                    }

                    let distortionDelta = max(
                        0,
                        block.distortion(threshold: nextThreshold) - block.distortion(threshold: currentThreshold)
                    )
                    let score = distortionDelta / Float(saving)
                    if best == nil ||
                        score < best!.score ||
                        (score == best!.score && saving > best!.saving) ||
                        (score == best!.score && saving == best!.saving && planeIndex < best!.plane) ||
                        (score == best!.score && saving == best!.saving && planeIndex == best!.plane && blockIndex < best!.block) {
                        best = (planeIndex, blockIndex, nextThreshold, saving, score)
                    }
                }
            }

            guard let operation = best else {
                return nil
            }

            thresholds[operation.plane][operation.block] = operation.nextThreshold
            currentBytes -= operation.saving
        }

        return thresholds
    }

    static func estimateFrameBytes(
        blocksByPlane: [[PyrowaveRateControlBlock]],
        thresholdsByPlane: [[Int]],
        fixedHeaderBytes: Int
    ) -> Int {
        var byteCount = fixedHeaderBytes
        for (blocks, thresholds) in zip(blocksByPlane, thresholdsByPlane) {
            for (block, threshold) in zip(blocks, thresholds) {
                byteCount += block.packetByteCost(threshold: threshold)
            }
        }
        return byteCount
    }

    private static func make8x8Stats(
        coefficients: [Int16],
        stride: Int,
        originX: Int,
        originY: Int,
        validWidth: Int,
        validHeight: Int
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

        for threshold in 0..<PyrowaveBlockStats.candidateCount {
            var squareError = Float(0)
            var encodeCostBits = 0
            var retainedValues = 0

            for y in 0..<validHeight {
                for x in 0..<validWidth {
                    let value = Int(coefficients[(originY + y) * stride + originX + x])
                    let magnitude = abs(value)
                    if magnitude > threshold {
                        retainedValues += 1
                        encodeCostBits += significantBitCount(magnitude)
                    } else {
                        squareError += Float(magnitude * magnitude)
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
