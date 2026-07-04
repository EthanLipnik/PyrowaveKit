import Foundation
import Testing
@testable import PyrowaveKit

@Test func roundTripSynthetic420() throws {
    let frame = try TestFrames.synthetic420(width: 160, height: 96)
    let codec = PyrowaveCodec()
    let encoded = try codec.encode(frame, configuration: CodecConfiguration(quantizationStep: 1.0 / 2048.0))
    let decoded = try codec.decode(encoded)
    let metrics = try Metrics.compare(frame, decoded)

    #expect(decoded.width == frame.width)
    #expect(decoded.height == frame.height)
    #expect(decoded.chroma == .yuv420)
    #expect(metrics.y.psnr > 46.0)
    #expect(metrics.weightedPSNR > 44.0)
}

@Test func waveletPaddingUsesMirroredRepeat() throws {
    let plane = try Plane8(width: 3, height: 2, data: [
        0, 50, 100,
        150, 200, 250
    ])
    let padded = Wavelet.padPlane(plane, paddedWidth: 6, paddedHeight: 5)
    let denormalized = padded.samples.map { UInt8((($0 + 0.5) * 255.0).rounded()) }

    #expect(padded.width == 6)
    #expect(padded.height == 5)
    #expect(Array(denormalized[0..<6]) == [0, 50, 100, 50, 0, 50])
    #expect(Array(denormalized[6..<12]) == [150, 200, 250, 200, 150, 200])
    #expect(Array(denormalized[12..<18]) == [0, 50, 100, 50, 0, 50])
}

@Test func roundTripNonAlignedSynthetic420UsesMirroredPadding() throws {
    let frame = try TestFrames.synthetic420(width: 130, height: 74)
    let codec = PyrowaveCodec(useMetalAcceleration: false)
    let encoded = try codec.encode(frame, configuration: CodecConfiguration(quantizationStep: 1.0 / 2048.0))
    let decoded = try codec.decode(encoded)
    let metrics = try Metrics.compare(frame, decoded)

    #expect(decoded.width == frame.width)
    #expect(decoded.height == frame.height)
    #expect(decoded.chroma == .yuv420)
    #expect(metrics.weightedPSNR > 36.0)
}

@Test func yuv4mpegReadWrite() throws {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("sample.y4m")
    let frame = try TestFrames.synthetic420(width: 64, height: 64)

    var writer = try YUV4MPEGWriter(url: url, width: frame.width, height: frame.height, chroma: frame.chroma)
    try writer.writeFrame(frame)

    var reader = try YUV4MPEGReader(url: url)
    let decoded = try #require(try reader.readFrame())
    #expect(decoded == frame)
}

@Test func metalBackendCompilesKernelsWhenDeviceExists() throws {
    #if canImport(Metal)
    do {
        let backend = try MetalPyrowaveBackend()
        _ = try backend.makeFunction(named: "pyrowave_quantize")
        _ = try backend.makeFunction(named: "pyrowave_dequantize")
    } catch PyrowaveError.externalToolUnavailable {
        return
    }
    #endif
}

@Test func metalQuantizationMatchesCPUReferenceWhenDeviceExists() throws {
    #if canImport(Metal)
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let samples: [Float] = [-2.0, -1.25, -0.5, -0.0004, 0.0, 0.0004, 0.5, 1.25, 2.0]
    let step: Float = 1.0 / 1024.0
    let metal = try backend.quantize(samples, quantizationStep: step)
    let cpu = samples.map { sample -> Int16 in
        let quantized = Int((sample / step).rounded())
        return Int16(max(Int(Int16.min), min(Int(Int16.max), quantized)))
    }
    #expect(metal == cpu)

    let dequantized = try backend.dequantize(metal, quantizationStep: step)
    #expect(dequantized == cpu.map { Float($0) * step })
    #endif
}

@Test func metalCodecMatchesCPUReferenceWhenDeviceExists() throws {
    #if canImport(Metal)
    do {
        _ = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let frame = try TestFrames.synthetic420(width: 160, height: 96)
    let configuration = CodecConfiguration(quantizationStep: 1.0 / 2048.0)
    let cpu = try PyrowaveCodec(useMetalAcceleration: false).encode(frame, configuration: configuration)
    let metal = try PyrowaveCodec(useMetalAcceleration: true).encode(frame, configuration: configuration)
    #expect(metal.data == cpu.data)

    let decodedCPU = try PyrowaveCodec(useMetalAcceleration: false).decode(cpu)
    let decodedMetal = try PyrowaveCodec(useMetalAcceleration: true).decode(metal)
    #expect(decodedMetal == decodedCPU)
    #endif
}

@Test func metalWaveletMatchesCPUReferenceWhenDeviceExists() throws {
    #if canImport(Metal)
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let width = 128
    let height = 128
    let levels = 3
    var samples = Array(repeating: Float(0), count: width * height)
    for y in 0..<height {
        for x in 0..<width {
            samples[y * width + x] = Float((x * 13 + y * 7) % 251) / 251.0 - 0.5
        }
    }

    var cpuForward = samples
    Wavelet.forward2D(&cpuForward, width: width, height: height, levels: levels)
    let metalForward = try backend.forwardWavelet(samples, width: width, height: height, levels: levels)

    let forwardError = zip(cpuForward, metalForward).map { abs($0 - $1) }.max() ?? 0
    #expect(forwardError < 0.0001)

    var cpuInverse = cpuForward
    Wavelet.inverse2D(&cpuInverse, width: width, height: height, levels: levels)
    let metalInverse = try backend.inverseWavelet(metalForward, width: width, height: height, levels: levels)

    let inverseError = zip(cpuInverse, metalInverse).map { abs($0 - $1) }.max() ?? 0
    #expect(inverseError < 0.0001)
    #endif
}

@Test func sparseRateControlCapsEncodedFrameSize() throws {
    let frame = try TestFrames.synthetic420(width: 256, height: 144)
    let codec = PyrowaveCodec(useMetalAcceleration: false)
    let uncapped = try codec.encode(frame, configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0))
    let cap = max(512, uncapped.data.count / 2)
    let capped = try codec.encode(
        frame,
        configuration: CodecConfiguration(
            quantizationStep: 1.0 / 1024.0,
            maximumEncodedBytes: cap
        )
    )

    #expect(capped.data.count <= cap)
    #expect(capped.data.count < uncapped.data.count)

    let decoded = try codec.decode(capped)
    let metrics = try Metrics.compare(frame, decoded)
    #expect(metrics.weightedPSNR > 20.0)
}

@Test func codecUsesPyrowaveSequenceHeaderStreamOnly() throws {
    let frame = try TestFrames.synthetic420(width: 64, height: 64)
    let codec = PyrowaveCodec(useMetalAcceleration: false)
    var encoded = try codec.encode(frame, configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0)).data

    var reader = BinaryReader(encoded)
    let sequence = try PyrowaveSequenceHeader(reader: &reader)
    #expect(sequence.width == frame.width)
    #expect(sequence.height == frame.height)
    #expect(sequence.chroma == .yuv420)
    #expect(sequence.sequence == 1)
    #expect(sequence.totalBlocks > 0)

    encoded[3] &= 0x7f
    #expect(throws: PyrowaveError.invalidBitstream("sequence header missing extended bit")) {
        _ = try codec.decode(EncodedFrame(data: encoded))
    }
}

@Test func codecAdvancesPyrowaveSequenceCounterModuloEight() throws {
    let frame = try TestFrames.synthetic420(width: 64, height: 64)
    let codec = PyrowaveCodec(useMetalAcceleration: false)
    var observedSequences = [UInt8]()

    for _ in 0..<9 {
        let encoded = try codec.encode(frame, configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0))
        var reader = BinaryReader(encoded.data)
        let sequence = try PyrowaveSequenceHeader(reader: &reader)
        observedSequences.append(sequence.sequence)

        while reader.offset < encoded.data.count {
            let block = try PyrowaveCoefficientBlockCodec.decodeBlock(reader: &reader)
            #expect(block.sequence == sequence.sequence)
        }
    }

    #expect(observedSequences == [1, 2, 3, 4, 5, 6, 7, 0, 1])
}

@Test func encodedFramePacketizesOnPyrowavePacketBoundaries() throws {
    let frame = try TestFrames.synthetic420(width: 64, height: 64)
    let encoded = try PyrowaveCodec(useMetalAcceleration: false).encode(
        frame,
        configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0)
    )
    let packets = try encoded.packetized(maximumPacketBytes: 8)

    var sequenceReader = BinaryReader(encoded.data)
    let sequence = try PyrowaveSequenceHeader(reader: &sequenceReader)
    #expect(packets.count == sequence.totalBlocks + 1)

    var reassembled = Data()
    for packet in packets {
        reassembled.append(packet.data)
    }
    #expect(reassembled == encoded.data)
}

@Test func packetStreamDecoderReconstructsCompletePacketizedFrame() throws {
    let frame = try TestFrames.synthetic420(width: 96, height: 64)
    let codec = PyrowaveCodec(useMetalAcceleration: false)
    let encoded = try codec.encode(frame, configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0))
    let expected = try codec.decode(encoded)
    let packets = try encoded.packetized(maximumPacketBytes: 8)
    let stream = PyrowavePacketStreamDecoder(useMetalAcceleration: false)

    for packet in packets.dropLast() {
        try stream.pushPacket(packet)
        #expect(!stream.decodeIsReady())
    }
    try stream.pushPacket(try #require(packets.last))
    #expect(stream.decodeIsReady())
    #expect(try stream.decode() == expected)
    #expect(!stream.decodeIsReady())
}

@Test func packetStreamDecoderAllowsPartialFrameAfterHalfTheBlocks() throws {
    let frame = try TestFrames.synthetic420(width: 96, height: 64)
    let encoded = try PyrowaveCodec(useMetalAcceleration: false).encode(
        frame,
        configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0)
    )
    let packets = try encoded.packetized(maximumPacketBytes: 8)
    let sequencePacket = try #require(packets.first)
    var sequenceReader = BinaryReader(sequencePacket.data)
    let sequence = try PyrowaveSequenceHeader(reader: &sequenceReader)
    let stream = PyrowavePacketStreamDecoder(useMetalAcceleration: false)

    try stream.pushPacket(sequencePacket)
    for packet in packets.dropFirst().prefix(sequence.totalBlocks / 2) {
        try stream.pushPacket(packet)
    }
    #expect(!stream.decodeIsReady(allowPartialFrame: true))

    try stream.pushPacket(packets[1 + sequence.totalBlocks / 2])
    #expect(stream.decodeIsReady(allowPartialFrame: true))
    let decoded = try stream.decode(allowPartialFrame: true)
    #expect(decoded.width == frame.width)
    #expect(decoded.height == frame.height)
    #expect(decoded.chroma == frame.chroma)
}

@Test func codecPreservesSequenceVideoSignalMetadata() throws {
    let source = try TestFrames.synthetic420(width: 64, height: 64)
    let frame = try YUVFrame(
        width: source.width,
        height: source.height,
        chroma: source.chroma,
        y: source.y,
        cb: source.cb,
        cr: source.cr,
        videoSignal: VideoSignalMetadata(
            colorPrimaries: .bt2020,
            transferFunction: .pq,
            yCbCrTransform: .bt2020,
            yCbCrRange: .limited,
            chromaSiting: .left
        )
    )

    let codec = PyrowaveCodec(useMetalAcceleration: false)
    let encoded = try codec.encode(frame, configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0))
    let decoded = try codec.decode(encoded)

    #expect(decoded.videoSignal == frame.videoSignal)
}

@Test func codecPacketsUseGlobalPyrowaveBlockOrder() throws {
    let frame = try TestFrames.synthetic420(width: 160, height: 96)
    let encoded = try PyrowaveCodec(useMetalAcceleration: false).encode(
        frame,
        configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0)
    )

    var reader = BinaryReader(encoded.data)
    let sequence = try PyrowaveSequenceHeader(reader: &reader)
    let layout = try PyrowaveBlockLayout(width: sequence.width, height: sequence.height, chroma: sequence.chroma)
    var blockIndices = [Int]()

    while reader.offset < encoded.data.count {
        let block = try PyrowaveCoefficientBlockCodec.decodeBlock(reader: &reader)
        blockIndices.append(block.blockIndex)
        #expect(block.sequence == sequence.sequence)
        #expect(block.blockIndex >= 0)
        #expect(block.blockIndex < layout.descriptors.count)
    }

    #expect(blockIndices.count == sequence.totalBlocks)
    #expect(blockIndices == blockIndices.sorted())
    #expect(blockIndices.contains { index in
        layout.descriptors[index].component == 1
    })
}

@Test func codecPacketsUsePerBandPyrowaveQuantCodes() throws {
    let frame = try TestFrames.synthetic420(width: 160, height: 96)
    let configuration = CodecConfiguration(quantizationStep: 1.0 / 1024.0)
    let encoded = try PyrowaveCodec(useMetalAcceleration: false).encode(frame, configuration: configuration)

    var reader = BinaryReader(encoded.data)
    let sequence = try PyrowaveSequenceHeader(reader: &reader)
    let layout = try PyrowaveBlockLayout(width: sequence.width, height: sequence.height, chroma: sequence.chroma)
    var observedQuantCodes = Set<UInt8>()
    var observedQScaleCodes = Set<UInt8>()

    while reader.offset < encoded.data.count {
        let packetStart = reader.offset
        let block = try PyrowaveCoefficientBlockCodec.decodeBlock(reader: &reader)
        let descriptor = layout.descriptors[block.blockIndex]
        let expectedStep = PyrowaveQuantization.quantizationStep(
            level: descriptor.level,
            component: descriptor.component,
            band: descriptor.band,
            baseStep: configuration.quantizationStep
        )
        #expect(block.quantCode == (try PyrowaveQuantization.encodeBlockScale(expectedStep)))
        observedQuantCodes.insert(block.quantCode)
        observedQScaleCodes.formUnion(block.qScaleCodes)
        #expect(reader.offset > packetStart)
    }

    #expect(observedQuantCodes.count > 1)
    #expect(observedQScaleCodes.contains { $0 != PyrowaveQuantization.identityQScaleCode })
}

@Test func codecRequiresSpecDecompositionLevelCount() throws {
    let frame = try TestFrames.synthetic420(width: 64, height: 64)
    let codec = PyrowaveCodec(useMetalAcceleration: false)
    #expect(throws: PyrowaveError.invalidDimensions) {
        _ = try codec.encode(frame, configuration: CodecConfiguration(decompositionLevels: 4))
    }
}

@Test func pyrowavePacketHeadersRoundTripPackedFields() throws {
    var writer = BinaryWriter()
    let packet = try PyrowavePacketHeader(
        ballot: 0x8421,
        payloadWords: 37,
        sequence: 5,
        extended: false,
        quantCode: 19,
        blockIndex: 0x00ab_cdef
    )
    packet.write(to: &writer)
    #expect(writer.data.count == 8)

    var reader = BinaryReader(writer.data)
    let decoded = try PyrowavePacketHeader(reader: &reader)
    #expect(decoded == packet)

    var sequenceWriter = BinaryWriter()
    let sequence = try PyrowaveSequenceHeader(
        width: 6144,
        height: 3456,
        sequence: 7,
        totalBlocks: 12345,
        chroma: .yuv444,
        videoSignal: VideoSignalMetadata(
            colorPrimaries: .bt2020,
            transferFunction: .pq,
            yCbCrTransform: .bt2020,
            yCbCrRange: .limited,
            chromaSiting: .left
        )
    )
    sequence.write(to: &sequenceWriter)
    #expect(sequenceWriter.data.count == 8)

    let bytes = [UInt8](sequenceWriter.data)
    let secondWord = UInt32(bytes[4]) |
        (UInt32(bytes[5]) << 8) |
        (UInt32(bytes[6]) << 16) |
        (UInt32(bytes[7]) << 24)
    let upperMetadataBits = secondWord >> 26
    #expect(upperMetadataBits == 0b11_1111)

    var sequenceReader = BinaryReader(sequenceWriter.data)
    #expect(try PyrowaveSequenceHeader(reader: &sequenceReader) == sequence)
}

@Test func pyrowaveBlockLayoutFollowsSpecOrdering() throws {
    let layout = try PyrowaveBlockLayout(width: 256, height: 256, chroma: .yuv420)
    let first = try #require(layout.descriptors.first)
    #expect(first.blockIndex == 0)
    #expect(first.level == 4)
    #expect(first.component == 0)
    #expect(first.band == 0)
    #expect(first.originX == 0)
    #expect(first.originY == 0)

    let firstLevel4Chroma = try #require(layout.descriptors.first { $0.level == 4 && $0.component == 1 })
    #expect(firstLevel4Chroma.band == 0)

    #expect(layout.descriptors.contains { $0.level == 0 && $0.component == 0 && $0.band == 1 })
    #expect(!layout.descriptors.contains { $0.level == 0 && $0.component == 1 })
    #expect(!layout.descriptors.contains { $0.level == 0 && $0.component == 2 })

    let blockIndices = layout.descriptors.map(\.blockIndex)
    #expect(blockIndices == Array(0..<layout.descriptors.count))
}

@Test func pyrowaveCoefficientBlockPayloadRoundTripsBitPlanes() throws {
    let stride = 32
    var coefficients = Array(repeating: Int16(0), count: stride * stride)
    coefficients[0] = 1
    coefficients[1] = -2
    coefficients[2] = 3
    coefficients[8 * stride + 8] = -17
    coefficients[31 * stride + 31] = 255

    let payload = try #require(try PyrowaveCoefficientBlockCodec.encodeBlock(
        blockIndex: 42,
        coefficients: coefficients,
        stride: stride,
        originX: 0,
        originY: 0,
        validWidth: 32,
        validHeight: 32,
        threshold: 0,
        sequence: 3,
        quantCode: 11
    ))

    var reader = BinaryReader(payload)
    let header = try PyrowavePacketHeader(reader: &reader)
    #expect(header.blockIndex == 42)
    #expect(header.sequence == 3)
    #expect(header.quantCode == 11)
    #expect(header.ballot == 0x8021)

    var decodeReader = BinaryReader(payload)
    let decoded = try PyrowaveCoefficientBlockCodec.decodeBlock(reader: &decodeReader)
    let decodedMap = Dictionary(uniqueKeysWithValues: decoded.coefficients.map { (Int($0.offset), $0.value) })

    #expect(decoded.blockIndex == 42)
    #expect(decoded.quantCode == 11)
    #expect(decoded.qScaleCodes == [PyrowaveQuantization.identityQScaleCode, PyrowaveQuantization.identityQScaleCode, PyrowaveQuantization.identityQScaleCode])
    #expect(decoded.coefficients.allSatisfy { $0.qScaleCode == PyrowaveQuantization.identityQScaleCode })
    #expect(decodedMap[0] == 1)
    #expect(decodedMap[1] == -2)
    #expect(decodedMap[2] == 3)
    #expect(decodedMap[8 * stride + 8] == -17)
    #expect(decodedMap[31 * stride + 31] == 255)
    #expect(decoded.coefficients.count == 5)
    #expect(decodeReader.offset == payload.count)
}

@Test func pyrowaveCoefficientBlockPayloadCarriesPer8x8QScales() throws {
    let stride = 32
    var coefficients = Array(repeating: Int16(0), count: stride * stride)
    coefficients[0] = 4
    coefficients[9] = 5
    coefficients[16] = 6

    var qScaleCodes = Array(repeating: PyrowaveQuantization.identityQScaleCode, count: 16)
    qScaleCodes[0] = 7
    qScaleCodes[2] = 8

    let payload = try #require(try PyrowaveCoefficientBlockCodec.encodeBlock(
        blockIndex: 3,
        coefficients: coefficients,
        stride: stride,
        originX: 0,
        originY: 0,
        validWidth: 32,
        validHeight: 32,
        threshold: 0,
        quantCode: 11,
        qScaleCodes: qScaleCodes
    ))

    var reader = BinaryReader(payload)
    let decoded = try PyrowaveCoefficientBlockCodec.decodeBlock(reader: &reader)
    #expect(decoded.qScaleCodes == [7, PyrowaveQuantization.identityQScaleCode, 8])
    #expect(decoded.coefficients.contains { $0.offset == 0 && $0.qScaleCode == 7 })
    #expect(decoded.coefficients.contains { $0.offset == 16 && $0.qScaleCode == 8 })
}

@Test func pyrowaveCoefficientBlockQuantLevelDropsBitplanesAndAdjustsScale() throws {
    let stride = 32
    var coefficients = Array(repeating: Int16(0), count: stride * stride)
    coefficients[0] = 7
    coefficients[1] = -8
    coefficients[9] = 3

    let quantCode = try PyrowaveQuantization.encodeBlockScale(1.0 / 1024.0)
    let payload = try #require(try PyrowaveCoefficientBlockCodec.encodeBlock(
        blockIndex: 9,
        coefficients: coefficients,
        stride: stride,
        originX: 0,
        originY: 0,
        validWidth: 32,
        validHeight: 32,
        threshold: 0,
        quantLevel: 2,
        quantCode: quantCode
    ))

    var reader = BinaryReader(payload)
    let decoded = try PyrowaveCoefficientBlockCodec.decodeBlock(reader: &reader)
    let decodedMap = Dictionary(uniqueKeysWithValues: decoded.coefficients.map { (Int($0.offset), $0.value) })
    #expect(decoded.quantCode == PyrowaveQuantization.modifyQuantCode(quantCode, droppingBitplanes: 2))
    #expect(decodedMap[0] == 1)
    #expect(decodedMap[1] == -2)
    #expect(decodedMap[9] == nil)
}

@Test func pyrowaveQuantizationHelpersMatchSpecFormulas() throws {
    let code = try PyrowaveQuantization.encodeBlockScale(1.0 / 1024.0)
    #expect(code == 112)
    #expect(abs(PyrowaveQuantization.decodeBlockScale(code) - (1.0 / 1024.0)) < 0.000001)
    #expect(PyrowaveQuantization.identityQScaleCode == 6)
    #expect(PyrowaveQuantization.decode8x8Scale(PyrowaveQuantization.identityQScaleCode) == 1.0)
    #expect(PyrowaveQuantization.encode8x8Scale(1.0) == PyrowaveQuantization.identityQScaleCode)

    let positive = PyrowaveQuantization.dequantize(coefficient: 2, quantCode: code, qScaleCode: PyrowaveQuantization.identityQScaleCode)
    let negative = PyrowaveQuantization.dequantize(coefficient: -2, quantCode: code, qScaleCode: PyrowaveQuantization.identityQScaleCode)
    #expect(abs(positive - 2.5 / 1024.0) < 0.000001)
    #expect(abs(negative + 2.5 / 1024.0) < 0.000001)

    #expect(PyrowaveQuantization.noisePowerNormalizedResolution(level: 0, component: 0, band: 1) == 128)
    #expect(PyrowaveQuantization.quantizationResolution(level: 4, component: 0, band: 0) == 512)
    #expect(PyrowaveQuantization.quantizationResolution(level: 1, component: 1, band: 1) == 128)
    #expect(PyrowaveQuantization.quantizationStep(level: 4, component: 0, band: 0, baseStep: 1.0 / 1024.0) == 1.0 / 512.0)
    let lumaDistortion = PyrowaveQuantization.rdoDistortionScale(level: 1, component: 0, band: 1, chroma: .yuv420)
    let chromaDistortion = PyrowaveQuantization.rdoDistortionScale(level: 1, component: 1, band: 1, chroma: .yuv420)
    #expect(abs((chromaDistortion / lumaDistortion) - 0.09) < 0.0001)
    #expect(PyrowaveQuantization.encode8x8ScaleCode(maxScaledCoefficient: 0.999) == PyrowaveQuantization.identityQScaleCode)
    #expect(PyrowaveQuantization.encode8x8ScaleCode(maxScaledCoefficient: 1.75) == PyrowaveQuantization.identityQScaleCode)
    #expect(PyrowaveQuantization.encode8x8ScaleCode(maxScaledCoefficient: 2.0) == 8)
    #expect(abs(PyrowaveQuantization.quantScale(for8x8ScaleCode: 8) - (1.0 / 1.25)) < 0.000001)
    #expect(PyrowaveQuantization.modifyQuantCode(code, droppingBitplanes: 2) == 96)
    #expect(PyrowaveQuantization.modifyQuantCode(code, droppingBitplanes: 99) == 0)
}

@Test func pyrowaveBlockStatsUseOriginalPackedShape() throws {
    let stats = PyrowaveBlockStats(
        numPlanes: 3,
        stats: (0..<PyrowaveBlockStats.candidateCount).map {
            PyrowaveQuantStats(squareError: Float($0 * $0), encodeCostBits: 100 - $0)
        }
    )

    let packed = stats.packedData()
    #expect(packed.count == PyrowaveBlockStats.packedByteCount)
    #expect(stats.stats.count == 15)
    #expect(stats.stats[4].encodeCostBits == 96)
}

@Test func pyrowaveRateControlBuildsMonotonicPacketCandidates() throws {
    let stride = 32
    var coefficients = Array(repeating: Int16(0), count: stride * stride)
    coefficients[0] = 1
    coefficients[1] = -2
    coefficients[9] = 4
    coefficients[8 * stride + 2] = -8
    coefficients[15 * stride + 15] = 14

    let block = try PyrowaveRateController.makeBlock(
        blockIndex: 0,
        coefficients: coefficients,
        stride: stride,
        originX: 0,
        originY: 0,
        validWidth: 32,
        validHeight: 32,
        quantCode: PyrowaveQuantization.identityQScaleCode,
        qScaleCode: PyrowaveQuantization.identityQScaleCode
    )

    #expect(block.eightByEightStats.count == 16)
    #expect(block.packetByteCosts[0] > block.packetByteCosts[14])
    #expect(block.distortion(quantLevel: 14) > block.distortion(quantLevel: 0))
    for threshold in 1..<PyrowaveBlockStats.candidateCount {
        #expect(block.packetByteCosts[threshold] <= block.packetByteCosts[threshold - 1])
    }
}

@Test func pyrowaveRateControlWeightsBitplaneDistortionLikeQuantShader() throws {
    let stride = 32
    var coefficients = Array(repeating: Int16(0), count: stride * stride)
    coefficients[0] = 8
    let quantCode = try PyrowaveQuantization.encodeBlockScale(1.0)

    let block = try PyrowaveRateController.makeBlock(
        blockIndex: 0,
        coefficients: coefficients,
        stride: stride,
        originX: 0,
        originY: 0,
        validWidth: 32,
        validHeight: 32,
        quantCode: quantCode,
        qScaleCode: PyrowaveQuantization.identityQScaleCode,
        rdoDistortionScale: 4.0
    )

    let stats = block.eightByEightStats[0].stats
    #expect(stats[0].squareError == 0)
    #expect(stats[1].squareError == 4)
    #expect(stats[2].squareError == 16)
    #expect(stats[4].squareError == 256)
}

@Test func pyrowaveRateControlUsesClusteredRDBuckets() throws {
    let stride = 32
    var coefficients = Array(repeating: Int16(0), count: stride * stride)
    coefficients[0] = 1
    coefficients[1] = -2
    coefficients[9] = 4
    coefficients[8 * stride + 2] = -8
    coefficients[15 * stride + 15] = 14

    let block = try PyrowaveRateController.makeBlock(
        blockIndex: 3,
        coefficients: coefficients,
        stride: stride,
        originX: 0,
        originY: 0,
        validWidth: 32,
        validHeight: 32,
        quantCode: PyrowaveQuantization.identityQScaleCode,
        qScaleCode: PyrowaveQuantization.identityQScaleCode
    )

    let buckets = PyrowaveRateController.inclusiveBucketIndices(for: block)
    #expect(buckets.count == PyrowaveBlockStats.candidateCount)
    #expect(buckets[0] == 0)
    #expect(buckets.allSatisfy { (0..<128).contains($0) })
    for quantLevel in 1..<buckets.count {
        #expect(buckets[quantLevel] >= buckets[quantLevel - 1] + 1)
    }

    let operations = PyrowaveRateController.makeRDOperations(blocksByPlane: [[block]])
    #expect(!operations.isEmpty)
    #expect(operations == operations.sorted {
        if $0.bucket != $1.bucket {
            return $0.bucket < $1.bucket
        }
        if $0.planeIndex != $1.planeIndex {
            return $0.planeIndex < $1.planeIndex
        }
        if $0.blockIndex != $1.blockIndex {
            return $0.blockIndex < $1.blockIndex
        }
        return $0.quantLevel < $1.quantLevel
    })
    #expect(operations.allSatisfy { $0.planeIndex == 0 && $0.blockIndex == 0 })
    #expect(operations.allSatisfy { $0.quantLevel > 0 && $0.saving > 0 })
    #expect(operations.allSatisfy { $0.bucket == buckets[$0.quantLevel] })
}

@Test func pyrowaveRateControlBucketIndexMatchesShaderFormulaShape() {
    #expect(PyrowaveRateController.distortionBucketIndex(
        distortion: 10,
        cost: 16,
        baseDistortion: 0,
        baseCost: 16
    ) == 0)

    let lowDistortion = PyrowaveRateController.distortionBucketIndex(
        distortion: 1,
        cost: 8,
        baseDistortion: 0,
        baseCost: 16
    )
    let highDistortion = PyrowaveRateController.distortionBucketIndex(
        distortion: 16,
        cost: 8,
        baseDistortion: 0,
        baseCost: 16
    )

    #expect(highDistortion > lowDistortion)
    #expect(lowDistortion >= 0)
    #expect(highDistortion < 128)
}

@Test func pyrowaveRateControllerSelectsThresholdsToMeetCap() throws {
    let stride = 32
    var coefficients = Array(repeating: Int16(0), count: stride * stride)
    coefficients[0] = 1
    coefficients[1] = 2
    coefficients[2] = 4
    coefficients[3] = 8
    coefficients[4] = 14

    let block = try PyrowaveRateController.makeBlock(
        blockIndex: 0,
        coefficients: coefficients,
        stride: stride,
        originX: 0,
        originY: 0,
        validWidth: 32,
        validHeight: 32,
        quantCode: PyrowaveQuantization.identityQScaleCode,
        qScaleCode: PyrowaveQuantization.identityQScaleCode
    )

    let targetBytes = block.packetByteCosts[14]
    let thresholds = try #require(PyrowaveRateController.selectThresholds(
        blocksByPlane: [[block]],
        fixedHeaderBytes: 0,
        maximumEncodedBytes: targetBytes
    ))
    let estimatedBytes = PyrowaveRateController.estimateFrameBytes(
        blocksByPlane: [[block]],
        thresholdsByPlane: thresholds,
        fixedHeaderBytes: 0
    )

    #expect(estimatedBytes <= targetBytes)
    #expect((thresholds.first?.first ?? 0) > 0)
}
