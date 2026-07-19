import Foundation
import Testing
@testable import PyrowaveKit

import CoreVideo
import Metal

private extension String {
    func requiredSlice(from start: String, to end: String) throws -> String {
        guard let startRange = range(of: start) else {
            throw PyrowaveError.processFailed("missing source slice start: \(start)")
        }
        guard let endRange = self[startRange.upperBound...].range(of: end) else {
            throw PyrowaveError.processFailed("missing source slice end: \(end)")
        }
        return String(self[startRange.lowerBound..<endRange.lowerBound])
    }
}

@Test func hardCutoverSourceTreeContainsOnlySwiftAndMetalSources() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let ignoredDirectories = Set([".build", ".git", ".swiftpm", ".pyrowave-results"])
    let originalSourceExtensions = Set(["c", "cc", "cpp", "cxx", "h", "hpp", "hxx", "m", "mm"])
    let originalShaderExtensions = Set(["glsl", "vert", "frag", "comp", "spv", "spirv"])
    let originalBuildFiles = Set(["CMakeLists.txt", "Makefile"])
    var forbiddenFiles = [String]()

    let enumerator = try #require(FileManager.default.enumerator(
        at: packageRoot,
        includingPropertiesForKeys: [.isDirectoryKey],
        options: [.skipsHiddenFiles]
    ))
    for case let fileURL as URL in enumerator {
        let resourceValues = try fileURL.resourceValues(forKeys: [.isDirectoryKey])
        if resourceValues.isDirectory == true {
            if ignoredDirectories.contains(fileURL.lastPathComponent) {
                enumerator.skipDescendants()
            }
            continue
        }

        let fileName = fileURL.lastPathComponent
        let fileExtension = fileURL.pathExtension.lowercased()
        if originalSourceExtensions.contains(fileExtension)
            || originalShaderExtensions.contains(fileExtension)
            || originalBuildFiles.contains(fileName) {
            forbiddenFiles.append(fileURL.path.replacingOccurrences(of: packageRoot.path + "/", with: ""))
        }
    }

    #expect(forbiddenFiles.isEmpty, "Original-language or Vulkan-era files remain: \(forbiddenFiles.sorted())")
}

@Test func publicCodecSurfaceExcludesCPUFrameFallbacks() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let codecSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/Codec.swift"), encoding: .utf8)
    let typesSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/Types.swift"), encoding: .utf8)
    let testFramesSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/TestFrames.swift"), encoding: .utf8)
    let metricsSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/Metrics.swift"), encoding: .utf8)
    let yuv4mpegSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/YUV4MPEG.swift"), encoding: .utf8)
    let hevcSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/HEVCComparison.swift"), encoding: .utf8)
    let benchmarkSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/BenchmarkSupport.swift"), encoding: .utf8)
    let coreVideoSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/CoreVideoSupport.swift"), encoding: .utf8)
    let metalTextureSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/MetalTextureSupport.swift"), encoding: .utf8)
    let metalBackendSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/MetalBackend.swift"), encoding: .utf8)
    let streamSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/PyrowaveStreamFile.swift"), encoding: .utf8)

    #expect(!typesSource.contains("public struct Plane8"))
    #expect(!typesSource.contains("public struct YUVFrame"))
    #expect(!codecSource.contains("public func encode(_ frame: YUVFrame"))
    #expect(!codecSource.contains("public func decode(_ frame: EncodedFrame) throws -> YUVFrame"))
    #expect(!codecSource.contains("public func decode(allowPartialFrame: Bool = false) throws -> YUVFrame"))
    #expect(!testFramesSource.contains("public enum TestFrames"))
    #expect(!metricsSource.contains("public enum Metrics"))
    #expect(!yuv4mpegSource.contains("public struct YUV4MPEG"))
    #expect(!hevcSource.contains("public enum HEVCComparison"))
    #expect(!benchmarkSource.contains("public struct PyrowaveBenchmarkFrames"))
    #expect(!benchmarkSource.contains("public enum PyrowaveBenchmarkRunner"))
    #expect(!coreVideoSource.contains("public init(\n        cvPixelBuffer: CVPixelBuffer"))
    #expect(!coreVideoSource.contains("public func makeCVPixelBuffer"))
    #expect(!coreVideoSource.contains("public func copy(to cvPixelBuffer: CVPixelBuffer)"))
    #expect(!metalTextureSource.contains("public init(texture: MTLTexture)"))
    #expect(!metalTextureSource.contains("public func makeMetalTexture"))
    #expect(!metalTextureSource.contains("public func makeMetalTextures"))
    #expect(!metalBackendSource.contains("public final class MetalPyrowaveBackend"))
    #expect(!metalBackendSource.contains("public func quantize(_ samples: [Float]"))
    #expect(!metalBackendSource.contains("public func dequantize(_ coefficients: [Int16]"))
    #expect(!metalBackendSource.contains("public func forwardWavelet(_ samples: [Float]"))
    #expect(!metalBackendSource.contains("public func inverseWavelet(_ coefficients: [Float]"))
    #expect(!streamSource.contains("public init(frame: YUVFrame"))
}

@Test func coreVideoCodecEntryPointsStayOnMetalTexturePath() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let source = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/CoreVideoSupport.swift"), encoding: .utf8)

    let gpuEncodeBody = try source.requiredSlice(
        from: "public func encodeGPUFrame(\n        _ cvPixelBuffer: CVPixelBuffer",
        to: "    public func encode(\n        _ cvPixelBuffer: CVPixelBuffer"
    )
    #expect(gpuEncodeBody.contains("let textures = try makeNV12MetalTexturesAndSignal("))
    #expect(gpuEncodeBody.contains("return try encodeGPUFrame("))
    #expect(!gpuEncodeBody.contains("exportGPUFrame"))
    #expect(!gpuEncodeBody.contains("YUVFrame("))
    #expect(!gpuEncodeBody.contains("CVPixelBufferLockBaseAddress"))

    let exportEncodeBody = try source.requiredSlice(
        from: "public func encode(\n        _ cvPixelBuffer: CVPixelBuffer",
        to: "    private func makeMetalTexture("
    )
    #expect(exportEncodeBody.contains("try exportGPUFrame(try encodeGPUFrame("))
    #expect(exportEncodeBody.contains("private func makeNV12MetalTexturesAndSignal("))
    #expect(!exportEncodeBody.contains("YUVFrame("))
    #expect(!exportEncodeBody.contains("nv12Planes"))
    #expect(!exportEncodeBody.contains("CVPixelBufferLockBaseAddress"))

    let exportDecodeBody = try source.requiredSlice(
        from: "public func decodeToCVPixelBuffer(",
        to: "    public func decodeGPUFrameToCVPixelBuffer("
    )
    #expect(exportDecodeBody.contains("try decodeGPUFrameToCVPixelBuffer(try importGPUFrame(frame), pixelFormat: pixelFormat)"))
    #expect(!exportDecodeBody.contains("BinaryReader"))
    #expect(!exportDecodeBody.contains("decodeToNV12Textures("))
    #expect(!exportDecodeBody.contains("YUVFrame("))
    #expect(!exportDecodeBody.contains("copy(to:"))

    let gpuDecodeAllocatingBody = try source.requiredSlice(
        from: "public func decodeGPUFrameToCVPixelBuffer(",
        to: "    public func decodeGPUFrame(\n        _ frame: PyrowaveGPUFrame"
    )
    #expect(gpuDecodeAllocatingBody.contains("CVPixelBufferCreate("))
    #expect(gpuDecodeAllocatingBody.contains("try decodeGPUFrame(frame, to: pixelBuffer)"))
    #expect(!gpuDecodeAllocatingBody.contains("importGPUFrame"))
    #expect(!gpuDecodeAllocatingBody.contains("YUVFrame("))
    #expect(!gpuDecodeAllocatingBody.contains("CVPixelBufferLockBaseAddress"))

    let gpuDecodeReusableBody = try source.requiredSlice(
        from: "public func decodeGPUFrame(\n        _ frame: PyrowaveGPUFrame",
        to: "    func decodeContiguousFrame("
    )
    #expect(gpuDecodeReusableBody.contains("makeMetalTexture("))
    #expect(gpuDecodeReusableBody.contains("try decodeGPUFrameToNV12Textures("))
    #expect(!gpuDecodeReusableBody.contains("importGPUFrame"))
    #expect(!gpuDecodeReusableBody.contains("decodeToNV12Textures("))
    #expect(!gpuDecodeReusableBody.contains("YUVFrame("))
    #expect(!gpuDecodeReusableBody.contains("CVPixelBufferLockBaseAddress"))

    let contiguousDecodeBody = try source.requiredSlice(
        from: "func decodeContiguousFrame(",
        to: "    }\n}"
    )
    #expect(contiguousDecodeBody.contains("try await decodeToNV12Textures("))
    #expect(!contiguousDecodeBody.contains("importGPUFrame"))

    let sessionsSource = try String(
        contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/PyrowaveSessions.swift"),
        encoding: .utf8
    )
    let productionDecodeBody = try sessionsSource.requiredSlice(
        from: "public func decode(_ frame: EncodedFrame) async throws",
        to: "    private static func applyVideoSignalAttachments("
    )
    #expect(productionDecodeBody.contains("try await codec.decodeContiguousFrame("))
    #expect(productionDecodeBody.contains("guard !isDecodingFrame"))
    #expect(!productionDecodeBody.contains("importGPUFrame"))
    #expect(!productionDecodeBody.contains("decodeToCVPixelBuffer"))

    let metalSource = try String(
        contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/MetalBackend.swift"),
        encoding: .utf8
    )
    let asynchronousCompletionBody = try metalSource.requiredSlice(
        from: "private func finishAsync(",
        to: "    private func validateWaveletInput("
    )
    #expect(asynchronousCompletionBody.contains("commandBuffer.addCompletedHandler"))
    #expect(asynchronousCompletionBody.contains("commandBuffer.commit()"))
    #expect(!asynchronousCompletionBody.contains("waitUntilCompleted"))
}

@Test func nv12TextureEncodedFrameAPIExportsGPUFrame() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let source = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/Codec.swift"), encoding: .utf8)
    let body = try source.requiredSlice(
        from: "public func encode(\n        yTexture: MTLTexture,\n        cbCrTexture: MTLTexture",
        to: "    public func encodeGPUFrame("
    )
    #expect(body.contains("try exportGPUFrame(try encodeGPUFrame("))
    #expect(!body.contains("encodeFrame("))
    #expect(!body.contains("encodeTexturePlanes("))
}

@Test func pyrowaveBenchmarkTimedScopeExcludesArtifactsMetricsAndCPUFrames() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let source = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/BenchmarkSupport.swift"), encoding: .utf8)
    let timedScope = try source.requiredSlice(
        from: "let encodeSources = try inputPixelBuffers.map",
        to: "        let metric: FrameMetrics?"
    )

    #expect(timedScope.contains("codec.encodeGPUFrame("))
    #expect(timedScope.contains("codec.decodeGPUFrameToNV12Textures("))
    #expect(timedScope.contains("reusesPacketBuffers: true"))
    #expect(timedScope.contains("let timedDecodeTarget = try makeDecodeTarget("))
    #expect(timedScope.contains("yTexture: timedDecodeTarget.yTexture"))
    #expect(!timedScope.contains("makeDecodeTargets("))
    let encodeEnd = try #require(timedScope.range(of: "encodeSeconds += stopwatch.lapSeconds()")?.lowerBound)
    let decodeStart = try #require(timedScope.range(of: "codec.decodeGPUFrameToNV12Textures(")?.lowerBound)
    let decodeEnd = try #require(timedScope.range(of: "decodeSeconds += stopwatch.lapSeconds()")?.lowerBound)
    let byteInspection = try #require(timedScope.range(of: "encodedByteCountForInspection()")?.lowerBound)
    #expect(encodeEnd < decodeStart)
    #expect(decodeEnd < byteInspection)
    #expect(!timedScope.contains("codec.encode(pixelBuffer"))
    #expect(!timedScope.contains("decodeToNV12Textures("))
    #expect(!timedScope.contains("gpuFrames.append"))
    #expect(!timedScope.contains("encodedFrames.append"))
    #expect(!timedScope.contains("YUVFrame("))
    #expect(!timedScope.contains("YUV4MPEGWriter"))
    #expect(!timedScope.contains("PyrowaveStreamWriter"))
    #expect(!timedScope.contains("Metrics.compare"))
    #expect(!timedScope.contains("JSONEncoder"))
    #expect(!timedScope.contains("exportGPUFrame"))
    #expect(!timedScope.contains("importGPUFrame"))
    #expect(!timedScope.contains("writeFrame"))
    #expect(!timedScope.contains("decodeToCVPixelBuffer"))
    #expect(!timedScope.contains("makeCVPixelBuffer"))
}

@Test func pyrowaveBenchmarkReviewArtifactsExportGPUFramesOutsideTimedPath() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let source = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/BenchmarkSupport.swift"), encoding: .utf8)
    let artifactScope = try source.requiredSlice(
        from: "if writesArtifactsAndMetrics {",
        to: "        } else {"
    )
    #expect(artifactScope.contains("artifactCodec.encodeGPUFrame("))
    #expect(artifactScope.contains("artifactCodec.exportGPUFrame("))
    #expect(artifactScope.contains("artifactCodec.decodeGPUFrameToNV12Textures("))
    #expect(artifactScope.contains("makeDecodeTargets("))
    #expect(artifactScope.contains("PyrowaveStreamWriter"))
    #expect(!artifactScope.contains("artifactCodec.encode($0"))
}

@Test func gpuFrameEncodeDoesNotUsePyrowaveByteCaps() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let codecSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/Codec.swift"), encoding: .utf8)
    let gpuEncodeBody = try codecSource.requiredSlice(
        from: "public func encodeGPUFrame(",
        to: "    public func decodeGPUFrameToNV12Textures("
    )
    #expect(!gpuEncodeBody.contains("maximumEncodedBytes"))
    #expect(!gpuEncodeBody.contains("metalSparsePacketByteCosts("))
    #expect(!gpuEncodeBody.contains("selectSparseRateControlPlan("))
    #expect(gpuEncodeBody.contains("packetByteCostsByPlane: nil"))
}

@Test func pyrowaveSourcesDoNotContainByteCapSelectors() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let sourceRoot = packageRoot.appendingPathComponent("Sources/PyrowaveKit")
    let sourceFiles = [
        "BenchmarkSupport.swift",
        "Codec.swift",
        "MetalBackend.swift",
        "PyrowaveRateControl.swift",
        "Metal/PyrowaveKernels.metal"
    ]
    let sources = try sourceFiles.map {
        try String(contentsOf: sourceRoot.appendingPathComponent($0), encoding: .utf8)
    }.joined(separator: "\n")

    #expect(!sources.contains("maximumEncodedBytes"))
    #expect(!sources.contains("pyrowaveFrameBudgetBytes"))
    #expect(!sources.contains("max-pyrowave-bytes"))
    #expect(!sources.contains("match-hevc-frame-budget"))
    #expect(!sources.contains("selectSparseRateControlPlan("))
    #expect(!sources.contains("rateControlUniformQuantLevel("))
    #expect(!sources.contains("pyrowave_rate_control_select_uniform_quant_level"))
}

@Test func gpuFrameEncodeKeepsQScaleMetadataResidentForPacketEmission() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let codecSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/Codec.swift"), encoding: .utf8)
    let gpuEncodeBody = try codecSource.requiredSlice(
        from: "public func encodeGPUFrame(",
        to: "    public func decodeGPUFrameToNV12Textures("
    )
    #expect(gpuEncodeBody.contains("readsQScaleCodes: false"))

    let gpuFramePlaneBody = try codecSource.requiredSlice(
        from: "private func makeGPUFramePlanes(",
        to: "    private func sparseBlocksWithMetal("
    )
    #expect(gpuFramePlaneBody.contains("usesResidentQScaleBuffers"))
    #expect(gpuFramePlaneBody.contains("encodeSparsePacketBuffersBatchResidentQScales"))
    #expect(gpuFramePlaneBody.contains("if !usesResidentQScaleBuffers"))

    let metalSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/MetalBackend.swift"), encoding: .utf8)
    let residentPacketBody = try metalSource.requiredSlice(
        from: "func encodeSparsePacketBuffersBatchResidentQScales(",
        to: "    private func emptySparsePacketOutputBuffer()"
    )
    #expect(residentPacketBody.contains("qScaleBuffer: MTLBuffer"))
    #expect(!residentPacketBody.contains("flatQScaleCodes"))
    #expect(residentPacketBody.contains("encoder.setComputePipelineState(sparsePacketEncodeThreadgroupPipeline)"))
    #expect(!residentPacketBody.contains("threadgroupDescriptorLimit"))
    #expect(!residentPacketBody.contains("sparsePacketEncodeSerialPipeline"))
}

@Test func gpuFrameQuantizationSkipsInterstageWaitWhenQScaleReadbackIsDisabled() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let metalSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/MetalBackend.swift"), encoding: .utf8)
    let quantizeBody = try metalSource.requiredSlice(
        from: "func quantizePlaneBufferResults(",
        to: "    func applySparseCoefficients("
    )
    let commit = try #require(quantizeBody.range(of: "commandBuffer.commit()")?.lowerBound)
    let readbackBranch = try #require(quantizeBody.range(of: "if readsQScaleCodes")?.lowerBound)
    let wait = try #require(quantizeBody.range(of: "commandBuffer.waitUntilCompleted()")?.lowerBound)
    #expect(commit < readbackBranch)
    #expect(readbackBranch < wait)
}

@Test func gpuFrameEncodeSkipsIntermediateDWTWaitBeforeResidentQuantization() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let codecSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/Codec.swift"), encoding: .utf8)
    let gpuEncodeBody = try codecSource.requiredSlice(
        from: "public func encodeGPUFrame(",
        to: "    public func decodeGPUFrameToNV12Textures("
    )
    #expect(gpuEncodeBody.contains("readsQScaleCodes: false"))
    #expect(gpuEncodeBody.contains("waitsForDWTCompletion: false"))

    let metalSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/MetalBackend.swift"), encoding: .utf8)
    let padDWTBody = try metalSource.requiredSlice(
        from: "func padTexturePlaneBuffersAndForwardWaveletBuffers(",
        to: "    func cropPlane("
    )
    #expect(padDWTBody.contains("waitsForCompletion: Bool = true"))
    #expect(padDWTBody.contains("if waitsForCompletion"))
    #expect(padDWTBody.contains("commandBuffer.commit()"))
}

@Test func residentQuantizationUsesCachedDescriptorBuffers() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let codecSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/Codec.swift"), encoding: .utf8)
    let residentQuantizeBody = try codecSource.requiredSlice(
        from: "private func quantizeResidentBuffers(",
        to: "    private func quantizeWithMetal("
    )
    #expect(residentQuantizeBody.contains("quantizePlaneBufferResultsResidentDescriptors"))
    #expect(residentQuantizeBody.contains("descriptorBuffer: descriptorInputs[index].descriptorBuffer"))

    let descriptorCacheBody = try codecSource.requiredSlice(
        from: "private func makeQuantizationDescriptors(",
        to: "    private func quantizationDescriptorCacheKey("
    )
    #expect(descriptorCacheBody.contains("makeStaticSharedBuffer"))
    #expect(descriptorCacheBody.contains("cached.descriptorBuffer"))
}

@Test func sparsePacketEncodeDescriptorDoesNotCarryFrameSequence() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let metalBackendSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/MetalBackend.swift"), encoding: .utf8)
    let swiftDescriptor = try metalBackendSource.requiredSlice(
        from: "struct MetalSparsePacketEncodeDescriptor {",
        to: "private struct PadPlaneConstants {"
    )
    #expect(!swiftDescriptor.contains("sequence"))

    let kernelSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/Metal/PyrowaveKernels.metal"), encoding: .utf8)
    let metalDescriptor = try kernelSource.requiredSlice(
        from: "struct SparsePacketEncodeDescriptor {",
        to: "struct SparsePacketEncodeConstants {"
    )
    let metalConstants = try kernelSource.requiredSlice(
        from: "struct SparsePacketEncodeConstants {",
        to: "struct RateControlBucketConstants {"
    )
    #expect(!metalDescriptor.contains("sequence"))
    #expect(metalConstants.contains("uint sequence"))
    #expect(kernelSource.contains("constants.sequence & 7u"))
}

@Test func gpuFrameEncodeUsesCachedSparsePacketDescriptorBuffers() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let codecSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/Codec.swift"), encoding: .utf8)
    let gpuFramePlaneBody = try codecSource.requiredSlice(
        from: "private func makeGPUFramePlanes(",
        to: "    private func cachedSparsePacketEncodeDescriptors("
    )
    #expect(gpuFramePlaneBody.contains("usesCachedPacketDescriptors"))
    #expect(gpuFramePlaneBody.contains("cachedSparsePacketEncodeDescriptors("))
    #expect(gpuFramePlaneBody.contains("packetDescriptorCountsByPlane"))
    #expect(gpuFramePlaneBody.contains("descriptorCount: packetDescriptorCountsByPlane[index]"))
    #expect(gpuFramePlaneBody.contains("descriptors: packetDescriptorBuffersByPlane[index] == nil ? packetDescriptorsByPlane[index] : nil"))
    #expect(gpuFramePlaneBody.contains("descriptorBuffer: packetDescriptorBuffersByPlane[index]"))
    #expect(gpuFramePlaneBody.contains("outputStorageMode: packetOutputStorageMode"))

    let exportBody = try codecSource.requiredSlice(
        from: "public func exportGPUFrame(",
        to: "    public func importGPUFrame("
    )
    #expect(exportBody.contains("sharedReadbackBuffer("))

    let cacheEntry = try codecSource.requiredSlice(
        from: "private struct SparsePacketEncodeDescriptorCacheEntry {",
        to: "    private struct DecodedPlaneTemplateCacheKey"
    )
    #expect(cacheEntry.contains("packetDescriptorCount: Int"))
    #expect(!cacheEntry.contains("packetDescriptors: [MetalSparsePacketEncodeDescriptor]"))

    let cacheBody = try codecSource.requiredSlice(
        from: "private func cachedSparsePacketEncodeDescriptors(",
        to: "    private func sparsePacketEncodeDescriptorCacheKey("
    )
    #expect(cacheBody.contains("makeStaticSharedBuffer(bytes: packetDescriptors)"))
    #expect(cacheBody.contains("packetDescriptorCount: packetDescriptors.count"))
    #expect(cacheBody.contains("sparsePacketEncodeDescriptorCache[key]"))

    let metalSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/MetalBackend.swift"), encoding: .utf8)
    let residentPacketBody = try metalSource.requiredSlice(
        from: "func encodeSparsePacketBuffersBatchResidentQScales(",
        to: "    private func emptySparsePacketOutputBuffer()"
    )
    #expect(residentPacketBody.contains("descriptorBuffer: MTLBuffer?"))
    #expect(residentPacketBody.contains("cachedDescriptorBuffer"))
    #expect(residentPacketBody.contains("outputStorageMode: MTLStorageMode"))
    #expect(residentPacketBody.contains("reusesOutputBuffers: Bool = false"))
    #expect(residentPacketBody.contains("purpose: .sparsePacketOutput"))
    #expect(residentPacketBody.contains(".storageModePrivate"))

    let readbackBody = try metalSource.requiredSlice(
        from: "func sharedReadbackBuffer(",
        to: "    private func emptySparsePacketOutputBuffer()"
    )
    #expect(readbackBody.contains("makeBlitCommandEncoder()"))
    #expect(readbackBody.contains("blitEncoder.copy"))
}

@Test func retainedGPUFramesDoNotShareSparsePacketPayloadOrSizeBuffers() throws {
    do {
        let backend = try MetalPyrowaveBackend()
        let codec = try PyrowaveCodec()
        let first = try TestFrames.synthetic420(width: 64, height: 64, frameIndex: 0)
        let second = try TestFrames.synthetic420(width: 64, height: 64, frameIndex: 1)
        let firstTextures = try first.makeMetalTextures(device: backend.device)
        let secondTextures = try second.makeMetalTextures(device: backend.device)
        let firstCbCrTexture = try makeNV12ChromaTexture(cb: first.cb, cr: first.cr, device: backend.device)
        let secondCbCrTexture = try makeNV12ChromaTexture(cb: second.cb, cr: second.cr, device: backend.device)
        let firstFrame = try codec.encodeGPUFrame(
            yTexture: firstTextures.y,
            cbCrTexture: firstCbCrTexture,
            videoSignal: first.videoSignal
        )
        let secondFrame = try codec.encodeGPUFrame(
            yTexture: secondTextures.y,
            cbCrTexture: secondCbCrTexture,
            videoSignal: second.videoSignal
        )
        let planeCount = min(firstFrame.planes.count, secondFrame.planes.count)
        #expect(planeCount > 0)
        for index in 0..<planeCount {
            #expect(firstFrame.planes[index].encoded.outputBuffer !== secondFrame.planes[index].encoded.outputBuffer)
            #expect(firstFrame.planes[index].encoded.sizeBuffer !== secondFrame.planes[index].encoded.sizeBuffer)
        }
    } catch PyrowaveError.externalToolUnavailable {
        return
    }
}

@Test func gpuFrameDecodeUsesResidentSparsePacketDecodeDescriptorBuffers() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let codecSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/Codec.swift"), encoding: .utf8)
    let gpuFramePlane = try codecSource.requiredSlice(
        from: "struct PyrowaveGPUFramePlane {",
        to: "public final class PyrowaveCodec"
    )
    #expect(gpuFramePlane.contains("decodeDescriptorCount: Int"))
    #expect(gpuFramePlane.contains("decodeDescriptorBuffer: MTLBuffer"))
    #expect(!gpuFramePlane.contains("decodeDescriptorsCoverFullPlane"))
    #expect(!gpuFramePlane.contains("decodeDescriptors: [MetalSparsePacketDecodeDescriptor]"))

    let gpuDecodeBody = try codecSource.requiredSlice(
        from: "public func decodeGPUFrameToNV12Textures(",
        to: "    public func exportGPUFrame("
    )
    #expect(gpuDecodeBody.contains("descriptorCount: $0.decodeDescriptorCount"))
    #expect(gpuDecodeBody.contains("descriptorBuffer: $0.decodeDescriptorBuffer"))
    #expect(!gpuDecodeBody.contains("zeroOutputFromDescriptors"))
    #expect(!gpuDecodeBody.contains("decodeDescriptors: [MetalSparsePacketDecodeDescriptor]"))

    let gpuFramePlaneBody = try codecSource.requiredSlice(
        from: "private func makeGPUFramePlanes(",
        to: "    private func cachedSparsePacketEncodeDescriptors("
    )
    #expect(gpuFramePlaneBody.contains("decodeDescriptorCountsByPlane"))
    #expect(gpuFramePlaneBody.contains("makeStaticSharedBuffer(bytes: decodeDescriptors)"))
    #expect(gpuFramePlaneBody.contains("cached.decodeDescriptorBuffer"))
    #expect(!gpuFramePlaneBody.contains("decodeDescriptorsCoverFullPlane"))
    #expect(!gpuFramePlaneBody.contains("decodeDescriptorsByPlane"))

    let importBody = try codecSource.requiredSlice(
        from: "public func importGPUFrame(",
        to: "    private func encodeFrame("
    )
    #expect(!importBody.contains("decodeDescriptorsCoverFullPlane"))

    let metalSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/MetalBackend.swift"), encoding: .utf8)
    let gpuDecodeBackend = try metalSource.requiredSlice(
        from: "func decodeSparsePacketBuffersInverseAndCropToNV12Textures(",
        to: "    func quantize("
    )
    #expect(gpuDecodeBackend.contains("descriptorCount: Int"))
    #expect(gpuDecodeBackend.contains("descriptorBuffer: MTLBuffer"))
    #expect(gpuDecodeBackend.contains("outputStorageMode: .private"))
    #expect(!gpuDecodeBackend.contains("zeroOutputFromDescriptors"))
    #expect(!gpuDecodeBackend.contains("descriptors: [MetalSparsePacketDecodeDescriptor]"))

    let kernelsSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/Metal/PyrowaveKernels.metal"), encoding: .utf8)
    let sparsePacketDecodeKernel = try kernelsSource.requiredSlice(
        from: "kernel void pyrowave_decode_sparse_packets_threadgroup(",
        to: "kernel void pyrowave_rate_control_bucket_indices("
    )
    #expect(!sparsePacketDecodeKernel.contains("constants.zeroOutputFromDescriptors"))

    let sparseOutputKey = try metalSource.requiredSlice(
        from: "private struct SparseCoefficientOutputKey",
        to: "private enum ReusableBufferPurpose"
    )
    #expect(sparseOutputKey.contains("storageModeRawValue"))

    let sparseOutputBody = try metalSource.requiredSlice(
        from: "private func sparseCoefficientOutput(",
        to: "    private func reusableSharedBuffer<T>"
    )
    #expect(sparseOutputBody.contains("storageMode: MTLStorageMode"))
    #expect(sparseOutputBody.contains(".storageModePrivate"))
}

@Test func gpuFrameKeepsSelectedQuantLevelsResidentForInspection() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let codecSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/Codec.swift"), encoding: .utf8)
    let gpuFramePlane = try codecSource.requiredSlice(
        from: "struct PyrowaveGPUFramePlane {",
        to: "public final class PyrowaveCodec"
    )
    #expect(gpuFramePlane.contains("selectedQuantLevelCount: Int"))
    #expect(gpuFramePlane.contains("selectedQuantLevelBuffer: MTLBuffer"))
    #expect(!gpuFramePlane.contains("selectedQuantLevels: [Int]"))

    let inspectionAccessor = try codecSource.requiredSlice(
        from: "public var selectedQuantLevelsByPlane: [[Int]] {",
        to: "    public func encodedByteCountForInspection()"
    )
    #expect(inspectionAccessor.contains("selectedQuantLevelBuffer.contents()"))
    #expect(inspectionAccessor.contains("bindMemory(to: UInt32.self"))

    let gpuFramePlaneBody = try codecSource.requiredSlice(
        from: "private func makeGPUFramePlanes(",
        to: "    private func cachedSparsePacketEncodeDescriptors("
    )
    #expect(gpuFramePlaneBody.contains("selectedQuantLevelBuffersByPlane"))
    #expect(gpuFramePlaneBody.contains("makeStaticSharedBuffer(bytes: selectedQuantLevels)"))
    #expect(gpuFramePlaneBody.contains("cached.selectedQuantLevelBuffer"))

    let cacheEntry = try codecSource.requiredSlice(
        from: "private struct SparsePacketEncodeDescriptorCacheEntry {",
        to: "    private struct DecodedPlaneTemplateCacheKey"
    )
    #expect(cacheEntry.contains("selectedQuantLevelCount: Int"))
    #expect(cacheEntry.contains("selectedQuantLevelBuffer: MTLBuffer"))
    #expect(!cacheEntry.contains("selectedQuantLevels: [Int]"))
}

@Test func gpuFrameKeepsBlockIndicesResidentForExportInspection() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let codecSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/Codec.swift"), encoding: .utf8)
    let gpuFramePlane = try codecSource.requiredSlice(
        from: "struct PyrowaveGPUFramePlane {",
        to: "public final class PyrowaveCodec"
    )
    #expect(gpuFramePlane.contains("blockIndexCount: Int"))
    #expect(gpuFramePlane.contains("blockIndexBuffer: MTLBuffer"))
    #expect(gpuFramePlane.contains("blockIndicesForInspection()"))
    #expect(!gpuFramePlane.contains("blockIndices: [Int]"))

    let exportBody = try codecSource.requiredSlice(
        from: "public func exportGPUFrame(",
        to: "    public func importGPUFrame("
    )
    #expect(exportBody.contains("blockIndicesForInspection()"))
    #expect(exportBody.contains("plane.encoded.descriptorCount == plane.blockIndexCount"))

    let gpuFramePlaneBody = try codecSource.requiredSlice(
        from: "private func makeGPUFramePlanes(",
        to: "    private func cachedSparsePacketEncodeDescriptors("
    )
    #expect(gpuFramePlaneBody.contains("blockIndexBuffersByPlane"))
    #expect(gpuFramePlaneBody.contains("makeStaticSharedBuffer(bytes: blockIndices)"))
    #expect(gpuFramePlaneBody.contains("cached.blockIndexBuffer"))

    let cacheEntry = try codecSource.requiredSlice(
        from: "private struct SparsePacketEncodeDescriptorCacheEntry {",
        to: "    private struct DecodedPlaneTemplateCacheKey"
    )
    #expect(cacheEntry.contains("blockIndexCount: Int"))
    #expect(cacheEntry.contains("blockIndexBuffer: MTLBuffer"))
    #expect(!cacheEntry.contains("blockIndices: [Int]"))
}

@Test func residentQuantizedCoefficientBuffersAreReusableBuffers() throws {
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let source = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/MetalBackend.swift"), encoding: .utf8)
    let backendBody = try source.requiredSlice(
        from: "func quantizePlaneBufferResults(",
        to: "    func applySparseCoefficients("
    )
    #expect(backendBody.contains("reusesOutputBuffers: Bool = false"))
    #expect(backendBody.contains("if reusesOutputBuffers"))
    #expect(backendBody.contains("purpose: .quantizedCoefficient"))

    let codecSource = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/Codec.swift"), encoding: .utf8)
    let residentBody = try codecSource.requiredSlice(
        from: "private func quantizeResidentBuffers(",
        to: "    private func quantizeWithMetal("
    )
    #expect(residentBody.contains("reusesOutputBuffers: true"))
}

@Test func yuvFrameImportsCoreVideoNV12PixelBuffer() throws {
    var pixelBuffer: CVPixelBuffer?
    let attributes = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary
    let status = CVPixelBufferCreate(
        nil,
        4,
        4,
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        attributes,
        &pixelBuffer
    )
    #expect(status == kCVReturnSuccess)
    let importedPixelBuffer = try #require(pixelBuffer)

    CVPixelBufferLockBaseAddress(importedPixelBuffer, [])
    let yStride = CVPixelBufferGetBytesPerRowOfPlane(importedPixelBuffer, 0)
    let cbCrStride = CVPixelBufferGetBytesPerRowOfPlane(importedPixelBuffer, 1)
    let yBase = try #require(CVPixelBufferGetBaseAddressOfPlane(importedPixelBuffer, 0)?.assumingMemoryBound(to: UInt8.self))
    let cbCrBase = try #require(CVPixelBufferGetBaseAddressOfPlane(importedPixelBuffer, 1)?.assumingMemoryBound(to: UInt8.self))
    let yRows: [[UInt8]] = [
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33]
    ]
    let cbCrRows: [[UInt8]] = [
        [100, 150, 101, 151],
        [102, 152, 103, 153]
    ]
    for row in 0..<4 {
        let destination = yBase.advanced(by: row * yStride)
        for column in 0..<4 {
            destination[column] = yRows[row][column]
        }
    }
    for row in 0..<2 {
        let destination = cbCrBase.advanced(by: row * cbCrStride)
        for column in 0..<4 {
            destination[column] = cbCrRows[row][column]
        }
    }
    CVPixelBufferUnlockBaseAddress(importedPixelBuffer, [])

    let frame = try YUVFrame(cvPixelBuffer: importedPixelBuffer)
    #expect(frame.chroma == ChromaSubsampling.yuv420)
    #expect(frame.y.data == yRows.flatMap { $0 })
    #expect(frame.cb.data == [100, 101, 102, 103])
    #expect(frame.cr.data == [150, 151, 152, 153])
    #expect(frame.videoSignal.yCbCrRange == YCbCrRange.limited)

    let exportedPixelBuffer = try frame.makeCVPixelBuffer()
    #expect(CVPixelBufferGetPixelFormatType(exportedPixelBuffer) == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
    let exportedFrame = try YUVFrame(cvPixelBuffer: exportedPixelBuffer)
    #expect(exportedFrame == frame)
}

@Test func codecEncodesAndDecodesCoreVideoPixelBuffers() throws {
    let source = try TestFrames.synthetic420(width: 64, height: 64)
    let sourcePixelBuffer = try source.makeCVPixelBuffer()
    let codec = try PyrowaveCodec()

    let gpuFrame = try codec.encodeGPUFrame(
        sourcePixelBuffer,
        configuration: CodecConfiguration(quantizationStep: 1.0 / 2048.0)
    )
    let gpuDecodedPixelBuffer = try codec.decodeGPUFrameToCVPixelBuffer(gpuFrame)
    try codec.decodeGPUFrame(gpuFrame, to: gpuDecodedPixelBuffer)
    let gpuDecoded = try YUVFrame(cvPixelBuffer: gpuDecodedPixelBuffer)
    #expect(CVPixelBufferGetPixelFormatType(gpuDecodedPixelBuffer) == YUVFrame.cvPixelFormat(for: source.videoSignal))
    #expect(gpuDecoded.width == source.width)
    #expect(gpuDecoded.height == source.height)
    #expect(gpuDecoded.chroma == .yuv420)
    #expect(try Metrics.compare(source, gpuDecoded).weightedPSNR > 44.0)

    let encoded = try codec.encode(
        sourcePixelBuffer,
        configuration: CodecConfiguration(quantizationStep: 1.0 / 2048.0)
    )
    let decodedPixelBuffer = try codec.decodeToCVPixelBuffer(encoded)
    let decoded = try YUVFrame(cvPixelBuffer: decodedPixelBuffer)
    let metrics = try Metrics.compare(source, decoded)

    #expect(encoded.data.count > 0)
    #expect(decoded.width == source.width)
    #expect(decoded.height == source.height)
    #expect(decoded.chroma == .yuv420)
    #expect(metrics.weightedPSNR > 44.0)

    let packets = try encoded.packetized(maximumPacketBytes: 64)
    let stream = try PyrowavePacketStreamDecoder(width: source.width, height: source.height, chroma: source.chroma)
    for packet in packets {
        try stream.pushPacket(packet)
    }
    let streamedPixelBuffer = try stream.decodeToCVPixelBuffer()
    let streamed = try YUVFrame(cvPixelBuffer: streamedPixelBuffer)
    #expect(try Metrics.compare(source, streamed).weightedPSNR > 44.0)
}

@Test func codecOwnedQualityMappingUsesNearLosslessHighestPreset() {
    #expect(PyrowaveQuality(normalized: 1).codecConfiguration.quantizationStep == 1.0 / 2048.0)
    #expect(PyrowaveQuality(normalized: 0).codecConfiguration.quantizationStep == 1.0 / 64.0)
    #expect(PyrowaveQuality(configuration: CodecConfiguration(quantizationStep: 1.0 / 2048.0)) == .highest)
    #expect(PyrowaveQuality(normalized: -1).normalized == 0)
    #expect(PyrowaveQuality(normalized: 2).normalized == 1)
    #expect(PyrowaveQuality(normalized: .nan).normalized == 1)
}

@Test func productionDecoderRestoresVideoSignalAttachments() async throws {
    let source = try TestFrames.synthetic420(width: 64, height: 64)
    let signal = VideoSignalMetadata(
        colorPrimaries: .bt709,
        transferFunction: .sRGB,
        yCbCrTransform: .bt709,
        yCbCrRange: .full,
        chromaSiting: .center
    )
    let descriptor = try PyrowaveSessionDescriptor(width: 64, height: 64, videoSignal: signal)
    let encoder = try PyrowaveEncoderSession(descriptor: descriptor)
    let decoder = try PyrowaveDecoderSession(descriptor: descriptor)
    let encoded = try await encoder.encode(source.makeCVPixelBuffer())
    let decoded = try await decoder.decode(encoded.frame)

    #expect(
        CVBufferCopyAttachment(
            decoded.pixelBuffer,
            kCVImageBufferColorPrimariesKey,
            nil
        ) as? String == kCVImageBufferColorPrimaries_ITU_R_709_2 as String
    )
    #expect(
        CVBufferCopyAttachment(
            decoded.pixelBuffer,
            kCVImageBufferTransferFunctionKey,
            nil
        ) as? String == kCVImageBufferTransferFunction_sRGB as String
    )
    #expect(
        CVBufferCopyAttachment(
            decoded.pixelBuffer,
            kCVImageBufferYCbCrMatrixKey,
            nil
        ) as? String == kCVImageBufferYCbCrMatrix_ITU_R_709_2 as String
    )
    #expect(CVBufferCopyAttachment(decoded.pixelBuffer, kCVImageBufferCGColorSpaceKey, nil) != nil)
}

@Test func serialSessionsRoundTripExistingBitstreamWithPooledOutput() async throws {
    let source = try TestFrames.synthetic420(width: 64, height: 64)
    let sourcePixelBuffer = try source.makeCVPixelBuffer()
    let descriptor = try PyrowaveSessionDescriptor(
        width: source.width,
        height: source.height,
        pixelFormat: YUVFrame.cvPixelFormat(for: source.videoSignal),
        videoSignal: source.videoSignal
    )
    let encoder = try PyrowaveEncoderSession(descriptor: descriptor)
    let decoder = try PyrowaveDecoderSession(descriptor: descriptor)

    let encoded = try await encoder.encode(sourcePixelBuffer, quality: PyrowaveQuality(normalized: 0.75))
    let oldDecoderOutput = try PyrowaveCodec().decodeToCVPixelBuffer(encoded.frame)
    let sessionDecoded = try await decoder.decode(encoded.frame)

    #expect(encoded.metrics.encodedBytes == encoded.frame.data.count)
    #expect(encoded.metrics.payloadBytesCopied + 8 == encoded.metrics.encodedBytes)
    #expect(encoded.metrics.payloadBytesCopied < encoded.metrics.packetSlotCapacityBytes)
    #expect(encoded.metrics.totalMilliseconds >= encoded.metrics.encodeMilliseconds)
    #expect(encoded.metrics.totalMilliseconds >= encoded.metrics.exportMilliseconds)
    #expect(sessionDecoded.metrics.encodedBytes == encoded.frame.data.count)
    #expect(sessionDecoded.metrics.totalMilliseconds >= sessionDecoded.metrics.contiguousDecodeMilliseconds)
    #expect(CVPixelBufferGetWidth(sessionDecoded.pixelBuffer) == source.width)
    #expect(CVPixelBufferGetHeight(sessionDecoded.pixelBuffer) == source.height)

    let oldDecoded = try YUVFrame(cvPixelBuffer: oldDecoderOutput)
    let newDecoded = try YUVFrame(cvPixelBuffer: sessionDecoded.pixelBuffer)
    #expect(oldDecoded == newDecoded)
    #expect(try Metrics.compare(source, newDecoded).weightedPSNR > 30)
}

@Test func decoderSessionClassifiesCorruptFramesAsRecoverable() async throws {
    let descriptor = try PyrowaveSessionDescriptor(width: 64, height: 64)
    let decoder = try PyrowaveDecoderSession(descriptor: descriptor)

    do {
        _ = try await decoder.decode(EncodedFrame(data: Data([0, 1, 2])))
        Issue.record("Corrupt frame unexpectedly decoded")
    } catch let error as PyrowaveSessionError {
        #expect(error.isRecoverable)
    }
}

@Test func roundTripSynthetic420() throws {
    let frame = try TestFrames.synthetic420(width: 160, height: 96)
    let codec = try PyrowaveCodec()
    let encoded = try codec.encode(frame, configuration: CodecConfiguration(quantizationStep: 1.0 / 2048.0))
    let decoded = try codec.decode(encoded)
    let metrics = try Metrics.compare(frame, decoded)

    #expect(decoded.width == frame.width)
    #expect(decoded.height == frame.height)
    #expect(decoded.chroma == .yuv420)
    #expect(metrics.y.psnr > 46.0)
    #expect(metrics.weightedPSNR > 44.0)
}

@Test func roundTripSynthetic444() throws {
    let frame = try TestFrames.synthetic444(width: 128, height: 96)
    let codec = try PyrowaveCodec()
    let encoded = try codec.encode(frame, configuration: CodecConfiguration(quantizationStep: 1.0 / 2048.0))
    let decoded = try codec.decode(encoded)
    let metrics = try Metrics.compare(frame, decoded)

    #expect(decoded.width == frame.width)
    #expect(decoded.height == frame.height)
    #expect(decoded.chroma == .yuv444)
    #expect(metrics.y.psnr > 42.0)
    #expect(metrics.weightedPSNR > 38.0)
}

@Test func waveletPaddingExtendsSourceEdgesBeforeDWTMirrorFiltering() throws {
    let plane = try Plane8(width: 3, height: 2, data: [
        0, 50, 100,
        150, 200, 250
    ])
    let padded = Wavelet.padPlane(plane, paddedWidth: 6, paddedHeight: 5)
    let denormalized = padded.samples.map { UInt8((($0 + 0.5) * 255.0).rounded()) }

    #expect(padded.width == 6)
    #expect(padded.height == 5)
    #expect(Array(denormalized[0..<6]) == [0, 50, 100, 100, 100, 100])
    #expect(Array(denormalized[6..<12]) == [150, 200, 250, 250, 250, 250])
    #expect(Array(denormalized[12..<18]) == [150, 200, 250, 250, 250, 250])
}

@Test func roundTripNonAlignedSynthetic420UsesEdgeExtendedPadding() throws {
    let frame = try TestFrames.synthetic420(width: 130, height: 74)
    let codec = try PyrowaveCodec()
    let encoded = try codec.encode(frame, configuration: CodecConfiguration(quantizationStep: 1.0 / 2048.0))
    let decoded = try codec.decode(encoded)
    let metrics = try Metrics.compare(frame, decoded)

    #expect(decoded.width == frame.width)
    #expect(decoded.height == frame.height)
    #expect(decoded.chroma == .yuv420)
    #expect(metrics.weightedPSNR > 36.0)
}

@Test func sequenceMetricsIncludeEveryFrame() throws {
    let black = try YUVFrame(
        width: 2,
        height: 2,
        chroma: .yuv420,
        y: Plane8(width: 2, height: 2, data: [0, 0, 0, 0]),
        cb: Plane8(width: 1, height: 1, data: [0]),
        cr: Plane8(width: 1, height: 1, data: [0])
    )
    let shifted = try YUVFrame(
        width: 2,
        height: 2,
        chroma: .yuv420,
        y: Plane8(width: 2, height: 2, data: [10, 10, 10, 10]),
        cb: Plane8(width: 1, height: 1, data: [10]),
        cr: Plane8(width: 1, height: 1, data: [10])
    )

    let firstFrameOnly = try Metrics.compare(black, black)
    let sequence = try Metrics.compare([black, black], [black, shifted])

    #expect(firstFrameOnly.weightedPSNR == 999.0)
    #expect(sequence.y.mse == 50.0)
    #expect(sequence.cb.mse == 50.0)
    #expect(sequence.cr.mse == 50.0)
    #expect(sequence.weightedPSNR < firstFrameOnly.weightedPSNR)
    #expect(throws: PyrowaveError.invalidDimensions) {
        _ = try Metrics.compare([black], [black, shifted])
    }
}

@Test func yuv4mpegReadWrite() throws {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("sample.y4m")
    let frame = try TestFrames.synthetic420(width: 64, height: 64)

    var writer = try YUV4MPEGWriter(url: url, width: frame.width, height: frame.height, chroma: frame.chroma)
    try writer.writeFrame(frame)

    var reader = try YUV4MPEGReader(url: url)
    #expect(reader.frameRateNumerator == 60)
    #expect(reader.frameRateDenominator == 1)
    let decoded = try #require(try reader.readFrame())
    #expect(decoded == frame)
}

@Test func yuvFrameImportsStrideAwareNV12CPUBuffer() throws {
    let y: [UInt8] = [
        0, 1, 2, 3, 90, 91,
        10, 11, 12, 13, 92, 93,
        20, 21, 22, 23, 94, 95,
        30, 31, 32, 33, 96, 97
    ]
    let cbCr: [UInt8] = [
        100, 150, 101, 151, 80, 81, 82, 83,
        102, 152, 103, 153, 84, 85, 86, 87
    ]
    let videoSignal = VideoSignalMetadata(
        colorPrimaries: .bt2020,
        transferFunction: .pq,
        yCbCrTransform: .bt2020,
        yCbCrRange: .limited,
        chromaSiting: .left
    )

    let frame = try YUVFrame(
        width: 4,
        height: 4,
        nv12Y: y,
        nv12CbCr: cbCr,
        yRowStride: 6,
        cbCrRowStride: 8,
        videoSignal: videoSignal
    )

    #expect(frame.chroma == .yuv420)
    #expect(frame.y.data == [
        0, 1, 2, 3,
        10, 11, 12, 13,
        20, 21, 22, 23,
        30, 31, 32, 33
    ])
    #expect(frame.cb.data == [100, 101, 102, 103])
    #expect(frame.cr.data == [150, 151, 152, 153])
    #expect(frame.videoSignal == videoSignal)

    let exported = try frame.nv12Planes(yRowStride: 6, cbCrRowStride: 8)
    #expect(exported.y == [
        0, 1, 2, 3, 0, 0,
        10, 11, 12, 13, 0, 0,
        20, 21, 22, 23, 0, 0,
        30, 31, 32, 33, 0, 0
    ])
    #expect(exported.cbCr == [
        100, 150, 101, 151, 0, 0, 0, 0,
        102, 152, 103, 153, 0, 0, 0, 0
    ])

    #expect(throws: PyrowaveError.invalidDimensions) {
        _ = try YUVFrame(width: 3, height: 4, nv12Y: y, nv12CbCr: cbCr)
    }
    #expect(throws: PyrowaveError.invalidDimensions) {
        _ = try YUVFrame(width: 4, height: 4, nv12Y: y, nv12CbCr: cbCr, yRowStride: 3)
    }
    #expect(throws: PyrowaveError.invalidDimensions) {
        _ = try YUVFrame(width: 4, height: 4, nv12Y: Array(y.dropLast(3)), nv12CbCr: cbCr, yRowStride: 6)
    }
    #expect(throws: PyrowaveError.invalidDimensions) {
        _ = try YUVFrame(width: 4, height: 4, nv12Y: y, nv12CbCr: Array(cbCr.dropLast(5)), cbCrRowStride: 8)
    }
    let yuv444 = try TestFrames.synthetic444(width: 4, height: 4)
    #expect(throws: PyrowaveError.unsupportedFormat("NV12 export expects yuv420 frames")) {
        _ = try yuv444.nv12Planes()
    }
}

@Test func yuv4mpegReadWrite444AndRejectsMismatchedWriterChroma() throws {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("sample444.y4m")
    let frame = try TestFrames.synthetic444(width: 48, height: 32)

    var writer = try YUV4MPEGWriter(url: url, width: frame.width, height: frame.height, chroma: frame.chroma)
    try writer.writeFrame(frame)

    var reader = try YUV4MPEGReader(url: url)
    let decoded = try #require(try reader.readFrame())
    #expect(decoded == frame)

    let mismatched = try TestFrames.synthetic420(width: 48, height: 32)
    #expect(throws: PyrowaveError.invalidDimensions) {
        try writer.writeFrame(mismatched)
    }
}

@Test func yuv4mpegSequenceWriterPreservesFrameRateAndTruncatesExistingFile() throws {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("sequence.y4m")
    let first = try TestFrames.synthetic420(width: 32, height: 32, frameIndex: 0)
    let second = try TestFrames.synthetic420(width: 32, height: 32, frameIndex: 1)

    try YUV4MPEGWriter.write(
        frames: [first, second],
        to: url,
        frameRateNumerator: 30000,
        frameRateDenominator: 1001
    )
    try YUV4MPEGWriter.write(
        frames: [second],
        to: url,
        frameRateNumerator: 30000,
        frameRateDenominator: 1001
    )

    var reader = try YUV4MPEGReader(url: url)
    #expect(reader.frameRateNumerator == 30000)
    #expect(reader.frameRateDenominator == 1001)
    #expect(try reader.readFrame() == second)
    #expect(try reader.readFrame() == nil)
}

@Test func yuv4mpegReadsHighBitDepth420AndRangeMetadata() throws {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("sample10.y4m")

    var data = Data("YUV4MPEG2 W2 H2 F30000:1001 Ip A1:1 C420p10 XCOLORRANGE=LIMITED\nFRAME\n".utf8)
    for sample in [0, 512, 1023, 256, 128, 768] {
        data.append(UInt8(sample & 0xff))
        data.append(UInt8((sample >> 8) & 0xff))
    }
    try data.write(to: url)

    var reader = try YUV4MPEGReader(url: url)
    #expect(reader.bitDepth == 10)
    #expect(reader.chroma == .yuv420)
    #expect(reader.frameRateNumerator == 30000)
    #expect(reader.frameRateDenominator == 1001)
    let frame = try #require(try reader.readFrame())
    #expect(frame.videoSignal.yCbCrRange == .limited)
    #expect(frame.y.data == [0, 128, 255, 64])
    #expect(frame.cb.data == [32])
    #expect(frame.cr.data == [191])
}

@Test func yuv4mpegReadsHighBitDepth444FullRange() throws {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("sample444-12.y4m")

    var data = Data("YUV4MPEG2 W1 H1 F60:1 Ip A1:1 C444p12 XCOLORRANGE=FULL\nFRAME\n".utf8)
    for sample in [4095, 2048, 0] {
        data.append(UInt8(sample & 0xff))
        data.append(UInt8((sample >> 8) & 0xff))
    }
    try data.write(to: url)

    var reader = try YUV4MPEGReader(url: url)
    #expect(reader.bitDepth == 12)
    #expect(reader.chroma == .yuv444)
    let frame = try #require(try reader.readFrame())
    #expect(frame.videoSignal.yCbCrRange == .full)
    #expect(frame.y.data == [255])
    #expect(frame.cb.data == [128])
    #expect(frame.cr.data == [0])
}

@Test func yuv4mpegRejectsBadFrameRate() throws {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("bad-rate.y4m")
    try Data("YUV4MPEG2 W2 H2 F60:0 Ip A1:1 C420jpeg\n".utf8).write(to: url)

    #expect(throws: PyrowaveError.unsupportedFormat("bad frame rate F60:0")) {
        _ = try YUV4MPEGReader(url: url)
    }
}

@Test func hevcComparisonUsesExactFrameDuration() throws {
    let sixty = try HEVCComparison.frameDuration(numerator: 60, denominator: 1)
    #expect(sixty.value == 1)
    #expect(sixty.timescale == 60)

    let ntsc = try HEVCComparison.frameDuration(numerator: 30000, denominator: 1001)
    #expect(ntsc.value == 1001)
    #expect(ntsc.timescale == 30000)

    #expect(throws: PyrowaveError.invalidDimensions) {
        _ = try HEVCComparison.frameDuration(numerator: 0, denominator: 1)
    }
    #expect(throws: PyrowaveError.invalidDimensions) {
        _ = try HEVCComparison.frameDuration(numerator: 60, denominator: 0)
    }
}

@Test func hevcComparisonUsesMirageRealtimeVideoToolboxPolicy() throws {
    #expect(HEVCComparison.defaultQuality == 0.8)
    let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let source = try String(contentsOf: packageRoot.appendingPathComponent("Sources/PyrowaveKit/HEVCComparison.swift"), encoding: .utf8)
    let encodeBody = try source.requiredSlice(
        from: "private static func encodeMirageHEVC(",
        to: "    private static func decodeMirageHEVC("
    )
    let configurationBody = try source.requiredSlice(
        from: "private static func configureMirageCompressionSession(",
        to: "    private static func qualitySettings("
    )
    #expect(encodeBody.contains("VTCompressionSessionEncodeFrame"))
    #expect(!source.contains("AVAssetWriter"))
    #expect(configurationBody.contains("kVTCompressionPropertyKey_RealTime"))
    #expect(configurationBody.contains("kVTCompressionPropertyKey_AllowFrameReordering"))
    #expect(configurationBody.contains("kVTCompressionPropertyKey_MaxFrameDelayCount"))
    #expect(configurationBody.contains("kVTCompressionPropertyKey_PrioritizeEncodingSpeedOverQuality"))
    #expect(configurationBody.contains("kVTCompressionPropertyKey_Quality"))
    #expect(configurationBody.contains("kVTCompressionPropertyKey_MinAllowedFrameQP"))
    #expect(configurationBody.contains("kVTCompressionPropertyKey_MaxAllowedFrameQP"))
    #expect(configurationBody.contains("kVTCompressionPropertyKey_AverageBitRate"))
    #expect(configurationBody.contains("kVTCompressionPropertyKey_DataRateLimits"))
    #expect(HEVCComparison.mirageTimingNote.contains("Quality capped at 0.8"))
    #expect(HEVCComparison.mirageTimingNote.contains("AverageBitRate plus DataRateLimits"))
}

@Test func hevcComparisonRejectsInvalidInputsBeforeEncoding() throws {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString, isDirectory: true)
    let frame = try TestFrames.synthetic420(width: 64, height: 64)
    let mismatched = try TestFrames.synthetic420(width: 96, height: 64)
    let yuv444 = try TestFrames.synthetic444(width: 64, height: 64)

    #expect(throws: PyrowaveError.truncatedInput) {
        _ = try HEVCComparison.runMirageHEVCComparison(
            referenceFrames: [],
            workingDirectory: directory,
            bitrate: 1
        )
    }
    #expect(throws: PyrowaveError.invalidDimensions) {
        _ = try HEVCComparison.runMirageHEVCComparison(
            referenceFrames: [frame],
            workingDirectory: directory,
            bitrate: 0
        )
    }
    #expect(throws: PyrowaveError.invalidDimensions) {
        _ = try HEVCComparison.runMirageHEVCComparison(
            referenceFrames: [frame, mismatched],
            workingDirectory: directory,
            bitrate: 1
        )
    }
    #expect(throws: PyrowaveError.unsupportedFormat("HEVC comparison expects yuv420 frames")) {
        _ = try HEVCComparison.runMirageHEVCComparison(
            referenceFrames: [yuv444],
            workingDirectory: directory,
            bitrate: 1
        )
    }
}

@Test func codecBenchmarkResultReportsPerFrameNormalization() throws {
    let result = CodecBenchmarkResult(
        codec: "sample",
        frameCount: 4,
        encodedBytes: 100,
        encodeSeconds: 0.020,
        decodeSeconds: 0.012,
        metrics: nil,
        note: nil
    )

    #expect(result.frameCount == 4)
    #expect(result.encodedBytesPerFrame == 25.0)
    #expect(abs(result.encodeMillisecondsPerFrame - 5.0) < 0.0001)
    #expect(abs(result.decodeMillisecondsPerFrame - 3.0) < 0.0001)

    let empty = CodecBenchmarkResult(
        codec: "empty",
        frameCount: 0,
        encodedBytes: 100,
        encodeSeconds: 0.020,
        decodeSeconds: 0.012,
        metrics: nil,
        note: nil
    )
    #expect(empty.encodedBytesPerFrame == 0)
    #expect(empty.encodeMillisecondsPerFrame == 0)
    #expect(empty.decodeMillisecondsPerFrame == 0)
}

@Test func benchmarkArgumentsDefaultToGitIgnoredSixKBaseline() throws {
    let arguments = try PyrowaveBenchmarkArguments([])
    #expect(arguments.input == nil)
    #expect(arguments.frames == 60)
    #expect(arguments.width == 6144)
    #expect(arguments.height == 3456)
    #expect(arguments.bitrate == 80_000_000)
    #expect(arguments.hevcQuality == 0.8)
    #expect(arguments.quantizationStep == 1.0 / 1024.0)
    #expect(arguments.outputDirectory.path.hasSuffix(".pyrowave-results"))
    #expect(arguments.requiredPyrowaveEncodeSpeedup == nil)
    #expect(arguments.requiredPyrowaveDecodeSpeedup == nil)
    #expect(!arguments.pyrowaveOnly)
    #expect(!arguments.shouldShowHelp)
}

@Test func benchmarkArgumentsParsePresetsAndQualityModes() throws {
    let custom = try PyrowaveBenchmarkArguments([
        "--preset", "4k",
        "--frames", "12",
        "--output-dir", ".pyrowave-results/custom",
        "--bitrate", "40000000",
        "--hevc-quality", "0.75",
        "--quantization-step", "0.002",
        "--pyrowave-only",
        "--require-pyrowave-encode-speedup", "1.25",
        "--require-pyrowave-decode-speedup", "1.5"
    ])
    #expect(custom.width == 3840)
    #expect(custom.height == 2160)
    #expect(custom.frames == 12)
    #expect(custom.outputDirectory.path.hasSuffix(".pyrowave-results/custom"))
    #expect(custom.bitrate == 40_000_000)
    #expect(custom.hevcQuality == 0.75)
    #expect(abs(custom.quantizationStep - 0.002) < 0.000001)
    #expect(custom.requiredPyrowaveEncodeSpeedup == 1.25)
    #expect(custom.requiredPyrowaveDecodeSpeedup == 1.5)
    #expect(custom.pyrowaveOnly)

    let size = try PyrowaveBenchmarkArguments([
        "--size", "1920x1080",
        "--require-pyrowave-faster-than-hevc"
    ])
    #expect(size.width == 1920)
    #expect(size.height == 1080)
    #expect(size.requiredPyrowaveEncodeSpeedup == 1)
    #expect(size.requiredPyrowaveDecodeSpeedup == 1)
}

@Test func benchmarkArgumentsValidateInputsWithoutRunningBenchmark() throws {
    #expect(throws: PyrowaveError.invalidDimensions) {
        _ = try PyrowaveBenchmarkArguments(["--frames", "0"])
    }
    #expect(throws: PyrowaveError.invalidDimensions) {
        _ = try PyrowaveBenchmarkArguments(["--size", "bad"])
    }
    #expect(throws: PyrowaveError.invalidDimensions) {
        _ = try PyrowaveBenchmarkArguments(["--hevc-quality", "0.81"])
    }
    #expect(throws: PyrowaveError.invalidDimensions) {
        _ = try PyrowaveBenchmarkArguments(["--hevc-quality", "-0.1"])
    }
    #expect(throws: PyrowaveError.unsupportedFormat("unknown preset 8k")) {
        _ = try PyrowaveBenchmarkArguments(["--preset", "8k"])
    }
    #expect(throws: PyrowaveError.invalidDimensions) {
        _ = try PyrowaveBenchmarkArguments(["--require-pyrowave-encode-speedup", "0"])
    }
    #expect(throws: PyrowaveError.invalidDimensions) {
        _ = try PyrowaveBenchmarkArguments(["--require-pyrowave-decode-speedup", "nan"])
    }

    let help = try PyrowaveBenchmarkArguments(["--help"])
    #expect(help.shouldShowHelp)
    #expect(PyrowaveBenchmarkArguments.usage.contains("--preset 6k|4k|1080p|720p"))
    #expect(PyrowaveBenchmarkArguments.usage.contains("--pyrowave-only"))
    #expect(PyrowaveBenchmarkArguments.usage.contains("--hevc-quality Q<=0.8"))
    #expect(!PyrowaveBenchmarkArguments.usage.contains("--max-pyrowave-bytes"))
    #expect(!PyrowaveBenchmarkArguments.usage.contains("--match-hevc-frame-budget"))
    #expect(!PyrowaveBenchmarkArguments.usage.contains("--unbounded-pyrowave"))
    #expect(!PyrowaveBenchmarkArguments.usage.contains("pyrowave-frame-budget"))
    #expect(PyrowaveBenchmarkArguments.usage.contains("--pyrowave-only"))
    #expect(PyrowaveBenchmarkArguments.usage.contains("--require-pyrowave-faster-than-hevc"))
}

@Test func benchmarkInputMustProvideRequestedFrameCount() throws {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("short.y4m")
    let frame = try TestFrames.synthetic420(width: 64, height: 64)
    try YUV4MPEGWriter.write(
        frames: [frame],
        to: url,
        frameRateNumerator: 24000,
        frameRateDenominator: 1001
    )

    let oneFrame = try PyrowaveBenchmarkRunner.loadFrames(
        arguments: PyrowaveBenchmarkArguments(["--input", url.path, "--frames", "1"])
    )
    #expect(oneFrame.frames == [frame])
    #expect(oneFrame.frameRateNumerator == 24000)
    #expect(oneFrame.frameRateDenominator == 1001)

    #expect(throws: PyrowaveError.truncatedInput) {
        _ = try PyrowaveBenchmarkRunner.loadFrames(
            arguments: PyrowaveBenchmarkArguments(["--input", url.path, "--frames", "2"])
        )
    }
}

@Test func benchmarkReportSchemaNamesReviewArtifacts() throws {
    let pyrowave = CodecBenchmarkResult(
        codec: "pyrowavekit-swift-metal",
        frameCount: 60,
        encodedBytes: 240,
        encodeSeconds: 0.25,
        decodeSeconds: 0.10,
        metrics: nil,
        note: nil
    )
    let hevc = CodecBenchmarkResult(
        codec: "hevc_videotoolbox_mirage",
        frameCount: 60,
        encodedBytes: 80,
        encodeSeconds: 2.0,
        decodeSeconds: 0.50,
        metrics: nil,
        note: nil
    )
    let report = PyrowaveBenchmarkReport(
        generatedAt: "2026-07-03T00:00:00Z",
        width: PyrowaveBenchmarkArguments.defaultWidth,
        height: PyrowaveBenchmarkArguments.defaultHeight,
        frames: PyrowaveBenchmarkArguments.defaultFrames,
        frameRateNumerator: 60,
        frameRateDenominator: 1,
        bitrate: PyrowaveBenchmarkArguments.defaultBitrate,
        hevcQuality: PyrowaveBenchmarkArguments.defaultHEVCQuality,
        pyrowave: pyrowave,
        hevc: hevc,
        comparison: CodecBenchmarkComparison(pyrowave: pyrowave, hevc: hevc)
    )

    #expect(report.artifacts.referenceY4M == PyrowaveBenchmarkArtifactNames.referenceY4M)
    #expect(report.artifacts.pyrowaveStream == PyrowaveBenchmarkArtifactNames.pyrowaveStream)
    #expect(report.artifacts.pyrowaveDecodedY4M == PyrowaveBenchmarkArtifactNames.pyrowaveDecodedY4M)
    #expect(report.artifacts.hevcStream == PyrowaveBenchmarkArtifactNames.hevcStream)
    #expect(report.artifacts.hevcDecodedY4M == PyrowaveBenchmarkArtifactNames.hevcDecodedY4M)
    #expect(PyrowaveBenchmarkArtifactNames.hevcStream == "hevc-videotoolbox.mirage-hevc")
    #expect(PyrowaveBenchmarkArtifactNames.report == "benchmark-report.json")
    #expect(report.hevcQuality == 0.8)
    #expect(PyrowaveBenchmarkRunner.timedBenchmarkScopeNote.contains("artifact writes"))
    #expect(HEVCComparison.mirageTimingNote.contains("direct VideoToolbox realtime path"))
    #expect(HEVCComparison.mirageTimingNote.contains("planar conversion"))
    #expect(report.comparison.pyrowaveBytesPerFrame == 4)
    #expect(report.comparison.hevcBytesPerFrame == 80.0 / 60.0)
    #expect(report.comparison.pyrowaveToHEVCBytesPerFrameRatio == 3)

    let encoded = try JSONEncoder().encode(report)
    let json = try #require(String(data: encoded, encoding: .utf8))
    #expect(json.contains("\"pyrowaveBytesPerFrame\""))
    #expect(json.contains("\"hevcBytesPerFrame\""))
    #expect(json.contains("\"pyrowaveToHEVCBytesPerFrameRatio\""))
    let decoded = try JSONDecoder().decode(PyrowaveBenchmarkReport.self, from: encoded)
    #expect(decoded == report)
}

@Test func benchmarkArgumentsValidateRequiredPyrowaveSpeedups() throws {
    let pyrowave = CodecBenchmarkResult(
        codec: "pyrowavekit-swift-metal",
        frameCount: 60,
        encodedBytes: 240,
        encodeSeconds: 0.25,
        decodeSeconds: 0.10,
        metrics: nil,
        note: nil
    )
    let hevc = CodecBenchmarkResult(
        codec: "hevc_videotoolbox_mirage",
        frameCount: 60,
        encodedBytes: 80,
        encodeSeconds: 2.0,
        decodeSeconds: 0.50,
        metrics: nil,
        note: nil
    )
    let report = PyrowaveBenchmarkReport(
        generatedAt: "2026-07-03T00:00:00Z",
        width: 1920,
        height: 1080,
        frames: 60,
        frameRateNumerator: 60,
        frameRateDenominator: 1,
        bitrate: 20_000_000,
        hevcQuality: PyrowaveBenchmarkArguments.defaultHEVCQuality,
        pyrowave: pyrowave,
        hevc: hevc,
        comparison: CodecBenchmarkComparison(pyrowave: pyrowave, hevc: hevc)
    )

    try PyrowaveBenchmarkArguments([
        "--require-pyrowave-encode-speedup", "8",
        "--require-pyrowave-decode-speedup", "5"
    ]).validate(report: report)

    #expect(throws: PyrowaveError.processFailed("Pyrowave encode speedup over HEVC 8.0 is below required 8.1")) {
        try PyrowaveBenchmarkArguments(["--require-pyrowave-encode-speedup", "8.1"]).validate(report: report)
    }
    #expect(throws: PyrowaveError.processFailed("Pyrowave decode speedup over HEVC 5.0 is below required 5.1")) {
        try PyrowaveBenchmarkArguments(["--require-pyrowave-decode-speedup", "5.1"]).validate(report: report)
    }

    let unavailable = PyrowaveBenchmarkReport(
        generatedAt: "2026-07-03T00:00:00Z",
        width: 1920,
        height: 1080,
        frames: 0,
        frameRateNumerator: 60,
        frameRateDenominator: 1,
        bitrate: 20_000_000,
        hevcQuality: PyrowaveBenchmarkArguments.defaultHEVCQuality,
        pyrowave: CodecBenchmarkResult(codec: "pyrowave", frameCount: 0, encodedBytes: 0, encodeSeconds: 0, decodeSeconds: 0, metrics: nil, note: nil),
        hevc: CodecBenchmarkResult(codec: "hevc", frameCount: 0, encodedBytes: 0, encodeSeconds: 0, decodeSeconds: 0, metrics: nil, note: nil),
        comparison: CodecBenchmarkComparison(
            pyrowave: CodecBenchmarkResult(codec: "pyrowave", frameCount: 0, encodedBytes: 0, encodeSeconds: 0, decodeSeconds: 0, metrics: nil, note: nil),
            hevc: CodecBenchmarkResult(codec: "hevc", frameCount: 0, encodedBytes: 0, encodeSeconds: 0, decodeSeconds: 0, metrics: nil, note: nil)
        )
    )
    #expect(throws: PyrowaveError.processFailed("Pyrowave encode speedup over HEVC is unavailable")) {
        try PyrowaveBenchmarkArguments(["--require-pyrowave-encode-speedup", "1"]).validate(report: unavailable)
    }
}

@Test func pyrowaveBenchmarkRunnerWritesReviewArtifactsWithoutHEVC() throws {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let frames = [
        try TestFrames.synthetic420(width: 96, height: 64, frameIndex: 0),
        try TestFrames.synthetic420(width: 96, height: 64, frameIndex: 1)
    ]
    let loaded = try PyrowaveBenchmarkFrames(
        frames: frames,
        frameRateNumerator: 60,
        frameRateDenominator: 1,
        bitDepth: 8
    )

    let result = try PyrowaveBenchmarkRunner.runPyrowave(
        loaded: loaded,
        configuration: CodecConfiguration(quantizationStep: 1.0 / 2048.0),
        outputDirectory: directory
    )

    #expect(result.codec == PyrowaveBenchmarkRunner.pyrowaveCodecName)
    #expect(result.frameCount == frames.count)
    #expect(result.encodedBytes > 0)
    #expect(result.metrics?.weightedPSNR ?? 0 > 40)
    #expect(result.note?.contains(PyrowaveBenchmarkRunner.pyrowaveImplementationNote) == true)
    #expect(result.note?.contains("PyrowaveGPUFrame") == true)
    #expect(result.note?.contains("compatibility stream export") == true)

    let streamURL = directory.appendingPathComponent(PyrowaveBenchmarkArtifactNames.pyrowaveStream)
    var streamReader = try PyrowaveStreamReader(url: streamURL)
    #expect(streamReader.header.width == 96)
    #expect(streamReader.header.height == 64)
    #expect(streamReader.header.frameRateNumerator == 60)
    #expect(streamReader.header.frameRateDenominator == 1)
    #expect(try streamReader.readFrame() != nil)
    #expect(try streamReader.readFrame() != nil)
    #expect(try streamReader.readFrame() == nil)

    let decodedURL = directory.appendingPathComponent(PyrowaveBenchmarkArtifactNames.pyrowaveDecodedY4M)
    var decodedReader = try YUV4MPEGReader(url: decodedURL)
    #expect(decodedReader.frameRateNumerator == 60)
    #expect(decodedReader.frameRateDenominator == 1)
    #expect(try decodedReader.readFrame() != nil)
    #expect(try decodedReader.readFrame() != nil)
    #expect(try decodedReader.readFrame() == nil)
}

@Test func pyrowaveBenchmarkRunnerCanSkipArtifactsAndMetricsForProfiling() throws {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let frames = [
        try TestFrames.synthetic420(width: 96, height: 64, frameIndex: 0),
        try TestFrames.synthetic420(width: 96, height: 64, frameIndex: 1)
    ]
    let loaded = try PyrowaveBenchmarkFrames(
        frames: frames,
        frameRateNumerator: 60,
        frameRateDenominator: 1,
        bitDepth: 8
    )

    let result = try PyrowaveBenchmarkRunner.runPyrowave(
        loaded: loaded,
        configuration: CodecConfiguration(quantizationStep: 1.0 / 2048.0),
        outputDirectory: directory,
        writesArtifactsAndMetrics: false
    )

    #expect(result.codec == PyrowaveBenchmarkRunner.pyrowaveCodecName)
    #expect(result.frameCount == frames.count)
    #expect(result.encodedBytes > 0)
    #expect(result.metrics == nil)
    #expect(!FileManager.default.fileExists(atPath: directory.appendingPathComponent(PyrowaveBenchmarkArtifactNames.pyrowaveStream).path))
    #expect(!FileManager.default.fileExists(atPath: directory.appendingPathComponent(PyrowaveBenchmarkArtifactNames.pyrowaveDecodedY4M).path))
}

@Test func codecBenchmarkComparisonReportsHEVCDeltas() throws {
    let pyrowave = CodecBenchmarkResult(
        codec: "pyrowave",
        frameCount: 60,
        encodedBytes: 240,
        encodeSeconds: 0.25,
        decodeSeconds: 0.10,
        metrics: FrameMetrics(
            y: ComponentMetrics(mse: 1, psnr: 52),
            cb: ComponentMetrics(mse: 2, psnr: 49),
            cr: ComponentMetrics(mse: 3, psnr: 48),
            weightedPSNR: 50
        ),
        note: nil
    )
    let hevc = CodecBenchmarkResult(
        codec: "hevc",
        frameCount: 60,
        encodedBytes: 80,
        encodeSeconds: 2.0,
        decodeSeconds: 0.50,
        metrics: FrameMetrics(
            y: ComponentMetrics(mse: 2, psnr: 47),
            cb: ComponentMetrics(mse: 3, psnr: 45),
            cr: ComponentMetrics(mse: 4, psnr: 44),
            weightedPSNR: 46
        ),
        note: nil
    )

    let comparison = CodecBenchmarkComparison(pyrowave: pyrowave, hevc: hevc)
    #expect(comparison.pyrowaveToHEVCByteRatio == 3)
    #expect(comparison.pyrowaveBytesPerFrame == 4)
    #expect(abs(comparison.hevcBytesPerFrame - 1.3333333333) < 0.0001)
    #expect(comparison.pyrowaveToHEVCBytesPerFrameRatio == 3)
    #expect(comparison.pyrowaveEncodeSpeedupOverHEVC == 8)
    #expect(comparison.pyrowaveDecodeSpeedupOverHEVC == 5)
    #expect(comparison.weightedPSNRDelta == 4)
    #expect(comparison.note.contains("Pyrowave"))

    let unavailable = CodecBenchmarkComparison(
        pyrowave: CodecBenchmarkResult(codec: "pyrowave", frameCount: 0, encodedBytes: 0, encodeSeconds: 0, decodeSeconds: 0, metrics: nil, note: nil),
        hevc: CodecBenchmarkResult(codec: "hevc", frameCount: 0, encodedBytes: 0, encodeSeconds: 0, decodeSeconds: 0, metrics: nil, note: nil)
    )
    #expect(unavailable.pyrowaveToHEVCByteRatio == nil)
    #expect(unavailable.pyrowaveBytesPerFrame == 0)
    #expect(unavailable.hevcBytesPerFrame == 0)
    #expect(unavailable.pyrowaveToHEVCBytesPerFrameRatio == nil)
    #expect(unavailable.pyrowaveEncodeSpeedupOverHEVC == nil)
    #expect(unavailable.pyrowaveDecodeSpeedupOverHEVC == nil)
    #expect(unavailable.weightedPSNRDelta == nil)
}

@Test func metalBackendCompilesKernelsWhenDeviceExists() throws {
    do {
        let backend = try MetalPyrowaveBackend()
        _ = try backend.makeFunction(named: "pyrowave_pad_plane")
        _ = try backend.makeFunction(named: "pyrowave_pad_texture_plane")
        _ = try backend.makeFunction(named: "pyrowave_crop_plane")
        _ = try backend.makeFunction(named: "pyrowave_crop_texture_plane")
        _ = try backend.makeFunction(named: "pyrowave_crop_nv12_textures")
        _ = try backend.makeFunction(named: "pyrowave_quantize")
        _ = try backend.makeFunction(named: "pyrowave_dequantize")
        _ = try backend.makeFunction(named: "pyrowave_quantize_plane_tiles")
        _ = try backend.makeFunction(named: "pyrowave_apply_sparse_coefficients")
        _ = try backend.makeFunction(named: "pyrowave_decode_sparse_packets")
        _ = try backend.makeFunction(named: "pyrowave_decode_sparse_packets_threadgroup")
        _ = try backend.makeFunction(named: "pyrowave_dwt_copy_active_rect")
        _ = try backend.makeFunction(named: "pyrowave_dwt_tiled_level0")
        _ = try backend.makeFunction(named: "pyrowave_idwt_tiled_level0")
        _ = try backend.makeFunction(named: "pyrowave_rate_control_tile_stats")
        _ = try backend.makeFunction(named: "pyrowave_packet_byte_costs")
        _ = try backend.makeFunction(named: "pyrowave_packet_byte_costs_smallblocks")
        _ = try backend.makeFunction(named: "pyrowave_packet_byte_costs_finalize")
        _ = try backend.makeFunction(named: "pyrowave_encode_sparse_packets")
        _ = try backend.makeFunction(named: "pyrowave_rate_control_bucket_indices")
        _ = try backend.makeFunction(named: "pyrowave_rate_control_bucket_savings")
        _ = try backend.makeFunction(named: "pyrowave_rate_control_bucket_savings_prefix")
        _ = try backend.makeFunction(named: "pyrowave_rate_control_select_quant_levels")
    } catch PyrowaveError.externalToolUnavailable {
        return
    }
}

@Test func metalTexturesRoundTripYUVFramesWhenDeviceExists() throws {
    do {
        let backend = try MetalPyrowaveBackend()
        let frame420 = try TestFrames.synthetic420(width: 64, height: 64)
        let textures420 = try frame420.makeMetalTextures(device: backend.device)
        let imported420 = try YUVFrame(
            yTexture: textures420.y,
            cbTexture: textures420.cb,
            crTexture: textures420.cr,
            videoSignal: frame420.videoSignal
        )
        #expect(imported420 == frame420)

        let frame444 = try TestFrames.synthetic444(width: 32, height: 24)
        let textures444 = try frame444.makeMetalTextures(device: backend.device)
        let imported444 = try YUVFrame(
            yTexture: textures444.y,
            cbTexture: textures444.cb,
            crTexture: textures444.cr,
            videoSignal: frame444.videoSignal
        )
        #expect(imported444 == frame444)
    } catch PyrowaveError.externalToolUnavailable {
        return
    }
}

private func makeNV12ChromaTexture(
    cb: Plane8,
    cr: Plane8,
    device: MTLDevice,
    usage: MTLTextureUsage = [.shaderRead, .shaderWrite]
) throws -> MTLTexture {
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rg8Unorm,
        width: cb.width,
        height: cb.height,
        mipmapped: false
    )
    descriptor.usage = usage
    descriptor.storageMode = .shared
    let texture = try #require(device.makeTexture(descriptor: descriptor))
    var cbCr = Array(repeating: UInt8(0), count: cb.data.count * 2)
    for index in cb.data.indices {
        cbCr[index * 2] = cb.data[index]
        cbCr[index * 2 + 1] = cr.data[index]
    }
    cbCr.withUnsafeBytes { bytes in
        texture.replace(
            region: MTLRegionMake2D(0, 0, cb.width, cb.height),
            mipmapLevel: 0,
            withBytes: bytes.baseAddress!,
            bytesPerRow: cb.width * 2
        )
    }
    return texture
}

private func makeTexture(
    device: MTLDevice,
    pixelFormat: MTLPixelFormat,
    width: Int,
    height: Int,
    usage: MTLTextureUsage = [.shaderRead, .shaderWrite]
) throws -> MTLTexture {
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: pixelFormat,
        width: width,
        height: height,
        mipmapped: false
    )
    descriptor.usage = usage
    descriptor.storageMode = .shared
    return try #require(device.makeTexture(descriptor: descriptor))
}

private func makeYUVFrame(
    yTexture: MTLTexture,
    cbCrTexture: MTLTexture,
    videoSignal: VideoSignalMetadata
) throws -> YUVFrame {
    let yByteCount = yTexture.width * yTexture.height
    var y = Array(repeating: UInt8(0), count: yByteCount)
    y.withUnsafeMutableBytes { bytes in
        yTexture.getBytes(
            bytes.baseAddress!,
            bytesPerRow: yTexture.width,
            from: MTLRegionMake2D(0, 0, yTexture.width, yTexture.height),
            mipmapLevel: 0
        )
    }

    let chromaSampleCount = cbCrTexture.width * cbCrTexture.height
    var cbCr = Array(repeating: UInt8(0), count: chromaSampleCount * 2)
    cbCr.withUnsafeMutableBytes { bytes in
        cbCrTexture.getBytes(
            bytes.baseAddress!,
            bytesPerRow: cbCrTexture.width * 2,
            from: MTLRegionMake2D(0, 0, cbCrTexture.width, cbCrTexture.height),
            mipmapLevel: 0
        )
    }
    var cb = Array(repeating: UInt8(0), count: chromaSampleCount)
    var cr = Array(repeating: UInt8(0), count: chromaSampleCount)
    for index in 0..<chromaSampleCount {
        cb[index] = cbCr[index * 2]
        cr[index] = cbCr[index * 2 + 1]
    }

    return try YUVFrame(
        width: yTexture.width,
        height: yTexture.height,
        chroma: .yuv420,
        y: Plane8(width: yTexture.width, height: yTexture.height, data: y),
        cb: Plane8(width: cbCrTexture.width, height: cbCrTexture.height, data: cb),
        cr: Plane8(width: cbCrTexture.width, height: cbCrTexture.height, data: cr),
        videoSignal: videoSignal
    )
}

@Test func codecEncodesAndDecodesMetalTexturesWhenDeviceExists() throws {
    do {
        let backend = try MetalPyrowaveBackend()
        let source = try TestFrames.synthetic420(width: 64, height: 64)
        let sourceTextures = try source.makeMetalTextures(device: backend.device)
        let referenceCodec = try PyrowaveCodec()
        let reference = try referenceCodec.encode(source, configuration: CodecConfiguration(quantizationStep: 1.0 / 2048.0))
        let codec = try PyrowaveCodec()
        let encoded = try codec.encode(
            yTexture: sourceTextures.y,
            cbTexture: sourceTextures.cb,
            crTexture: sourceTextures.cr,
            configuration: CodecConfiguration(quantizationStep: 1.0 / 2048.0),
            videoSignal: source.videoSignal
        )
        #expect(encoded.data == reference.data)
        let decodedTextures = try codec.decodeToMetalTextures(encoded, device: backend.device)
        let decoded = try YUVFrame(
            yTexture: decodedTextures.y,
            cbTexture: decodedTextures.cb,
            crTexture: decodedTextures.cr,
            videoSignal: source.videoSignal
        )
        #expect(try Metrics.compare(source, decoded).weightedPSNR > 44.0)

        let packets = try encoded.packetized(maximumPacketBytes: 64)
        let stream = try PyrowavePacketStreamDecoder(width: source.width, height: source.height, chroma: source.chroma)
        for packet in packets {
            try stream.pushPacket(packet)
        }
        let streamedTextures = try stream.decodeToMetalTextures(device: backend.device)
        let streamed = try YUVFrame(
            yTexture: streamedTextures.y,
            cbTexture: streamedTextures.cb,
            crTexture: streamedTextures.cr,
            videoSignal: source.videoSignal
        )
        #expect(try Metrics.compare(source, streamed).weightedPSNR > 44.0)
    } catch PyrowaveError.externalToolUnavailable {
        return
    }
}

@Test func codecEncodesAndDecodesGPUFrameWithoutEncodedDataWhenDeviceExists() throws {
    do {
        let backend = try MetalPyrowaveBackend()
        let source = try TestFrames.synthetic420(width: 64, height: 64)
        let sourceTextures = try source.makeMetalTextures(device: backend.device)
        let sourceCbCrTexture = try makeNV12ChromaTexture(
            cb: source.cb,
            cr: source.cr,
            device: backend.device
        )
        let decodedYTexture = try makeTexture(
            device: backend.device,
            pixelFormat: .r8Unorm,
            width: source.width,
            height: source.height
        )
        let decodedCbCrTexture = try makeTexture(
            device: backend.device,
            pixelFormat: .rg8Unorm,
            width: source.cb.width,
            height: source.cb.height
        )

        let codec = try PyrowaveCodec()
        let gpuFrame = try codec.encodeGPUFrame(
            yTexture: sourceTextures.y,
            cbCrTexture: sourceCbCrTexture,
            configuration: CodecConfiguration(quantizationStep: 1.0 / 2048.0),
            videoSignal: source.videoSignal
        )
        #expect(gpuFrame.width == source.width)
        #expect(gpuFrame.height == source.height)
        #expect(gpuFrame.chroma == .yuv420)
        #expect(gpuFrame.estimatedPacketCapacityBytes > 0)
        #expect(gpuFrame.encodedByteCountForInspection() > 0)
        #expect(gpuFrame.selectedQuantLevelsByPlane.count == 3)
        #expect(gpuFrame.selectedQuantLevelsByPlane.flatMap { $0 }.allSatisfy { $0 == 0 })
        #expect(!gpuFrame.selectedQuantLevelsByPlane.flatMap { $0 }.isEmpty)

        try codec.decodeGPUFrameToNV12Textures(
            gpuFrame,
            yTexture: decodedYTexture,
            cbCrTexture: decodedCbCrTexture
        )
        let decoded = try makeYUVFrame(
            yTexture: decodedYTexture,
            cbCrTexture: decodedCbCrTexture,
            videoSignal: source.videoSignal
        )
        #expect(try Metrics.compare(source, decoded).weightedPSNR > 44.0)
    } catch PyrowaveError.externalToolUnavailable {
        return
    }
}

@Test func codecExportsAndImportsGPUFrameOnlyAsSecondaryBoundaryAPI() throws {
    do {
        let backend = try MetalPyrowaveBackend()
        let source = try TestFrames.synthetic420(width: 64, height: 64)
        let sourceTextures = try source.makeMetalTextures(device: backend.device)
        let sourceCbCrTexture = try makeNV12ChromaTexture(
            cb: source.cb,
            cr: source.cr,
            device: backend.device
        )
        let decodedYTexture = try makeTexture(
            device: backend.device,
            pixelFormat: .r8Unorm,
            width: source.width,
            height: source.height
        )
        let decodedCbCrTexture = try makeTexture(
            device: backend.device,
            pixelFormat: .rg8Unorm,
            width: source.cb.width,
            height: source.cb.height
        )

        let codec = try PyrowaveCodec()
        let gpuFrame = try codec.encodeGPUFrame(
            yTexture: sourceTextures.y,
            cbCrTexture: sourceCbCrTexture,
            configuration: CodecConfiguration(quantizationStep: 1.0 / 2048.0),
            videoSignal: source.videoSignal
        )
        let exported = try codec.exportGPUFrame(gpuFrame)
        #expect(exported.data.count > gpuFrame.encodedByteCountForInspection())

        let decodedExport = try codec.decode(exported)
        #expect(try Metrics.compare(source, decodedExport).weightedPSNR > 44.0)

        let imported = try codec.importGPUFrame(exported)
        #expect(imported.selectedQuantLevelsByPlane.count == gpuFrame.selectedQuantLevelsByPlane.count)
        #expect(!imported.selectedQuantLevelsByPlane.flatMap { $0 }.isEmpty)
        #expect(imported.selectedQuantLevelsByPlane.flatMap { $0 }.allSatisfy { $0 == 0 })
        let reexported = try codec.exportGPUFrame(imported)
        #expect(reexported == exported)

        try codec.decodeGPUFrameToNV12Textures(
            imported,
            yTexture: decodedYTexture,
            cbCrTexture: decodedCbCrTexture
        )
        let decodedImport = try makeYUVFrame(
            yTexture: decodedYTexture,
            cbCrTexture: decodedCbCrTexture,
            videoSignal: source.videoSignal
        )
        #expect(try Metrics.compare(source, decodedImport).weightedPSNR > 44.0)
    } catch PyrowaveError.externalToolUnavailable {
        return
    }
}

@Test func codecEncodesNV12MetalChromaTextureLikeSeparatePlanes() throws {
    do {
        let backend = try MetalPyrowaveBackend()
        let source = try TestFrames.synthetic420(width: 64, height: 64)
        let sourceTextures = try source.makeMetalTextures(device: backend.device)
        let cbCrTexture = try makeNV12ChromaTexture(cb: source.cb, cr: source.cr, device: backend.device, usage: [.shaderRead])

        let configuration = CodecConfiguration(quantizationStep: 1.0 / 2048.0)
        let separate = try PyrowaveCodec().encode(
            yTexture: sourceTextures.y,
            cbTexture: sourceTextures.cb,
            crTexture: sourceTextures.cr,
            configuration: configuration,
            videoSignal: source.videoSignal
        )
        let nv12 = try PyrowaveCodec().encode(
            yTexture: sourceTextures.y,
            cbCrTexture: cbCrTexture,
            configuration: configuration,
            videoSignal: source.videoSignal
        )
        #expect(nv12.data == separate.data)
    } catch PyrowaveError.externalToolUnavailable {
        return
    }
}

@Test func metalPlanePaddingMatchesCPUReferenceWhenDeviceExists() throws {
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let plane = try Plane8(width: 3, height: 2, data: [
        0, 64, 128,
        192, 224, 255
    ])
    let cpu = Wavelet.padPlane(plane, paddedWidth: 9, paddedHeight: 7).samples
    let metal = try backend.padPlane(plane, paddedWidth: 9, paddedHeight: 7)
    #expect(metal.count == cpu.count)
    let maxError = zip(metal, cpu).map { abs($0 - $1) }.max() ?? 0
    #expect(maxError < 0.000001)
}

@Test func metalTexturePlanePaddingMatchesCPUReferenceWhenDeviceExists() throws {
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let plane = try Plane8(width: 3, height: 2, data: [
        0, 64, 128,
        192, 224, 255
    ])
    let texture = try plane.makeMetalTexture(device: backend.device)
    let cpu = Wavelet.padPlane(plane, paddedWidth: 9, paddedHeight: 7).samples
    let metal = try backend.padTexturePlane(texture, paddedWidth: 9, paddedHeight: 7)
    #expect(metal.count == cpu.count)
    let maxError = zip(metal, cpu).map { abs($0 - $1) }.max() ?? 0
    #expect(maxError < 0.000001)
}

@Test func metalTexturePlanePaddingBatchMatchesSinglePlaneResultsWhenDeviceExists() throws {
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let plane = try Plane8(width: 3, height: 2, data: [
        0, 64, 128,
        192, 224, 255
    ])
    let firstTexture = try plane.makeMetalTexture(device: backend.device)
    let secondTexture = try plane.makeMetalTexture(device: backend.device)
    let batchedBuffers = try backend.padTexturePlaneBuffers([
        (texture: firstTexture, channel: 0, paddedWidth: 9, paddedHeight: 7),
        (texture: secondTexture, channel: 0, paddedWidth: 9, paddedHeight: 7)
    ])
    #expect(batchedBuffers.count == 2)

    let single = try backend.padTexturePlane(firstTexture, paddedWidth: 9, paddedHeight: 7)
    for buffer in batchedBuffers {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: single.count)
        let batched = Array(UnsafeBufferPointer(start: pointer, count: single.count))
        #expect(batched == single)
    }
}

@Test func metalPlaneCropMatchesCPUReferenceWhenDeviceExists() throws {
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let paddedWidth = 5
    let samples: [Float] = [
        -0.75, -0.5, -0.25, 0.0, 0.5,
        0.001, 0.249, 0.251, 0.499, 0.75,
        -0.49, -0.001, 0.123, 0.333, 0.9
    ]
    let cpu = try Wavelet.cropPlane(samples, paddedWidth: paddedWidth, width: 4, height: 3)
    let metal = try backend.cropPlane(samples, paddedWidth: paddedWidth, width: 4, height: 3)
    #expect(metal == cpu)
}

@Test func metalQuantizationMatchesCPUReferenceWhenDeviceExists() throws {
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
}

@Test func metalPlaneQuantizationBatchMatchesSinglePlaneResultsWhenDeviceExists() throws {
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let width = 32
    let height = 32
    let sampleCount = width * height
    let quantCode = try PyrowaveQuantization.encodeBlockScale(1.0 / 1024.0)
    let baseScale = 1.0 / PyrowaveQuantization.decodeBlockScale(quantCode)
    let descriptors = [
        MetalPlaneQuantizationDescriptor(
            originX: 0,
            originY: 0,
            validWidth: UInt32(width),
            validHeight: UInt32(height),
            stride: UInt32(width),
            quantCode: UInt32(quantCode),
            baseScale: baseScale
        ),
        MetalPlaneQuantizationDescriptor(
            originX: 8,
            originY: 8,
            validWidth: 16,
            validHeight: 16,
            stride: UInt32(width),
            quantCode: UInt32(quantCode),
            baseScale: baseScale * 0.75
        )
    ]
    let planes = (0..<3).map { planeIndex in
        (0..<sampleCount).map { index in
            Float((index * (17 + planeIndex * 5) + planeIndex * 29) % 1021) / 1024.0 - 0.5
        }
    }
    let buffers = try planes.map { samples -> MTLBuffer in
        try #require(backend.device.makeBuffer(
            bytes: samples,
            length: samples.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
    }
    let singles = try buffers.map {
        try backend.quantizePlaneBufferResult(
            $0,
            sampleCount: sampleCount,
            stride: width,
            descriptors: descriptors
        )
    }
    let batched = try backend.quantizePlaneBufferResults(buffers.map {
        (samples: $0, sampleCount: sampleCount, stride: width, descriptors: descriptors)
    })

    #expect(batched.count == singles.count)
    for index in singles.indices {
        #expect(batched[index].coefficientCount == singles[index].coefficientCount)
        #expect(batched[index].qScaleCodesByDescriptor == singles[index].qScaleCodesByDescriptor)
        let singlePointer = singles[index].coefficientBuffer.contents().bindMemory(to: Int16.self, capacity: sampleCount)
        let batchPointer = batched[index].coefficientBuffer.contents().bindMemory(to: Int16.self, capacity: sampleCount)
        let singleCoefficients = Array(UnsafeBufferPointer(start: singlePointer, count: sampleCount))
        let batchCoefficients = Array(UnsafeBufferPointer(start: batchPointer, count: sampleCount))
        #expect(batchCoefficients == singleCoefficients)
    }
}

@Test func metalSparseApplyMatchesCPUReferenceWhenDeviceExists() throws {
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let quantCode = UInt8(32)
    let entries = [
        MetalSparseCoefficientEntry(destinationOffset: 7, coefficient: -9, quantCode: UInt32(quantCode), qScaleCode: 3),
        MetalSparseCoefficientEntry(destinationOffset: 2, coefficient: 14, quantCode: UInt32(quantCode), qScaleCode: 6),
        MetalSparseCoefficientEntry(destinationOffset: 11, coefficient: 1, quantCode: UInt32(quantCode), qScaleCode: 15),
        MetalSparseCoefficientEntry(destinationOffset: 13, coefficient: -3, quantCode: 0, qScaleCode: 6),
        MetalSparseCoefficientEntry(destinationOffset: 14, coefficient: 5, quantCode: 96, qScaleCode: 9)
    ]
    let metal = try backend.applySparseCoefficients(sampleCount: 16, entries: entries)
    var cpu = Array(repeating: Float(0), count: 16)
    for entry in entries {
        cpu[Int(entry.destinationOffset)] = PyrowaveQuantization.dequantize(
            coefficient: Int16(entry.coefficient),
            quantCode: UInt8(entry.quantCode),
            qScaleCode: UInt8(entry.qScaleCode)
        )
    }

    let maxError = zip(metal, cpu).map { abs($0 - $1) }.max() ?? 0
    #expect(maxError < 0.000001)
    #expect(try backend.applySparseCoefficients(sampleCount: 5, entries: []) == Array(repeating: Float(0), count: 5))
}

@Test func metalSparseApplyBatchMatchesSinglePlaneResultsWhenDeviceExists() throws {
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let planes = [
        (
            sampleCount: 16,
            entries: [
                MetalSparseCoefficientEntry(destinationOffset: 7, coefficient: -9, quantCode: 32, qScaleCode: 3),
                MetalSparseCoefficientEntry(destinationOffset: 2, coefficient: 14, quantCode: 32, qScaleCode: 6)
            ]
        ),
        (
            sampleCount: 8,
            entries: [MetalSparseCoefficientEntry(destinationOffset: 5, coefficient: 4, quantCode: 96, qScaleCode: 9)]
        ),
        (
            sampleCount: 5,
            entries: []
        )
    ]

    let batchedBuffers = try backend.applySparseCoefficientBuffers(planes)
    #expect(batchedBuffers.count == planes.count)

    for index in planes.indices {
        let singlePlane = try backend.applySparseCoefficients(
            sampleCount: planes[index].sampleCount,
            entries: planes[index].entries
        )
        let pointer = batchedBuffers[index].contents().bindMemory(to: Float.self, capacity: planes[index].sampleCount)
        let batched = Array(UnsafeBufferPointer(start: pointer, count: planes[index].sampleCount))
        #expect(batched == singlePlane)
    }
}

@Test func metalRateControlStatsMatchCPUReferenceWhenDeviceExists() throws {
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let stride = PyrowaveBitstream.coefficientBlockSize
    var coefficients = Array(repeating: Int16(0), count: stride * stride)
    for y in 0..<stride {
        for x in 0..<stride {
            let raw = ((x * 37 + y * 19 + x * y) % 257) - 128
            coefficients[y * stride + x] = Int16(raw)
        }
    }

    let quantCode = try PyrowaveQuantization.encodeBlockScale(1.0 / 1024.0)
    let qScaleCodes = (0..<16).map { UInt8(3 + ($0 % 9)) }
    let validWidth = 29
    let validHeight = 27
    let distortionScale = Float(1.375)
    let cpu = try PyrowaveRateController.makeBlock(
        blockIndex: 0,
        coefficients: coefficients,
        stride: stride,
        originX: 0,
        originY: 0,
        validWidth: validWidth,
        validHeight: validHeight,
        quantCode: quantCode,
        qScaleCodes: qScaleCodes,
        rdoDistortionScale: distortionScale
    )

    var descriptors = [MetalRateControlStatsDescriptor]()
    for tileY in 0..<4 {
        for tileX in 0..<4 {
            let tileIndex = tileY * 4 + tileX
            descriptors.append(MetalRateControlStatsDescriptor(
                originX: UInt32(tileX * PyrowaveBitstream.smallBlockSize),
                originY: UInt32(tileY * PyrowaveBitstream.smallBlockSize),
                validWidth: UInt32(max(0, min(PyrowaveBitstream.smallBlockSize, validWidth - tileX * PyrowaveBitstream.smallBlockSize))),
                validHeight: UInt32(max(0, min(PyrowaveBitstream.smallBlockSize, validHeight - tileY * PyrowaveBitstream.smallBlockSize))),
                stride: UInt32(stride),
                quantCode: UInt32(quantCode),
                qScaleCode: UInt32(qScaleCodes[tileIndex]),
                distortionScale: distortionScale
            ))
        }
    }

    let metal = try backend.rateControlTileStats(coefficients: coefficients, descriptors: descriptors)
    #expect(metal.count == cpu.eightByEightStats.count)
    for index in metal.indices {
        #expect(metal[index].numPlanes == cpu.eightByEightStats[index].numPlanes)
        #expect(metal[index].stats.count == PyrowaveBlockStats.candidateCount)
        for quantLevel in 0..<PyrowaveBlockStats.candidateCount {
            let metalStat = metal[index].stats[quantLevel]
            let cpuStat = cpu.eightByEightStats[index].stats[quantLevel]
            #expect(metalStat.encodeCostBits == UInt32(cpuStat.encodeCostBits))
            let converted = PyrowaveQuantStats(squareError: metalStat.squareError, encodeCostBits: Int(metalStat.encodeCostBits))
            #expect(abs(converted.squareError - cpuStat.squareError) < 0.0001)
        }
    }
}

@Test func metalPacketByteCostsMatchCPUReferenceWhenDeviceExists() throws {
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let stride = 40
    var coefficients = Array(repeating: Int16(0), count: stride * 36)
    for y in 0..<27 {
        for x in 0..<29 {
            let raw = ((x * 43 + y * 17 + x * y * 3) % 511) - 255
            coefficients[(2 + y) * stride + 3 + x] = Int16(raw)
        }
    }
    let quantCode = try PyrowaveQuantization.encodeBlockScale(1.0 / 1024.0)
    let qScaleCodes = (0..<16).map { UInt8(($0 * 5) % 16) }
    let descriptors = [
        MetalPacketByteCostDescriptor(
            originX: 3,
            originY: 2,
            validWidth: 29,
            validHeight: 27,
            stride: UInt32(stride)
        ),
        MetalPacketByteCostDescriptor(
            originX: 0,
            originY: 32,
            validWidth: 32,
            validHeight: 4,
            stride: UInt32(stride)
        )
    ]

    let metal = try backend.packetByteCosts(coefficients: coefficients, descriptors: descriptors)
    let cpu = try PyrowaveRateController.makePacketByteCosts(
        blockIndex: 11,
        coefficients: coefficients,
        stride: stride,
        originX: 3,
        originY: 2,
        validWidth: 29,
        validHeight: 27,
        quantCode: quantCode,
        qScaleCodes: qScaleCodes
    )

    #expect(metal.count == descriptors.count)
    #expect(metal[0] == cpu)
    #expect(metal[1] == Array(repeating: 0, count: PyrowaveBlockStats.candidateCount))
}

@Test func metalRateControlBatchMatchesSinglePlaneResultsWhenDeviceExists() throws {
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let stride = 40
    let coefficientSets: [[Int16]] = (0..<2).map { planeIndex in
        var coefficients = Array(repeating: Int16(0), count: stride * 36)
        for y in 0..<27 {
            for x in 0..<29 {
                let raw = ((x * (43 + planeIndex * 7) + y * 17 + x * y * 3) % 511) - 255
                coefficients[(2 + y) * stride + 3 + x] = Int16(raw)
            }
        }
        return coefficients
    }
    let statsDescriptors = [
        MetalRateControlStatsDescriptor(
            originX: 0,
            originY: 0,
            validWidth: 32,
            validHeight: 32,
            stride: UInt32(stride),
            quantCode: 32,
            qScaleCode: 6,
            distortionScale: 1.0
        ),
        MetalRateControlStatsDescriptor(
            originX: 8,
            originY: 8,
            validWidth: 8,
            validHeight: 8,
            stride: UInt32(stride),
            quantCode: 40,
            qScaleCode: 9,
            distortionScale: 0.25
        )
    ]
    let packetDescriptors = [
        MetalPacketByteCostDescriptor(originX: 3, originY: 2, validWidth: 29, validHeight: 27, stride: UInt32(stride)),
        MetalPacketByteCostDescriptor(originX: 0, originY: 32, validWidth: 32, validHeight: 4, stride: UInt32(stride))
    ]
    let buffers = try coefficientSets.map { coefficients -> MTLBuffer in
        let byteLength = coefficients.count * MemoryLayout<Int16>.stride
        return try #require(backend.device.makeBuffer(bytes: coefficients, length: byteLength, options: .storageModeShared))
    }

    let batchedStats = try backend.rateControlTileStatsBatch(buffers.indices.map {
        (coefficientBuffer: buffers[$0], coefficientCount: coefficientSets[$0].count, descriptors: statsDescriptors)
    })
    let batchedCosts = try backend.packetByteCostsBatch(buffers.indices.map {
        (coefficientBuffer: buffers[$0], coefficientCount: coefficientSets[$0].count, descriptors: packetDescriptors)
    })

    for index in coefficientSets.indices {
        let singleStats = try backend.rateControlTileStats(coefficients: coefficientSets[index], descriptors: statsDescriptors)
        let singleCosts = try backend.packetByteCosts(coefficients: coefficientSets[index], descriptors: packetDescriptors)
        #expect(batchedStats[index].count == singleStats.count)
        for statIndex in singleStats.indices {
            #expect(batchedStats[index][statIndex].numPlanes == singleStats[statIndex].numPlanes)
            #expect(batchedStats[index][statIndex].stats.count == singleStats[statIndex].stats.count)
            for quantLevel in singleStats[statIndex].stats.indices {
                #expect(batchedStats[index][statIndex].stats[quantLevel].squareError == singleStats[statIndex].stats[quantLevel].squareError)
                #expect(batchedStats[index][statIndex].stats[quantLevel].encodeCostBits == singleStats[statIndex].stats[quantLevel].encodeCostBits)
            }
        }
        #expect(batchedCosts[index] == singleCosts)
    }
}

@Test func metalSparsePacketEncodingMatchesCPUReferenceWhenDeviceExists() throws {
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let stride = 40
    var coefficients = Array(repeating: Int16(0), count: stride * 40)
    for y in 0..<27 {
        for x in 0..<29 {
            let raw = ((x * 47 + y * 23 + x * y * 5) % 1021) - 510
            coefficients[(2 + y) * stride + 3 + x] = Int16(raw)
        }
    }
    coefficients[5 * stride + 7] = -32768

    let quantCode = try PyrowaveQuantization.encodeBlockScale(1.0 / 1024.0)
    let qScaleCodes = (0..<16).map { UInt8(($0 * 7) % 16) }
    let descriptors = [
        MetalSparsePacketEncodeDescriptor(
            originX: 3,
            originY: 2,
            validWidth: 29,
            validHeight: 27,
            stride: UInt32(stride),
            blockIndex: 37,
            quantLevel: 2,
            quantCode: UInt32(quantCode)
        ),
        MetalSparsePacketEncodeDescriptor(
            originX: 32,
            originY: 32,
            validWidth: 8,
            validHeight: 8,
            stride: UInt32(stride),
            blockIndex: 38,
            quantLevel: 0,
            quantCode: UInt32(quantCode)
        )
    ]

    let metal = try backend.encodeSparsePackets(
        coefficients: coefficients,
        descriptors: descriptors,
        qScaleCodes: [qScaleCodes, qScaleCodes],
        sequence: 5
    )
    let cpu = try PyrowaveCoefficientBlockCodec.encodeBlock(
        blockIndex: 37,
        coefficients: coefficients,
        stride: stride,
        originX: 3,
        originY: 2,
        validWidth: 29,
        validHeight: 27,
        threshold: 0,
        quantLevel: 2,
        sequence: 5,
        quantCode: quantCode,
        qScaleCodes: qScaleCodes
    )

    #expect(metal.count == descriptors.count)
    #expect(metal[0] == cpu)
    #expect(metal[1] == nil)

    let secondCoefficients = coefficients.enumerated().map { index, value in
        index % 5 == 0 ? Int16(value / 2) : value
    }
    let coefficientSets = [coefficients, secondCoefficients]
    let buffers = try coefficientSets.map { values -> MTLBuffer in
        let byteLength = values.count * MemoryLayout<Int16>.stride
        return try #require(backend.device.makeBuffer(bytes: values, length: byteLength, options: .storageModeShared))
    }
    let batched = try backend.encodeSparsePacketsBatch(buffers.indices.map {
        (
            coefficientBuffer: buffers[$0],
            coefficientCount: coefficientSets[$0].count,
            descriptors: descriptors,
            qScaleCodes: [qScaleCodes, qScaleCodes]
        )
    })
    #expect(batched.count == coefficientSets.count)
    for index in coefficientSets.indices {
        let single = try backend.encodeSparsePackets(
            coefficients: coefficientSets[index],
            descriptors: descriptors,
            qScaleCodes: [qScaleCodes, qScaleCodes]
        )
        #expect(batched[index] == single)
    }
}

@Test func metalRateControlBucketsMatchCPUReferenceWhenDeviceExists() throws {
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let stride = 40
    var coefficients = Array(repeating: Int16(0), count: stride * 40)
    for y in 0..<32 {
        for x in 0..<32 {
            let raw = ((x * 29 + y * 31 + x * y * 7) % 767) - 383
            coefficients[(4 + y) * stride + 5 + x] = Int16(raw)
        }
    }

    let quantCode = try PyrowaveQuantization.encodeBlockScale(1.0 / 1024.0)
    let blocks = [
        try PyrowaveRateController.makeBlock(
            blockIndex: 0,
            coefficients: coefficients,
            stride: stride,
            originX: 5,
            originY: 4,
            validWidth: 32,
            validHeight: 32,
            quantCode: quantCode,
            rdoDistortionScale: 1.0
        ),
        try PyrowaveRateController.makeBlock(
            blockIndex: 1,
            coefficients: coefficients,
            stride: stride,
            originX: 5,
            originY: 4,
            validWidth: 17,
            validHeight: 23,
            quantCode: quantCode,
            qScaleCodes: (0..<16).map { UInt8(($0 * 3) % 16) },
            rdoDistortionScale: 1.75
        )
    ]

    let distortions = blocks.map { block in
        (0..<PyrowaveBlockStats.candidateCount).map { block.distortion(quantLevel: $0) }
    }
    let packetByteCosts = blocks.map(\.packetByteCosts)
    let metal = try backend.rateControlBucketIndices(
        distortions: distortions,
        packetByteCosts: packetByteCosts
    )
    let cpu = blocks.map { PyrowaveRateController.inclusiveBucketIndices(for: $0) }

    #expect(metal == cpu)
    let batched = try backend.rateControlBucketIndicesBatch([
        (distortions: distortions, packetByteCosts: packetByteCosts),
        (distortions: Array(distortions.reversed()), packetByteCosts: Array(packetByteCosts.reversed())),
        (distortions: [], packetByteCosts: [])
    ])
    #expect(batched.count == 3)
    #expect(batched[0] == metal)
    #expect(batched[1] == Array(cpu.reversed()))
    #expect(batched[2].isEmpty)

    let metalOperations = PyrowaveRateController.makeRDOperations(
        blocksByPlane: [blocks],
        bucketIndicesByPlane: [metal]
    )
    let cpuOperations = PyrowaveRateController.makeRDOperations(blocksByPlane: [blocks])
    #expect(metalOperations == cpuOperations)

    let metalSavings = try backend.rateControlCumulativeBucketSavings(
        bucketIndices: metal,
        packetByteCosts: packetByteCosts
    )
    let cpuSavings = PyrowaveRateController.cumulativeBucketSavings(
        blocksByPlane: [blocks],
        bucketIndicesByPlane: [metal]
    )
    #expect(metalSavings == cpuSavings)
    #expect(metalSavings.count == 128)
    for index in 1..<metalSavings.count {
        #expect(metalSavings[index] >= metalSavings[index - 1])
    }
    let requiredSavings = metalSavings.first { $0 > 0 } ?? 0
    let metalQuantLevels = try backend.rateControlSelectedQuantLevels(
        bucketIndices: metal,
        packetByteCosts: packetByteCosts,
        cumulativeSavings: metalSavings,
        requiredSavings: requiredSavings
    )
    var remainingSavings = requiredSavings
    var expectedQuantLevels = Array(repeating: 0, count: packetByteCosts.count)
    for bucket in 0..<128 where remainingSavings > 0 {
        for blockIndex in packetByteCosts.indices where remainingSavings > 0 {
            for quantLevel in 1..<PyrowaveBlockStats.candidateCount where remainingSavings > 0 {
                let currentLevel = expectedQuantLevels[blockIndex]
                let transitionSaving = packetByteCosts[blockIndex][quantLevel - 1] - packetByteCosts[blockIndex][quantLevel]
                if quantLevel > currentLevel,
                   transitionSaving > 0,
                   metal[blockIndex][quantLevel] == bucket {
                    let actualSaving = packetByteCosts[blockIndex][currentLevel] - packetByteCosts[blockIndex][quantLevel]
                    if actualSaving > 0 {
                        expectedQuantLevels[blockIndex] = quantLevel
                        remainingSavings = max(0, remainingSavings - actualSaving)
                    }
                }
            }
        }
    }
    #expect(metalQuantLevels == expectedQuantLevels)

    let fused = try backend.rateControlBucketDataBatch([
        (distortions: distortions, packetByteCosts: packetByteCosts),
        (distortions: Array(distortions.reversed()), packetByteCosts: Array(packetByteCosts.reversed())),
        (distortions: [], packetByteCosts: [])
    ])
    #expect(fused.bucketIndicesByPlane == batched)
    let fusedCPUSavings = PyrowaveRateController.cumulativeBucketSavings(
        blocksByPlane: [blocks, Array(blocks.reversed()), []],
        bucketIndicesByPlane: fused.bucketIndicesByPlane
    )
    #expect(fused.cumulativeSavings == fusedCPUSavings)
}

@Test func metalCodecMatchesCPUReferenceWhenDeviceExists() throws {
    do {
        _ = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let configuration = CodecConfiguration(quantizationStep: 1.0 / 2048.0)
    let frames = [
        try TestFrames.synthetic420(width: 160, height: 96),
        try TestFrames.synthetic420(width: 130, height: 74),
        try TestFrames.synthetic444(width: 128, height: 96)
    ]

    for frame in frames {
        let cpu = try PyrowaveCodec().encode(frame, configuration: configuration)
        let metal = try PyrowaveCodec().encode(frame, configuration: configuration)
        #expect(metal.data == cpu.data)

        let decodedCPU = try PyrowaveCodec().decode(cpu)
        let decodedMetal = try PyrowaveCodec().decode(metal)
        #expect(decodedMetal == decodedCPU)
    }
}

@Test func metalWaveletMatchesCPUReferenceWhenDeviceExists() throws {
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let width = 512
    let height = 256
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
}

@Test func metalForwardWaveletBatchMatchesSinglePlaneResultsWhenDeviceExists() throws {
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let width = 128
    let height = 128
    let levels = 3
    let planes = (0..<3).map { planeIndex in
        (0..<(width * height)).map { index in
            Float((index * (13 + planeIndex * 5) + planeIndex * 19) % 251) / 251.0 - 0.5
        }
    }
    let single = try planes.map {
        try backend.forwardWavelet($0, width: width, height: height, levels: levels)
    }
    let buffers = try planes.map { samples -> MTLBuffer in
        let byteLength = samples.count * MemoryLayout<Float>.stride
        return try #require(backend.device.makeBuffer(bytes: samples, length: byteLength, options: .storageModeShared))
    }

    let batchedBuffers = try backend.forwardWaveletBuffers(buffers.map {
        (buffer: $0, sampleCount: width * height, width: width, height: height, levels: levels)
    })
    #expect(batchedBuffers.count == buffers.count)
    for index in batchedBuffers.indices {
        let pointer = batchedBuffers[index].contents().bindMemory(to: Float.self, capacity: width * height)
        let batched = Array(UnsafeBufferPointer(start: pointer, count: width * height))
        let maxError = zip(batched, single[index]).map { abs($0 - $1) }.max() ?? 0
        #expect(maxError < 0.0001)
    }
}

@Test func metalInverseWaveletBatchMatchesSinglePlaneResultsWhenDeviceExists() throws {
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let width = 128
    let height = 128
    let levels = 3
    let planes = (0..<3).map { planeIndex in
        (0..<(width * height)).map { index in
            Float((index * (11 + planeIndex * 3) + planeIndex * 17) % 257) / 257.0 - 0.5
        }
    }
    let transformed = try planes.map {
        try backend.forwardWavelet($0, width: width, height: height, levels: levels)
    }
    let single = try transformed.map {
        try backend.inverseWavelet($0, width: width, height: height, levels: levels)
    }
    let buffers = try transformed.map { samples -> MTLBuffer in
        let byteLength = samples.count * MemoryLayout<Float>.stride
        return try #require(backend.device.makeBuffer(bytes: samples, length: byteLength, options: .storageModeShared))
    }

    let batchedBuffers = try backend.inverseWaveletBuffers(buffers.map {
        (buffer: $0, sampleCount: width * height, width: width, height: height, levels: levels)
    })
    #expect(batchedBuffers.count == buffers.count)
    for index in batchedBuffers.indices {
        let pointer = batchedBuffers[index].contents().bindMemory(to: Float.self, capacity: width * height)
        let batched = Array(UnsafeBufferPointer(start: pointer, count: width * height))
        let maxError = zip(batched, single[index]).map { abs($0 - $1) }.max() ?? 0
        #expect(maxError < 0.0001)
    }
}

@Test func tiledMetalInverseWaveletMatchesCPUReferenceWhenDeviceExists() throws {
    let backend: MetalPyrowaveBackend
    do {
        backend = try MetalPyrowaveBackend()
    } catch PyrowaveError.externalToolUnavailable {
        return
    }

    let width = 128
    let height = 128
    let levels = 3
    let samples = (0..<(width * height)).map { index in
        Float((index * 37 + (index / width) * 19) % 4099) / 4099.0 - 0.5
    }
    var coefficients = samples
    Wavelet.forward2D(&coefficients, width: width, height: height, levels: levels)
    var cpuReference = coefficients
    Wavelet.inverse2D(&cpuReference, width: width, height: height, levels: levels)

    let byteLength = coefficients.count * MemoryLayout<Float>.stride
    let input = try #require(backend.device.makeBuffer(bytes: coefficients, length: byteLength, options: .storageModeShared))
    let output = try backend.inverseWaveletBuffer(
        input,
        sampleCount: coefficients.count,
        width: width,
        height: height,
        levels: levels,
        useTiledLevelZero: true
    )
    let readback = try #require(backend.device.makeBuffer(length: byteLength, options: .storageModeShared))
    guard let commandBuffer = backend.commandQueue.makeCommandBuffer(),
          let blit = commandBuffer.makeBlitCommandEncoder() else {
        throw PyrowaveError.processFailed("failed to create Metal readback command buffer")
    }
    blit.copy(from: output, sourceOffset: 0, to: readback, destinationOffset: 0, size: byteLength)
    blit.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error {
        throw PyrowaveError.processFailed("Metal tiled iDWT readback failed: \(error.localizedDescription)")
    }

    let pointer = readback.contents().bindMemory(to: Float.self, capacity: coefficients.count)
    let metal = Array(UnsafeBufferPointer(start: pointer, count: coefficients.count))
    let maxError = zip(metal, cpuReference).map { abs($0 - $1) }.max() ?? 0
    #expect(maxError < 0.0001)
}

@Test func codecUsesPyrowaveSequenceHeaderStreamOnly() throws {
    let frame = try TestFrames.synthetic420(width: 64, height: 64)
    let codec = try PyrowaveCodec()
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
    let codec = try PyrowaveCodec()
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
    let encoded = try PyrowaveCodec().encode(
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
    let codec = try PyrowaveCodec()
    let encoded = try codec.encode(frame, configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0))
    let expected = try codec.decode(encoded)
    let packets = try encoded.packetized(maximumPacketBytes: 8)
    let stream = try PyrowavePacketStreamDecoder()

    for packet in packets.dropLast() {
        try stream.pushPacket(packet)
        #expect(!stream.decodeIsReady())
    }
    try stream.pushPacket(try #require(packets.last))
    #expect(stream.decodeIsReady())
    #expect(try stream.decode() == expected)
    #expect(!stream.decodeIsReady())
}

@Test func packetStreamDecoderReconstructsComplete444Frame() throws {
    let frame = try TestFrames.synthetic444(width: 96, height: 64)
    let codec = try PyrowaveCodec()
    let encoded = try codec.encode(frame, configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0))
    let expected = try codec.decode(encoded)
    let packets = try encoded.packetized(maximumPacketBytes: 64)
    let stream = try PyrowavePacketStreamDecoder()

    for packet in packets {
        try stream.pushPacket(packet)
    }

    #expect(stream.decodeIsReady())
    #expect(try stream.decode() == expected)
}

@Test func pyrowaveStreamFileRoundTripsMultipleEncodedFrames() throws {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("sample.pwks")
    let frames = [
        try TestFrames.synthetic420(width: 96, height: 64, frameIndex: 0),
        try TestFrames.synthetic420(width: 96, height: 64, frameIndex: 1)
    ]
    let codec = try PyrowaveCodec()
    let encoded = try frames.map { try codec.encode($0, configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0)) }

    var writer = try PyrowaveStreamWriter(
        url: url,
        header: PyrowaveStreamHeader(
            frame: frames[0],
            frameRateNumerator: 120,
            frameRateDenominator: 1
        )
    )
    for frame in encoded {
        try writer.writeFrame(frame)
    }

    var reader = try PyrowaveStreamReader(url: url)
    #expect(reader.header.width == 96)
    #expect(reader.header.height == 64)
    #expect(reader.header.chroma == .yuv420)
    #expect(reader.header.frameRateNumerator == 120)
    #expect(reader.header.frameRateDenominator == 1)
    #expect(reader.header.bitDepth == 8)

    let first = try #require(try reader.readFrame())
    let second = try #require(try reader.readFrame())
    #expect(try reader.readFrame() == nil)
    let decodedFirst = try codec.decode(first)
    let expectedFirst = try codec.decode(encoded[0])
    let decodedSecond = try codec.decode(second)
    let expectedSecond = try codec.decode(encoded[1])
    #expect(decodedFirst == expectedFirst)
    #expect(decodedSecond == expectedSecond)
}

@Test func pyrowaveStreamFilePreservesExactHighBitDepth() throws {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("high-depth.pwks")
    let frame = try TestFrames.synthetic420(width: 96, height: 64)
    let encoded = try PyrowaveCodec().encode(
        frame,
        configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0)
    )

    var writer = try PyrowaveStreamWriter(
        url: url,
        header: PyrowaveStreamHeader(frame: frame, bitDepth: 10)
    )
    try writer.writeFrame(encoded)

    var reader = try PyrowaveStreamReader(url: url)
    #expect(reader.header.bitDepth == 10)
    #expect(try reader.readFrame() == encoded)
}

@Test func pyrowaveStreamFilePreservesVideoSignalMetadata() throws {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("video-signal.pwks")
    let source = try TestFrames.synthetic420(width: 96, height: 64)
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
    let encoded = try PyrowaveCodec().encode(
        frame,
        configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0)
    )

    var writer = try PyrowaveStreamWriter(url: url, header: PyrowaveStreamHeader(frame: frame))
    try writer.writeFrame(encoded)

    let reader = try PyrowaveStreamReader(url: url)
    #expect(reader.header.videoSignal == frame.videoSignal)
}

@Test func pyrowaveStreamFileRejectsMissingMagic() throws {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("bad.pwks")
    try Data("NOTWAVE!".utf8).write(to: url)

    #expect(throws: PyrowaveError.unsupportedFormat("missing PYROWAVE stream magic")) {
        _ = try PyrowaveStreamReader(url: url)
    }
}

@Test func pyrowaveStreamFileRejectsFrameThatDoesNotMatchHeader() throws {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("mismatch-write.pwks")
    let headerFrame = try TestFrames.synthetic420(width: 96, height: 64)
    let mismatchedFrame = try TestFrames.synthetic444(width: 96, height: 64)
    let mismatchedEncoded = try PyrowaveCodec().encode(
        mismatchedFrame,
        configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0)
    )

    var writer = try PyrowaveStreamWriter(url: url, header: PyrowaveStreamHeader(frame: headerFrame))
    #expect(throws: PyrowaveError.invalidBitstream("encoded frame does not match stream header")) {
        try writer.writeFrame(mismatchedEncoded)
    }
}

@Test func pyrowaveStreamFileReaderRejectsMismatchedEmbeddedFrame() throws {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("mismatch-read.pwks")
    let header = try PyrowaveStreamHeader(width: 96, height: 64, chroma: .yuv420)
    let mismatchedFrame = try TestFrames.synthetic444(width: 96, height: 64)
    let mismatchedEncoded = try PyrowaveCodec().encode(
        mismatchedFrame,
        configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0)
    )

    var writer = BinaryWriter()
    writer.append(data: Data("PYROWAVE".utf8))
    for word in [
        UInt32(header.width),
        UInt32(header.height),
        UInt32(0),
        UInt32(header.bitDepth),
        UInt32(header.videoSignal.colorPrimaries.rawValue) << 0 |
            UInt32(header.videoSignal.transferFunction.rawValue) << 1 |
            UInt32(header.videoSignal.yCbCrTransform.rawValue) << 2 |
            UInt32(header.videoSignal.yCbCrRange.rawValue) << 3 |
            UInt32(header.videoSignal.chromaSiting.rawValue) << 4,
        UInt32(header.frameRateNumerator),
        UInt32(header.frameRateDenominator),
        0
    ] {
        writer.append(word)
    }
    writer.append(UInt32(mismatchedEncoded.data.count))
    writer.append(data: mismatchedEncoded.data)
    try writer.data.write(to: url)

    var reader = try PyrowaveStreamReader(url: url)
    #expect(throws: PyrowaveError.invalidBitstream("encoded frame does not match stream header")) {
        _ = try reader.readFrame()
    }
}

@Test func packetStreamDecoderAllowsPartialFrameAfterHalfTheBlocks() throws {
    let frame = try TestFrames.synthetic420(width: 96, height: 64)
    let encoded = try PyrowaveCodec().encode(
        frame,
        configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0)
    )
    let packets = try encoded.packetized(maximumPacketBytes: 8)
    let sequencePacket = try #require(packets.first)
    var sequenceReader = BinaryReader(sequencePacket.data)
    let sequence = try PyrowaveSequenceHeader(reader: &sequenceReader)
    let stream = try PyrowavePacketStreamDecoder()

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

@Test func geometryAwarePacketStreamDecoderAcceptsCoefficientRestartWithoutSequenceHeader() throws {
    let width = 64
    let height = 64
    let layout = try PyrowaveBlockLayout(width: width, height: height, chroma: .yuv420)
    let requiredPackets = layout.descriptors.count / 2 + 1
    let stream = try PyrowavePacketStreamDecoder(width: width, height: height, chroma: .yuv420)
    var coefficients = Array(repeating: Int16(0), count: PyrowaveBitstream.coefficientBlockSize * PyrowaveBitstream.coefficientBlockSize)
    coefficients[0] = 1

    for descriptor in layout.descriptors.prefix(requiredPackets) {
        let packet = try #require(try PyrowaveCoefficientBlockCodec.encodeBlock(
            blockIndex: descriptor.blockIndex,
            coefficients: coefficients,
            stride: PyrowaveBitstream.coefficientBlockSize,
            originX: 0,
            originY: 0,
            validWidth: PyrowaveBitstream.coefficientBlockSize,
            validHeight: PyrowaveBitstream.coefficientBlockSize,
            threshold: 0,
            sequence: 4,
            quantCode: try PyrowaveQuantization.encodeBlockScale(1.0 / 1024.0)
        ))
        try stream.pushPacket(EncodedPacket(data: packet))
    }

    #expect(!stream.decodeIsReady())
    #expect(stream.decodeIsReady(allowPartialFrame: true))
    let decoded = try stream.decode(allowPartialFrame: true)
    #expect(decoded.width == width)
    #expect(decoded.height == height)
    #expect(decoded.chroma == .yuv420)
}

@Test func geometryAwarePacketStreamDecoderRejectsMismatchedSequenceHeader() throws {
    let stream = try PyrowavePacketStreamDecoder(width: 64, height: 64, chroma: .yuv420)
    var writer = BinaryWriter()
    try PyrowaveSequenceHeader(
        width: 96,
        height: 64,
        sequence: 1,
        totalBlocks: 1,
        chroma: .yuv420
    ).write(to: &writer)

    #expect(throws: PyrowaveError.invalidBitstream("sequence header does not match decoder geometry")) {
        try stream.pushPacket(EncodedPacket(data: writer.data))
    }
}

@Test func packetStreamDecoderRejectsCoefficientPacketShorterThanHeader() throws {
    let stream = try PyrowavePacketStreamDecoder(width: 64, height: 64, chroma: .yuv420)
    var writer = BinaryWriter()
    try PyrowavePacketHeader(
        ballot: 1,
        payloadWords: 1,
        sequence: 1,
        extended: false,
        quantCode: 0,
        blockIndex: 0
    ).write(to: &writer)

    #expect(throws: PyrowaveError.invalidBitstream("payload_words is not large enough")) {
        try stream.pushPacket(EncodedPacket(data: writer.data))
    }
}

@Test func packetStreamDecoderRejectsSameSequenceHeaderMutation() throws {
    let frame = try TestFrames.synthetic420(width: 64, height: 64)
    let encoded = try PyrowaveCodec().encode(
        frame,
        configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0)
    )
    let packets = try encoded.packetized(maximumPacketBytes: 8)
    let sequencePacket = try #require(packets.first)
    var sequenceReader = BinaryReader(sequencePacket.data)
    let sequence = try PyrowaveSequenceHeader(reader: &sequenceReader)
    let stream = try PyrowavePacketStreamDecoder()

    try stream.pushPacket(sequencePacket)
    try stream.pushPacket(try #require(packets.dropFirst().first))

    var mutatedWriter = BinaryWriter()
    try PyrowaveSequenceHeader(
        width: sequence.width,
        height: sequence.height,
        sequence: sequence.sequence,
        totalBlocks: sequence.totalBlocks + 1,
        chroma: sequence.chroma
    ).write(to: &mutatedWriter)

    #expect(throws: PyrowaveError.invalidBitstream("sequence header changed within active sequence")) {
        try stream.pushPacket(EncodedPacket(data: mutatedWriter.data))
    }
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

    let codec = try PyrowaveCodec()
    let encoded = try codec.encode(frame, configuration: CodecConfiguration(quantizationStep: 1.0 / 1024.0))
    let decoded = try codec.decode(encoded)

    #expect(decoded.videoSignal == frame.videoSignal)
}

@Test func codecPacketsUseGlobalPyrowaveBlockOrder() throws {
    let frame = try TestFrames.synthetic420(width: 160, height: 96)
    let encoded = try PyrowaveCodec().encode(
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
    let encoded = try PyrowaveCodec().encode(frame, configuration: configuration)

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
    let codec = try PyrowaveCodec()
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

@Test func pyrowaveBlockLayoutIncludesTopLevelChromaFor444() throws {
    let layout = try PyrowaveBlockLayout(width: 256, height: 256, chroma: .yuv444)
    #expect(layout.descriptors.contains { $0.level == 0 && $0.component == 1 && $0.band == 1 })
    #expect(layout.descriptors.contains { $0.level == 0 && $0.component == 2 && $0.band == 3 })
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

@Test func pyrowaveCoefficientBlockRejectsNonZeroWordPadding() throws {
    let stride = 32
    var coefficients = Array(repeating: Int16(0), count: stride * stride)
    coefficients[0] = 1

    var payload = try #require(try PyrowaveCoefficientBlockCodec.encodeBlock(
        blockIndex: 5,
        coefficients: coefficients,
        stride: stride,
        originX: 0,
        originY: 0,
        validWidth: 32,
        validHeight: 32,
        threshold: 0
    ))

    #expect(payload.count == 16)
    payload[payload.count - 1] = 0xff

    var reader = BinaryReader(payload)
    #expect(throws: PyrowaveError.invalidBitstream("non-zero coefficient packet padding")) {
        _ = try PyrowaveCoefficientBlockCodec.decodeBlock(reader: &reader)
    }
}

@Test func pyrowaveCoefficientBlockRejectsPayloadShorterThanHeader() throws {
    var writer = BinaryWriter()
    try PyrowavePacketHeader(
        ballot: 1,
        payloadWords: 1,
        sequence: 0,
        extended: false,
        quantCode: 0,
        blockIndex: 0
    ).write(to: &writer)

    var reader = BinaryReader(writer.data)
    #expect(throws: PyrowaveError.invalidBitstream("payload_words is not large enough")) {
        _ = try PyrowaveCoefficientBlockCodec.decodeBlock(reader: &reader)
    }
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
