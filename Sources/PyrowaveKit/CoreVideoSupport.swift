import Foundation
import CoreVideo
import Metal

extension YUVFrame {
    init(
        cvPixelBuffer: CVPixelBuffer,
        videoSignal: VideoSignalMetadata? = nil
    ) throws {
        let pixelFormat = CVPixelBufferGetPixelFormatType(cvPixelBuffer)
        let supportedFormats: Set<OSType> = [
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        ]
        guard supportedFormats.contains(pixelFormat) else {
            throw PyrowaveError.unsupportedFormat("CVPixelBuffer pixel format \(pixelFormat)")
        }

        CVPixelBufferLockBaseAddress(cvPixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(cvPixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidthOfPlane(cvPixelBuffer, 0)
        let height = CVPixelBufferGetHeightOfPlane(cvPixelBuffer, 0)
        guard width > 0, height > 0,
              CVPixelBufferGetPlaneCount(cvPixelBuffer) >= 2,
              CVPixelBufferGetWidthOfPlane(cvPixelBuffer, 1) == width / 2,
              CVPixelBufferGetHeightOfPlane(cvPixelBuffer, 1) == height / 2,
              let yBase = CVPixelBufferGetBaseAddressOfPlane(cvPixelBuffer, 0),
              let cbCrBase = CVPixelBufferGetBaseAddressOfPlane(cvPixelBuffer, 1) else {
            throw PyrowaveError.invalidDimensions
        }

        let yStride = CVPixelBufferGetBytesPerRowOfPlane(cvPixelBuffer, 0)
        let cbCrStride = CVPixelBufferGetBytesPerRowOfPlane(cvPixelBuffer, 1)
        let yByteCount = yStride * (height - 1) + width
        let cbCrByteCount = cbCrStride * (height / 2 - 1) + width
        let y = Array(UnsafeBufferPointer(start: yBase.assumingMemoryBound(to: UInt8.self), count: yByteCount))
        let cbCr = Array(UnsafeBufferPointer(start: cbCrBase.assumingMemoryBound(to: UInt8.self), count: cbCrByteCount))
        let inferredSignal = VideoSignalMetadata(
            yCbCrRange: pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ? .limited : .full
        )

        try self.init(
            width: width,
            height: height,
            nv12Y: y,
            nv12CbCr: cbCr,
            yRowStride: yStride,
            cbCrRowStride: cbCrStride,
            videoSignal: videoSignal ?? inferredSignal
        )
    }

    func makeCVPixelBuffer(pixelFormat: OSType? = nil) throws -> CVPixelBuffer {
        let format = pixelFormat ?? Self.cvPixelFormat(for: videoSignal)
        var pixelBuffer: CVPixelBuffer?
        let attributes = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ] as CFDictionary
        let status = CVPixelBufferCreate(
            nil,
            width,
            height,
            format,
            attributes,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer else {
            throw PyrowaveError.processFailed("failed to allocate CVPixelBuffer")
        }
        try copy(to: pixelBuffer)
        return pixelBuffer
    }

    func copy(to cvPixelBuffer: CVPixelBuffer) throws {
        let pixelFormat = CVPixelBufferGetPixelFormatType(cvPixelBuffer)
        let supportedFormats: Set<OSType> = [
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        ]
        guard supportedFormats.contains(pixelFormat) else {
            throw PyrowaveError.unsupportedFormat("CVPixelBuffer pixel format \(pixelFormat)")
        }
        guard chroma == .yuv420 else {
            throw PyrowaveError.unsupportedFormat("CVPixelBuffer export expects yuv420 frames")
        }

        CVPixelBufferLockBaseAddress(cvPixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(cvPixelBuffer, []) }
        guard CVPixelBufferGetPlaneCount(cvPixelBuffer) >= 2,
              CVPixelBufferGetWidthOfPlane(cvPixelBuffer, 0) == width,
              CVPixelBufferGetHeightOfPlane(cvPixelBuffer, 0) == height,
              CVPixelBufferGetWidthOfPlane(cvPixelBuffer, 1) == width / 2,
              CVPixelBufferGetHeightOfPlane(cvPixelBuffer, 1) == height / 2,
              let yBase = CVPixelBufferGetBaseAddressOfPlane(cvPixelBuffer, 0),
              let cbCrBase = CVPixelBufferGetBaseAddressOfPlane(cvPixelBuffer, 1) else {
            throw PyrowaveError.invalidDimensions
        }

        let yStride = CVPixelBufferGetBytesPerRowOfPlane(cvPixelBuffer, 0)
        let cbCrStride = CVPixelBufferGetBytesPerRowOfPlane(cvPixelBuffer, 1)
        let yDestination = yBase.assumingMemoryBound(to: UInt8.self)
        y.data.withUnsafeBufferPointer { source in
            for row in 0..<height {
                let sourceStart = row * width
                yDestination.advanced(by: row * yStride).update(from: source.baseAddress!.advanced(by: sourceStart), count: width)
            }
        }

        let cbCrDestination = cbCrBase.assumingMemoryBound(to: UInt8.self)
        for row in 0..<(height / 2) {
            let sourceStart = row * (width / 2)
            let destinationRow = cbCrDestination.advanced(by: row * cbCrStride)
            for column in 0..<(width / 2) {
                destinationRow[column * 2] = cb.data[sourceStart + column]
                destinationRow[column * 2 + 1] = cr.data[sourceStart + column]
            }
        }
    }

    static func cvPixelFormat(for videoSignal: VideoSignalMetadata) -> OSType {
        videoSignal.yCbCrRange == .limited ? kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange : kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
    }
}

extension PyrowaveCodec {
    public func encodeGPUFrame(
        _ cvPixelBuffer: CVPixelBuffer,
        configuration: CodecConfiguration = CodecConfiguration(),
        videoSignal: VideoSignalMetadata? = nil
    ) throws -> PyrowaveGPUFrame {
        let textures = try makeNV12MetalTexturesAndSignal(from: cvPixelBuffer, videoSignal: videoSignal)
        return try encodeGPUFrame(
            yTexture: textures.yTexture,
            cbCrTexture: textures.cbCrTexture,
            configuration: configuration,
            videoSignal: textures.videoSignal
        )
    }

    public func encode(
        _ cvPixelBuffer: CVPixelBuffer,
        configuration: CodecConfiguration = CodecConfiguration(),
        videoSignal: VideoSignalMetadata? = nil
    ) throws -> EncodedFrame {
        try exportGPUFrame(try encodeGPUFrame(
            cvPixelBuffer,
            configuration: configuration,
            videoSignal: videoSignal
        ))
    }

    private func makeNV12MetalTexturesAndSignal(
        from cvPixelBuffer: CVPixelBuffer,
        videoSignal: VideoSignalMetadata?
    ) throws -> (yTexture: MTLTexture, cbCrTexture: MTLTexture, videoSignal: VideoSignalMetadata, yReference: CVMetalTexture, cbCrReference: CVMetalTexture) {
        let pixelFormat = CVPixelBufferGetPixelFormatType(cvPixelBuffer)
        let supportedFormats: Set<OSType> = [
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        ]
        guard supportedFormats.contains(pixelFormat),
              CVPixelBufferGetPlaneCount(cvPixelBuffer) >= 2 else {
            throw PyrowaveError.unsupportedFormat("CVPixelBuffer encode expects 8-bit NV12")
        }

        let width = CVPixelBufferGetWidthOfPlane(cvPixelBuffer, 0)
        let height = CVPixelBufferGetHeightOfPlane(cvPixelBuffer, 0)
        let chromaWidth = CVPixelBufferGetWidthOfPlane(cvPixelBuffer, 1)
        let chromaHeight = CVPixelBufferGetHeightOfPlane(cvPixelBuffer, 1)
        guard width > 0,
              height > 0,
              chromaWidth == width / 2,
              chromaHeight == height / 2 else {
            throw PyrowaveError.invalidDimensions
        }

        let yTexture = try makeMetalTexture(
            from: cvPixelBuffer,
            pixelFormat: .r8Unorm,
            width: width,
            height: height,
            planeIndex: 0
        )
        let cbCrTexture = try makeMetalTexture(
            from: cvPixelBuffer,
            pixelFormat: .rg8Unorm,
            width: chromaWidth,
            height: chromaHeight,
            planeIndex: 1
        )
        let inferredSignal = VideoSignalMetadata(
            yCbCrRange: pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ? .limited : .full
        )
        return (
            yTexture: yTexture.texture,
            cbCrTexture: cbCrTexture.texture,
            videoSignal: videoSignal ?? inferredSignal,
            yReference: yTexture.reference,
            cbCrReference: cbCrTexture.reference
        )
    }

    private func makeMetalTexture(
        from cvPixelBuffer: CVPixelBuffer,
        pixelFormat: MTLPixelFormat,
        width: Int,
        height: Int,
        planeIndex: Int
    ) throws -> (texture: MTLTexture, reference: CVMetalTexture) {
        var textureReference: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            nil,
            coreVideoTextureCache,
            cvPixelBuffer,
            nil,
            pixelFormat,
            width,
            height,
            planeIndex,
            &textureReference
        )
        guard status == kCVReturnSuccess,
              let textureReference,
              let texture = CVMetalTextureGetTexture(textureReference) else {
            throw PyrowaveError.processFailed("failed to create CVMetalTexture for plane \(planeIndex)")
        }
        return (texture, textureReference)
    }

    public func decodeToCVPixelBuffer(
        _ frame: EncodedFrame,
        pixelFormat: OSType? = nil
    ) throws -> CVPixelBuffer {
        try decodeGPUFrameToCVPixelBuffer(try importGPUFrame(frame), pixelFormat: pixelFormat)
    }

    public func decodeGPUFrameToCVPixelBuffer(
        _ frame: PyrowaveGPUFrame,
        pixelFormat: OSType? = nil
    ) throws -> CVPixelBuffer {
        guard frame.chroma == .yuv420 else {
            throw PyrowaveError.unsupportedFormat("CVPixelBuffer decode expects yuv420 frames")
        }
        let format = pixelFormat ?? YUVFrame.cvPixelFormat(for: frame.videoSignal)
        let supportedFormats: Set<OSType> = [
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        ]
        guard supportedFormats.contains(format) else {
            throw PyrowaveError.unsupportedFormat("CVPixelBuffer pixel format \(format)")
        }

        var pixelBuffer: CVPixelBuffer?
        let attributes = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            kCVPixelBufferMetalCompatibilityKey as String: true
        ] as CFDictionary
        let status = CVPixelBufferCreate(
            nil,
            frame.width,
            frame.height,
            format,
            attributes,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer else {
            throw PyrowaveError.processFailed("failed to allocate decode CVPixelBuffer")
        }

        try decodeGPUFrame(frame, to: pixelBuffer)
        return pixelBuffer
    }

    public func decodeGPUFrame(
        _ frame: PyrowaveGPUFrame,
        to cvPixelBuffer: CVPixelBuffer
    ) throws {
        let pixelFormat = CVPixelBufferGetPixelFormatType(cvPixelBuffer)
        let supportedFormats: Set<OSType> = [
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        ]
        guard frame.chroma == .yuv420,
              supportedFormats.contains(pixelFormat),
              CVPixelBufferGetPlaneCount(cvPixelBuffer) >= 2,
              CVPixelBufferGetWidthOfPlane(cvPixelBuffer, 0) == frame.width,
              CVPixelBufferGetHeightOfPlane(cvPixelBuffer, 0) == frame.height,
              CVPixelBufferGetWidthOfPlane(cvPixelBuffer, 1) == frame.width / 2,
              CVPixelBufferGetHeightOfPlane(cvPixelBuffer, 1) == frame.height / 2 else {
            throw PyrowaveError.invalidDimensions
        }

        let yTexture = try makeMetalTexture(
            from: cvPixelBuffer,
            pixelFormat: .r8Unorm,
            width: frame.width,
            height: frame.height,
            planeIndex: 0
        )
        let cbCrTexture = try makeMetalTexture(
            from: cvPixelBuffer,
            pixelFormat: .rg8Unorm,
            width: frame.width / 2,
            height: frame.height / 2,
            planeIndex: 1
        )
        try decodeGPUFrameToNV12Textures(
            frame,
            yTexture: yTexture.texture,
            cbCrTexture: cbCrTexture.texture
        )
    }
}
