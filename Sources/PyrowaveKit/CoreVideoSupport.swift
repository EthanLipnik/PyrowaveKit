import Foundation

#if canImport(CoreVideo)
import CoreVideo

extension YUVFrame {
    public init(
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

    public func makeCVPixelBuffer(pixelFormat: OSType? = nil) throws -> CVPixelBuffer {
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

    public func copy(to cvPixelBuffer: CVPixelBuffer) throws {
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

    public static func cvPixelFormat(for videoSignal: VideoSignalMetadata) -> OSType {
        videoSignal.yCbCrRange == .limited ? kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange : kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
    }
}
#endif
