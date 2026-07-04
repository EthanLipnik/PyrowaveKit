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
}
#endif
