import Foundation

public enum Metrics {
    public static func compare(_ reference: YUVFrame, _ candidate: YUVFrame) throws -> FrameMetrics {
        guard reference.width == candidate.width,
              reference.height == candidate.height,
              reference.chroma == candidate.chroma else {
            throw PyrowaveError.invalidDimensions
        }

        let y = comparePlane(reference.y, candidate.y)
        let cb = comparePlane(reference.cb, candidate.cb)
        let cr = comparePlane(reference.cr, candidate.cr)
        let lumaPixels = Double(reference.y.data.count)
        let chromaPixels = Double(reference.cb.data.count)
        let totalPixels = lumaPixels + 2.0 * chromaPixels
        let weightedMSE = (y.mse * lumaPixels + cb.mse * chromaPixels + cr.mse * chromaPixels) / totalPixels
        let weightedPSNR = psnr(mse: weightedMSE)
        return FrameMetrics(y: y, cb: cb, cr: cr, weightedPSNR: weightedPSNR)
    }

    public static func comparePlane(_ reference: Plane8, _ candidate: Plane8) -> ComponentMetrics {
        precondition(reference.width == candidate.width && reference.height == candidate.height)
        var sum = 0.0
        for index in reference.data.indices {
            let delta = Double(Int(reference.data[index]) - Int(candidate.data[index]))
            sum += delta * delta
        }

        let mse = sum / Double(reference.data.count)
        return ComponentMetrics(mse: mse, psnr: psnr(mse: mse))
    }

    private static func psnr(mse: Double) -> Double {
        if mse == 0 {
            return 999.0
        }
        return 10.0 * log10((255.0 * 255.0) / mse)
    }
}

public struct Stopwatch {
    private var start: UInt64

    public init() {
        start = DispatchTime.now().uptimeNanoseconds
    }

    public mutating func lapSeconds() -> Double {
        let now = DispatchTime.now().uptimeNanoseconds
        let elapsed = Double(now - start) / 1_000_000_000.0
        start = now
        return elapsed
    }
}
