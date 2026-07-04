import Foundation

enum Metrics {
    static func compare(_ reference: YUVFrame, _ candidate: YUVFrame) throws -> FrameMetrics {
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

    static func compare(_ referenceFrames: [YUVFrame], _ candidateFrames: [YUVFrame]) throws -> FrameMetrics {
        guard !referenceFrames.isEmpty else {
            throw PyrowaveError.truncatedInput
        }
        guard referenceFrames.count == candidateFrames.count else {
            throw PyrowaveError.invalidDimensions
        }

        var yError = 0.0
        var cbError = 0.0
        var crError = 0.0
        var ySamples = 0
        var cbSamples = 0
        var crSamples = 0

        for (reference, candidate) in zip(referenceFrames, candidateFrames) {
            guard reference.width == candidate.width,
                  reference.height == candidate.height,
                  reference.chroma == candidate.chroma else {
                throw PyrowaveError.invalidDimensions
            }

            yError += squaredError(reference.y, candidate.y)
            cbError += squaredError(reference.cb, candidate.cb)
            crError += squaredError(reference.cr, candidate.cr)
            ySamples += reference.y.data.count
            cbSamples += reference.cb.data.count
            crSamples += reference.cr.data.count
        }

        let yMSE = yError / Double(ySamples)
        let cbMSE = cbError / Double(cbSamples)
        let crMSE = crError / Double(crSamples)
        let totalSamples = Double(ySamples + cbSamples + crSamples)
        let weightedMSE = (yError + cbError + crError) / totalSamples
        return FrameMetrics(
            y: ComponentMetrics(mse: yMSE, psnr: psnr(mse: yMSE)),
            cb: ComponentMetrics(mse: cbMSE, psnr: psnr(mse: cbMSE)),
            cr: ComponentMetrics(mse: crMSE, psnr: psnr(mse: crMSE)),
            weightedPSNR: psnr(mse: weightedMSE)
        )
    }

    static func comparePlane(_ reference: Plane8, _ candidate: Plane8) -> ComponentMetrics {
        precondition(reference.width == candidate.width && reference.height == candidate.height)
        let mse = squaredError(reference, candidate) / Double(reference.data.count)
        return ComponentMetrics(mse: mse, psnr: psnr(mse: mse))
    }

    private static func squaredError(_ reference: Plane8, _ candidate: Plane8) -> Double {
        precondition(reference.width == candidate.width && reference.height == candidate.height)
        var sum = 0.0
        for index in reference.data.indices {
            let delta = Double(Int(reference.data[index]) - Int(candidate.data[index]))
            sum += delta * delta
        }
        return sum
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
