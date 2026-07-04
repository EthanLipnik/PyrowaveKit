import Foundation

enum Wavelet {
    static let alignment = 32
    static let minimumSize = 128
    private static let alpha: Float = -1.586134342059924
    private static let beta: Float = -0.052980118572961
    private static let gamma: Float = 0.882911075530934
    private static let delta: Float = 0.443506852043971
    private static let k: Float = 1.230174104914001
    private static let invK: Float = 1.0 / 1.230174104914001

    static func alignedDimension(_ value: Int) -> Int {
        max(minimumSize, (value + alignment - 1) & ~(alignment - 1))
    }

    static func usableLevels(width: Int, height: Int, requested: Int) -> Int {
        var w = width
        var h = height
        var levels = 0
        while levels < requested, w >= 2, h >= 2, w % 2 == 0, h % 2 == 0 {
            levels += 1
            w /= 2
            h /= 2
        }
        return levels
    }

    static func padPlane(_ plane: Plane8) -> (samples: [Float], width: Int, height: Int) {
        let paddedWidth = alignedDimension(plane.width)
        let paddedHeight = alignedDimension(plane.height)
        var samples = Array(repeating: Float(0), count: paddedWidth * paddedHeight)

        for y in 0..<paddedHeight {
            let sourceY = min(y, plane.height - 1)
            for x in 0..<paddedWidth {
                let sourceX = min(x, plane.width - 1)
                samples[y * paddedWidth + x] = Float(plane.data[sourceY * plane.width + sourceX]) / 255.0 - 0.5
            }
        }

        return (samples, paddedWidth, paddedHeight)
    }

    static func cropPlane(_ samples: [Float], paddedWidth: Int, width: Int, height: Int) throws -> Plane8 {
        var output = Array(repeating: UInt8(0), count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let normalized = min(1.0, max(0.0, samples[y * paddedWidth + x] + 0.5))
                output[y * width + x] = UInt8((normalized * 255.0).rounded())
            }
        }
        return try Plane8(width: width, height: height, data: output)
    }

    static func forward2D(_ samples: inout [Float], width: Int, height: Int, levels: Int) {
        var currentWidth = width
        var currentHeight = height
        for _ in 0..<levels {
            transformRows(&samples, width: width, activeWidth: currentWidth, activeHeight: currentHeight, inverse: false)
            transformColumns(&samples, width: width, activeWidth: currentWidth, activeHeight: currentHeight, inverse: false)
            currentWidth /= 2
            currentHeight /= 2
        }
    }

    static func inverse2D(_ samples: inout [Float], width: Int, height: Int, levels: Int) {
        var sizes = [(Int, Int)]()
        var currentWidth = width
        var currentHeight = height
        for _ in 0..<levels {
            sizes.append((currentWidth, currentHeight))
            currentWidth /= 2
            currentHeight /= 2
        }

        for (activeWidth, activeHeight) in sizes.reversed() {
            transformColumns(&samples, width: width, activeWidth: activeWidth, activeHeight: activeHeight, inverse: true)
            transformRows(&samples, width: width, activeWidth: activeWidth, activeHeight: activeHeight, inverse: true)
        }
    }

    private static func transformRows(_ samples: inout [Float], width: Int, activeWidth: Int, activeHeight: Int, inverse: Bool) {
        var line = Array(repeating: Float(0), count: activeWidth)
        for y in 0..<activeHeight {
            let base = y * width
            for x in 0..<activeWidth {
                line[x] = samples[base + x]
            }

            if inverse {
                inverse1D(&line)
            } else {
                forward1D(&line)
            }

            for x in 0..<activeWidth {
                samples[base + x] = line[x]
            }
        }
    }

    private static func transformColumns(_ samples: inout [Float], width: Int, activeWidth: Int, activeHeight: Int, inverse: Bool) {
        var line = Array(repeating: Float(0), count: activeHeight)
        for x in 0..<activeWidth {
            for y in 0..<activeHeight {
                line[y] = samples[y * width + x]
            }

            if inverse {
                inverse1D(&line)
            } else {
                forward1D(&line)
            }

            for y in 0..<activeHeight {
                samples[y * width + x] = line[y]
            }
        }
    }

    private static func mirror(_ index: Int, count: Int) -> Int {
        guard count > 1 else { return 0 }
        if index < 0 {
            return -index
        }
        if index >= count {
            return 2 * count - index - 2
        }
        return index
    }

    private static func forward1D(_ values: inout [Float]) {
        let n = values.count
        guard n >= 2 else { return }

        var lifted = values
        var i = 1
        while i < n {
            lifted[i] += alpha * (lifted[i - 1] + lifted[mirror(i + 1, count: n)])
            i += 2
        }

        i = 0
        while i < n {
            lifted[i] += beta * (lifted[mirror(i - 1, count: n)] + lifted[mirror(i + 1, count: n)])
            i += 2
        }

        i = 1
        while i < n {
            lifted[i] += gamma * (lifted[i - 1] + lifted[mirror(i + 1, count: n)])
            i += 2
        }

        i = 0
        while i < n {
            lifted[i] += delta * (lifted[mirror(i - 1, count: n)] + lifted[mirror(i + 1, count: n)])
            i += 2
        }

        i = 0
        while i < n {
            lifted[i] *= invK
            if i + 1 < n {
                lifted[i + 1] *= k
            }
            i += 2
        }

        let lowCount = (n + 1) / 2
        for x in 0..<lowCount {
            values[x] = lifted[x * 2]
        }
        for x in 0..<(n / 2) {
            values[lowCount + x] = lifted[x * 2 + 1]
        }
    }

    private static func inverse1D(_ values: inout [Float]) {
        let n = values.count
        guard n >= 2 else { return }

        let lowCount = (n + 1) / 2
        var lifted = Array(repeating: Float(0), count: n)
        for x in 0..<lowCount {
            lifted[x * 2] = values[x] * k
        }
        for x in 0..<(n / 2) {
            lifted[x * 2 + 1] = values[lowCount + x] * invK
        }

        var i = 0
        while i < n {
            lifted[i] -= delta * (lifted[mirror(i - 1, count: n)] + lifted[mirror(i + 1, count: n)])
            i += 2
        }

        i = 1
        while i < n {
            lifted[i] -= gamma * (lifted[i - 1] + lifted[mirror(i + 1, count: n)])
            i += 2
        }

        i = 0
        while i < n {
            lifted[i] -= beta * (lifted[mirror(i - 1, count: n)] + lifted[mirror(i + 1, count: n)])
            i += 2
        }

        i = 1
        while i < n {
            lifted[i] -= alpha * (lifted[i - 1] + lifted[mirror(i + 1, count: n)])
            i += 2
        }

        values = lifted
    }
}
