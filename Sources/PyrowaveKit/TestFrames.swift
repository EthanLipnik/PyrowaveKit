import Foundation

public enum TestFrames {
    public static func synthetic420(width: Int, height: Int, frameIndex: Int = 0) throws -> YUVFrame {
        guard width > 0, height > 0, width % 2 == 0, height % 2 == 0 else {
            throw PyrowaveError.invalidDimensions
        }

        var y = Array(repeating: UInt8(0), count: width * height)
        var cb = Array(repeating: UInt8(0), count: (width / 2) * (height / 2))
        var cr = cb

        for row in 0..<height {
            for column in 0..<width {
                let gradient = (column * 255) / max(1, width - 1)
                let wave = Int(32.0 * sin(Double(row + frameIndex * 3) / 9.0))
                y[row * width + column] = UInt8(clamping: gradient + wave)
            }
        }

        for row in 0..<(height / 2) {
            for column in 0..<(width / 2) {
                cb[row * (width / 2) + column] = UInt8(clamping: 128 + ((row + frameIndex) % 24) - 12)
                cr[row * (width / 2) + column] = UInt8(clamping: 128 + ((column + frameIndex) % 32) - 16)
            }
        }

        return try YUVFrame(
            width: width,
            height: height,
            chroma: .yuv420,
            y: Plane8(width: width, height: height, data: y),
            cb: Plane8(width: width / 2, height: height / 2, data: cb),
            cr: Plane8(width: width / 2, height: height / 2, data: cr)
        )
    }
}
