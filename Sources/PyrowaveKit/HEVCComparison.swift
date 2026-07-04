import Foundation

public struct CodecBenchmarkResult: Codable, Equatable, Sendable {
    public var codec: String
    public var encodedBytes: Int
    public var encodeSeconds: Double
    public var decodeSeconds: Double
    public var metrics: FrameMetrics?
    public var note: String?

    public init(
        codec: String,
        encodedBytes: Int,
        encodeSeconds: Double,
        decodeSeconds: Double,
        metrics: FrameMetrics?,
        note: String?
    ) {
        self.codec = codec
        self.encodedBytes = encodedBytes
        self.encodeSeconds = encodeSeconds
        self.decodeSeconds = decodeSeconds
        self.metrics = metrics
        self.note = note
    }
}

public enum HEVCComparison {
    public static func runFFmpegVideoToolboxComparison(
        reference: YUVFrame,
        workingDirectory: URL,
        bitrate: Int
    ) throws -> CodecBenchmarkResult {
        guard reference.chroma == .yuv420 else {
            throw PyrowaveError.unsupportedFormat("HEVC comparison currently expects yuv420p")
        }
        guard let ffmpeg = findExecutable("ffmpeg") else {
            return CodecBenchmarkResult(
                codec: "hevc_videotoolbox",
                encodedBytes: 0,
                encodeSeconds: 0,
                decodeSeconds: 0,
                metrics: nil,
                note: "ffmpeg not found; install ffmpeg with VideoToolbox support to run HEVC comparison"
            )
        }

        try FileManager.default.createDirectory(at: workingDirectory, withIntermediateDirectories: true)
        let rawURL = workingDirectory.appendingPathComponent("reference.yuv")
        let hevcURL = workingDirectory.appendingPathComponent("hevc-videotoolbox.mov")
        let decodedURL = workingDirectory.appendingPathComponent("hevc-decoded.yuv")

        try writeRaw420(reference, to: rawURL)

        var stopwatch = Stopwatch()
        try run(ffmpeg, arguments: [
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-s", "\(reference.width)x\(reference.height)",
            "-r", "60",
            "-i", rawURL.path,
            "-c:v", "hevc_videotoolbox",
            "-b:v", "\(bitrate)",
            "-tag:v", "hvc1",
            hevcURL.path
        ])
        let encodeSeconds = stopwatch.lapSeconds()

        try run(ffmpeg, arguments: [
            "-hide_banner", "-loglevel", "error", "-y",
            "-i", hevcURL.path,
            "-f", "rawvideo", "-pix_fmt", "yuv420p",
            decodedURL.path
        ])
        let decodeSeconds = stopwatch.lapSeconds()

        let decoded = try readRaw420(url: decodedURL, width: reference.width, height: reference.height)
        let encodedBytes = (try FileManager.default.attributesOfItem(atPath: hevcURL.path)[.size] as? NSNumber)?.intValue ?? 0

        return try CodecBenchmarkResult(
            codec: "hevc_videotoolbox",
            encodedBytes: encodedBytes,
            encodeSeconds: encodeSeconds,
            decodeSeconds: decodeSeconds,
            metrics: Metrics.compare(reference, decoded),
            note: nil
        )
    }

    static func writeRaw420(_ frame: YUVFrame, to url: URL) throws {
        var data = Data()
        data.reserveCapacity(frame.y.data.count + frame.cb.data.count + frame.cr.data.count)
        data.append(contentsOf: frame.y.data)
        data.append(contentsOf: frame.cb.data)
        data.append(contentsOf: frame.cr.data)
        try data.write(to: url)
    }

    static func readRaw420(url: URL, width: Int, height: Int) throws -> YUVFrame {
        let data = try Data(contentsOf: url)
        let ySize = width * height
        let cSize = (width / 2) * (height / 2)
        guard data.count >= ySize + 2 * cSize else {
            throw PyrowaveError.truncatedInput
        }
        let y = [UInt8](data[0..<ySize])
        let cb = [UInt8](data[ySize..<ySize + cSize])
        let cr = [UInt8](data[ySize + cSize..<ySize + 2 * cSize])
        return try YUVFrame(
            width: width,
            height: height,
            chroma: .yuv420,
            y: Plane8(width: width, height: height, data: y),
            cb: Plane8(width: width / 2, height: height / 2, data: cb),
            cr: Plane8(width: width / 2, height: height / 2, data: cr)
        )
    }

    static func findExecutable(_ name: String) -> String? {
        let paths = (ProcessInfo.processInfo.environment["PATH"] ?? "").split(separator: ":").map(String.init)
        for path in paths {
            let candidate = URL(fileURLWithPath: path).appendingPathComponent(name).path
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return candidate
            }
        }
        return nil
    }

    static func run(_ executable: String, arguments: [String]) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        let errorPipe = Pipe()
        process.standardError = errorPipe
        try process.run()
        process.waitUntilExit()
        if process.terminationStatus != 0 {
            let error = String(decoding: errorPipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            throw PyrowaveError.processFailed(error)
        }
    }
}
