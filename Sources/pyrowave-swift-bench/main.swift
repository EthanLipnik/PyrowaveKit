import Foundation
import PyrowaveKit

do {
    let arguments = try PyrowaveBenchmarkArguments()
    if arguments.shouldShowHelp {
        print(PyrowaveBenchmarkArguments.usage)
        exit(0)
    }
    try FileManager.default.createDirectory(at: arguments.outputDirectory, withIntermediateDirectories: true)
    let loaded = try PyrowaveBenchmarkRunner.loadFrames(arguments: arguments)
    let frames = loaded.frames
    try YUV4MPEGWriter.write(
        frames: frames,
        to: arguments.outputDirectory.appendingPathComponent(PyrowaveBenchmarkArtifactNames.referenceY4M),
        frameRateNumerator: loaded.frameRateNumerator,
        frameRateDenominator: loaded.frameRateDenominator
    )
    let pyrowaveBudget: Int?
    if let maximumPyrowaveBytes = arguments.maximumPyrowaveBytes {
        pyrowaveBudget = maximumPyrowaveBytes
    } else if arguments.matchHEVCFrameBudget {
        pyrowaveBudget = try HEVCComparison.matchedFrameByteBudget(
            bitrate: arguments.bitrate,
            frameRateNumerator: loaded.frameRateNumerator,
            frameRateDenominator: loaded.frameRateDenominator
        )
    } else {
        pyrowaveBudget = nil
    }
    let configuration = CodecConfiguration(
        quantizationStep: arguments.quantizationStep,
        maximumEncodedBytes: pyrowaveBudget
    )
    let pyrowave = try PyrowaveBenchmarkRunner.runPyrowave(
        loaded: loaded,
        configuration: configuration,
        outputDirectory: arguments.outputDirectory
    )
    let hevc = try HEVCComparison.runAVKitHEVCComparison(
        referenceFrames: frames,
        workingDirectory: arguments.outputDirectory,
        bitrate: arguments.bitrate,
        frameRateNumerator: loaded.frameRateNumerator,
        frameRateDenominator: loaded.frameRateDenominator
    )

    let report = PyrowaveBenchmarkReport(
        generatedAt: ISO8601DateFormatter().string(from: Date()),
        width: frames[0].width,
        height: frames[0].height,
        frames: frames.count,
        frameRateNumerator: loaded.frameRateNumerator,
        frameRateDenominator: loaded.frameRateDenominator,
        bitrate: arguments.bitrate,
        pyrowaveFrameBudgetBytes: pyrowaveBudget,
        artifacts: PyrowaveBenchmarkArtifacts(),
        pyrowave: pyrowave,
        hevc: hevc,
        comparison: CodecBenchmarkComparison(pyrowave: pyrowave, hevc: hevc)
    )

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let reportData = try encoder.encode(report)
    let reportURL = arguments.outputDirectory.appendingPathComponent(PyrowaveBenchmarkArtifactNames.report)
    try reportData.write(to: reportURL)
    print("Wrote \(reportURL.path)")
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
