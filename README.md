# PyrowaveKit

PyrowaveKit is an Apple-only Swift and Metal port of PyroWave. The project is intentionally a hard cutover from the original Vulkan/C++ codebase: the build root is Swift Package Manager, the GPU backend is Metal, and benchmark artifacts are written outside git.

The codec remains intra-only and wavelet based. It targets low-latency local video transport where high throughput and predictable frame budgets matter more than broad platform compatibility.

## Requirements

- macOS 14 or newer
- Swift 6 toolchain
- Metal-capable Apple GPU
- AVKit/AVFoundation-capable Apple platform runtime for HEVC comparison benchmarks

## Build

```shell
swift build
```

## Test

```shell
swift test
swift test -c release
```

The test suite covers the Swift packet format, packet stream decoding, mirrored wavelet padding, 4:2:0 and 4:4:4 round trips, rate-control decisions, and Metal parity against the CPU reference path when a Metal device is available.

## Benchmark

The benchmark executable compares PyrowaveKit against a multi-frame HEVC encode/decode path using AVKit and AVFoundation:

```shell
swift run -c release pyrowave-swift-bench --frames 60 --preset 6k --output-dir .pyrowave-results
```

Without arguments, the benchmark uses the same 60-frame 6k baseline and writes to `.pyrowave-results/`.
Outputs are written to `.pyrowave-results/`, which is git ignored. The benchmark report is saved as `.pyrowave-results/benchmark-report.json` alongside `reference.y4m`, `pyrowave-sample.pwks`, `pyrowave-decoded.y4m`, `hevc-avkit.mov`, and `hevc-decoded.y4m`.
The report includes raw Pyrowave and HEVC results plus comparison ratios for byte usage, encode speed, decode speed, and weighted PSNR delta.
When Pyrowave is matched to the HEVC bitrate, its per-frame cap uses the same 60 Hz quality-reference rule as Mirage: high-refresh streams above 60 Hz do not halve per-frame quality.

Do not treat benchmark numbers as final until the Swift/Metal port is complete. The current benchmark harness exists so quality and performance deltas can be measured repeatedly during the port.
