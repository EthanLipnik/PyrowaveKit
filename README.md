# PyrowaveKit

PyrowaveKit is an Apple-only Swift and Metal port of PyroWave. The project is intentionally a hard cutover from the original Vulkan/C++ codebase: the build root is Swift Package Manager, the GPU backend is Metal, and benchmark artifacts are written outside git.

The codec remains intra-only and wavelet based. It targets low-latency local video transport where high throughput and predictable frame budgets matter more than broad platform compatibility.

## Requirements

- macOS 14 or newer
- Swift 6 toolchain
- Metal-capable Apple GPU
- Optional: `ffmpeg` with VideoToolbox support for HEVC comparison benchmarks

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

The benchmark executable compares PyrowaveKit against a VideoToolbox HEVC encode/decode path through `ffmpeg`:

```shell
swift run -c release pyrowave-swift-bench --frames 60 --preset 1080p --output .pyrowave-results
```

Outputs are written to `.pyrowave-results/`, which is git ignored. The benchmark report is saved as `.pyrowave-results/benchmark-report.json` alongside the generated reference, Pyrowave, HEVC, and decoded sample artifacts.

Do not treat benchmark numbers as final until the Swift/Metal port is complete. The current benchmark harness exists so quality and performance deltas can be measured repeatedly during the port.
