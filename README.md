# PyrowaveKit

PyrowaveKit is an Apple-only Swift and Metal port of PyroWave. The project is intentionally a hard cutover from the original Vulkan/C++ codebase: the build root is Swift Package Manager, the GPU backend is Metal, and benchmark artifacts are written outside git.

The codec remains intra-only and wavelet based. It targets low-latency local video transport where high throughput and predictable frame budgets matter more than broad platform compatibility.

## Requirements

- macOS 14 or newer
- Swift 6 toolchain
- Metal-capable Apple GPU
- VideoToolbox-capable Apple platform runtime for HEVC comparison benchmarks

## Build

```shell
swift build
```

## Test

```shell
swift test
swift test -c release
```

The test suite covers the Swift packet format, packet stream decoding, edge-extended source padding plus mirrored wavelet filtering, 4:2:0 and 4:4:4 round trips, rate-control decisions, and Metal parity against internal reference routines when a Metal device is available. Normal codec entry points are Metal texture and CoreVideo pixel-buffer based.

## Benchmark

The benchmark compares the Metal GPU-frame Pyrowave path with a Mirage-style realtime HEVC path built directly on VideoToolbox.

```shell
swift run -c release pyrowave-swift-bench --frames 60 --preset 6k --output-dir .pyrowave-results
```

Benchmark outputs are written to `.pyrowave-results/`, which is git ignored. Timed encode/decode excludes input loading, pixel-buffer setup, planar conversion for metrics, artifact writes, report serialization, and PSNR calculation.

Policy:
- Pyrowave is not byte capped.
- HEVC quality is capped at `0.8`.
- HEVC uses realtime VideoToolbox settings: no frame reordering, low delay, Mirage-style QP bounds, `AverageBitRate`, and `DataRateLimits`.

Latest 60-frame results from July 4, 2026:

```shell
swift run -c release pyrowave-swift-bench --preset 1080p --frames 60 --output-dir .pyrowave-results/2026-07-04-second-pass-final-hevc-1080p-60
swift run -c release pyrowave-swift-bench --preset 4k --frames 60 --output-dir .pyrowave-results/2026-07-04-second-pass-final-hevc-4k-60
swift run -c release pyrowave-swift-bench --preset 6k --frames 60 --output-dir .pyrowave-results/2026-07-04-second-pass-final-hevc-6k-60
```

Performance:

| Size | Py encode | Py decode | HEVC encode | HEVC decode | Py encode speedup | Py decode speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1080p | 1.067 ms | 0.937 ms | 5.865 ms | 0.398 ms | 5.497x | 0.425x |
| 4K | 5.017 ms | 1.533 ms | 19.602 ms | 1.388 ms | 3.907x | 0.905x |
| 6K | 8.211 ms | 3.019 ms | 44.543 ms | 4.107 ms | 5.425x | 1.360x |

Size and quality:

| Size | Py bytes/frame | HEVC bytes/frame | Byte ratio | PSNR delta |
| --- | ---: | ---: | ---: | ---: |
| 1080p | 115,066.3 | 29,302.0 | 3.927x | +1.449 dB |
| 4K | 450,154.7 | 111,520.8 | 4.037x | +0.237 dB |
| 6K | 1,144,178.2 | 164,876.0 | 6.940x | +6.263 dB |

Pyrowave currently encodes faster at every tested size. Decode is still behind HEVC at 1080p and slightly behind at 4K, but is faster at 6K. Speedups above `1.0x` mean Pyrowave is faster than HEVC.
