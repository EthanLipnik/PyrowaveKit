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

The test suite covers the Swift packet format, packet stream decoding, edge-extended source padding plus mirrored wavelet filtering, 4:2:0 and 4:4:4 round trips, rate-control decisions, and Metal parity against internal reference routines when a Metal device is available. Normal codec entry points are Metal texture and CoreVideo pixel-buffer based.

## Benchmark

The benchmark executable compares PyrowaveKit against a multi-frame HEVC encode/decode path using AVKit and AVFoundation:

```shell
swift run -c release pyrowave-swift-bench --frames 60 --preset 6k --output-dir .pyrowave-results
```

Without arguments, the benchmark uses the same 60-frame 6k baseline and writes to `.pyrowave-results/`.
Outputs are written to `.pyrowave-results/`, which is git ignored. The benchmark report is saved as `.pyrowave-results/benchmark-report.json` alongside `reference.y4m`, `pyrowave-sample.pwks`, `pyrowave-decoded.y4m`, `hevc-avkit.mov`, and `hevc-decoded.y4m`.
The report includes raw Pyrowave and HEVC results plus comparison ratios for byte usage, encode speed, decode speed, and weighted PSNR delta.
Timed encode/decode sections exclude input loading, CoreVideo pixel-buffer preparation, planar conversion for quality analysis, Y4M/stream artifact writes, report serialization, and PSNR metric generation. The HEVC timing stops at the AVFoundation pixel-buffer append/read boundary a normal Apple app would use; review artifacts are produced afterward.
Pyrowave is not byte capped in the benchmark; quality is controlled by Pyrowave quantization settings. HEVC uses AVFoundation with `AVVideoQualityKey` capped at `0.8`.

Current guarded results from July 4, 2026 were generated with:

```shell
swift run -c release pyrowave-swift-bench --preset 1080p --frames 3 --output-dir .pyrowave-results/2026-07-04-current-guarded-1080p --require-pyrowave-faster-than-hevc
swift run -c release pyrowave-swift-bench --preset 4k --frames 3 --output-dir .pyrowave-results/2026-07-04-current-guarded-4k --require-pyrowave-faster-than-hevc
swift run -c release pyrowave-swift-bench --preset 6k --frames 3 --output-dir .pyrowave-results/2026-07-04-current-guarded-6k --require-pyrowave-faster-than-hevc
```

| Size | Pyrowave encode | Pyrowave decode | HEVC encode | HEVC decode | Encode speedup | Decode speedup | Pyrowave bytes/frame | HEVC bytes/frame | Py/HEVC bytes | PSNR delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1080p | 2.894 ms/frame | 1.499 ms/frame | 15.331 ms/frame | 16.212 ms/frame | 5.297x | 10.816x | 117,709.3 | 27,192.7 | 4.329x | +1.005 dB |
| 4K | 8.128 ms/frame | 2.873 ms/frame | 23.219 ms/frame | 19.809 ms/frame | 2.857x | 6.894x | 461,210.7 | 94,314.3 | 4.890x | +0.510 dB |
| 6K | 14.019 ms/frame | 5.876 ms/frame | 39.990 ms/frame | 24.844 ms/frame | 2.853x | 4.228x | 1,173,789.3 | 244,573.7 | 4.799x | +0.539 dB |

These runs use the GPU-frame Pyrowave path and keep review artifact export, stream serialization, and metrics generation outside the timed encode/decode sections. Pyrowave is currently faster than the HEVC comparison in this matrix while spending about 4.3x to 4.9x the bytes per frame in the uncapped configuration.
