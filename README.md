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

The benchmark executable compares PyrowaveKit against a multi-frame Mirage-style realtime HEVC encode/decode path using VideoToolbox directly:

```shell
swift run -c release pyrowave-swift-bench --frames 60 --preset 6k --output-dir .pyrowave-results
```

Without arguments, the benchmark uses the same 60-frame 6k baseline and writes to `.pyrowave-results/`.
Outputs are written to `.pyrowave-results/`, which is git ignored. The benchmark report is saved as `.pyrowave-results/benchmark-report.json` alongside `reference.y4m`, `pyrowave-sample.pwks`, `pyrowave-decoded.y4m`, `hevc-videotoolbox.mirage-hevc`, and `hevc-decoded.y4m`.
The report includes raw Pyrowave and HEVC results plus comparison ratios for byte usage, encode speed, decode speed, and weighted PSNR delta.
Timed encode/decode sections exclude input loading, CoreVideo pixel-buffer preparation, planar conversion for quality analysis, Y4M/stream artifact writes, report serialization, and PSNR metric generation. The HEVC timing uses the same VideoToolbox boundary a realtime Mirage-like app would use: `VTCompressionSessionEncodeFrame` through output callbacks, then `VTDecompressionSessionDecodeFrame` through output callbacks.
Pyrowave is not byte capped in the benchmark; quality is controlled by Pyrowave quantization settings. HEVC uses a direct VideoToolbox session with Mirage-style realtime settings, `Quality` capped at `0.8`, Mirage-style QP bounds, `AverageBitRate`, and `DataRateLimits`.

Current Mirage-style HEVC results from July 4, 2026 were generated with:

```shell
swift run -c release pyrowave-swift-bench --preset 1080p --frames 60 --output-dir .pyrowave-results/2026-07-04-second-pass-final-hevc-1080p-60
swift run -c release pyrowave-swift-bench --preset 4k --frames 60 --output-dir .pyrowave-results/2026-07-04-second-pass-final-hevc-4k-60
swift run -c release pyrowave-swift-bench --preset 6k --frames 60 --output-dir .pyrowave-results/2026-07-04-second-pass-final-hevc-6k-60
```

| Size | Pyrowave encode | Pyrowave decode | HEVC encode | HEVC decode | Encode speedup | Decode speedup | Pyrowave bytes/frame | HEVC bytes/frame | Py/HEVC bytes | PSNR delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1080p | 1.067 ms/frame | 0.937 ms/frame | 5.865 ms/frame | 0.398 ms/frame | 5.497x | 0.425x | 115,066.3 | 29,302.0 | 3.927x | +1.449 dB |
| 4K | 5.017 ms/frame | 1.533 ms/frame | 19.602 ms/frame | 1.388 ms/frame | 3.907x | 0.905x | 450,154.7 | 111,520.8 | 4.037x | +0.237 dB |
| 6K | 8.211 ms/frame | 3.019 ms/frame | 44.543 ms/frame | 4.107 ms/frame | 5.425x | 1.360x | 1,144,178.2 | 164,876.0 | 6.940x | +6.263 dB |

These runs use the GPU-frame Pyrowave path and the Mirage-style direct VideoToolbox HEVC path. Pyrowave encodes faster across the matrix. Decode remains slower than HEVC at 1080p and 4K, but 4K is now close to parity and the 60-frame 6K run decodes faster than HEVC after Metal DWT parity dispatch plus tiled inverse-DWT copyback for aligned higher levels. Decode speedup below `1.0x` means Pyrowave is slower to decode; above `1.0x` means Pyrowave is faster. Pyrowave spends about 3.9x to 6.9x the HEVC bytes per frame in this uncapped configuration.
