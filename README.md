# PyrowaveKit

PyrowaveKit is an Apple-only Swift and Metal port of PyroWave for low-latency local video transport.

The port is intentionally native: Swift Package Manager, Metal compute, CoreVideo pixel buffers, VideoToolbox HEVC benchmarks, and no CPU fallback path.

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

Coverage includes packet decoding, wavelet padding/filtering, 4:2:0 and 4:4:4 round trips, rate control, and Metal parity checks when a Metal device is available.

## Benchmarks

Pyrowave is compared against a Mirage-style realtime HEVC path built on VideoToolbox.

```shell
swift run -c release pyrowave-swift-bench --frames 60 --preset 6k --output-dir .pyrowave-results
```

Reports are written to `.pyrowave-results/`, which is git ignored.

Rules:
- Pyrowave has no byte cap.
- HEVC quality is capped at `0.8`.
- HEVC uses realtime VideoToolbox settings: no frame reordering, low delay, Mirage-style QP bounds, `AverageBitRate`, and `DataRateLimits`.
- Timings exclude loading, pixel-buffer setup, metric conversions, artifact writes, report serialization, and PSNR calculation.

Latest 60-frame results from July 4, 2026:

```shell
swift run -c release pyrowave-swift-bench --preset 1080p --frames 60 --output-dir .pyrowave-results/2026-07-04-second-pass-final-hevc-1080p-60
swift run -c release pyrowave-swift-bench --preset 4k --frames 60 --output-dir .pyrowave-results/2026-07-04-second-pass-final-hevc-4k-60
swift run -c release pyrowave-swift-bench --preset 6k --frames 60 --output-dir .pyrowave-results/2026-07-04-second-pass-final-hevc-6k-60
```

### Performance

| Size | Py encode | Py decode | HEVC encode | HEVC decode | Py encode speedup | Py decode speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1080p | 1.067 ms | 0.937 ms | 5.865 ms | 0.398 ms | 5.497x | 0.425x |
| 4K | 5.017 ms | 1.533 ms | 19.602 ms | 1.388 ms | 3.907x | 0.905x |
| 6K | 8.211 ms | 3.019 ms | 44.543 ms | 4.107 ms | 5.425x | 1.360x |

### Size and Quality

| Size | Py bytes/frame | HEVC bytes/frame | Byte ratio | PSNR delta |
| --- | ---: | ---: | ---: | ---: |
| 1080p | 115,066.3 | 29,302.0 | 3.927x | +1.449 dB |
| 4K | 450,154.7 | 111,520.8 | 4.037x | +0.237 dB |
| 6K | 1,144,178.2 | 164,876.0 | 6.940x | +6.263 dB |

Speedups above `1.0x` mean Pyrowave is faster than HEVC. Current results: faster encode at every size, faster decode at 6K, slower decode at 1080p and 4K.
