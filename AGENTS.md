# Agent Instructions

## Project Direction

PyrowaveKit is an Apple-only Swift and Metal port of PyroWave. Treat the rewrite as a native Apple codec project, not as a compatibility layer for the old Vulkan/C++ implementation.

- Keep the package Swift Package Manager based.
- Keep codec entry points centered on Metal textures and CoreVideo pixel buffers.
- Do not add CPU fallback paths.
- Do not preserve legacy compatibility unless the user explicitly asks for it.
- Convert any old Vulkan-style GPU work, including sparse block application, into Metal compute kernels or Metal-backed Swift orchestration.

## Performance Priorities

Performance work should focus on real app behavior:

- Avoid benchmark-only shortcuts that a normal app could not use.
- Do not include input loading, artifact writing, report serialization, PSNR calculation, or metric pixel conversion in encode/decode timings.
- Keep Pyrowave uncapped by byte size or quality ceiling unless a specific test requests a limit.
- Compare HEVC using the strongest realtime VideoToolbox path available in this repo, aligned with the Mirage-style benchmark behavior.
- HEVC quality should remain capped at `0.8`.
- HEVC should use realtime settings such as no frame reordering, low delay, QP bounds, `AverageBitRate`, and `DataRateLimits` where supported.

## Benchmarks

Benchmark output belongs under `.pyrowave-results/`, which is git ignored.

Useful commands:

```shell
swift run -c release pyrowave-swift-bench --preset 1080p --frames 60 --output-dir .pyrowave-results
swift run -c release pyrowave-swift-bench --preset 4k --frames 60 --output-dir .pyrowave-results
swift run -c release pyrowave-swift-bench --preset 6k --frames 60 --output-dir .pyrowave-results
```

When changing codec, Metal, HEVC comparison, or benchmark code, run at least:

```shell
swift test -c release
```

If the only remaining work is validation, run the benchmark and save the report under `.pyrowave-results/`.

## Code Guidelines

- Prefer existing local types and patterns in `Sources/PyrowaveKit`.
- Keep GPU work explicit and inspectable in `Sources/PyrowaveKit/Metal/PyrowaveKernels.metal`.
- Avoid broad refactors while optimizing; isolate one performance hypothesis at a time.
- Keep generated benchmark reports out of git.
- Update `README.md` when benchmark policy or headline results change.

## Validation Notes

Before reporting performance changes, compare:

- Pyrowave encode ms/frame
- Pyrowave decode ms/frame
- HEVC encode ms/frame
- HEVC decode ms/frame
- bytes/frame
- PSNR delta

Speedups above `1.0x` mean Pyrowave is faster than HEVC.
