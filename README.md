# Softmax Kernel Lab

CUDA learning and benchmarking project for row-wise FP32 softmax on contiguous tensors with logical shape `[rows, cols]`.

The project contains a simple CUDA baseline, user CUDA variants, a CUB/Thrust composed baseline, a cuDNN baseline, a PyTorch framework baseline, correctness tests, sanitizer scripts, benchmark scripts, profiler entrypoints, and public performance display artifacts.

## Contents

- `include/`: public CUDA/C++ interfaces.
- `kernels/`: CUDA implementations and library baseline integrations.
- `reference/`: CPU softmax reference.
- `tests/`: correctness test executable source.
- `benchmark/`: benchmark executable source.
- `scripts/`: build, correctness, benchmark, sanitizer, profiler, and plotting helpers.
- `reports/benchmarks/`: public benchmark CSV outputs.
- `reports/final_overview/`: public summary CSV files and SVG performance figures.
- `reports/trends/`: public sweep trend figures.

## Requirements

- NVIDIA GPU with CUDA support.
- CUDA toolkit 12.8 was used for the archived results.
- CMake 3.24 or newer.
- C++17 and CUDA C++17 compiler support.
- cuDNN 9.19.0 was used for the archived cuDNN baseline.
- Python with PyTorch CUDA support is required only for the `framework_torch` baseline scripts.

Archived measurements were collected on:

- GPU: NVIDIA GeForce RTX 5070
- Driver: 580.159.03
- CUDA toolkit: `/usr/local/cuda-12.8`
- PyTorch baseline: `torch 2.9.0+cu128`

## Build

```bash
./scripts/build_debug.sh
./scripts/build_release.sh
```

The debug build uses `-G -g -lineinfo` and enables `DEBUG_CUDA_SYNC`. The release build uses `-O3 -lineinfo`.

## Verify

Run correctness for one mode:

```bash
./scripts/run_correctness.sh user_v2
```

Run sanitizer checks for one C++/CUDA mode:

```bash
./scripts/run_memcheck.sh user_v2
./scripts/run_racecheck.sh user_v2
./scripts/run_synccheck.sh user_v2
```

The PyTorch framework baseline is correctness-tested but is not used as a Compute Sanitizer target.

## Benchmark

Run a release benchmark for one mode:

```bash
./scripts/run_benchmark.sh user_v2 4096 1024 200 20
```

Timing scope: steady-state GPU operator timing. Device allocation, host input generation, workspace allocation, and host-to-device copy are outside the timed loop. Nsight Compute single-launch reports are profiler evidence and are not used as headline benchmark timing.

Available modes:

- `baseline`
- `library`
- `cudnn`
- `framework_torch`
- `user`
- `user_v1`
- `user_v2`

## Performance Display Artifacts

The repository keeps public display artifacts and summary CSVs, while raw profiler reports, logs, sanitizer outputs, correctness logs, and analysis notes are excluded from Git tracking.

Key public files:

- `reports/final_overview/softmax_best_user_vs_baseline.csv`
- `reports/final_overview/softmax_full_series_heatmap_values.csv`
- `reports/final_overview/softmax_ncu_mode_metrics.csv`
- `reports/final_overview/softmax_stall_top5_by_stage.csv`
- `reports/final_overview/images/*.svg`
- `reports/trends/*.png`
- `reports/benchmarks/*.csv`

## License

Apache License 2.0.
