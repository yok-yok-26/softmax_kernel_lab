#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import torch


def make_input(rows, cols, pattern, seed):
    torch.manual_seed(seed)
    if pattern == "zeros":
        return torch.zeros((rows, cols), device="cuda", dtype=torch.float32)
    if pattern == "ones":
        return torch.ones((rows, cols), device="cuda", dtype=torch.float32)
    if pattern == "negative":
        idx = torch.arange(rows * cols, device="cuda", dtype=torch.float32).reshape(rows, cols)
        return -1.0 - torch.remainder(idx, 17.0) * 0.25
    if pattern == "alternating":
        idx = torch.arange(rows * cols, device="cuda").reshape(rows, cols)
        return torch.where(idx % 2 == 0, torch.tensor(3.0, device="cuda"), torch.tensor(-3.0, device="cuda"))
    if pattern == "impulse":
        x = torch.full((rows, cols), -12.0, device="cuda", dtype=torch.float32)
        x[:, cols // 2] = 12.0
        return x
    if pattern == "wide":
        return torch.empty((rows, cols), device="cuda", dtype=torch.float32).uniform_(-80.0, 80.0)
    return torch.empty((rows, cols), device="cuda", dtype=torch.float32).uniform_(-2.0, 2.0)


def correctness():
    Path("reports/correctness").mkdir(parents=True, exist_ok=True)
    log_path = Path("reports/correctness/latest_framework_torch.log")
    cols_cases = [1, 2, 7, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4097]
    rows_cases = [1, 2, 5]
    patterns = ["zeros", "ones", "negative", "alternating", "impulse", "random", "wide"]
    failures = 0
    with log_path.open("w") as log:
        log.write("mode=framework_torch dtype=float32 layout=contiguous shape=[rows,cols] axis=last\n")
        for rows in rows_cases:
            for cols in cols_cases:
                for pattern in patterns:
                    x = make_input(rows, cols, pattern, rows * 100003 + cols)
                    y = torch.softmax(x, dim=-1)
                    ref = torch.softmax(x.double(), dim=-1).float()
                    diff = (y - ref).abs()
                    rel = diff / ref.abs().clamp_min(1e-12)
                    max_abs = float(diff.max().item())
                    max_rel = float(rel.max().item())
                    bad = int(((diff > 2e-5) & (rel > 2e-5)).sum().item())
                    if bad:
                        failures += 1
                        worst = int(diff.reshape(-1).argmax().item())
                        log.write(f"FAIL rows={rows} cols={cols} pattern={pattern} bad_count={bad} max_abs={max_abs} max_rel={max_rel} worst_idx={worst}\n")
                    else:
                        log.write(f"PASS rows={rows} cols={cols} pattern={pattern} max_abs={max_abs} max_rel={max_rel}\n")
        log.write(f"summary failures={failures} skipped=0\n")
    print(f"correctness log: {log_path}")
    return 0 if failures == 0 else 1


def benchmark(rows, cols, iters, warmup, single_launch):
    if single_launch:
        iters = 1
        warmup = 0
    Path("reports/benchmark").mkdir(parents=True, exist_ok=True)
    Path("reports/benchmarks").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260527)
    x = torch.empty((rows, cols), device="cuda", dtype=torch.float32).uniform_(-8.0, 8.0)
    for _ in range(warmup):
        torch.softmax(x, dim=-1)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    if single_launch:
        torch.cuda.cudart().cudaProfilerStart()
    start.record()
    for _ in range(iters):
        y = torch.softmax(x, dim=-1)
    stop.record()
    torch.cuda.synchronize()
    if single_launch:
        torch.cuda.cudart().cudaProfilerStop()
    ms = start.elapsed_time(stop) / iters
    bytes_ = rows * cols * 4
    gbps = (bytes_ * 2.0 / 1e9) / (ms / 1000.0)
    line = f"mode=framework_torch rows={rows} cols={cols} iters={iters} ms={ms:.7g} approx_gbps={gbps:.7g}"
    if not single_launch:
        Path("reports/benchmark/latest_framework_torch.log").write_text(line + "\nbenchmark csv: reports/benchmarks/latest_framework_torch.csv\n")
        csv_path = Path("reports/benchmarks/latest_framework_torch.csv")
        new = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["mode", "rows", "cols", "iters", "ms", "approx_gbps"])
            w.writerow(["framework_torch", rows, cols, iters, ms, gbps])
    print(line)
    if single_launch:
        print("single-launch profiling run: benchmark CSV not updated")
    else:
        print("benchmark csv: reports/benchmarks/latest_framework_torch.csv")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["correctness", "benchmark"])
    parser.add_argument("rows", nargs="?", type=int, default=4096)
    parser.add_argument("cols", nargs="?", type=int, default=1024)
    parser.add_argument("iters", nargs="?", type=int, default=100)
    parser.add_argument("warmup", nargs="?", type=int, default=10)
    parser.add_argument("--single-launch", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available")
    if args.command == "correctness":
        raise SystemExit(correctness())
    raise SystemExit(benchmark(args.rows, args.cols, args.iters, args.warmup, args.single_launch))


if __name__ == "__main__":
    main()
