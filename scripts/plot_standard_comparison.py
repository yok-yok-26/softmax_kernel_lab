#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "final_overview"
OUT.mkdir(parents=True, exist_ok=True)
BENCH = ROOT / "reports" / "benchmarks"
TRENDS = ROOT / "reports" / "trends"
NCU = ROOT / "reports" / "ncu"

MODE_ORDER = ["framework_torch", "cudnn", "baseline", "library", "user", "user_v1", "user_v2"]
USER_MODES = ["user", "user_v1", "user_v2"]
BASELINE_CANDIDATES = ["framework_torch", "cudnn", "library", "baseline"]
MODE_IDEA = {
    "framework_torch": "L1 framework torch.softmax",
    "cudnn": "L1 cuDNN softmax",
    "baseline": "custom simple CUDA baseline",
    "library": "L2 CUB/Thrust composed",
    "user": "user kernel",
    "user_v1": "user block-loop v1",
    "user_v2": "user block-loop v2",
}
COLORS = {
    "framework_torch": "#1f77b4",
    "cudnn": "#2ca02c",
    "baseline": "#7f7f7f",
    "library": "#9467bd",
    "user": "#d62728",
    "user_v1": "#ff7f0e",
    "user_v2": "#17becf",
}

def mode_sort_key(mode: str) -> Tuple[int, str]:
    return (MODE_ORDER.index(mode) if mode in MODE_ORDER else 999, mode)

def read_latest_benchmark() -> pd.DataFrame:
    frames = []
    for p in sorted(BENCH.glob("latest_*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if not df.empty:
            df["source"] = str(p.relative_to(ROOT))
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["mode", "rows", "cols", "iters", "ms", "approx_gbps"])
    df = pd.concat(frames, ignore_index=True)
    for c in ["rows", "cols", "iters", "ms", "approx_gbps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["mode", "rows", "cols", "ms", "approx_gbps"])

def read_sweep() -> pd.DataFrame:
    p = TRENDS / "softmax_param_sweep_raw.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for c in ["rows", "cols", "iters", "ms", "approx_gbps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["mode", "rows", "cols", "ms", "approx_gbps"])

def compact_config(row) -> str:
    return f"{int(row.rows)}x{int(row.cols)}"

def pick_baseline(group: pd.DataFrame) -> pd.Series | None:
    for mode in BASELINE_CANDIDATES:
        sub = group[group["mode"] == mode]
        if not sub.empty:
            return sub.loc[sub["ms"].idxmin()]
    return None

def pick_best_user(group: pd.DataFrame) -> pd.Series | None:
    sub = group[group["mode"].isin(USER_MODES)]
    if sub.empty:
        return None
    return sub.loc[sub["ms"].idxmin()]

def write_best_user_tables(raw: pd.DataFrame):
    if raw.empty:
        return pd.DataFrame()
    rows = []
    for (r, c), g in raw.groupby(["rows", "cols"], sort=True):
        b = pick_baseline(g)
        u = pick_best_user(g)
        if b is None or u is None:
            continue
        rows.append({
            "config": f"{int(r)}x{int(c)}",
            "rows": int(r),
            "cols": int(c),
            "baseline_mode": b["mode"],
            "baseline_ms": float(b["ms"]),
            "baseline_gbps": float(b["approx_gbps"]),
            "best_user_mode": u["mode"],
            "best_user_ms": float(u["ms"]),
            "best_user_gbps": float(u["approx_gbps"]),
            "latency_speedup_user_vs_baseline": float(b["ms"]) / float(u["ms"]),
            "throughput_ratio_user_vs_baseline": float(u["approx_gbps"]) / float(b["approx_gbps"]),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(OUT / "softmax_best_user_vs_baseline.csv", index=False)
    return out

def plot_best_user_vs_baseline(best: pd.DataFrame):
    if best.empty:
        return
    best = best.copy()
    best["label"] = best["config"]
    x = np.arange(len(best))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(11, len(best) * 0.9), 6))
    ax.bar(x - width / 2, best["baseline_gbps"], width, label="fair baseline", color="#65707a")
    ax.bar(x + width / 2, best["best_user_gbps"], width, label="best user", color="#17becf")
    for i, row in best.iterrows():
        ax.text(i - width / 2, row["baseline_gbps"], str(row["baseline_mode"]), ha="center", va="bottom", fontsize=8, rotation=80)
        ax.text(i + width / 2, row["best_user_gbps"], str(row["best_user_mode"]), ha="center", va="bottom", fontsize=8, rotation=80)
    ax.set_xticks(x)
    ax.set_xticklabels(best["label"], rotation=45, ha="right")
    ax.set_ylabel("Release benchmark throughput (GB/s)")
    ax.set_title("Softmax best user kernel vs fair baseline")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(OUT / "softmax_best_user_vs_baseline.png", dpi=180)
    fig.savefig(OUT / "softmax_best_user_vs_baseline.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(11, len(best) * 0.85), 4.8))
    vals = best["throughput_ratio_user_vs_baseline"]
    colors = ["#2ca02c" if v > 1.05 else "#ffbf00" if v >= 0.95 else "#d62728" for v in vals]
    ax.bar(x, vals, color=colors)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(best["label"], rotation=45, ha="right")
    ax.set_ylabel("Best user / baseline throughput ratio")
    ax.set_title("Softmax best user ratio against fair baseline")
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(OUT / "softmax_best_user_ratio.png", dpi=180)
    fig.savefig(OUT / "softmax_best_user_ratio.svg")
    plt.close(fig)

def plot_full_series_heatmap(raw: pd.DataFrame):
    if raw.empty:
        return
    df = raw.copy()
    df["config"] = df.apply(compact_config, axis=1)
    order = sorted(df["config"].unique(), key=lambda s: tuple(map(int, s.split("x"))))
    modes = sorted(df["mode"].unique(), key=mode_sort_key)
    piv = df.pivot_table(index="config", columns="mode", values="approx_gbps", aggfunc="max").reindex(index=order, columns=modes)
    if piv.empty:
        return
    norm = piv.div(piv.max(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(max(9, len(modes) * 1.25), max(5.5, len(order) * 0.42)))
    im = ax.imshow(norm.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels(modes, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.set_title("Softmax full-series heatmap, row-normalized throughput")
    for i, cfg in enumerate(order):
        row = piv.loc[cfg]
        winner = row.idxmax() if row.notna().any() else None
        for j, mode in enumerate(modes):
            val = row.get(mode)
            if pd.notna(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color="black")
                if mode == winner:
                    ax.add_patch(plt.Rectangle((j - 0.49, i - 0.49), 0.98, 0.98, fill=False, edgecolor="#d62728", linewidth=2))
    fig.colorbar(im, ax=ax, label="row-normalized throughput")
    ax.set_xlabel("mode")
    ax.set_ylabel("rows x cols")
    fig.tight_layout()
    fig.savefig(OUT / "softmax_full_series_heatmap.png", dpi=180)
    fig.savefig(OUT / "softmax_full_series_heatmap.svg")
    plt.close(fig)
    piv.to_csv(OUT / "softmax_full_series_heatmap_values.csv")

METRIC_LABELS = [
    "Duration", "Compute (SM) Throughput", "Memory Throughput", "DRAM Throughput",
    "L1/TEX Cache Throughput", "L2 Cache Throughput", "Issue Slots Busy",
    "Achieved Occupancy", "Eligible Warps Per Scheduler", "Warp Cycles Per Issued Instruction",
    "Registers Per Thread", "Static Shared Memory Per Block", "Dynamic Shared Memory Per Block",
    "Executed Instructions", "Executed Ipc Active", "L1/TEX Hit Rate", "L2 Hit Rate",
]

def parse_number(s: str) -> float | None:
    s = s.strip().replace(",", "")
    if s.lower() in {"n/a", "nan", ""}:
        return None
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else None

def short_kernel_name(name: str) -> str:
    if "softmax_warp_forward" in name:
        return "torch softmax_warp_forward"
    if "softmax_fw" in name or "softmax" in name and "cudnn" in name.lower():
        return "cuDNN softmax"
    if "DeviceSegmentedReduce" in name and "Max" in name:
        return "CUB row max"
    if "DeviceSegmentedReduce" in name and "plus" in name:
        return "CUB row sum"
    if "ExpWithRowMax" in name:
        return "Thrust exp"
    if "NormalizeWithRowSum" in name:
        return "Thrust normalize"
    if "baseline" in name:
        return "baseline kernel"
    if "softmax_block_loop" in name:
        return "block_loop"
    if len(name) > 70:
        return name[:67] + "..."
    return name

def parse_ncu_details(mode: str) -> List[Dict[str, object]]:
    p = NCU / f"latest_{mode}_details.txt"
    if not p.exists():
        return []
    rows = []
    current = None
    for line in p.read_text(errors="replace").splitlines():
        if line.startswith("  ") and "(" in line and "Section:" not in line and "Metric" not in line and not line.strip().startswith(("-", "INF", "OPT")):
            stripped = line.strip()
            if not stripped.startswith(("Section", "Metric")) and len(stripped) > 8:
                current = {"mode": mode, "kernel": stripped, "stage": short_kernel_name(stripped)}
                rows.append(current)
            continue
        if current is None:
            continue
        for label in METRIC_LABELS:
            if label in line:
                val = parse_number(line.split(label, 1)[1])
                if val is not None:
                    current[label] = val
    return rows

def collect_ncu() -> Tuple[pd.DataFrame, pd.DataFrame]:
    stages = []
    for mode in MODE_ORDER:
        stages.extend(parse_ncu_details(mode))
    stage_df = pd.DataFrame(stages)
    if stage_df.empty:
        return stage_df, pd.DataFrame()
    stage_df.to_csv(OUT / "softmax_ncu_stage_metrics.csv", index=False)
    mode_rows = []
    for mode, g in stage_df.groupby("mode", sort=False):
        row = {"mode": mode, "idea": MODE_IDEA.get(mode, "")}
        durations = pd.to_numeric(g.get("Duration"), errors="coerce") if "Duration" in g else pd.Series(dtype=float)
        weights = durations.fillna(0)
        row["Duration"] = float(weights.sum()) if not weights.empty else np.nan
        for label in METRIC_LABELS:
            if label == "Duration" or label not in g:
                continue
            vals = pd.to_numeric(g[label], errors="coerce")
            valid = vals.dropna()
            if valid.empty:
                continue
            if not weights.empty and weights.sum() > 0 and label not in {"Registers Per Thread", "Static Shared Memory Per Block", "Dynamic Shared Memory Per Block"}:
                aligned = vals.fillna(valid.mean())
                row[label] = float((aligned * weights).sum() / weights.sum())
            elif label in {"Registers Per Thread", "Static Shared Memory Per Block", "Dynamic Shared Memory Per Block"}:
                row[label] = float(valid.max())
            else:
                row[label] = float(valid.mean())
        mode_rows.append(row)
    mode_df = pd.DataFrame(mode_rows).sort_values("mode", key=lambda s: s.map(lambda x: mode_sort_key(x)[0]))
    mode_df.to_csv(OUT / "softmax_ncu_mode_metrics.csv", index=False)
    return stage_df, mode_df

def plot_ncu_by_algorithm(mode_df: pd.DataFrame):
    metrics = [m for m in ["Duration", "Compute (SM) Throughput", "Memory Throughput", "DRAM Throughput", "L1/TEX Cache Throughput", "L2 Cache Throughput", "Issue Slots Busy", "Achieved Occupancy", "Eligible Warps Per Scheduler", "Warp Cycles Per Issued Instruction", "Registers Per Thread", "Executed Instructions", "Executed Ipc Active", "L1/TEX Hit Rate", "L2 Hit Rate"] if m in mode_df]
    if mode_df.empty or not metrics:
        return
    modes = list(mode_df["mode"])
    fig_h = max(8, len(modes) * len(metrics) * 0.18 + 2.5)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    y = 0
    yticks, ylabels = [], []
    metric_index = {m: i + 1 for i, m in enumerate(metrics)}
    for mode in modes:
        sub = mode_df[mode_df["mode"] == mode].iloc[0]
        base = COLORS.get(mode, "#555555")
        yticks.append(y + len(metrics) / 2 - 0.5)
        ylabels.append(mode)
        for k, metric in enumerate(metrics):
            val = sub.get(metric, np.nan)
            maxv = pd.to_numeric(mode_df[metric], errors="coerce").max()
            norm = float(val) / float(maxv) if pd.notna(val) and maxv and maxv > 0 else 0
            ax.barh(y + k, norm, color=base, alpha=0.35 + 0.55 * (k + 1) / len(metrics), height=0.72)
            ax.text(-0.025, y + k, str(metric_index[metric]), va="center", ha="right", fontsize=7)
            raw = "" if pd.isna(val) else f"{val:.3g}"
            ax.text(1.015, y + k, raw, va="center", ha="left", fontsize=7)
        y += len(metrics) + 1
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlim(-0.08, 1.22)
    ax.set_xlabel("Normalized within each metric, raw values at right")
    ax.set_title("Softmax NCU metrics grouped by algorithm")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    legend_text = "  ".join(f"{i}: {m}" for m, i in metric_index.items())
    fig.text(0.02, 0.985, legend_text, ha="left", va="top", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "softmax_ncu_metrics_by_algorithm.png", dpi=180)
    fig.savefig(OUT / "softmax_ncu_metrics_by_algorithm.svg")
    plt.close(fig)

def plot_ncu_by_metric(mode_df: pd.DataFrame):
    metrics = [m for m in ["Duration", "Compute (SM) Throughput", "Memory Throughput", "DRAM Throughput", "Achieved Occupancy", "Eligible Warps Per Scheduler", "Warp Cycles Per Issued Instruction", "Registers Per Thread", "Executed Instructions", "Executed Ipc Active", "L2 Hit Rate"] if m in mode_df]
    if mode_df.empty or not metrics:
        return
    modes = list(mode_df["mode"])
    fig_h = max(8, len(metrics) * len(modes) * 0.22 + 3.0)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    y = 0
    yticks, ylabels = [], []
    top_table = " | ".join(f"{m}: {MODE_IDEA.get(m, )}" for m in modes)
    for metric in metrics:
        vals = pd.to_numeric(mode_df[metric], errors="coerce")
        maxv = vals.max()
        yticks.append(y + len(modes) / 2 - 0.5)
        ylabels.append(metric)
        for k, mode in enumerate(modes):
            val = mode_df[mode_df["mode"] == mode].iloc[0].get(metric, np.nan)
            norm = float(val) / float(maxv) if pd.notna(val) and maxv and maxv > 0 else 0
            ax.barh(y + k, norm, color=COLORS.get(mode, "#555"), height=0.72)
            ax.text(-0.025, y + k, mode, va="center", ha="right", fontsize=7)
            raw = "" if pd.isna(val) else f"{val:.3g}"
            ax.text(1.015, y + k, raw, va="center", ha="left", fontsize=7)
        y += len(modes) + 1
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlim(-0.22, 1.22)
    ax.set_xlabel("Normalized within each metric, raw values at right")
    ax.set_title("Softmax NCU metrics grouped by metric")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.text(0.02, 0.985, top_table, ha="left", va="top", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "softmax_ncu_metrics_by_metric.png", dpi=180)
    fig.savefig(OUT / "softmax_ncu_metrics_by_metric.svg")
    plt.close(fig)

def read_stalls() -> pd.DataFrame:
    frames = []
    for p in sorted(NCU.glob("stall_*.csv")):
        mode = p.stem.replace("stall_", "")
        try:
            df = pd.read_csv(p, comment="=")
        except Exception:
            # Files may start with profiler log lines before the CSV header.
            lines = p.read_text(errors="replace").splitlines()
            start = next((i for i,l in enumerate(lines) if l.startswith(chr(34) + "ID" + chr(34))), None)
            if start is None:
                continue
            import io
            df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
        if df.empty or "Metric Name" not in df:
            continue
        df["mode"] = mode
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["Metric Name"].astype(str).str.contains("warp_issue_stalled")].copy()
    df["stall_reason"] = df["Metric Name"].astype(str).str.replace("smsp__warp_issue_stalled_", "", regex=False).str.replace("_per_warp_active.pct", "", regex=False)
    df["value_pct"] = pd.to_numeric(df["Metric Value"], errors="coerce")
    df = df.dropna(subset=["value_pct"])
    df["stage"] = df["Kernel Name"].map(short_kernel_name) if "Kernel Name" in df else "kernel"
    df.to_csv(OUT / "softmax_stall_all_metrics.csv", index=False)
    top = df.sort_values("value_pct", ascending=False).groupby(["mode", "stage"], sort=False).head(5)
    top.to_csv(OUT / "softmax_stall_top5_by_stage.csv", index=False)
    return top

def plot_stalls(top: pd.DataFrame):
    if top.empty:
        return
    # Keep the most representative stage per mode for the visual, preserve every stage in CSV.
    picks = []
    for mode, g in top.groupby("mode", sort=False):
        stage_order = g.groupby("stage")["value_pct"].max().sort_values(ascending=False)
        stage = stage_order.index[0]
        picks.append(g[g["stage"] == stage])
    vis = pd.concat(picks, ignore_index=True)
    modes = sorted(vis["mode"].unique(), key=mode_sort_key)
    fig, ax = plt.subplots(figsize=(14, max(6, len(modes) * 1.2)))
    y = 0
    yticks, ylabels = [], []
    for mode in modes:
        g = vis[vis["mode"] == mode].sort_values("value_pct", ascending=True)
        maxv = g["value_pct"].max() if not g.empty else 1
        yticks.append(y + len(g) / 2 - 0.5)
        stage_label = g["stage"].iloc[0] if not g.empty else ""
        ylabels.append(f"{mode}\n{stage_label}")
        for k, (_, row) in enumerate(g.iterrows()):
            norm = row["value_pct"] / maxv if maxv else 0
            ax.barh(y + k, norm, color=COLORS.get(mode, "#555"), alpha=0.35 + 0.55 * (k + 1) / max(1, len(g)), height=0.72)
            ax.text(0.01, y + k, row["stall_reason"], va="center", ha="left", fontsize=8)
            stall_pct = row["value_pct"]
            ax.text(norm + 0.015, y + k, f"{stall_pct:.2f}%", va="center", ha="left", fontsize=8)
        y += len(g) + 1
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Normalized within each algorithm/stage; labels show real stall percentage")
    ax.set_title("Softmax top-5 warp stall reasons")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUT / "softmax_stall_reasons_top5.png", dpi=180)
    fig.savefig(OUT / "softmax_stall_reasons_top5.svg")
    plt.close(fig)

def main():
    latest = read_latest_benchmark()
    sweep = read_sweep()
    raw_for_overview = sweep if not sweep.empty else latest
    best = write_best_user_tables(raw_for_overview)
    plot_best_user_vs_baseline(best)
    plot_full_series_heatmap(raw_for_overview)
    stage_df, mode_df = collect_ncu()
    plot_ncu_by_algorithm(mode_df)
    plot_ncu_by_metric(mode_df)
    stalls = read_stalls()
    plot_stalls(stalls)
    produced = sorted(p.name for p in OUT.iterdir() if p.is_file())
    (OUT / "README.md").write_text("# Softmax Standard Comparison Figures\n\n" + "\n".join(f"- `{p}`" for p in produced) + "\n")
    print("\n".join(str(OUT / p) for p in produced))

if __name__ == "__main__":
    main()
