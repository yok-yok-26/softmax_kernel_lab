#!/usr/bin/env python3
from pathlib import Path
import csv
import math

root = Path(__file__).resolve().parents[1]
roof = root / 'reports' / 'roofline'
roof.mkdir(parents=True, exist_ok=True)
bench_dir = root / 'reports' / 'benchmarks'
points = []
for csv_path in bench_dir.glob('latest_*.csv'):
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            rows = int(row['rows']); cols = int(row['cols'])
            ms = float(row['ms'])
            bytes_rw = rows * cols * 4 * 2
            est_flops = rows * cols * 8
            ai = est_flops / bytes_rw if bytes_rw else 0.0
            perf = est_flops / (ms / 1000.0) / 1e12 if ms > 0 else 0.0
            points.append((row['mode'], ai, perf, ms, float(row['approx_gbps'])))
md = roof / 'softmax_latest.md'
md.write_text('# Softmax Roofline Notes\n\nData source: benchmark CSV plus optional Nsight Compute raw reports. Current FLOP counts are estimates until NCU exported metrics are parsed.\n\n' + '\n'.join(f'- {p[0]}: AI={p[1]:.4f} FLOP/byte, perf={p[2]:.6f} TFLOP/s, duration={p[3]:.6f} ms, approx_mem_bw={p[4]:.3f} GB/s' for p in points) + '\n')
def draw_pil_png(path, points, xlim):
    from PIL import Image, ImageDraw
    w, h = 1000, 640
    margin_l, margin_r, margin_t, margin_b = 90, 40, 50, 80
    img = Image.new('RGB', (w, h), 'white')
    d = ImageDraw.Draw(img)
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b
    peak_bw_gbs = 672.0
    peak_tflops = 31.0
    ymin, ymax = 1e-4, 100.0
    def sx(x):
        x = max(min(x, xlim[1]), xlim[0])
        return margin_l + (math.log10(x) - math.log10(xlim[0])) / (math.log10(xlim[1]) - math.log10(xlim[0])) * plot_w
    def sy(y):
        y = max(min(y, ymax), ymin)
        return margin_t + (math.log10(ymax) - math.log10(y)) / (math.log10(ymax) - math.log10(ymin)) * plot_h
    d.rectangle([margin_l, margin_t, margin_l + plot_w, margin_t + plot_h], outline='black')
    for tick in [0.001, 0.01, 0.1, 1, 10, 100]:
        if xlim[0] <= tick <= xlim[1]:
            x = sx(tick)
            d.line([x, margin_t, x, margin_t + plot_h], fill=(230, 230, 230))
            d.text((x - 12, margin_t + plot_h + 8), str(tick), fill='black')
    for tick in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
        y = sy(tick)
        d.line([margin_l, y, margin_l + plot_w, y], fill=(230, 230, 230))
        d.text((8, y - 7), f'{tick:g}', fill='black')
    xs = [10 ** (math.log10(xlim[0]) + i * (math.log10(xlim[1]/xlim[0]) / 200.0)) for i in range(201)]
    roof_pts = [(sx(x), sy(min(peak_tflops, x * peak_bw_gbs / 1000.0))) for x in xs]
    d.line(roof_pts, fill='black', width=3)
    colors = [(20, 96, 170), (180, 70, 30), (40, 140, 80), (120, 70, 160)]
    if not points:
        points = [('no_data', 0.01, 0.001, 0.0, 0.0)]
    for i, (mode, ai, perf, ms, gbps) in enumerate(points):
        x, y = sx(max(ai, 1e-6)), sy(max(perf, 1e-9))
        c = colors[i % len(colors)]
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=c, outline='black')
        d.text((x + 9, y - 18), f'{mode} {ms:.3f}ms {gbps:.1f}GB/s', fill=c)
    d.text((margin_l, 16), 'Softmax roofline (estimated ceilings)', fill='black')
    d.text((w // 2 - 110, h - 35), 'Arithmetic intensity (FLOP/byte, log scale)', fill='black')
    d.text((12, 18), 'TFLOP/s', fill='black')
    img.save(path)

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    (roof / 'softmax_latest.txt').write_text(f'matplotlib unavailable: {exc}\nFallback PNG generated with PIL. See softmax_latest.md for numeric notes.\n')
    draw_pil_png(roof / 'softmax_latest.png', points, (0.001, 10.0))
    draw_pil_png(roof / 'softmax_global_latest.png', points, (0.001, 100.0))
    print(f'roofline notes: {md}')
    print(f'roofline png: {roof / "softmax_latest.png"}')
    print(f'global roofline png: {roof / "softmax_global_latest.png"}')
    raise SystemExit(0)
if not points:
    points = [('no_data', 0.01, 0.001, 0.0, 0.0)]
peak_bw_gbs = 672.0
peak_tflops = 31.0
for name, xlim in [('softmax_latest.png', (0.001, 10.0)), ('softmax_global_latest.png', (0.001, 100.0))]:
    xs = [10 ** (math.log10(xlim[0]) + i * (math.log10(xlim[1]/xlim[0]) / 200.0)) for i in range(201)]
    ys = [min(peak_tflops, x * peak_bw_gbs / 1000.0) for x in xs]
    plt.figure(figsize=(8, 5))
    plt.loglog(xs, ys, label='estimated roofline', color='black')
    for mode, ai, perf, ms, gbps in points:
        ai = max(ai, 1e-6); perf = max(perf, 1e-9)
        plt.scatter([ai], [perf], s=60)
        plt.annotate(f'{mode}\n{ms:.3f} ms\n{gbps:.1f} GB/s', (ai, perf), textcoords='offset points', xytext=(8, 6), fontsize=8)
    plt.xlabel('Arithmetic intensity (FLOP/byte)')
    plt.ylabel('Achieved performance (TFLOP/s)')
    plt.title('Softmax roofline (estimated until NCU metrics are parsed)')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(roof / name, dpi=160)
    plt.close()
print(f'roofline notes: {md}')
print(f'roofline png: {roof / "softmax_latest.png"}')
print(f'global roofline png: {roof / "softmax_global_latest.png"}')
