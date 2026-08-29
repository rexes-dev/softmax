#!/usr/bin/env python3
"""Plot safe vs online timings from softmax_bench output.

    ./build/softmax_bench > results.txt
    python3 plot_bench.py results.txt -o bench.png
    ./build/softmax_bench | python3 plot_bench.py -o bench.png

One column of panels per "Sweep:" block: time per launch on top (log-log),
safe / online speedup below. Rows of any other type (naive) are ignored.
"""

import argparse
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

COLOR = {"safe": "#2a78d6", "online": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e4e3df"
MARK = dict(marker="o", markersize=6, markeredgecolor="white",
            markeredgewidth=1.5, linewidth=2, clip_on=False)


def parse(text):
    """-> [(title, xlabel, {type: {x: us}})], x = whichever of V / batch varies."""
    sweeps = []
    for line in text.splitlines():
        tok = line.split()
        if line.lstrip().startswith("Sweep:"):
            sweeps.append((line.split(":", 1)[1].strip(), defaultdict(list)))
        elif sweeps and len(tok) >= 4 and tok[2] in COLOR:
            sweeps[-1][1][tok[2]].append((int(tok[0]), int(tok[1]), float(tok[3])))

    out = []
    for title, rows in sweeps:
        flat = [r for rs in rows.values() for r in rs]
        idx = 0 if len({r[0] for r in flat}) > 1 else 1  # V varies, else batch
        out.append((title, ["V", "batch_size"][idx],
                    {t: {r[idx]: r[2] for r in rs} for t, rs in rows.items()}))
    return out


def fmt(x):
    return f"{x / 1000:.0f}K" if x >= 10_000 else f"{x:.0f}"


def setup(ax, xs):
    """Log x with a tick at every measured value; labels stagger onto a
    second line when neighbours are < 1.4x apart (128K next to 152K)."""
    crowded = any(b / a < 1.4 for a, b in zip(xs, xs[1:]))
    ax.set_xscale("log")
    ax.set_xticks(xs, [("\n" if crowded and i % 2 else "") + fmt(x)
                       for i, x in enumerate(xs)])
    ax.xaxis.set_minor_locator(NullLocator())
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(length=0, labelcolor=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_sweep(ax_time, ax_speed, title, xlabel, series):
    xs = sorted(set().union(*series.values()))

    for name, color in COLOR.items():
        pts = sorted(series.get(name, {}).items())
        if not pts:
            continue
        x, y = zip(*pts)
        ax_time.plot(x, y, color=color, label=name, **MARK)
        ax_time.annotate(name, (x[-1], y[-1]), (6, 0), textcoords="offset points",
                         va="center", fontsize=9, color=INK)

    common = sorted(set(series.get("safe", {})) & set(series.get("online", {})))
    ratio = [series["safe"][x] / series["online"][x] for x in common]
    if ratio:
        ax_speed.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax_speed.plot(common, ratio, color=INK, **MARK)
        for i, (x, r) in enumerate(zip(common, ratio)):
            up = i % 2 == 0  # alternate so 1.31x next to 1.31x don't collide
            ax_speed.annotate(f"{r:.2f}x", (x, r), (0, 8 if up else -8),
                              textcoords="offset points", ha="center",
                              va="bottom" if up else "top", fontsize=8, color=MUTED)
        lo, hi = min(ratio + [1.0]), max(ratio + [1.0])
        pad = 0.3 * (hi - lo or 0.1)
        ax_speed.set_ylim(lo - pad, hi + pad)

    for ax in (ax_time, ax_speed):
        setup(ax, xs)
    ax_time.set_yscale("log")
    ax_time.set_title(title, loc="left", fontsize=11)
    ax_time.set_ylabel("time per launch (µs)", color=MUTED)
    ax_time.legend(frameon=False, fontsize=9, loc="upper left")
    ax_speed.set_ylabel("speedup  safe ÷ online", color=MUTED)
    ax_speed.set_xlabel(xlabel, color=MUTED)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("results", nargs="?", help="bench output (default: stdin)")
    p.add_argument("-o", "--output", help="write image here instead of showing")
    args = p.parse_args()

    text = open(args.results).read() if args.results else sys.stdin.read()
    sweeps = parse(text)
    if not sweeps:
        sys.exit("no safe/online rows found in input")

    n = len(sweeps)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 7.2), sharex="col",
                             squeeze=False, gridspec_kw={"height_ratios": [3, 2]})
    for col, (title, xlabel, series) in enumerate(sweeps):
        plot_sweep(axes[0][col], axes[1][col], title, xlabel, series)

    sm = next((l.strip() for l in text.splitlines() if l.startswith("SM count")), "")
    fig.suptitle(f"Softmax: safe vs online   ({sm})", x=0.02, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if args.output:
        fig.savefig(args.output, dpi=150)
        print("wrote", args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
