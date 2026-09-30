"""Render the latency charts used in the README.

Two panels. The top one compares the engine's own kernels on a linear scale,
which is the point: at 62 ms against 3579 ms the GPU bar is a sliver, and that
is easier to read than the numbers are. The bottom one places the engine against
production runtimes on a log scale, because covering 4.9 ms to 3579 ms linearly
would collapse everything but the slowest bar.

Writes a light and a dark variant so the README can serve whichever matches the
reader's GitHub theme.

    python docs/make_speedup_chart.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, milliseconds, is_highlight)
KERNELS = [
    ("CUDA 8 bit, fused\n(dp4a, T4)", 62.1, True),
    ("Vectorized 8 bit\n(AVX-VNNI)", 233.5, False),
    ("Plain loop, 8 bit", 3578.7, False),
    ("Plain loop, 32 bit", 3105.9, False),
]

RUNTIMES = [
    ("This engine, plain loop", 3578.7, False),
    ("This engine, vectorized", 233.5, False),
    ("This engine, CUDA 8 bit", 62.1, True),
    ("PyTorch, 32 bit", 42.4, False),
    ("ONNX Runtime, 32 bit", 24.1, False),
    ("OpenVINO, 8 bit", 13.9, False),
    ("TensorRT, 8 bit, T4", 4.9, False),
]

THEMES = {
    "light": {
        "bg": "#ffffff",
        "fg": "#0f172a",
        "muted": "#64748b",
        "bar": "#cbd5e1",
        "bar_edge": "#94a3b8",
        "hi": "#ea580c",
        "grid": "#e2e8f0",
    },
    "dark": {
        "bg": "#0d1117",
        "fg": "#e6edf3",
        "muted": "#9198a1",
        "bar": "#30363d",
        "bar_edge": "#484f58",
        "hi": "#f97316",
        "grid": "#21262d",
    },
}


def fmt(ms):
    return f"{ms:,.0f} ms" if ms >= 100 else f"{ms:.1f} ms"


def draw(theme_name, colors):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 7.2), dpi=150,
        gridspec_kw={"height_ratios": [3, 5], "hspace": 0.55},
    )
    fig.patch.set_facecolor(colors["bg"])

    for ax in (ax1, ax2):
        ax.set_facecolor(colors["bg"])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(colors["grid"])
        ax.tick_params(colors=colors["muted"], labelsize=11, length=0)
        ax.set_axisbelow(True)

    # ---- Panel 1, the engine's own kernels, linear ------------------------
    labels = [k[0] for k in KERNELS]
    vals = [k[1] for k in KERNELS]
    highs = [k[2] for k in KERNELS]
    ypos = range(len(vals))

    ax1.barh(
        list(ypos), vals, height=0.62,
        color=[colors["hi"] if h else colors["bar"] for h in highs],
        edgecolor=[colors["hi"] if h else colors["bar_edge"] for h in highs],
        linewidth=1.0,
    )
    ax1.set_yticks(list(ypos))
    ax1.set_yticklabels(labels, color=colors["fg"], fontsize=11)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(vals) * 1.18)
    ax1.xaxis.grid(True, color=colors["grid"], linewidth=0.8)
    ax1.set_xlabel("milliseconds per image, lower is better", color=colors["muted"],
                   fontsize=10, labelpad=8)

    for i, (v, h) in enumerate(zip(vals, highs)):
        ax1.text(v + max(vals) * 0.015, i, fmt(v), va="center", fontsize=11,
                 color=colors["hi"] if h else colors["fg"],
                 fontweight="bold" if h else "normal")

    ax1.set_title(
        "The same model, the same weights, four ways of doing the arithmetic",
        color=colors["fg"], fontsize=13, fontweight="bold", loc="left", pad=14)

    # headline ratio: the GPU bar against the vectorized CPU one
    ax1.annotate(
        "", xy=(vals[0] * 1.04, 0.5), xytext=(vals[1], 0.5),
        arrowprops=dict(arrowstyle="->", color=colors["hi"], linewidth=1.8,
                        shrinkA=0, shrinkB=0),
    )
    ax1.text(vals[1] * 1.10, 0.5, f"{vals[1] / vals[0]:.1f}x faster",
             color=colors["hi"], fontsize=12, fontweight="bold", ha="left",
             va="center")

    # ---- Panel 2, against the libraries, log ------------------------------
    rlabels = [r[0] for r in RUNTIMES]
    rvals = [r[1] for r in RUNTIMES]
    rhigh = [r[2] for r in RUNTIMES]
    rpos = range(len(rvals))

    ax2.barh(
        list(rpos), rvals, height=0.62,
        color=[colors["hi"] if h else colors["bar"] for h in rhigh],
        edgecolor=[colors["hi"] if h else colors["bar_edge"] for h in rhigh],
        linewidth=1.0,
    )
    ax2.set_xscale("log")
    # Plain numbers rather than powers of ten, since the point of the chart is
    # that someone can read it without decoding scientific notation.
    ax2.set_xticks([10, 100, 1000, 10000])
    ax2.set_xticklabels(["10", "100", "1,000", "10,000"])
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_yticks(list(rpos))
    ax2.set_yticklabels(rlabels, color=colors["fg"], fontsize=11)
    ax2.invert_yaxis()
    ax2.set_xlim(3, 12000)
    ax2.xaxis.grid(True, color=colors["grid"], linewidth=0.8, which="both")
    ax2.set_xlabel("milliseconds per image, lower is better. Each gridline is ten "
                   "times the one before it", color=colors["muted"], fontsize=10,
                   labelpad=8)

    for i, (v, h) in enumerate(zip(rvals, rhigh)):
        ax2.text(v * 1.14, i, fmt(v), va="center", fontsize=11,
                 color=colors["hi"] if h else colors["fg"],
                 fontweight="bold" if h else "normal")

    ax2.set_title("Where that leaves it against production runtimes",
                  color=colors["fg"], fontsize=13, fontweight="bold", loc="left",
                  pad=14)

    fig.text(0.012, 0.012,
             "CPU rows on an Intel Core Ultra 7 256V, mains power; GPU rows on a "
             "Colab Tesla T4. One configuration per process, fastest observed\n"
             "pass. Every row of this engine scores mAP 0.8826, the same "
             "detections. PyTorch and ONNX are 32 bit at 0.8859; TensorRT in "
             "8 bit scores 0.8656.",
             color=colors["muted"], fontsize=9)

    out = os.path.join(HERE, f"speedup-{theme_name}.png")
    fig.savefig(out, facecolor=colors["bg"], bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print("wrote", out)


for name, palette in THEMES.items():
    draw(name, palette)
