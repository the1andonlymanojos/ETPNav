"""Plot results/memory_summary.json (memory_anchor.py): accuracy, questions and corrections per day, with vs without memory.

python cleancmd/anchor/plot_memory.py   -> results/memory_curve.png
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval_anchor import RESULTS  # noqa: E402

SERIES = [(True, "with memory", "#2a78d6"), (False, "without memory", "#eb6834")]
PANELS = [("accuracy", "First-try accuracy", lambda v: f"{v:.0%}"),
          ("questions_per_day", "Questions per day (robot asked)", lambda v: f"{v:.1f}"),
          ("corrections_per_day", "Corrections per day (picked wrong)", lambda v: f"{v:.1f}")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    s = json.load(open(RESULTS / "memory_summary.json"))
    rows = s["rows"]
    days = sorted({r["day"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="#fcfcfb")
    for ax, (key, title, fmt) in zip(axes, PANELS):
        ax.set_facecolor("#fcfcfb")
        for mem, label, color in SERIES:
            per_day = [[r[key] for r in rows if r["memory"] == mem and r["day"] == d] for d in days]
            mean = [sum(v) / len(v) for v in per_day]
            ax.fill_between(days, [min(v) for v in per_day], [max(v) for v in per_day], color=color, alpha=0.15, lw=0)
            ax.plot(days, mean, color=color, lw=2, marker="o", ms=8, mec="#fcfcfb", mew=2, label=label)
            ax.annotate(fmt(mean[-1]), (days[-1], mean[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xticks(days)
        ax.set_xlabel("day", color=MUTED)
        ax.set_xlim(days[0] - 0.2, days[-1] + 0.6)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.tick_params(colors=MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.set_ylim(bottom=0)
    axes[0].set_ylim(0, 1.05)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].legend(frameon=False, loc="center right", labelcolor=INK)
    fig.suptitle(f"Method C (room-first) with vs without per-scene memory — {len(s['scenes'])} scenes, "
                 f"{s['day_cmds']} commands/day from a pool of {s['pool']}, 3 seeds (band = seed range)",
                 x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    out = RESULTS / "memory_curve.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(out)


if __name__ == "__main__":
    main()
