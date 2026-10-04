"""Plot results/memory_users_summary.json (memory_users.py) -> results/memory_users_curve.png.

Panels: accuracy, accuracy on rows whose target was detected, questions per day. One line per simulated user with
memory, plus one shared "without memory" line (it does not depend on the user: nothing they say is kept).

python cleancmd/anchor/plot_memory_users.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval_anchor import RESULTS  # noqa: E402

SERIES = [("consistent", True, "with memory: consistent user", "#2a78d6", "-"),
          ("inconsistent", True, "with memory: 30% inconsistent user", "#eb6834", "-"),
          ("random", True, "with memory: random user", "#1baf7a", "-"),
          ("consistent", False, "without memory (any user)", "#52514e", "--")]
PANELS = [("accuracy", "Accuracy", lambda v: f"{v:.0%}"),
          ("accuracy_detected", "Accuracy, target detected only", lambda v: f"{v:.0%}"),
          ("questions_per_day", "Questions per day per scene", lambda v: f"{v:.1f}")]
INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main():
    s = json.load(open(RESULTS / "memory_users_summary.json"))
    rows = s["rows"]
    days = sorted({r["day"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=BG)
    for ax, (key, title, fmt) in zip(axes, PANELS):
        ax.set_facecolor(BG)
        ends = []
        for user, mem, label, color, ls in SERIES:
            per_day = [[r[key] for r in rows if r["user"] == user and r["memory"] == mem and r["day"] == d] for d in days]
            mean = [sum(v) / len(v) for v in per_day]
            ax.fill_between(days, [min(v) for v in per_day], [max(v) for v in per_day], color=color, alpha=0.12, lw=0)
            ax.plot(days, mean, color=color, lw=2, ls=ls, marker="o", ms=8, mec=BG, mew=2, label=label)
            ends.append(mean[-1])
        if key != "questions_per_day":
            # end labels, nudged apart so close values stay readable
            order = sorted(range(len(ends)), key=lambda i: ends[i])
            ys = [ends[i] for i in order]
            for k in range(1, len(ys)):
                ys[k] = max(ys[k], ys[k - 1] + 0.045)
            for i, y in zip(order, ys):
                ax.annotate(fmt(ends[i]), (days[-1], y), xytext=(8, 0), textcoords="offset points",
                            va="center", color=INK, fontsize=9)
        else:
            ax.text(0.98, 0.55, "with memory: the three users' lines coincide\n(a phrase is never asked again once any\nanswer is remembered, right or wrong)",
                    transform=ax.transAxes, ha="right", va="center", fontsize=8.5, color=MUTED)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xticks(days)
        ax.set_xlabel("day", color=MUTED)
        ax.set_xlim(days[0] - 0.2, days[-1] + 0.7)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.tick_params(colors=MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.set_ylim(bottom=0)
    for ax in axes[:2]:
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, labelcolor=INK, fontsize=9)
    fig.suptitle(f"Memory controls: method {s['base']} (C-gated) on the open-vocabulary detector, {len(s['scenes'])} MP3D "
                 f"datasets, {s['day_cmds']} commands/day, 3 seeds (band = seed range)",
                 x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = RESULTS / "memory_users_curve.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    print(out)


if __name__ == "__main__":
    main()
