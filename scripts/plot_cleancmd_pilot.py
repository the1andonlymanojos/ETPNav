#!/usr/bin/env python3
"""Top-down figure of a CleanCmd pilot: for a few commands, the proxy target cells, the anchor, and each method's answer.

Needs habitat_sim (navmesh background) and matplotlib, e.g. inside `conda activate vlmaps_3.9`, from the repo root:
    python scripts/plot_cleancmd_pilot.py --ids zsNo4HB9uLZ-p00 zsNo4HB9uLZ-p07 zsNo4HB9uLZ-p15 zsNo4HB9uLZ-p22 \
        --out cleancmd/pilot/zsNo4HB9uLZ_pilot_topdown.png
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cleancmd.grid import Grid  # noqa: E402
from cleancmd.io import load_jsonl, to_cells  # noqa: E402

PILOT = "cleancmd/pilot"
NAVMESH = os.path.expanduser("~/vlmaps_cc/mp3d_habitat/mp3d/{0}/{0}.navmesh")
METHODS = [  # file suffix, label, colour, marker, marker size (filter-on drawn larger so a shared answer stays visible)
    ("etpnav", "ETPNav (path, end = x)", "tab:blue", "x", 11),
    ("vlmaps", "VLMaps, front filter", "tab:red", "o", 14),
    ("vlmaps_nofilter", "VLMaps, no front filter", "tab:orange", "s", 7),
    ("ours_v0", "ours v0 (region; anchor footprint dashed)", "tab:purple", "D", 8),
    ("vlmaps_oracleobj", "VLMaps, correct object", "tab:brown", "P", 12),
    ("ours_v0_oracleobj", "ours v0, correct object (annotated box dotted)", "tab:cyan", "D", 8),
]


def navigable_image(scene, grid, res=0.1):
    import habitat_sim

    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(NAVMESH.format(scene))
    xs = np.arange(grid.x0, grid.x0 + grid.cols * grid.cell, res)
    zs = np.arange(grid.z0, grid.z0 + grid.rows * grid.cell, res)
    img = np.zeros((len(zs), len(xs)), dtype=bool)
    for i, z in enumerate(zs):
        for j, x in enumerate(xs):
            img[i, j] = pf.is_navigable(np.array([x, grid.floor_y, z], dtype=np.float32))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="zsNo4HB9uLZ")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gt = {g["command_id"]: g for g in load_jsonl(os.path.join(PILOT, a.scene + "_gt_proxy.jsonl"))}
    probes = {p["command_id"]: p for p in load_jsonl(os.path.join(PILOT, a.scene + "_probes.jsonl"))}
    preds = {}
    for suffix, *_ in METHODS:  # noqa: B007
        path = os.path.join(PILOT, "%s_pred_%s.jsonl" % (a.scene, suffix))
        preds[suffix] = {p["command_id"]: p for p in load_jsonl(path)}

    grid = Grid.from_dict(gt[a.ids[0]]["grid"])
    nav = navigable_image(a.scene, grid)
    extent = [grid.x0, grid.x0 + grid.cols * grid.cell, grid.z0 + grid.rows * grid.cell, grid.z0]

    n = len(a.ids)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(11.5 * cols, 5.2 * rows), squeeze=False)
    for ax, cid in zip(axes.flat, a.ids):
        g, p = gt[cid], probes[cid]
        ax.imshow(np.where(nav, 0.95, 0.6), extent=extent, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for r, c in {(int(r), int(c)) for r, c in g["masks"][0]}:
            x0, z0 = grid.x0 + c * grid.cell, grid.z0 + r * grid.cell
            ax.add_patch(plt.Rectangle((x0, z0), grid.cell, grid.cell, color="tab:green", alpha=0.55, lw=0))
        ax.plot([], [], "s", color="tab:green", alpha=0.55, label="proxy target cells")
        if g.get("anchor"):
            ax.plot(*g["anchor"]["center"], "k*", ms=14, label="anchor (%s)" % p["anchor_category"])
        sx, _, sz = p["start_position"]
        yaw = np.deg2rad(p["start_yaw_deg"])
        ax.annotate("", xy=(sx - 0.9 * np.sin(yaw), sz - 0.9 * np.cos(yaw)), xytext=(sx, sz),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=2))
        ax.plot(sx, sz, "ko", ms=6, label="start + heading")

        answers = []
        for suffix, label, colour, marker, ms in METHODS:
            pr = preds[suffix].get(cid)
            if pr is None:
                continue
            if pr["kind"] == "path":
                path = np.array(pr["path"])
                ax.plot(path[:, 0], path[:, 1], "-", color=colour, lw=1.8, alpha=0.9)
                ax.plot(path[-1, 0], path[-1, 1], marker, color=colour, ms=ms, mew=3, label=label)
                answers.append(path[-1])
            elif pr["kind"] == "point":
                ax.plot(*pr["point"], marker, color=colour, ms=ms, mec="k", label=label)
                answers.append(pr["point"])
                for r, c in to_cells(pr, grid):
                    x0, z0 = grid.x0 + c * grid.cell, grid.z0 + r * grid.cell
                    ax.add_patch(plt.Rectangle((x0, z0), grid.cell, grid.cell, fill=False, ec=colour, lw=1))
            elif pr["kind"] == "region" and pr["cells"]:
                for r, c in pr["cells"]:
                    x0, z0 = grid.x0 + c * grid.cell, grid.z0 + r * grid.cell
                    ax.add_patch(plt.Rectangle((x0, z0), grid.cell, grid.cell, color=colour, alpha=0.35, lw=0))
                ctr = np.mean([grid.center(r, c) for r, c in pr["cells"]], axis=0)
                ax.plot(*ctr, marker, color=colour, ms=ms, mec="k", label=label + ", centre")
                answers.append(ctr)
            else:
                why = [k for k in ("unsupported", "not_found", "front_filtered") if pr["meta"].get(k)] or ["empty region"]
                ax.plot([], [], marker, color=colour, mfc="none", label="%s: no answer (%s)" % (label, ",".join(why)))
            fp = pr.get("meta", {}).get("footprint_xz")
            if fp:
                fp = np.array(fp + fp[:1])
                ax.plot(fp[:, 0], fp[:, 1], ":" if "oracle_object" in pr["meta"] else "--", color=colour, lw=1.2)

        pts = [g["anchor"]["center"]] if g.get("anchor") else []
        pts += [[sx, sz]] + [grid.center(int(r), int(c)) for r, c in g["masks"][0]] + [list(x) for x in answers]
        pts = np.array(pts)
        pad = 1.5
        ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
        ax.set_ylim(pts[:, 1].max() + pad, pts[:, 1].min() - pad)
        ax.set_aspect("equal")
        ax.set_title("%s: %r" % (cid, p["command"]), fontsize=11)
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world z (m)")
        ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.9)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("CleanCmd PILOT, %s: geometric proxy targets (not human annotation); left/right relative to the "
                 "robot's start view" % a.scene, fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
