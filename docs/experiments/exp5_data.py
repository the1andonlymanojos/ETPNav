"""Loads the Exp 5 radius sweep (docs/experiments/exp5_reopen/s{0,1,2}/r{R}, plus the reopen-off point already
recorded as exp4_split's clean_control_w20) into per-radius statistics. Run standalone to print the table."""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
X = f"{REPO}/docs/experiments"
TOURS = (0, 1, 2)
RADII = [-1.0, 0.0, 1.5, 3.0, 6.0, 30.0]   # -1 = reopening off (the validated clean control)


def _path(r, s):
    if r < 0:
        return f"{X}/exp4_split/s{s}/clean_control_w20"
    return f"{X}/exp5_reopen/s{s}/r{r}"   # directories are named r0.0, r1.5, r3.0, r6.0, r30.0 (bash %s, not %g)


def load():
    data = {}
    for r in RADII:
        data[r] = {s: json.load(open(f"{_path(r, s)}/results.json"))["episodes"] for s in TOURS}
    base = {s: json.load(open(f"{X}/exp3_diagnosis/s{s}/none/results.json"))["episodes"] for s in TOURS}
    return data, base


def stats(data, base):
    out = {}
    for r, tours in data.items():
        rows = [(s, e, st) for s in TOURS for e, st in tours[s].items()]
        n = len(rows)
        wins = losses = 0
        for s, e, st in rows:
            b = base[s][e]["success"] > .5
            x = st["success"] > .5
            wins += (x and not b)
            losses += (b and not x)
        out[r] = dict(
            n=n, k=sum(st["success"] > .5 for _, _, st in rows), spl=sum(st["spl"] for _, _, st in rows) / n,
            ne=sum(st["distance_to_goal"] for _, _, st in rows) / n, steps=sum(st["steps_taken"] for _, _, st in rows) / n,
            wins=wins, losses=losses, per_tour={s: sum(st["success"] > .5 for st in tours[s].values()) for s in TOURS})
    return out


def exact_zero_hits():
    """How many later-episode nodes sit at EXACTLY (bit-for-bit) the same (x,z) as some earlier-episode node, on the
    r=0.0 run -- i.e. how often even the tightest possible threshold still finds something to reopen."""
    hits, checked = 0, 0
    for s in TOURS:
        dbg = json.load(open(f"{_path(0.0, s)}/persist_debug.json"))
        ids = list(dbg.keys())
        for i, e in enumerate(ids):
            nodes = dbg[e]["nodes"]
            newpos = {(n[0], n[1]) for n in nodes if n[2] == i}
            oldpos = {(n[0], n[1]) for n in nodes if n[2] < i}
            checked += len(newpos)
            hits += len(newpos & oldpos)
    return hits, checked


if __name__ == "__main__":
    d, b = load()
    S = stats(d, b)
    print(f"{'radius':>8s} {'reached':>9s} {'SPL':>5s} {'miss':>6s} {'steps':>6s} {'+/-':>8s}  per-tour")
    for r in RADII:
        s = S[r]
        lbl = "off" if r < 0 else f"{r:g}m"
        print(f"{lbl:>8s} {s['k']:3d}/{s['n']:<5} {s['spl']:5.2f} {s['ne']:5.1f}m {s['steps']:6.0f} "
              f"{'+%d/-%d' % (s['wins'], s['losses']):>8s}  {s['per_tour']}")
    h, c = exact_zero_hits()
    print(f"\nexact-zero position coincidences at r=0.0: {h} of {c} later-episode nodes land bit-for-bit on an earlier one")
