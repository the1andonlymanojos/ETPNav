"""Loads the Exp 3 sweep (docs/experiments/exp3_diagnosis/s{0,1,2}/{arm}/results.json) into per-arm statistics.
Used by build_exp3.py; runnable on its own to print the table:  python docs/experiments/exp3_data.py"""
import json
import os
from math import factorial


def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = f"{REPO}/docs/experiments/exp3_diagnosis"
TOURS = (0, 1, 2)
# arm id -> (label, mode, frontiers)
ARMS = {
    "none": ("map wiped", "none", "all"),
    "w10": ("window 10", "window", "all"),
    "w20": ("window 20", "window", "all"),
    "w40": ("window 40", "window", "all"),
    "full": ("full graph", "full", "all"),
    "w20cur": ("window 20, old frontiers hidden", "window", "current"),
    "fullcur": ("full graph, old frontiers hidden", "full", "current"),
}


def sign_test(wins, losses):
    """Exact two-sided sign test on discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load():
    """{arm: {tour: {episode_id: stats}}} plus intended orders; missing arms/tours are simply absent."""
    data, order = {}, {}
    for a in ARMS:
        for s in TOURS:
            p = f"{DIR}/s{s}/{a}/results.json"
            if os.path.exists(p):
                r = json.load(open(p))
                data.setdefault(a, {})[s] = r["episodes"]
                order[s] = r["intended_order"]
    return data, order


def stats(data, order):
    out = {}
    base = data.get("none", {})
    for a, tours in data.items():
        rows = [(s, e, tours[s][e]) for s in TOURS if s in tours for e in order[s]]
        n = len(rows)
        wins = losses = 0
        for s, e, st in rows:
            if s in base:
                b = base[s][e]["success"] > .5
                x = st["success"] > .5
                wins += (x and not b)
                losses += (b and not x)
        old = [st.get("g_old_visible_mean", 0.0) for _, _, st in rows]
        out[a] = dict(
            n=n, tours=sorted(tours), k=sum(st["success"] > .5 for _, _, st in rows),
            spl=sum(st["spl"] for _, _, st in rows) / n, ne=sum(st["distance_to_goal"] for _, _, st in rows) / n,
            steps=sum(st["steps_taken"] for _, _, st in rows) / n,
            long=sum(st["steps_taken"] > 150 for _, _, st in rows), old=sum(old) / n,
            stop1=sum(st["steps_taken"] <= 1 for _, _, st in rows),
            no_vp=sum(st.get("g_stop_no_vp_left", 0) for _, _, st in rows),
            wins=int(wins), losses=int(losses), p=sign_test(int(wins), int(losses)),
            per_tour={s: sum(tours[s][e]["success"] > .5 for e in order[s]) for s in tours})
    return out


if __name__ == "__main__":
    d, o = load()
    S = stats(d, o)
    print(f"{'arm':9s} {'n':>3s} {'SR':>6s} {'SPL':>5s} {'NE':>5s} {'steps':>6s} {'>150':>4s} {'stop1':>5s} {'oldvis':>6s} {'+':>3s} {'-':>3s} {'p':>6s}  per-tour")
    for a in ARMS:
        if a in S:
            s = S[a]
            print(f"{a:9s} {s['n']:3d} {100 * s['k'] / s['n']:5.1f}% {s['spl']:5.2f} {s['ne']:5.1f} {s['steps']:6.0f} {s['long']:4d} {s['stop1']:5d} {s['old']:6.1f} "
                  f"{s['wins']:3d} {s['losses']:3d} {s['p']:6.3f}  {s['per_tour']}")


# ---------------------------------------------------------------- deeper cuts used by the Exp 3 write-up
def _pd(path):
    return json.load(open(path))


def hidden_frontier_stats():
    """Old-frontiers-hidden arms: how many frontiers did the planner have at decision 1, and how did those episodes end?"""
    out = {}
    for arm in ("w20cur", "fullcur"):
        rows = []
        for s in TOURS:
            r = json.load(open(f"{DIR}/s{s}/{arm}/results.json"))["episodes"]
            d = _pd(f"{DIR}/s{s}/{arm}/persist_debug.json")
            for e, st in r.items():
                rows.append((d[e]["trace"][0]["ghosts_visible"], st["success"] > .5, d[e]["stop_reason"], st["steps_taken"]))
        lo = [x for x in rows if x[0] <= 2]
        hi = [x for x in rows if x[0] >= 3]
        out[arm] = dict(n=len(rows), zero=sum(1 for x in rows if x[0] == 0), lo_n=len(lo), lo_k=sum(x[1] for x in lo),
                        hi_n=len(hi), hi_k=sum(x[1] for x in hi), no_vp=sum(1 for x in rows if x[2] == "no_vp_left"),
                        one_forced=sum(1 for x in rows if x[3] <= 1 and x[2] == "no_vp_left"),
                        one_chose=sum(1 for x in rows if x[3] <= 1 and x[2] == "stop"))
    return out


def decision_stats():
    """Window-20 reruns with per-decision logging (docs/experiments/exp3_diagnosis/decisions): what did the planner choose?"""
    ep, moves = [], []
    for s in TOURS:
        base = json.load(open(f"{DIR}/s{s}/none/results.json"))["episodes"]
        r = json.load(open(f"{DIR}/decisions/s{s}/w20/results.json"))["episodes"]
        d = _pd(f"{DIR}/decisions/s{s}/w20/persist_debug.json")
        for i, (e, st) in enumerate(r.items()):
            if i == 0:
                continue                                    # episode 1 starts from an empty map: identical to baseline
            ds = d[e]["decisions"]
            g = [x for x in ds if x["kind"] == "ghost"]
            first = ds[0]
            ep.append(dict(base=base[e]["success"] > .5, ok=st["success"] > .5, kind=first["kind"],
                           first_old=first["kind"] == "ghost" and first["ghost_fronts_old"] == first["ghost_fronts_all"],
                           old_moves=sum(x["ghost_fronts_old"] == x["ghost_fronts_all"] for x in g),
                           via_old=sum(x["path_old_nodes"] > 0 for x in g), moves=len(g)))
    tot = sum(x["moves"] for x in ep)
    old = sum(x["old_moves"] for x in ep)
    via = sum(x["via_old"] for x in ep)
    took = [x for x in ep if x["old_moves"] > 0]
    none_ = [x for x in ep if x["old_moves"] == 0]
    b = [x for x in ep if x["base"]]
    loc = [x for x in b if x["kind"] == "ghost" and not x["first_old"]]
    far = [x for x in b if x["kind"] == "ghost" and x["first_old"]]
    return dict(n_ep=len(ep), moves=tot, old=old, via=via, first_old=sum(x["first_old"] for x in ep),
                took_n=len(took), took_k=sum(x["ok"] for x in took), none_n=len(none_), none_k=sum(x["ok"] for x in none_),
                base_n=len(b), base_persist_k=sum(x["ok"] for x in b), loc_n=len(loc), loc_k=sum(x["ok"] for x in loc),
                far_n=len(far), far_k=sum(x["ok"] for x in far))


if __name__ == "__main__":
    print(hidden_frontier_stats())
    print(decision_stats())
