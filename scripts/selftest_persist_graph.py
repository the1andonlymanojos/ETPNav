#!/usr/bin/env python3
"""CPU-only sanity test of GraphMap persistence (no simulator, no model): a synthetic 1-D corridor walked in several
"episodes". Checks the invariants the planner input depends on. Run: python scripts/selftest_persist_graph.py"""
import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlnce_baselines.models.graph_utils import GraphMap  # noqa: E402

Y = 0.17


def walk(g, xs, z=0.0):
    """Walk the agent through x positions like the trainer does: new node each step, chosen ghost consumed, next node
    linked to the front of the chosen ghost."""
    prev_vp = None
    for step, x in enumerate(xs, 1):
        cur_vp = str(len(g.node_pos))
        cur_pos = np.array([x, Y, z], dtype=np.float64)
        cand_vp = [f"{cur_vp}_0"]
        cand_pos = [np.array([x + 2.0, Y, z])]
        g.update_graph(prev_vp, step, cur_vp, cur_pos, np.ones(4), cand_vp, cand_pos, [np.ones(4)], None)
        check(g, cur_vp)
        # trainer: pick the newest ghost, remember its front, consume it
        ghosts = g.visible_ghost_ids()
        if not ghosts:          # trainer: no_vp_left -> forced stop
            g.stop_reason = "no_vp_left"
            return cur_vp
        gv = ghosts[-1]
        _, front = g.front_to_ghost_dist(gv)
        g.delete_ghost(gv)
        prev_vp = front
    return cur_vp


def check(g, cur_vp):
    if g.persist == "none":
        assert g.active_nodes is None
        assert g.visible_ghost_ids() == list(g.ghost_pos)
        assert all(g.step_of(v) == g.node_stepId[v] for v in g.node_pos)
        return
    comp = nx.node_connected_component(g.graph_nx, cur_vp)
    here = {v for v in comp if g.node_episode[v] == g.episode_idx}
    assert g.active_nodes <= comp, "active must be connected to the current node"
    assert cur_vp in g.active_nodes and here <= g.active_nodes
    assert all(v in g.shortest_dist[cur_vp] for v in g.active_nodes), "every active node needs a path from cur"
    for gv in g.visible_ghost_ids():
        _, front = g.front_to_ghost_dist(gv)
        assert front in g.active_nodes and front in g.shortest_dist[cur_vp]
        if g.ghost_scope == 'current' and not gv.startswith('g_reopen_'):
            assert any(g.node_episode[f] == g.episode_idx for f in g.ghost_fronts[gv]), 'old-only frontier leaked'
    vis = g.visible_node_ids()
    assert cur_vp in vis and set(vis) <= g.active_nodes
    if g.node_scope == 'current':
        assert all(g.node_episode[v] == g.episode_idx for v in vis), 'old node token leaked'
    # get_node_embeds is what the trainer actually calls for every token id (real nodes AND ghosts) each decision;
    # it tells the two apart by an id prefix, which is exactly what broke the first real run of reopened ghosts
    for vp in vis:
        assert g.get_node_embeds(vp) is g.node_embeds[vp]
    for gv in g.visible_ghost_ids():   # a frontier's front need not be a token, but its distance must exist
        assert g.front_to_ghost_dist(gv)[1] in g.shortest_dist[cur_vp]
        assert g.get_node_embeds(gv) is not None
        if gv.startswith('g_reopen_'):
            old_vp = gv[len('g_reopen_'):]
            dis, front = g.front_to_ghost_dist(gv)
            assert front == old_vp and dis == 0.0, "a reopened ghost must sit exactly at its own old node"
            assert (g.ghost_embeds[gv][0] == g.node_embeds[old_vp]).all(), "reopened embedding must be the real one"
            assert old_vp not in g.reopen_consumed, "a still-visible reopened ghost can't already be consumed"
    if g.persist == "window":
        # cap is soft only by the connector nodes needed to keep every kept node reachable
        assert len(g.active_nodes) <= max(g.window_nodes, len(here)) + g.trace[-1]["connectors"]
    else:
        assert g.active_nodes == comp
    for v in g.active_nodes - here:
        assert g.step_of(v) == 1
    for v in here:
        assert g.step_of(v) == g.node_stepId[v]


def build(persist, window=4, scope='all', nscope='all', ascope='all', reopen=-1.0):
    return GraphMap(False, 0.5, True, 0, persist=persist, window_nodes=window, reloc_radius=1.0, ghost_scope=scope,
                     node_scope=nscope, absorb_scope=ascope, reopen_radius=reopen)


if __name__ == "__main__":
    # --- stock mode is untouched ---
    g = build("none")
    walk(g, [0, 2, 4, 6, 8, 10])
    assert g.shortest_dist["0"]["5"] == 10.0 and g.trace == [] and g.node_episode
    print("stock mode: unchanged behaviour, chain distance 0->5 = 10.0m  OK")

    for mode in ("window", "full"):
        g = build(mode)
        walk(g, [0, 2, 4, 6, 8, 10])                       # episode 0 builds a 6-node corridor
        n_old, ghosts_old = len(g.node_pos), set(g.ghost_pos)
        g.begin_episode()
        assert g.node_stop_scores == {} and g.trace == []
        walk(g, [4.4, 6.4])                                # episode 1 starts 0.4 m from old node at x=4
        t = g.trace[0]
        assert t["reloc"] >= 1, "new start should link to the old map"
        assert t["comp"] == n_old + 1, "graphs should now be one component"
        if mode == "window":
            assert t["active"] <= 4 and t["old_visible"] == 3, t
        else:
            assert t["active"] == n_old + 1 and t["old_visible"] == n_old, t
        print(f"{mode:6s}: episode-1 start linked to old map (reloc={t['reloc']}), planner sees "
              f"{t['active']} nodes ({t['old_visible']} old), {t['ghosts_visible']} ghosts  OK")

        g.begin_episode()
        walk(g, [50.0, 52.0])                              # episode 2 starts far from everything
        t = g.trace[0]
        assert t["reloc"] == 0 and t["comp"] == 1 and t["old_visible"] == 0
        print(f"{mode:6s}: episode-2 start far from the map -> no link, sees only its own node (no crash)  OK")
        assert len(set(g.node_pos)) == len(g.node_pos)     # ids never collide
    # --- regression for the first real-run crash: a current-episode node whose only route back to the rest of this
    # episode runs through old nodes that fall outside the window cap ---
    g = build("window", window=4)
    walk(g, [float(x) for x in range(0, 22, 2)])           # old corridor, 11 nodes, x = 0..20
    g.begin_episode()
    def add(g, prev, x):
        vp = str(len(g.node_pos)); pos = np.array([x, Y, 0.0])
        g.update_graph(prev, 1, vp, pos, np.ones(4), [f"{vp}_0"], [np.array([x + 2.0, Y, 0.0])], [np.ones(4)], None)
        return vp
    add(g, None, 20.3)                                     # A: start beside the far end of the old corridor
    b = add(g, None, 0.3)                                  # B: jumped (backtracked) to the near end; A now only reachable via old nodes
    check(g, b)
    assert g.trace[-1]["connectors"] > 0, "test must actually exercise the connector case"
    print(f"window: current node cut off from its own episode's start by the cap -> {g.trace[-1]['connectors']} connector nodes kept  OK")

    # --- absorption: a proposed waypoint that lands on an old node makes no frontier in stock; with absorb='current' it must ---
    for asc, expect in (("all", 0), ("current", 1)):
        g = build("window", window=8, scope="current", nscope="current", ascope=asc)
        walk(g, [0.0, 2.0, 4.0, 6.0, 8.0])                 # old corridor
        g.begin_episode()
        vp = str(len(g.node_pos))
        g.update_graph(None, 1, vp, np.array([4.3, Y, 0.0]), np.ones(4), [f"{vp}_0"], [np.array([6.1, Y, 0.0])], [np.ones(4)], None)
        n_front = sum(1 for gv in g.visible_ghost_ids())
        assert n_front == expect, (asc, n_front)
    print("absorb: candidate on an old node -> no frontier in stock (all), a frontier when absorb='current'  OK")

    # --- reopen: on the validated clean baseline (nodes/frontiers/absorption all scoped to this episode), a nearby
    # old node should come back as a frontier the agent can choose, and never reappear once consumed ---
    g = build("window", window=8, scope="current", nscope="current", ascope="current", reopen=3.0)
    walk(g, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])                # old corridor, x = 0..10
    g.begin_episode()
    vp = str(len(g.node_pos))
    g.update_graph(None, 1, vp, np.array([7.5, Y, 0.0]), np.ones(4), [f"{vp}_0"], [np.array([9.5, Y, 0.0])], [np.ones(4)], None)
    check(g, vp)
    reopened = [gv for gv in g.visible_ghost_ids() if gv.startswith('g_reopen_')]
    assert reopened, "a node 1.5-2.5 m away should be within reopen_radius=3.0"
    assert all(g.node_pos[g.front_to_ghost_dist(gv)[1]][0] >= 7.5 - 3.0 for gv in reopened)
    gv = reopened[0]
    old_vp = gv[len('g_reopen_'):]
    g.delete_ghost(gv)
    assert old_vp in g.reopen_consumed
    vp2 = str(len(g.node_pos))
    g.update_graph(g.front_to_ghost_dist(reopened[-1])[1] if len(reopened) > 1 else None, 2, vp2,
                    np.array([7.4, Y, 0.05]), np.ones(4), [f"{vp2}_0"], [np.array([9.4, Y, 0.05])], [np.ones(4)], None)
    assert f'g_reopen_{old_vp}' not in g.visible_ghost_ids(), "a consumed reopened node must not come back this episode"
    g.begin_episode()                                       # new episode: consumption resets
    vp3 = str(len(g.node_pos))
    g.update_graph(None, 1, vp3, np.array([7.5, Y, 0.0]), np.ones(4), [f"{vp3}_0"], [np.array([9.5, Y, 0.0])], [np.ones(4)], None)
    assert any(gv == f'g_reopen_{old_vp}' for gv in g.visible_ghost_ids()), "a new episode may reopen it again"
    print("reopen: nearby old node comes back as a frontier, is consumed once, and reopens fresh next episode  OK")

    # --- oracle anchor: reopening near a fixed goal, not the agent, must reopen a DIFFERENT node than the default ---
    g = build("window", window=8, scope="current", nscope="current", ascope="current", reopen=3.0)
    walk(g, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])   # old corridor, x = 0..14
    g.begin_episode()
    g.set_goal(np.array([14.0, Y, 0.0]))                    # far end of the corridor
    vp = str(len(g.node_pos))
    g.update_graph(None, 1, vp, np.array([1.0, Y, 0.0]), np.ones(4), [f"{vp}_0"], [np.array([3.0, Y, 0.0])], [np.ones(4)], None)
    check(g, vp)
    reopened = {gv[len('g_reopen_'):] for gv in g.visible_ghost_ids() if gv.startswith('g_reopen_')}
    assert reopened and all(g.node_pos[v][0] >= 14.0 - 3.0 for v in reopened), "oracle anchor must reopen near the GOAL, not the agent (near x=1)"
    print("oracle anchor: reopening tracks the true goal instead of the agent's own position  OK")

    # --- fuzz: random maps where this episode's nodes hang off old nodes (the case that crashed the first real run:
    # a current-episode node whose shortest path runs through an older node outside the window cap) ---
    import random
    rng = random.Random(0)
    n_states = 0
    for _trial in range(300):
        g = build(rng.choice(["window", "full"]), window=rng.randint(1, 4), scope=rng.choice(["all", "current"]),
                  nscope=rng.choice(["all", "current"]), ascope=rng.choice(["all", "current"]),
                  reopen=rng.choice([-1.0, 0.0, 1.0, 4.0, 20.0]))
        for _ in range(rng.randint(1, 4)):                 # several episodes on one growing map
            g.begin_episode()
            prev = None
            for step in range(1, rng.randint(2, 9)):
                cur_vp = str(len(g.node_pos))
                if g.node_pos and rng.random() < 0.5:      # start / step next to a random old node (relocalise)
                    pos = g.node_pos[rng.choice(list(g.node_pos))] + np.array([rng.uniform(-.9, .9), 0, rng.uniform(-.9, .9)])
                else:
                    pos = np.array([rng.uniform(-7, 7), Y, rng.uniform(-7, 7)])
                g.update_graph(prev, step, cur_vp, pos, np.ones(4), [f"{cur_vp}_0"],
                               [pos + np.array([rng.uniform(2, 6), 0, rng.uniform(2, 6)])], [np.ones(4)], None)
                check(g, cur_vp)
                n_states += 1
                ghosts = g.visible_ghost_ids()
                if not ghosts:
                    break
                gv = rng.choice(ghosts)
                _, prev = g.front_to_ghost_dist(gv)        # the front may be an OLD node in window mode
                g.delete_ghost(gv)
    print(f"fuzz: {n_states} planner-input states checked, every kept node reachable from the current node  OK")
    print("ALL SELF-TESTS PASSED")
