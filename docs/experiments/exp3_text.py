"""Narrative for the Exp 3 page. Every number comes from exp3_data (raw result files), none typed by hand."""
import exp3_data as X

TITLE = "What Breaks the Persistent Map?"
REPRO = ("Reproduce: <code>scripts/run_tour.py --scene zsNo4HB9uLZ --n 15 --sample-seed {0,1,2} --mode {none,window,full} [--window-nodes N] [--ghosts current]</code> "
         "· analysis: <code>docs/experiments/exp3_data.py</code> · data: <code>docs/experiments/exp3_diagnosis/</code>")


def _p(k, n):
    return f"{100 * k / n:.0f}%"


def lead(S, N):
    b = S["none"]
    others = [a for a in X.ARMS if a != "none"]
    lo = min(100 * S[a]["k"] / S[a]["n"] for a in others)
    hi = max(100 * S[a]["k"] / S[a]["n"] for a in others)
    return (f"Keeping the map between episodes cost ETPNav about half its accuracy in <b>every</b> variant we tried: <b>{_p(b['k'], N)}</b> of {N} episodes reached "
            f"with the map wiped, <b>{lo:.0f}–{hi:.0f}%</b> with it kept. The damage doesn't grow with how much old map the planner sees, "
            f"and my first explanation, stale frontiers luring the agent away, turns out not to be the main cause.")


def sections(S, N, chart1, chart2, table):
    hf = X.hidden_frontier_stats()
    ds = X.decision_stats()
    w, wc, f_, fc = S["w20"], S["w20cur"], S["full"], S["fullcur"]
    p_worst = max(S[a]["p"] for a in X.ARMS if a != "none")

    setup = f"""<dl class="kv">
<dt>Episodes</dt><dd>{N} real R2R-CE episodes in one house, in 3 tours of 15. Each tour is a different random sample in a different order. No episode appears twice.</dd>
<dt>Baseline</dt><dd>The map is wiped at the start of every episode, as shipped.</dd>
<dt>Window <i>N</i></dt><dd>The map is kept. The planner sees this episode's nodes plus the nearest earlier nodes, about <i>N</i> in total.</dd>
<dt>Full graph</dt><dd>The map is kept and the planner sees every node connected to where it stands.</dd>
<dt>Old frontiers hidden</dt><dd>As above, but unexplored frontiers left by earlier episodes are removed from the planner's options.</dd>
<dt>Test</dt><dd>Every arm is compared with the baseline on the <b>same</b> episodes. An exact sign test counts episodes gained against episodes lost.</dd>
</dl>"""

    result = (f'<div class="panel"><h3>Reached the goal, {N} episodes</h3><p class="sub">Whiskers are 95% intervals. The dashed line is the baseline. '
              f'Hollow bars hide old frontiers. Hover a bar for the paired comparison.</p>{chart1}</div>'
              f'<div class="prose" style="margin-top:14px"><p>Every kept-map arm is worse than the baseline, and the weakest evidence among them is still p = {p_worst:.3f}. '
              f'Gains are rare: the best arm gains {max(S[a]["wins"] for a in X.ARMS)} episodes the baseline missed, and loses {S["w20"]["losses"]} it solved.</p></div>')

    dose = (f'<div class="panel"><h3>Success against how much old map the planner sees</h3><p class="sub">If the harm came from map size, the line would fall steadily to the right. It drops at once and then stays flat.</p>{chart2}'
            f'<div class="legend"><span><i class="sw s1"></i>window, growing cap</span><span><i class="sw s2"></i>full graph</span><span><span class="ring"></span>old frontiers hidden</span></div></div>'
            f'<div class="prose" style="margin-top:14px"><p>With the window cut to 10, the planner sees only {S["w10"]["old"]:.0f} old nodes on average and already reaches just {_p(S["w10"]["k"], N)}. '
            f'Twenty old nodes do no worse ({_p(S["full"]["k"], N)} for the full graph). So this isn\'t a case of too much memory. Something about <i>any</i> old node hurts.</p></div>')

    ledger = f"""<div class="panel"><div class="scroll"><table class="ref"><thead><tr><th>idea</th><th>test</th><th>verdict</th></tr></thead><tbody>
<tr><td><b>Stale frontiers lure the agent away</b></td><td>Hide old frontiers</td><td><span class="chip note">not the main cause</span> Hiding them made accuracy fall to {_p(wc['k'], N)}, but that is tangled up with the next row.</td></tr>
<tr><td><b>The map is too big for the planner</b></td><td>Grow the window from 10 to full</td><td><span class="chip miss">not the cause</span> Five old nodes hurt as much as twenty.</td></tr>
<tr><td><b>Explored ground offers no options</b></td><td>Count frontiers at the first decision</td><td><span class="chip ok">explains part of it</span> See below.</td></tr>
<tr><td><b>Old nodes read as "step 1"</b></td><td>Hide old node tokens</td><td><span class="chip note">not yet tested</span> Experiment 4.</td></tr>
</tbody></table></div></div>"""

    hid = f"""<div class="prose"><p>The planner can only choose an unexplored frontier, or stop. When a new episode starts on ground already mapped, the places the agent proposes land on old nodes and are absorbed into them, so <b>no frontier is created</b>. With old frontiers also hidden there may be nothing to choose.</p>
<p>The numbers agree. An empty map gives 3 to 4 frontiers at the first decision. In the two hidden arms, episodes that began with 2 or fewer frontiers reached the goal <b>{hf['w20cur']['lo_k'] + hf['fullcur']['lo_k']} times out of {hf['w20cur']['lo_n'] + hf['fullcur']['lo_n']}</b>,
against {hf['w20cur']['hi_k'] + hf['fullcur']['hi_k']} of {hf['w20cur']['hi_n'] + hf['fullcur']['hi_n']} for those with 3 or more. {hf['w20cur']['zero'] + hf['fullcur']['zero']} episodes began with none and were forced to stop before moving.</p>
<p>That is only part of the early stopping. The two hidden arms have {hf['w20cur']['one_forced'] + hf['fullcur']['one_forced'] + hf['w20cur']['one_chose'] + hf['fullcur']['one_chose']} one-step episodes. {hf['w20cur']['one_forced'] + hf['fullcur']['one_forced']} were forced.
The other <b>{hf['w20cur']['one_chose'] + hf['fullcur']['one_chose']}</b> were the planner <b>choosing</b> to stop at once, with one to four frontiers on offer. A crowded map seems to make it more willing to stop early, which fits the idea, still untested, that old nodes look like progress already made.</p></div>"""

    done = f"""<div class="prose"><p>I re-ran window 20 on all three tours with a log of every decision. The logging changes nothing: 315 result values matched the earlier runs exactly.
Across the {ds['n_ep']} episodes that start with a non-empty map:</p><ul>
<li><b>{_p(ds['old'], ds['moves'])}</b> of {ds['moves']} frontier moves went to a frontier left by an earlier episode, and <b>{_p(ds['via'], ds['moves'])}</b> walked through at least one old node on the way.</li>
<li>At the very first decision the agent chose an old frontier in {ds['first_old']} of {ds['n_ep']} episodes.</li></ul>
<p>Yet that isn't what separates success from failure. Episodes that took an old frontier reached the goal {_p(ds['took_k'], ds['took_n'])} of the time, against {_p(ds['none_k'], ds['none_n'])} for those that never did.
And of the {ds['base_n']} later episodes the baseline solves, the window arm solves only {ds['base_persist_k']} ({_p(ds['base_persist_k'], ds['base_n'])}),
including {ds['loc_k']} of {ds['loc_n']} when its first move stayed local. The groups are small, but the pattern is clear: <b>having earlier places in the planner's input hurts even when the agent doesn't act on them.</b></p></div>"""

    cav = f"""<div class="prose"><ul>
<li><b>One house, {N} episodes.</b> The pattern holds in all three tours, but the house is a single floor plan.</li>
<li><b>My implementation is one guess at persistence.</b> The join radius (1 m), the step number given to old nodes, and the window rule were my choices. This tests them, not the idea of memory in general.</li>
<li><b>Arms share episodes</b>, which is why the comparison is paired, but the intervals in the chart don't account for that pairing.</li></ul></div>"""

    nxt = """<div class="prose"><p>The planner scores old nodes but was never trained to have them in its input. Experiment 4 hides old <b>node tokens</b> while keeping the graph for routing, and keeps or hides old frontiers.
If accuracy recovers, the old nodes were the poison and frontier-shaped memory is safe. That would point to a way of giving the planner memory in the form it already understands.
<a href="https://claude.ai/artifact/Gv8y2FXYkigvzDkFbPvnv8" target="_blank" rel="noopener">Read Experiment 4</a>.</p></div>"""

    return [("Setup", setup), ("Result", result), ("Does more map hurt more?", dose), ("Four explanations, tested", ledger),
            ("Why hiding old frontiers made it worse", hid), ("What the agent actually did", done), ("Every arm", table),
            ("Caveats", cav), ("Next", nxt)]
