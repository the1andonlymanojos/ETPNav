"""Shared page shell for the persistent-map experiment artifacts: one look, one ladder header, so each experiment
reads as a page of the same lab notebook. Usage: page(...) returns a full HTML fragment (the Artifact tool wraps it)."""
import html

LADDER = [
    (1, "Baseline", "Stock ETPNav, one house"),
    (2, "Mechanics", "Does the persisted map connect up?"),
    (3, "Diagnosis", "What breaks when the map persists?"),
    (4, "Node vs frontier", "Which kind of memory does the harm?"),
    (5, "Reopen places", "Give the planner memory it can use"),
    (6, "More houses", "Scale up"),
]

CSS = """
:root{
  --ground:#edf0f2; --surface:#ffffff; --ink:#16202a; --dim:#576573; --rule:#d2d9df; --soft:#e5e9ed;
  --ok:#0b7a85; --ok-soft:#d9eff0; --miss:#a13b3b; --miss-soft:#f4e0e0; --note:#a35a17; --note-soft:#f5e6d4;
  --seq-early:#c3d3ea; --seq-late:#183c88; --floor:#dde3e8; --floor-edge:#c6ced6; --link:#b45f10;
  --s1:#2b5cb0; --s2:#b8791f; --grid:#e3e8ec;
  --shadow:0 1px 2px rgba(20,30,40,.05), 0 6px 20px rgba(20,30,40,.04);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0d1319; --surface:#141c24; --ink:#e6ebef; --dim:#93a1ad; --rule:#28323c; --soft:#1e2832;
  --ok:#55d3c6; --ok-soft:rgba(85,211,198,.12); --miss:#f09490; --miss-soft:rgba(240,148,144,.12);
  --note:#efa75f; --note-soft:rgba(239,167,95,.12);
  --seq-early:#2a4270; --seq-late:#9dbfff; --floor:#1c262f; --floor-edge:#2a3641; --link:#f0a45d;
  --s1:#7fa7f2; --s2:#e5a04a; --grid:#222d37;
  --shadow:0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.28);}}
:root[data-theme="dark"]{
  --ground:#0d1319; --surface:#141c24; --ink:#e6ebef; --dim:#93a1ad; --rule:#28323c; --soft:#1e2832;
  --ok:#55d3c6; --ok-soft:rgba(85,211,198,.12); --miss:#f09490; --miss-soft:rgba(240,148,144,.12);
  --note:#efa75f; --note-soft:rgba(239,167,95,.12);
  --seq-early:#2a4270; --seq-late:#9dbfff; --floor:#1c262f; --floor-edge:#2a3641; --link:#f0a45d;
  --s1:#7fa7f2; --s2:#e5a04a; --grid:#222d37;
  --shadow:0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.28);}

*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font:15px/1.55 "IBM Plex Sans",system-ui,sans-serif;margin:0;
  padding-inline:20px;padding-block:44px 80px}
.page{max-width:860px;margin:0 auto;display:flex;flex-direction:column;gap:30px}
.mono{font-family:"IBM Plex Mono",ui-monospace,monospace}
h1{font-size:clamp(1.7rem,4.2vw,2.3rem);line-height:1.12;font-weight:650;letter-spacing:-.015em;margin:0;text-wrap:balance}
h2{font:500 .72rem/1 "IBM Plex Mono",monospace;text-transform:uppercase;letter-spacing:.13em;color:var(--dim);margin:0 0 12px}
p{margin:0}
.lead{font-size:1.12rem;line-height:1.5;max-width:64ch}
.lead b{font-weight:600}
.prose{max-width:66ch;color:var(--ink)} .prose p+p{margin-top:.7em} .prose ul{margin:.4em 0 0;padding-left:1.15em} .prose li+li{margin-top:.35em}
.dim{color:var(--dim)}
code{font:500 .86em "IBM Plex Mono",monospace;background:var(--soft);padding:.08em .38em;border-radius:4px}

/* header + ladder: the experiments really are a sequence, so the numbering carries information */
.eyebrow{font:500 .72rem/1 "IBM Plex Mono",monospace;text-transform:uppercase;letter-spacing:.14em;color:var(--ok)}
.ladder{display:flex;flex-wrap:wrap;gap:6px 22px;margin:14px 0 20px;padding:0;list-style:none;font:500 .74rem/1.2 "IBM Plex Mono",monospace}
.ladder li{display:flex;align-items:center;gap:7px;color:var(--dim)}
.ladder .dot{width:9px;height:9px;border-radius:50%;border:1.5px solid var(--dim);background:transparent;flex:none}
.ladder .done{color:var(--ink)} .ladder .done .dot{background:var(--ok);border-color:var(--ok)}
.ladder .current{color:var(--ink);font-weight:600} .ladder .current .dot{border-color:var(--ok);box-shadow:0 0 0 3px var(--ok-soft);background:var(--surface)}

/* setup: label / value pairs, no boxes */
.kv{display:grid;grid-template-columns:max-content 1fr;gap:7px 20px;margin:0;font-size:.92rem}
.kv dt{font:500 .72rem/1.9 "IBM Plex Mono",monospace;text-transform:uppercase;letter-spacing:.09em;color:var(--dim)}
.kv dd{margin:0;max-width:64ch}
@media (max-width:560px){.kv{grid-template-columns:1fr;gap:1px}.kv dd{margin-bottom:9px}}

/* readout: the few numbers that matter, plain */
.readout{display:flex;flex-wrap:wrap;gap:16px 34px;margin:0;padding:2px 0}
.readout div{min-width:88px}
.readout dt{font:500 .68rem/1.3 "IBM Plex Mono",monospace;text-transform:uppercase;letter-spacing:.09em;color:var(--dim)}
.readout dd{margin:2px 0 0;font:600 1.7rem/1.15 "IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
.readout dd small{font-size:.8rem;font-weight:500;color:var(--dim);margin-left:3px}

/* the one raised object: chart / table panels */
.panel{background:var(--surface);border:1px solid var(--rule);border-radius:10px;box-shadow:var(--shadow);padding:16px 18px}
.panel h3{font:600 .95rem/1.3 "IBM Plex Sans",sans-serif;margin:0 0 3px}
.panel .sub{color:var(--dim);font-size:.84rem;margin:0 0 12px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:10px;font-size:.82rem;color:var(--dim)}
.legend span{display:inline-flex;align-items:center;gap:7px}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
.sw.ok{background:var(--ok)} .sw.miss{background:var(--miss)} .sw.s1{background:var(--s1)} .sw.s2{background:var(--s2)}
svg text{font-family:"IBM Plex Mono",monospace;fill:var(--dim)}
.tip{position:fixed;z-index:20;pointer-events:none;max-width:290px;padding:9px 11px;border-radius:8px;font-size:.8rem;line-height:1.4;
  background:var(--ink);color:var(--ground);box-shadow:var(--shadow);opacity:0;transition:opacity .08s}
.tip.on{opacity:1} .tip b{font-weight:600} .tip .m{font-family:"IBM Plex Mono",monospace;font-size:.76rem;opacity:.85}

/* tables */
.scroll{overflow-x:auto}
table.eps{width:100%;border-collapse:collapse;font-size:.86rem;font-variant-numeric:tabular-nums}
table.eps th{font:500 .66rem/1.2 "IBM Plex Mono",monospace;text-transform:uppercase;letter-spacing:.09em;color:var(--dim);text-align:right;
  padding:0 10px 8px;border-bottom:1px solid var(--rule);white-space:nowrap}
table.eps th:nth-child(-n+3),table.eps td:nth-child(-n+3){text-align:left}
table.eps td{padding:8px 10px 2px;text-align:right;font-family:"IBM Plex Mono",monospace;font-size:.82rem;white-space:nowrap}
table.eps tr.i td{padding:0 10px 9px;text-align:left;white-space:normal;font:400 .8rem/1.4 "IBM Plex Sans",sans-serif;color:var(--dim);max-width:0}
table.eps tbody{border-bottom:1px solid var(--soft)} table.eps tbody:last-child{border-bottom:0}
.chip{display:inline-block;padding:1px 8px;border-radius:999px;font:500 .7rem/1.6 "IBM Plex Mono",monospace;letter-spacing:.02em}
.chip.ok{background:var(--ok-soft);color:var(--ok)} .chip.miss{background:var(--miss-soft);color:var(--miss)}
.chip.note{background:var(--note-soft);color:var(--note)}

pre.code{margin:0;padding:12px 14px;background:var(--surface);border:1px solid var(--rule);border-radius:8px;overflow-x:auto;
  font:400 .78rem/1.55 "IBM Plex Mono",monospace;color:var(--ink)}
.foot{font-size:.78rem;color:var(--dim);border-top:1px solid var(--rule);padding-top:14px}
.callout{border-left:3px solid var(--note);background:var(--note-soft);padding:10px 14px;border-radius:0 8px 8px 0;font-size:.9rem;max-width:66ch}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;650&display=swap" rel="stylesheet">')


def ladder(current, done):
    items = []
    for n, name, tip in LADDER:
        cls = "current" if n == current else ("done" if n in done else "")
        items.append(f'<li class="{cls}" title="{html.escape(tip)}"><span class="dot"></span>{n} {html.escape(name)}</li>')
    return '<ol class="ladder" aria-label="Experiment ladder">' + "".join(items) + "</ol>"


def page(title, exp_no, done, lead_html, sections_html, script_js="", reproduce=""):
    """sections_html: list of (heading, inner_html). Returns the full HTML document fragment."""
    body = "".join(f'<section><h2>{html.escape(h)}</h2>{inner}</section>' for h, inner in sections_html)
    foot = (f'<div class="foot">{reproduce}</div>' if reproduce else "")
    return (f'<title>{html.escape(title)}</title>{FONTS}<style>{CSS}</style>'
            f'<div class="page"><header><div class="eyebrow">ETPNav persistent map · Experiment {exp_no} of {len(LADDER)}</div>'
            f'{ladder(exp_no, done)}<h1>{html.escape(title)}</h1></header>'
            f'<p class="lead">{lead_html}</p>{body}{foot}</div><div class="tip" id="tip" role="tooltip"></div>'
            f'<script>{script_js}</script>')


# generic hover tooltip: any element with data-tip="html" shows it; tap works on touch
TIP_JS = """
(function(){var tip=document.getElementById('tip');
function show(e){var t=e.target.closest('[data-tip]');if(!t){tip.classList.remove('on');return;}
tip.innerHTML=t.getAttribute('data-tip');tip.classList.add('on');
var x=e.clientX+14,y=e.clientY+14,w=tip.offsetWidth,h=tip.offsetHeight;
if(x+w>innerWidth-8)x=e.clientX-w-14;if(y+h>innerHeight-8)y=e.clientY-h-14;tip.style.left=Math.max(8,x)+'px';tip.style.top=Math.max(8,y)+'px';}
document.addEventListener('mousemove',show);document.addEventListener('click',show);
document.addEventListener('mouseleave',function(){tip.classList.remove('on');});})();
"""


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h
