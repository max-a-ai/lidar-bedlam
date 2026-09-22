"""The scoreboard as a compact, phone-first interactive page.

Same runs, baselines, groups and schedule as :mod:`build_scoreboard`, read
through its loaders so every number matches; only the presentation
differs. One metric is shown at a time across the three evaluation splits
(a row is four cells wide), groups fold like chapters, a tapped row opens a
card with every metric and the validation curves, and a "side by side"
switch renders the full table for wide screens (on by default from 800 px,
which an iPhone 12 Pro in landscape reaches).

The data is embedded as JSON; the page is self-contained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lidar_bedlam.scripts import build_scoreboard as bs

SPARK_METRICS = ("mpjpe", "transl_err", "map")  # keys in metrics.jsonl
SPARK_POINTS = 40  # per series, subsampled evenly


def curves(root: Path, run: str) -> dict[str, list[list[float]]]:
    """Validation curves of one run: ``"<split tag>_<metric>"`` -> list of
    [step, value], at most :data:`SPARK_POINTS` points."""
    path = root / "runs" / run / "metrics.jsonl"
    if not path.exists():
        return {}
    series: dict[str, list[list[float]]] = {}
    for ln in path.read_text().splitlines():
        if '"val/' not in ln:
            continue
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        for tag, split in bs.SPLITS:
            for m in SPARK_METRICS:
                v = d.get(f"val/{split}/{m}")
                if v is None:
                    continue
                series.setdefault(f"{tag}_{m}", []).append(
                    [float(d["step"]), round(float(v), 4)]
                )
    for k, pts in series.items():
        if len(pts) > SPARK_POINTS:
            step = len(pts) / SPARK_POINTS
            keep = [pts[int(i * step)] for i in range(SPARK_POINTS - 1)]
            series[k] = [*keep, pts[-1]]
    return series


def live_rows(
    squeue: Path | None, root: Path, configs: Path
) -> list[dict[str, Any]]:
    """The queue as the running table sees it (step, target, eta, trend)."""
    from lidar_bedlam.scripts.running_table import collect

    if squeue is None or not squeue.exists() or not squeue.read_text().strip():
        return []
    out = []
    for r in collect(squeue.read_text(), root / "runs", configs, 1):
        out.append(
            {
                "name": r.name,
                "step": r.step,
                "target": r.target,
                "kind": r.target_kind,
                "eta": bs._eta_text(r.eta_s),
                "queued": r.queued,
                "state": r.state,
                "trends": {
                    k: None if v is None else [v[0], v[1]]
                    for k, v in r.trends.items()
                },
            }
        )
    return out


def _num(v: float | None) -> float | None:
    if v is None or v != v:
        return None
    return round(float(v), 3)


def page_data(
    root: Path,
    baselines: Path,
    stamp: str,
    squeue: Path | None,
    configs: Path,
) -> dict[str, Any]:
    """Everything the page renders, as one JSON-ready dict."""
    groups: list[dict[str, Any]] = []
    for i, g in enumerate(bs.GROUPS):
        rows = bs.group_rows(root, g.rows, g.pending)
        if i == 0:
            reported = [
                bs.Row(
                    label,
                    note,
                    "static",
                    dict.fromkeys(bs._keys()) | dict(vals),
                )
                for label, note, vals in bs.REPORTED_ROWS
            ]
            rows = (
                bs.static_rows(
                    baselines / "results_static.json", bs.STATIC_ROWS
                )
                + reported
                + rows
            )
        if not rows:
            continue
        runs_of: dict[str, list[str]] = dict(g.rows)
        out_rows: list[dict[str, Any]] = []
        for r in rows:
            runs: list[str] = runs_of.get(r.label, [])
            out_rows.append(
                {
                    "label": r.label,
                    "note": r.note,
                    "kind": r.kind,
                    "retrain": r.retrain,
                    "labels": r.labels,
                    "footnote": r.footnote,
                    "run": runs[0] if runs else None,
                    "v": {k: _num(v) for k, v in r.values.items()},
                }
            )
        groups.append({"title": g.title, "desc": g.desc, "rows": out_rows})
    run_names: list[str] = sorted(
        {str(r["run"]) for g in groups for r in g["rows"] if r["run"]}
    )
    return {
        "stamp": stamp,
        "splits": [
            {"tag": t, "name": n, "short": s}
            for (t, n), s in zip(
                bs.SPLITS, ("Waymo", "SLOPER4D", "3DPW"), strict=True
            )
        ],
        "metrics": [
            {"key": m, "label": lbl, "higher": m in bs.HIGHER_IS_BETTER}
            for m, lbl in zip(bs.METRICS, bs.LABELS, strict=True)
        ],
        "groups": groups,
        "curves": {run: c for run in run_names if (c := curves(root, run))},
        "live": live_rows(squeue, root, configs),
        "queue_known": squeue is not None
        and squeue.exists()
        and bool(squeue.read_text().strip()),
        "schedule": [
            {"block": b, "runs": r, "sched": s, "why": w}
            for b, r, s, w in bs.SCHEDULE
        ],
    }


def build_pocket(
    root: Path,
    baselines: Path,
    stamp: str,
    squeue: Path | None = None,
    configs: Path = Path("configs"),
) -> str:
    """The whole page."""
    data = page_data(root, baselines, stamp, squeue, configs)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return HTML.replace("__DATA__", payload)


HTML = r"""<title>LiDAR-BEDLAM Scoreboard</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--ground:#F6F8F6;--panel:#FFFFFF;--ink:#1A2421;--muted:#5E6A65;--rule:#D5DCD8;--accent:#0E6B64;--accent-ink:#FFFFFF;
--g-bg:#D9F0DD;--g-ink:#166534;--y-bg:#FBF0C2;--y-ink:#7A5A00;--r-bg:#F9DBD5;--r-ink:#9C3323;--b-bg:#D6E6FA;--b-ink:#1D4E9C;--ours:#EAF3F1;--grey-bg:#ECEDEC;--grey-ink:#7A827E;--chip:#E3ECE9;--spark:#0E6B64;--spark-dim:#B7CBC6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ground:#131917;--panel:#1B2320;--ink:#E7ECE9;--muted:#9BA8A2;--rule:#2B3733;--accent:#5FC7BC;--accent-ink:#0F1A17;
--g-bg:#1D3F28;--g-ink:#8FE0A5;--y-bg:#453A0F;--y-ink:#F1D46B;--r-bg:#4A2019;--r-ink:#F3A08E;--b-bg:#1B3556;--b-ink:#9CC4F5;--ours:#1E2B28;--grey-bg:#202422;--grey-ink:#7E8783;--chip:#22302C;--spark:#5FC7BC;--spark-dim:#2F423E}}
:root[data-theme="dark"]{--ground:#131917;--panel:#1B2320;--ink:#E7ECE9;--muted:#9BA8A2;--rule:#2B3733;--accent:#5FC7BC;--accent-ink:#0F1A17;
--g-bg:#1D3F28;--g-ink:#8FE0A5;--y-bg:#453A0F;--y-ink:#F1D46B;--r-bg:#4A2019;--r-ink:#F3A08E;--b-bg:#1B3556;--b-ink:#9CC4F5;--ours:#1E2B28;--grey-bg:#202422;--grey-ink:#7E8783;--chip:#22302C;--spark:#5FC7BC;--spark-dim:#2F423E}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:14px;line-height:1.4;padding-inline:12px;padding-block:0 48px;margin:0}
main{max-width:720px;margin:0 auto;display:grid;gap:14px}
body.wide main{max-width:1400px}
header.top{position:sticky;top:0;z-index:5;background:var(--ground);padding:12px 0 8px;border-bottom:1px solid var(--rule);display:grid;gap:8px}
h1{font-family:"Fraunces",Georgia,serif;font-weight:600;font-size:1.25rem;margin:0;line-height:1.15;letter-spacing:-0.01em}
.stamp{font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);font-weight:600}
.seg{display:flex;gap:4px;overflow-x:auto;scrollbar-width:none;-webkit-overflow-scrolling:touch}
.seg::-webkit-scrollbar{display:none}
.seg button{flex:0 0 auto;border:1px solid var(--rule);background:var(--panel);color:var(--ink);border-radius:999px;padding:5px 12px;font:inherit;font-size:.8rem;font-weight:500;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
.seg button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
body.wide .seg{display:none}
.tools{display:flex;gap:10px 14px;flex-wrap:wrap;align-items:center;font-size:.78rem;color:var(--muted)}
.tools label{display:flex;gap:5px;align-items:center;cursor:pointer;white-space:nowrap}
.queue{display:flex;gap:6px;flex-wrap:wrap;font-size:.75rem;align-items:center}
.queue .chip{background:var(--chip);border-radius:999px;padding:2px 9px;font-family:"IBM Plex Mono",monospace}
.queue .chip.run{background:var(--g-bg);color:var(--g-ink)}
.queue .chip.warn{background:var(--y-bg);color:var(--y-ink)}
.queue button.chip{border:1px solid var(--rule);cursor:pointer;font:inherit;font-family:"IBM Plex Mono",monospace;color:var(--ink)}
.live{display:grid;gap:4px;font-size:.74rem;font-family:"IBM Plex Mono",monospace;width:100%}
.live div{display:flex;gap:8px;flex-wrap:wrap;align-items:baseline}
.live b{font-family:"IBM Plex Sans",system-ui,sans-serif;font-weight:600}
.live .bar{flex:1 1 80px;height:6px;background:var(--rule);border-radius:3px;overflow:hidden;align-self:center}
.live .bar i{display:block;height:100%;background:var(--accent)}
details.grp{background:var(--panel);border:1px solid var(--rule);border-radius:10px;overflow:hidden}
details.grp>summary{list-style:none;cursor:pointer;padding:10px 12px;display:flex;gap:8px;align-items:baseline;font-family:"Fraunces",Georgia,serif;font-weight:600;font-size:1rem}
details.grp>summary::-webkit-details-marker{display:none}
details.grp>summary .t small{display:block;font-family:"IBM Plex Sans",system-ui,sans-serif;font-weight:400;font-size:.72rem;color:var(--muted);line-height:1.2}
details.grp>summary .n{margin-left:auto;font-family:"IBM Plex Mono",monospace;font-size:.72rem;color:var(--muted);font-weight:400;white-space:nowrap}
details.grp>summary::before{content:"▸";color:var(--accent);font-size:.8rem;transition:transform .15s}
details.grp[open]>summary::before{transform:rotate(90deg)}
details.grp .desc{padding:0 12px 8px;font-size:.78rem;color:var(--muted);margin:0}
.wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th{font-size:.64rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600;padding:6px 8px;text-align:right;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);background:var(--panel);white-space:nowrap}
th:first-child{text-align:left}
th.split{text-align:center;color:var(--ink);letter-spacing:.12em;border-bottom:none;padding-bottom:1px}
th button{all:unset;cursor:pointer}th button:focus-visible{outline:2px solid var(--accent)}
th .dir{color:var(--accent)}
td{padding:7px 6px;border-bottom:1px solid var(--rule);text-align:right;font-family:"IBM Plex Mono",monospace;font-size:.8rem;white-space:nowrap}
td.name{text-align:left;font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.84rem;font-weight:500;white-space:normal;line-height:1.25;max-width:0;width:46%;overflow-wrap:anywhere}
body.wide .wrap{overflow-x:visible}
body.wide table{table-layout:fixed}
body.wide th,body.wide td{padding:4px 2px;font-size:.6rem;letter-spacing:0}
body.wide th{font-size:.5rem;letter-spacing:.02em}
body.wide th:first-child{width:17%}
body.wide td.name{width:auto;max-width:none;min-width:0;white-space:normal;font-size:.66rem;line-height:1.15;overflow-wrap:anywhere}
body.wide td.name small{font-size:.54rem}
body.wide .tag{font-size:.5rem;padding:0 3px;margin-right:2px}
@media (min-width:1100px){body.wide th,body.wide td{padding:5px 6px;font-size:.76rem}body.wide th{font-size:.62rem;letter-spacing:.06em}body.wide td.name{font-size:.84rem}body.wide td.name small{font-size:.68rem}body.wide .tag{font-size:.62rem;padding:0 5px}}
td.name small{display:block;font-weight:400;color:var(--muted);font-size:.68rem;margin-top:1px}
tr.ours td{background:var(--ours)}
tr.retrain td{background:var(--grey-bg)!important;color:var(--grey-ink)}
tr.divider td{border-top:2px solid var(--accent)}
tr.row{cursor:pointer}
td.g{background:var(--g-bg)!important;color:var(--g-ink);font-weight:500}
td.y{background:var(--y-bg)!important;color:var(--y-ink);font-weight:500}
td.r{background:var(--r-bg)!important;color:var(--r-ink);font-weight:500}
td.b{background:var(--b-bg)!important;color:var(--b-ink);font-weight:600}
.tag{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:.62rem;padding:0 5px;border-radius:4px;margin-right:4px;vertical-align:1px}
.tag.yes{background:var(--g-bg);color:var(--g-ink)}.tag.no{background:var(--r-bg);color:var(--r-ink)}
tr.card td{background:var(--ground)!important;white-space:normal;padding:8px 10px 10px;font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.78rem;text-align:left}
.mini{display:grid;grid-template-columns:auto repeat(6,1fr);gap:2px 6px;font-family:"IBM Plex Mono",monospace;font-size:.72rem;margin-top:4px}
.mini .h{color:var(--muted);font-size:.6rem;letter-spacing:.06em;text-transform:uppercase;text-align:right}
.mini .s{color:var(--muted);text-align:left}
.mini .v{text-align:right}
.sparks{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px}
.spark{display:grid;gap:2px;font-size:.66rem;color:var(--muted)}
.spark svg{width:150px;height:40px;display:block}
.spark .cap{display:flex;justify-content:space-between;gap:6px}
.spark .cap b{color:var(--ink);font-family:"IBM Plex Mono",monospace;font-weight:500}
.why{color:var(--r-ink);margin-top:6px}
.fn{color:var(--muted);margin-top:6px}
.legend{font-size:.72rem;color:var(--muted);display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.legend .sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:3px}
table.sched td{white-space:normal;font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.78rem;text-align:left;vertical-align:top}
table.sched td.b{font-weight:600;white-space:nowrap}
@media (max-width:420px){td.name{font-size:.8rem}td.name small{font-size:.64rem}th{padding:6px 5px}}
@media (min-width:600px){body{font-size:15px}td.name{width:46%}}
@media (prefers-reduced-motion: reduce){details.grp>summary::before{transition:none}}
</style>
<main>
<header class="top">
  <div><div class="stamp" id="stamp"></div><h1>LiDAR-BEDLAM Scoreboard</h1></div>
  <div class="seg" id="metric" role="group" aria-label="metric"></div>
  <div class="tools">
    <label><input type="checkbox" id="wide"> all metrics side by side</label>
    <label><input type="checkbox" id="hideRetrain"> hide rows to retrain</label>
    <label><input type="checkbox" id="hideStatic"> hide published pipelines</label>
    <span id="unit"></span>
  </div>
  <div class="queue" id="queue"></div>
</header>
<div class="legend"><span><i class="sw" style="background:var(--g-bg)"></i>best</span><span><i class="sw" style="background:var(--y-bg)"></i>second</span><span><i class="sw" style="background:var(--r-bg)"></i>third</span><span><i class="sw" style="background:var(--b-bg)"></i>beats every pipeline</span><span><i class="sw" style="background:var(--grey-bg)"></i>to retrain</span><span>· <span class="tag yes">✓ v2</span>pseudo-GT mesh on Waymo (v3 = refit with hip-centre term and joint offsets), <span class="tag no">✗ kp</span>keypoints only · tap a row for every metric and its validation curves · tap a column to sort</span></div>
<div id="groups"></div>
<details class="grp" id="sched"><summary>Retraining schedule<span class="n" id="schedN"></span></summary><div class="wrap"><table class="sched"><tbody id="schedBody"></tbody></table></div></details>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const AUTO_WIDE_PX = 800;
const state = {metric: 'mpjpe', sort: {}, open: new Set(), openGroups: [], hideRetrain: false, hideStatic: false, wide: null, queueOpen: false};
try { Object.assign(state, JSON.parse(localStorage.getItem('pocket') || '{}'), {open: new Set()}); } catch (e) {}
function save(){ try { localStorage.setItem('pocket', JSON.stringify({metric: state.metric, sort: state.sort, openGroups: state.openGroups, hideRetrain: state.hideRetrain, hideStatic: state.hideStatic, wide: state.wide})); } catch (e) {} }
function isWide(){ return state.wide === null ? window.innerWidth >= AUTO_WIDE_PX : state.wide; }
const M = Object.fromEntries(D.metrics.map(m => [m.key, m]));
const SM = {mpjpe: 'mpjpe', transl_err_m: 'transl_err', map: 'map'};
function fmt(key, v){ if (v == null) return '–'; if (key === 'transl_err_m') return v.toFixed(3); if (key === 'map') return v.toFixed(2); return v.toFixed(1); }
function unit(key){ return key === 'transl_err_m' ? 'metres, lower is better' : key === 'map' ? '3D box mAP, higher is better' : 'mm, lower is better'; }
function finite(v){ return v != null && Number.isFinite(v); }
function ranks(rows, k, hi){ const valid = rows.map((r, i) => [i, r.v[k]]).filter(([i, v]) => finite(v) && !rows[i].retrain); valid.sort((a, b) => hi ? b[1] - a[1] : a[1] - b[1]); const out = {}; valid.slice(0, 3).forEach(([i], p) => out[i] = p); return out; }
function beats(rows, k, hi){ const ref = rows.filter(r => r.kind === 'static' && finite(r.v[k])).map(r => r.v[k]); if (!ref.length) return new Set(); const bar = hi ? Math.max(...ref) : Math.min(...ref); const s = new Set(); rows.forEach((r, i) => { if (r.kind === 'ours' && finite(r.v[k]) && (hi ? r.v[k] > bar : r.v[k] < bar)) s.add(i); }); return s; }
function splitName(label){ label = label.replace(/^ours · /, ''); let head = label, rest = ''; const cut = Math.min(...[label.indexOf(' ('), label.indexOf(', ')].filter(i => i > 0)); if (Number.isFinite(cut)) { head = label.slice(0, cut); rest = label.slice(cut).replace(/^[ ,(]+/, '').replace(/\)$/, ''); } if (head.length > 28) { const j = head.lastIndexOf(' · '); if (j > 8) { rest = (head.slice(j + 3) + (rest ? ' · ' + rest : '')); head = head.slice(0, j); } } return [head, rest]; }
function nameCell(r){ const [head, rest] = splitName(r.label); const sub = [rest, r.note].filter(Boolean).join(' · '); return `${tag(r)}${head}${sub ? `<small>${sub}</small>` : ''}`; }
function splitTitle(t){ const i = t.indexOf(':'); if (i > 0) return [t.slice(0, i), t.slice(i + 1).trim()]; const j = t.indexOf(' ('); if (j > 0) return [t.slice(0, j), t.slice(j + 2).replace(/\)$/, '')]; return [t, '']; }
function tag(r){ if (r.kind !== 'ours' || r.labels == null) return ''; return r.labels === '' ? '<span class="tag no">✗ kp</span>' : `<span class="tag yes">✓ ${r.labels}</span>`; }
function spark(pts, key){ const w = 150, h = 40, p = 3; const xs = pts.map(q => q[0]), ys = pts.map(q => q[1]); const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys); const sx = x => x1 > x0 ? p + (x - x0) / (x1 - x0) * (w - 2 * p) : w / 2; const sy = y => y1 > y0 ? h - p - (y - y0) / (y1 - y0) * (h - 2 * p) : h / 2; const d = pts.map((q, i) => (i ? 'L' : 'M') + sx(q[0]).toFixed(1) + ' ' + sy(q[1]).toFixed(1)).join(''); const last = pts[pts.length - 1]; const best = key === 'map' ? Math.max(...ys) : Math.min(...ys); return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="validation curve"><line x1="${p}" y1="${sy(best).toFixed(1)}" x2="${w - p}" y2="${sy(best).toFixed(1)}" stroke="var(--spark-dim)" stroke-dasharray="2 3"/><path d="${d}" fill="none" stroke="var(--spark)" stroke-width="1.5"/><circle cx="${sx(last[0]).toFixed(1)}" cy="${sy(last[1]).toFixed(1)}" r="2.5" fill="var(--spark)"/></svg>`; }
function sparks(r){ const c = r.run && D.curves[r.run]; if (!c) return ''; const key = SM[state.metric] || 'mpjpe'; const mk = Object.keys(SM).find(k => SM[k] === key); let h = ''; D.splits.forEach(s => { const pts = c[s.tag + '_' + key]; if (!pts || pts.length < 2) return; const last = pts[pts.length - 1]; h += `<div class="spark"><div class="cap"><span>${s.short} ${M[mk].label}</span><b>${fmt(mk, last[1])} @ ${(last[0] / 1000).toFixed(0)}k</b></div>${spark(pts, key)}</div>`; }); return h ? `<div class="sparks">${h}</div>` : ''; }
function card(r){ const cells = ['<span></span>', ...D.metrics.map(m => `<span class="h">${m.label}</span>`)]; D.splits.forEach(s => { cells.push(`<span class="s">${s.short}</span>`); D.metrics.forEach(m => cells.push(`<span class="v">${fmt(m.key, r.v[s.tag + '_' + m.key])}</span>`)); }); let h = isWide() ? '' : `<div class="mini">${cells.join('')}</div>`; h += sparks(r); if (r.retrain) h += `<div class="why">retrain: ${r.retrain}</div>`; if (r.footnote) h += `<div class="fn">${r.footnote}</div>`; if (r.run) h += `<div class="fn">run ${r.run}</div>`; return h || '<div class="fn">no further detail</div>'; }
function queueHtml(){ if (!D.queue_known) return '<span class="chip warn">queue state unknown: the cluster was not reached at this tick</span>'; if (!D.live.length) return '<span class="chip">queue empty</span>'; const nr = D.live.filter(j => !j.queued).length, np = D.live.length - nr; let h = `<button class="chip sum" id="qtog">${nr} running · ${np} pending ${state.queueOpen ? '▾' : '▸'}</button>`; if (state.queueOpen) { h += '<div class="live">'; D.live.filter(j => !j.queued).forEach(j => { const pct = j.target ? Math.min(100, 100 * j.step / j.target) : 0; const tr = Object.entries(j.trends).filter(([k, v]) => v).map(([k, v]) => `${k} ${v[0].toFixed(1)} (${v[1] >= 0 ? '+' : ''}${v[1].toFixed(1)} %)`).join(' · '); h += `<div><b>${j.name}</b><span>${j.step.toLocaleString()} / ${j.target.toLocaleString()} ${j.kind}</span><span class="bar"><i style="width:${pct.toFixed(0)}%"></i></span><span>${j.eta}</span>${tr ? `<span>${tr}</span>` : ''}</div>`; }); if (np) h += `<div><span>${np} queued, waiting for nodes</span></div>`; h += '</div>'; } return h; }
function render(){
  const wide = isWide(); document.body.classList.toggle('wide', wide); document.getElementById('wide').checked = wide;
  document.getElementById('stamp').textContent = D.stamp;
  document.getElementById('unit').textContent = wide ? 'mm unless stated; transl in m; mAP higher is better' : unit(state.metric);
  const seg = document.getElementById('metric'); seg.innerHTML = '';
  D.metrics.forEach(m => { const b = document.createElement('button'); b.textContent = m.label; b.setAttribute('aria-pressed', m.key === state.metric); b.onclick = () => { state.metric = m.key; save(); render(); }; seg.appendChild(b); });
  const q = document.getElementById('queue'); q.innerHTML = queueHtml(); const qt = document.getElementById('qtog'); if (qt) qt.onclick = () => { state.queueOpen = !state.queueOpen; render(); };
  const host = document.getElementById('groups'); host.innerHTML = '';
  const cols = wide ? D.splits.flatMap(s => D.metrics.map(m => ({k: s.tag + '_' + m.key, hi: m.higher, label: m.label, split: s.short}))) : D.splits.map(s => ({k: s.tag + '_' + state.metric, hi: M[state.metric].higher, label: s.short}));
  D.groups.forEach((g, gi) => {
    let rows = g.rows.map((r, i) => ({...r, i}));
    if (state.hideRetrain) rows = rows.filter(r => !r.retrain);
    if (state.hideStatic) rows = rows.filter(r => r.kind !== 'static');
    if (!rows.length) return;
    const sortKey = state.sort[gi]; const sc = cols.find(c => c.k === sortKey);
    if (sc) rows.sort((a, b) => { const x = a.v[sc.k], y = b.v[sc.k]; if (!finite(x)) return 1; if (!finite(y)) return -1; return sc.hi ? y - x : x - y; });
    const rk = Object.fromEntries(cols.map(c => [c.k, ranks(rows, c.k, c.hi)])); const bl = Object.fromEntries(cols.map(c => [c.k, beats(rows, c.k, c.hi)]));
    const det = document.createElement('details'); det.className = 'grp'; det.open = gi === 0 || state.openGroups.includes(gi);
    det.addEventListener('toggle', () => { const og = new Set(state.openGroups); det.open ? og.add(gi) : og.delete(gi); state.openGroups = [...og]; save(); });
    const [th, tsub] = splitTitle(g.title);
    let h = `<summary><span class="t">${th}${tsub ? `<small>${tsub}</small>` : ''}</span><span class="n">${rows.length} rows</span></summary><p class="desc">${g.desc}</p><div class="wrap"><table><thead>`;
    if (wide) { h += '<tr><th></th>' + D.splits.map(s => `<th class="split" colspan="${D.metrics.length}">${s.short}</th>`).join('') + '</tr>'; }
    h += '<tr><th>method</th>' + cols.map(c => `<th><button data-k="${c.k}">${c.label}${sortKey === c.k ? ' <span class="dir">▾</span>' : ''}</button></th>`).join('') + '</tr></thead><tbody>';
    let prev = null;
    rows.forEach((r, i) => {
      const div = prev === 'static' && r.kind === 'ours' ? ' divider' : ''; prev = r.kind;
      const id = gi + ':' + r.i;
      h += `<tr class="row ${r.kind}${div}${r.retrain ? ' retrain' : ''}" data-id="${id}"><td class="name">${nameCell(r)}</td>`;
      cols.forEach(c => { const p = rk[c.k][i]; const cls = p != null ? ['g', 'y', 'r'][p] : (bl[c.k].has(i) ? 'b' : ''); h += `<td class="${cls}">${fmt(c.k.slice(2), r.v[c.k])}</td>`; });
      h += '</tr>';
      if (state.open.has(id)) h += `<tr class="card"><td colspan="${cols.length + 1}">${card(r)}</td></tr>`;
    });
    h += '</tbody></table></div>';
    det.innerHTML = h; host.appendChild(det);
    det.querySelectorAll('tr.row').forEach(tr => tr.onclick = () => { const id = tr.dataset.id; state.open.has(id) ? state.open.delete(id) : state.open.add(id); render(); });
    det.querySelectorAll('th button').forEach(b => b.onclick = e => { e.stopPropagation(); state.sort[gi] = state.sort[gi] === b.dataset.k ? null : b.dataset.k; save(); render(); });
  });
  document.getElementById('schedBody').innerHTML = D.schedule.map(s => `<tr><td class="b">${s.block}</td><td>${s.runs}<br><small style="color:var(--muted)">${s.sched} · ${s.why}</small></td></tr>`).join('');
  document.getElementById('schedN').textContent = D.schedule.length + ' blocks';
}
document.getElementById('hideRetrain').checked = state.hideRetrain;
document.getElementById('hideStatic').checked = state.hideStatic;
document.getElementById('hideRetrain').onchange = e => { state.hideRetrain = e.target.checked; save(); render(); };
document.getElementById('hideStatic').onchange = e => { state.hideStatic = e.target.checked; save(); render(); };
document.getElementById('wide').onchange = e => { state.wide = e.target.checked; save(); render(); };
let lastAuto = window.innerWidth >= AUTO_WIDE_PX;
window.addEventListener('resize', () => { const now = window.innerWidth >= AUTO_WIDE_PX; if (now !== lastAuto) { lastAuto = now; if (state.wide === null) render(); } });
render();
</script>
"""
