"""Build the comparison scoreboard (one HTML page) from result files.

    bash lidar_bedlam/scripts/pull_helma_results.sh
    uv run python lidar_bedlam/scripts/build_scoreboard.py --out /tmp/scoreboard.html

Sources, in order of preference per run: ``outputs/helma/eval/<run>-last.json``
(re-evaluation on the full validation sets, has the absolute MPJPE), else
the last validation line of ``outputs/helma/runs/<run>/metrics.jsonl``.
Published pipelines come from ``outputs/baselines/results_static.json`` and
never change; the mirror test from ``results_mirror.json`` when present.
Rows of one group are ranked per column (green / yellow / red = best /
second / third; lower is better except mAP).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SPLITS = (("W", "waymo_val"), ("S", "sloper4d_test"))
METRICS = ("mpjpe", "pa_mpjpe", "pve", "abs_mpjpe", "transl_err_m", "map")
LABELS = ("MPJPE", "PA", "PVE", "abs", "transl", "mAP")
HIGHER_IS_BETTER = {"map"}

STATIC_ROWS = [
    ("hmr2-full", "HMR2.0 (4D Humans)", "image only"),
    ("tokenhmr-tight", "TokenHMR", "image only, tight crop"),
    ("camerahmr-full", "CameraHMR", "image only, GT intrinsics"),
    ("lidar-hmr", "LiDAR-HMR", "LiDAR only, Waymo weights"),
]

# group title, description, [(label, [run names to average])]
GROUPS: list[tuple[str, str, list[tuple[str, list[str]]]]] = [
    (
        "Headline: final schedule (150k steps) against the published pipelines",
        "Our runs use the 50/40/10 synthetic / Waymo / SLOPER4D mixture; "
        "full-main-mixed is the headline run, the others change one thing.",
        [
            ("ours · main-mixed", ["full-main-mixed-000"]),
            ("ours · mix80 (80/10/10)", ["full-mix80-000"]),
            ("ours · synth-only", ["full-synth-only-000"]),
            ("ours · real-only", ["full-real-only-000"]),
            ("ours · + pseudo-GT Waymo", ["full-pseudo-waymo-000"]),
            ("ours · + 3DPW mesh-LiDAR", ["full-3dpw-000"]),
            ("ours · gate none (plain sum)", ["full-gate-none-000"]),
            ("ours · gate hard (fixed priors)", ["full-gate-hard-000"]),
            ("ours · image only", ["full-image-only-000"]),
            ("ours · LiDAR only", ["full-lidar-only-000"]),
            (
                "ours · LiDAR at the SLOPER4D rig pose",
                ["full-rig-sloper4d-000"],
            ),
            ("ours · LiDAR at the Waymo rig pose", ["full-rig-waymo-000"]),
            ("ours · ball 0.25 m", ["full-ball025-000"]),
            ("ours · synthetic pool 2x (9k records)", ["full-scale-2x-000"]),
            ("ours · synthetic pool 16x", ["full-scale-16x-000"]),
            ("ours · Waymo resolution only", ["full-target-waymo-000"]),
            ("ours · main-mixed, seed 1", ["full-main-mixed-s1"]),
        ],
    ),
    (
        "Synthesis axis at the final schedule (150k steps or early stop)",
        "How the LiDAR is simulated on the synthetic records; the main run "
        "draws the sensor inside a 1 m ball and sweeps 12 resolutions.",
        [
            ("ball 1 m, 12 resolutions (main-mixed)", ["full-main-mixed-000"]),
            ("ball 1 m, 12 resolutions, seed 1", ["full-main-mixed-s1"]),
            ("LiDAR at the SLOPER4D rig pose", ["full-rig-sloper4d-000"]),
            ("LiDAR at the Waymo rig pose", ["full-rig-waymo-000"]),
            ("ball 0.25 m", ["full-ball025-000"]),
            ("Waymo resolution only", ["full-target-waymo-000"]),
        ],
    ),
    (
        "Fusion axis (1/3 schedule, 17 epochs, mean of 2 seeds)",
        "Reference abl-mixed-short: learned gates. Everything below is the "
        "same mixture and schedule with the gate changed.",
        [
            (
                "learned gates (reference)",
                ["abl-mixed-short-001", "abl-mixed-short-s1"],
            ),
            (
                "gate none (plain sum)",
                ["abl-gate-none-000", "abl-gate-none-s1"],
            ),
            (
                "gate hard (fixed priors)",
                ["abl-gate-hard-000", "abl-gate-hard-s1"],
            ),
            ("image only", ["abl-image-only-000", "abl-image-only-s1"]),
            ("LiDAR only", ["abl-lidar-only-000", "abl-lidar-only-s1"]),
        ],
    ),
    (
        "Synthesis axis (1/3 schedule)",
        "How the LiDAR is simulated on the synthetic records.",
        [
            (
                "ball 1 m, 12 resolutions (reference)",
                ["abl-mixed-short-001", "abl-mixed-short-s1"],
            ),
            ("ball 0.25 m", ["abl-ball025-000"]),
            ("LiDAR at the Waymo rig pose", ["abl-rig-waymo-000"]),
            ("Waymo resolution only", ["abl-target-waymo-000"]),
        ],
    ),
    (
        "Data axis (fixed 3,468 steps)",
        "Synthetic pool capped at k x the Waymo train count; the reference "
        "is the full pool (85x).",
        [
            ("2x (9k records)", ["abl-scale-2x-001"]),
            ("4x", ["abl-scale-4x-001"]),
            ("8x", ["abl-scale-8x-001"]),
            ("16x", ["abl-scale-16x-001"]),
            ("32x", ["abl-scale-32x-001"]),
            (
                "full pool (reference)",
                ["abl-mixed-short-001", "abl-mixed-short-s1"],
            ),
        ],
    ),
    (
        "Label sources (1/3 schedule)",
        "Extra supervision on top of the reference mixture.",
        [
            ("reference", ["abl-mixed-short-001", "abl-mixed-short-s1"]),
            ("+ pseudo-GT SMPL on Waymo", ["abl-pseudo-waymo-000"]),
            ("+ 3DPW mesh-LiDAR 10 %", ["abl-3dpw-000"]),
        ],
    ),
]

# numbers reported by other papers on their own protocol (no code or
# weights to run on our records): SLOPER4D paper, Table 4a, LiDARCap
# trained on SLOPER4D / on LiDARHuman26M + SLOPER4D, tested on SLOPER4D
REPORTED_ROWS = [
    (
        "LiDARCap (SLOPER4D-trained)",
        "LiDAR only, reported in the SLOPER4D paper, own protocol",
        {"S_mpjpe": 86.1, "S_pa_mpjpe": 65.1},
    ),
    (
        "LiDARCap (LH26M + SLOPER4D)",
        "LiDAR only, reported in the SLOPER4D paper, own protocol",
        {"S_mpjpe": 79.2, "S_pa_mpjpe": 60.1},
    ),
]

MIRROR_ROWS = [
    ("lidar-hmr", "LiDAR-HMR", "as is"),
    ("lidar-hmr-mirror", "LiDAR-HMR", "mirrored input"),
    ("camerahmr-full", "CameraHMR", "as is"),
    ("camerahmr-mirror", "CameraHMR", "mirrored input"),
]


@dataclass
class Row:
    """One table row: label, provenance note and metric values per split."""

    label: str
    note: str
    kind: str  # static | ours
    values: dict[str, float | None] = field(default_factory=dict)


def _from_results(res: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for tag, split in SPLITS:
        r = res.get(split, {})
        for m in METRICS:
            v = r.get(m)
            out[f"{tag}_{m}"] = float(v) if v is not None else None
    return out


def _from_metrics_line(line: dict[str, Any]) -> dict[str, float | None]:
    keys = {
        "mpjpe": "mpjpe",
        "pa_mpjpe": "pa_mpjpe",
        "abs_mpjpe": "abs_mpjpe",
        "transl_err_m": "transl_err",
        "map": "map",
    }
    out: dict[str, float | None] = {}
    for tag, split in SPLITS:
        for m, k in keys.items():
            v = line.get(f"val/{split}/{k}")
            out[f"{tag}_{m}"] = float(v) if v is not None else None
    return out


def load_run(
    root: Path, run: str
) -> tuple[dict[str, float | None], str] | None:
    """Values and a note for one run, or None when nothing exists yet."""
    ev = root / "eval" / f"{run}-last.json"
    if ev.exists():
        payload = json.loads(ev.read_text())
        step = payload.get("step", -1)
        return _from_results(
            payload["results"]
        ), f"final, step {step // 1000}k"
    metrics = root / "runs" / run / "metrics.jsonl"
    if not metrics.exists():
        return None
    last = None
    for ln in metrics.read_text().splitlines():
        if '"val/' in ln:
            last = json.loads(ln)
    if last is None:
        return None
    done = (root / "runs" / run / "DONE").exists()
    state = "final" if done else "running"
    values = _from_metrics_line(last)
    if values.get("W_abs_mpjpe") is None:
        # logged before the placement fix of 2026-09-14 (unlabelled hips
        # inflated the Waymo placement error): not comparable, wait for the
        # re-evaluation
        values["W_transl_err_m"] = None
    return values, f"{state}, step {int(last['step']) // 1000}k"


def average(parts: list[dict[str, float | None]]) -> dict[str, float | None]:
    """Mean per key over runs, ignoring missing values."""
    out: dict[str, float | None] = {}
    for k in parts[0]:
        vals = [p[k] for p in parts if p.get(k) is not None]
        out[k] = (
            sum(v for v in vals if v is not None) / len(vals) if vals else None
        )
    return out


def group_rows(root: Path, spec: list[tuple[str, list[str]]]) -> list[Row]:
    """Rows of one group (runs without any result are left out)."""
    rows = []
    for label, runs in spec:
        loaded = [(r, load_run(root, r)) for r in runs]
        found = [(r, x) for r, x in loaded if x is not None]
        if not found:
            continue
        values = average([x[0] for _, x in found])
        notes = sorted({x[1] for _, x in found})
        note = "; ".join(notes)
        if len(found) > 1:
            note = f"mean of {len(found)} seeds; " + note
        if len(found) < len(runs):
            note += f" ({len(runs) - len(found)} seed pending)"
        rows.append(Row(label, note, "ours", values))
    return rows


def static_rows(path: Path, spec: list[tuple[str, str, str]]) -> list[Row]:
    """Rows from a score_baselines results file."""
    if not path.exists():
        return []
    res = json.loads(path.read_text())
    rows = []
    for key, label, note in spec:
        if key in res:
            rows.append(Row(label, note, "static", _from_results(res[key])))
    return rows


def fmt(key: str, v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    if key.endswith("transl_err_m"):
        return f"{v:.3f} m"
    if key.endswith("map"):
        return f"{v:.2f}"
    return f"{v:.1f}"


def ranks(rows: list[Row], key: str) -> dict[int, int]:
    """Row index -> rank (0..2) of the best three values in a column."""
    higher = key.split("_", 1)[1] in HIGHER_IS_BETTER
    have = [(i, r.values.get(key)) for i, r in enumerate(rows)]
    valid = [(i, v) for i, v in have if v is not None and np.isfinite(v)]
    valid.sort(key=lambda t: -t[1] if higher else t[1])
    return {i: pos for pos, (i, _) in enumerate(valid[:3])}


def _keys() -> list[str]:
    return [f"{tag}_{m}" for tag, _ in SPLITS for m in METRICS]


def table_html(rows: list[Row]) -> str:
    """One ranked table."""
    keys = _keys()
    rank_by_key = {k: ranks(rows, k) for k in keys}
    cls = ["g", "y", "r"]
    h = [
        '<div class="wrap"><table><thead><tr><th></th>'
        '<th class="group" colspan="6">Waymo val · 894</th>'
        '<th class="group" colspan="6">SLOPER4D test · 9,904</th></tr><tr><th>method</th>'
    ]
    h.append("".join(f"<th>{lbl}</th>" for _ in SPLITS for lbl in LABELS))
    h.append("</tr></thead><tbody>")
    prev = None
    for i, r in enumerate(rows):
        div = " divider" if prev == "static" and r.kind == "ours" else ""
        h.append(
            f'<tr class="{r.kind}{div}"><td class="name">{r.label}'
            f"<small>{r.note}</small></td>"
        )
        for k in keys:
            rk = rank_by_key[k].get(i)
            c = f" {cls[rk]}" if rk is not None else ""
            h.append(f'<td class="num{c}">{fmt(k, r.values.get(k))}</td>')
        h.append("</tr>")
        prev = r.kind
    h.append("</tbody></table></div>")
    return "".join(h)


STYLE = """
:root{--ground:#F6F8F6;--panel:#FFFFFF;--ink:#1A2421;--muted:#5E6A65;--rule:#D5DCD8;--accent:#0E6B64;
--g-bg:#D9F0DD;--g-ink:#166534;--y-bg:#FBF0C2;--y-ink:#7A5A00;--r-bg:#F9DBD5;--r-ink:#9C3323;--ours:#EAF3F1}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ground:#131917;--panel:#1B2320;--ink:#E7ECE9;--muted:#9BA8A2;--rule:#2B3733;--accent:#5FC7BC;
--g-bg:#1D3F28;--g-ink:#8FE0A5;--y-bg:#453A0F;--y-ink:#F1D46B;--r-bg:#4A2019;--r-ink:#F3A08E;--ours:#1E2B28}}
:root[data-theme="dark"]{--ground:#131917;--panel:#1B2320;--ink:#E7ECE9;--muted:#9BA8A2;--rule:#2B3733;--accent:#5FC7BC;
--g-bg:#1D3F28;--g-ink:#8FE0A5;--y-bg:#453A0F;--y-ink:#F1D46B;--r-bg:#4A2019;--r-ink:#F3A08E;--ours:#1E2B28}
body{background:var(--ground);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.5;padding-inline:24px;padding-block:40px 64px}
main{max-width:1240px;margin:0 auto;display:grid;gap:40px}
h1{font-family:"Fraunces","Iowan Old Style",Georgia,serif;font-weight:600;font-size:2.1rem;line-height:1.15;margin:0;text-wrap:balance;letter-spacing:-0.01em}
h2{font-family:"Fraunces",Georgia,serif;font-weight:600;font-size:1.3rem;margin:0 0 4px}
p{margin:0;max-width:72ch}
.lede,.desc{color:var(--muted)}
.desc{margin-bottom:12px;font-size:.92rem}
.eyebrow{font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);font-weight:600;margin-bottom:8px}
.legend{display:flex;flex-wrap:wrap;gap:10px 18px;font-size:.85rem;color:var(--muted);align-items:center}
.chip{display:inline-block;padding:2px 10px;border-radius:999px;font-family:"IBM Plex Mono",monospace;font-size:.8rem;font-weight:500}
.chip.g{background:var(--g-bg);color:var(--g-ink)}.chip.y{background:var(--y-bg);color:var(--y-ink)}.chip.r{background:var(--r-bg);color:var(--r-ink)}
.wrap{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--panel)}
table{border-collapse:collapse;width:100%;min-width:900px;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap}
th{font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600;background:var(--panel)}
th.group{text-align:center;border-bottom:none;padding-bottom:2px;color:var(--ink);letter-spacing:.14em}
td:first-child,th:first-child{text-align:left}
td.name{font-weight:500}td.name small{display:block;font-weight:400;color:var(--muted);font-size:.76rem}
td.num{font-family:"IBM Plex Mono",monospace;font-size:.88rem}
tr:last-child td{border-bottom:none}tr.ours td{background:var(--ours)}tr.divider td{border-top:2px solid var(--accent)}
td.g{background:var(--g-bg)!important;color:var(--g-ink);font-weight:500}
td.y{background:var(--y-bg)!important;color:var(--y-ink);font-weight:500}
td.r{background:var(--r-bg)!important;color:var(--r-ink);font-weight:500}
.notes{display:grid;gap:8px;font-size:.88rem;color:var(--muted);max-width:80ch}.notes b{color:var(--ink);font-weight:600}
@media (max-width:600px){body{padding-inline:16px}h1{font-size:1.6rem}}
"""

FONTS = (
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Sans:wght@400;500;600"
    '&family=IBM+Plex+Mono:wght@400;500&display=swap">'
)


def build(root: Path, baselines: Path, stamp: str) -> str:
    """The whole page."""
    sections = []
    for i, (title, desc, spec) in enumerate(GROUPS):
        rows = group_rows(root, spec)
        if i == 0:
            reported = [
                Row(label, note, "static", dict.fromkeys(_keys()) | dict(vals))
                for label, note, vals in REPORTED_ROWS
            ]
            rows = (
                static_rows(baselines / "results_static.json", STATIC_ROWS)
                + reported
                + rows
            )
        if not rows:
            continue
        sections.append(
            f"<section><h2>{title}</h2><p class='desc'>{desc}</p>{table_html(rows)}</section>"
        )
    mirror = static_rows(baselines / "results_mirror.json", MIRROR_ROWS)
    if mirror:
        sections.append(
            "<section><h2>Mirror test (memorisation check)</h2>"
            "<p class='desc'>Inputs mirrored left/right (points x → −x, image "
            "flipped with the principal point), predictions mirrored back with "
            "the SMPL left/right vertex map. A model that memorised the "
            "validation records loses far more than one that generalises; "
            "CameraHMR, which never saw either set, is the control.</p>"
            + table_html(
                [Row(r.label, r.note, "static", r.values) for r in mirror]
            )
            + "</section>"
        )
    body = "\n".join(sections)
    return f"""<title>LiDAR-BEDLAM Scoreboard</title>
{FONTS}
<style>{STYLE}</style>
<main>
<header>
  <div class="eyebrow">{stamp}</div>
  <h1>Placement, box and pose against the published pipelines</h1>
  <p class="lede">Every row is scored on the same records with the same protocol: Waymo val (894 crops, 13 COCO joints, hip-centre placement) and the full SLOPER4D test split (9,904 crops, 24 SMPL joints, SMPL translation). MPJPE, PA-MPJPE and PVE (per-vertex, SMPL labels only) in mm are root-relative; abs is the mean joint error as predicted, pose and placement together, which exposes the depth ambiguity of image-only methods; placement in metres; mAP over 3D box IoU 0.25/0.5/0.7 (mesh-only methods get a box from the mesh extent, scaled to Waymo's padded box convention).</p>
</header>
<section><div class="legend"><span>Per column, within each table:</span>
<span class="chip g">best</span><span class="chip y">second</span><span class="chip r">third</span>
<span>· lower is better except mAP · shaded rows are our runs · the published pipelines are static</span></div></section>
{body}
<section class="notes">
<p><b>Image baselines</b> get our 256 px crop and the crop's true intrinsics; the weak-perspective camera is converted to metric translation with the real principal point. CameraHMR is given the ground-truth intrinsics instead of estimating them.</p>
<p><b>LiDAR-HMR</b> runs the Waymo release weights on 1,024 points centred on the box centre in a z-up frame; SLOPER4D is out of its training domain.</p>
<p><b>Our runs</b>: rows marked final are re-evaluated from <code>last.pt</code> on the full sets; running rows show their latest in-training evaluation; their Waymo placement and abs MPJPE appear once the run is re-evaluated with the current protocol (runs started before the placement fix logged an inflated Waymo placement). Ablation tables use the one-third schedule of the reference <code>abl-mixed-short</code> (17 epochs = 3,468 steps); the data axis fixes 3,468 steps on capped pools.</p>
</section>
</main>
"""


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("outputs/helma"))
    ap.add_argument(
        "--baselines", type=Path, default=Path("outputs/baselines")
    )
    ap.add_argument("--stamp", default="")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(args.root, args.baselines, args.stamp))
    sys.stdout.write(f"wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
