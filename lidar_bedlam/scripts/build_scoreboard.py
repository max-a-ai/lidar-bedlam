"""Build the comparison scoreboard (one HTML page) from result files.

    bash lidar_bedlam/scripts/pull_helma_results.sh
    uv run python lidar_bedlam/scripts/build_scoreboard.py --out /tmp/scoreboard.html

Sources, in order of preference per run: ``outputs/helma/eval/<run>-last.json``
(re-evaluation on the full validation sets, has the absolute MPJPE), else
the last validation line of ``outputs/helma/runs/<run>/metrics.jsonl``.
Published pipelines come from ``outputs/baselines/results_static.json``
(its sibling result files fill the splits it lacks) and never change.
Rows of one group are ranked per column (green / yellow / red = best /
second / third; lower is better except mAP). In the headline table a value of
ours that beats every compared pipeline is blue, unless it ranks in the
column's top three, where green, yellow and red take precedence.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SPLITS = (("W", "waymo_val"), ("S", "sloper4d_test"), ("P", "threedpw_test"))
METRICS = ("mpjpe", "pa_mpjpe", "pve", "abs_mpjpe", "transl_err_m", "map")
LABELS = ("MPJPE", "PA", "PVE", "abs", "transl", "mAP")
HIGHER_IS_BETTER = {"map"}

STATIC_ROWS = [
    ("hmr2-full", "HMR2.0 (4D Humans)", "image only"),
    ("tokenhmr-tight", "TokenHMR", "image only, tight crop"),
    ("camerahmr-full", "CameraHMR", "image only, GT intrinsics"),
    ("prompthmr", "PromptHMR", "image only, box prompt, GT intrinsics"),
    (
        "sam3d-body",
        "SAM 3D Body (DINOv3-H+)",
        "image only, GT intrinsics, MHR mesh fitted to SMPL",
    ),
    ("lidar-hmr", "LiDAR-HMR", "LiDAR only, Waymo release weights"),
    (
        "human3r",
        "Human3R",
        "image only, single frame on the crop",
    ),
]

MAIN = "ours · anchor · mix80"
MAIN_RUN = ["a-full-mix80-001"]
MAIN_NOTE = " (main, final schedule)"


@dataclass
class Group:
    """One table: title, description, rows and how it is shown."""

    title: str
    desc: str
    rows: list[tuple[str, list[str]]]
    pending: bool = False  # rows without results are shown as queued
    archive: bool = False  # collapsed at the end of the page


V2_ABL = "15k steps on the main v2 mixture (75/10/10/5), point anchor"

# runs that have to be retrained once the Waymo hand problem is fixed, and
# why; the scoreboard greys their rows and the schedule lists them
HAND = "trained before the Waymo hand fix"
PGT = "Waymo keypoints only; rerun on pseudo-GT v2 with the hand fix"
RETRAIN: dict[str, str] = {
    "a-full-main-v2-prior-000": HAND + "; prior variant of main v2",
    "a-full-mix80-prior-000": HAND + "; prior variant of mix80",
    "a-full-main-mixed-000": HAND + "; mixture row of ablation 2",
    "a-full-synth-only-000": HAND + "; data row of ablation 2",
    "a-full-real-only-000": HAND + "; data row of ablation 2",
    "a-full-lidar-only-000": HAND + "; modality row of ablation 1",
    "full-image-only-000": HAND
    + "; camera-frame translation and 50/40/10 mixture, rerun on the "
    "main v2 mixture",
    "v2-abl-ref-000": PGT,
    "v2-abl-gate-none-000": PGT,
    "v2-abl-gate-hard-000": PGT,
    "v2-abl-lidar-only-000": PGT,
    "v2-abl-image-only-000": PGT,
    "v2-abl-loss-chamfer-000": PGT,
    "v2-abl-loss-icp-000": PGT,
    "v2-abl-loss-mesh-000": PGT,
    "v2-abl-loss-chamfer-icp-000": PGT,
    "v2-abl-loss-icp-mesh-000": PGT,
    "v2-abl-loss-chamfer-icp-mesh-000": PGT,
    "v2-abl-pseudo-pedgen-000": HAND + "; label-source comparison",
    "v2-abl-pseudo-lhmr-000": HAND + "; label-source comparison",
    "v2-scale-2x-000": PGT,
    "v2-scale-4x-000": PGT,
    "v2-scale-8x-000": PGT,
    "v2-scale-16x-000": PGT,
    "v2-scale-32x-000": PGT,
    "v2-scale-full-000": PGT,
}

# what has to be trained, in order, once the hand fix is in: (block, runs,
# schedule, what for). The scoreboard renders it and --schedule-md writes it.
SCHEDULE: list[tuple[str, str, str, str]] = [
    (
        "0 · prerequisite",
        "t-short-real, t-short-simlidar (9k steps each)",
        "short",
        "find the Waymo hand problem: real returns against LiDAR simulated "
        "on the pseudo-GT mesh; no full training before the cause is fixed",
    ),
    (
        "1 · mains",
        "b-full-mix80, b-full-main-v2 (pseudo-GT v2 + fix); "
        "a-full-mix80, a-full-main-v2 (keypoints + fix)",
        "full, early stop",
        "headline rows; b-full-mix80 replaces the run on the first v2 labels",
    ),
    (
        "2 · label source (ablation 5)",
        "v2-abl-ref, v2-abl-pseudo-v2 (new), v2-abl-pseudo-pedgen, "
        "v2-abl-pseudo-lhmr",
        "15k",
        "the pseudo-GT v2 row is missing; the others predate the fix",
    ),
    (
        "3 · fusion and loss (ablations 3, 4)",
        "v2-abl-gate-none/hard, lidar-only, image-only; six v2-abl-loss-*",
        "15k",
        "rerun on pseudo-GT v2 so the axes match the mains",
    ),
    (
        "4 · modality and data (ablations 1, 2)",
        "a-full-lidar-only, image-only (main v2 mixture); "
        "a-full-synth-only, a-full-real-only, a-full-main-mixed",
        "full, early stop",
        "final-schedule rows predate the fix; image only also moves to the "
        "main v2 mixture",
    ),
    (
        "5 · pool size (ablation 7)",
        "v2-scale-2x … full",
        "60k to 80k, early stop",
        "predates the fix",
    ),
    (
        "6 · ratio (ablation 8)",
        "b-ratio-90/80/70/60",
        "full, early stop",
        "running now on the corrected labels but without the fix; rerun "
        "if the fix changes the training data or loss",
    ),
    (
        "7 · pose prior",
        "a-full-main-v2-prior, a-full-mix80-prior",
        "full, early stop",
        "only worth rerunning if the hand fix is not a prior itself",
    ),
    (
        "8 · LiDAR synthesis (ablation 6, removed from the board)",
        "rig SLOPER4D, rig Waymo, ball 0.25 m, Waymo resolution only, "
        "reference",
        "full, early stop",
        "never run on the point-anchored model; the old camera-frame rows "
        "were dropped",
    ),
]

GROUPS: list[Group] = [
    Group(
        "Headline: our main trainings against the published pipelines",
        "Final schedule (150k steps or early stop), evaluated on the full "
        "sets. Recipes in the footnotes; every other run of ours is in the "
        "ablation tables below.",
        [
            ("ours · main v2", ["a-full-main-v2-001"]),
            ("ours · main v2 + pose prior", ["a-full-main-v2-prior-001"]),
            (MAIN, MAIN_RUN),
            ("ours · anchor · mix80 + pose prior", ["a-full-mix80-prior-001"]),
            ("ours · mix90 (pseudo-GT v2)", ["b-ratio-90-001"]),
        ],
        pending=True,
    ),
    Group(
        "Main runs",
        "Every full-schedule training of the main recipe with the "
        "point-anchored translation (the model since 2026-09-15); the "
        "mixture is what differs.",
        [
            ("ours · main v2", ["a-full-main-v2-001"]),
            ("ours · main v2 + pose prior", ["a-full-main-v2-prior-001"]),
            (MAIN, MAIN_RUN),
            ("ours · anchor · mix80 + pose prior", ["a-full-mix80-prior-001"]),
            ("ours · mix90 (pseudo-GT v2)", ["b-ratio-90-001"]),
            (
                "ours · anchor · main-mixed (50/40/10)",
                ["a-full-main-mixed-000"],
            ),
        ],
        pending=True,
    ),
    Group(
        "Ablation 1: input modality (final schedule)",
        "One stream switched off at training and test time. Image only "
        "keeps the camera-frame translation (no points to anchor to).",
        [
            ("image only", ["full-image-only-000"]),
            ("LiDAR only", ["a-full-lidar-only-000"]),
            (MAIN + MAIN_NOTE, MAIN_RUN),
        ],
    ),
    Group(
        "Ablation 2: training data (final schedule)",
        "What the batches are drawn from; point-anchored translation "
        "throughout.",
        [
            ("synthetic only (BEDLAM)", ["a-full-synth-only-000"]),
            ("real only (Waymo, SLOPER4D)", ["a-full-real-only-000"]),
            ("mixed 50/40/10", ["a-full-main-mixed-000"]),
            (MAIN + MAIN_NOTE, MAIN_RUN),
        ],
    ),
    Group(
        "Ablation 3: fusion",
        V2_ABL + ". Reference: learned gates; image only keeps the "
        "camera-frame translation.",
        [
            ("learned gates (reference)", ["v2-abl-ref-000"]),
            ("gate none (plain sum)", ["v2-abl-gate-none-000"]),
            ("gate hard (fixed priors)", ["v2-abl-gate-hard-000"]),
            ("LiDAR only", ["v2-abl-lidar-only-000"]),
            ("image only", ["v2-abl-image-only-000"]),
            (MAIN + MAIN_NOTE, MAIN_RUN),
        ],
        pending=True,
    ),
    Group(
        "Ablation 4: loss terms",
        V2_ABL + ". Reference plus LiDAR-to-surface terms (chamfer with a "
        "learned clothing offset, rigid ICP residual) and the Pose2Mesh mesh "
        "terms (vertex L1, surface normal, edge length), alone and combined.",
        [
            ("reference", ["v2-abl-ref-000"]),
            ("+ chamfer", ["v2-abl-loss-chamfer-000"]),
            ("+ ICP", ["v2-abl-loss-icp-000"]),
            ("+ mesh terms", ["v2-abl-loss-mesh-000"]),
            ("+ chamfer + ICP", ["v2-abl-loss-chamfer-icp-000"]),
            ("+ ICP + mesh terms", ["v2-abl-loss-icp-mesh-000"]),
            (
                "+ chamfer + ICP + mesh terms",
                ["v2-abl-loss-chamfer-icp-mesh-000"],
            ),
            (MAIN + MAIN_NOTE, MAIN_RUN),
        ],
        pending=True,
    ),
    Group(
        "Ablation 5: Waymo label source",
        V2_ABL + ". What supervises the Waymo slice: the 13 keypoints "
        "(reference) or a pseudo-GT SMPL mesh on the matched records.",
        [
            ("reference: Waymo keypoints", ["v2-abl-ref-000"]),
            (
                "pseudo-GT SMPL v2, ours (TokenHMR latent fit, 4,070 records)",
                ["v2-abl-pseudo-v2-000"],
            ),
            (
                "pseudo-GT SMPL, pedestrian generation (1,489 records)",
                ["v2-abl-pseudo-pedgen-000"],
            ),
            (
                "pseudo-GT SMPL, LiDAR-HMR (4,586 records)",
                ["v2-abl-pseudo-lhmr-000"],
            ),
            (MAIN + MAIN_NOTE, MAIN_RUN),
        ],
        pending=True,
    ),
    Group(
        "Ablation 7: synthetic pool size",
        "Synthetic pool capped at k x the Waymo train count, main v2 mixture "
        "(75/10/10/5) and point anchor; 60k to 80k steps, stopping early "
        "once no validation source improves for five evaluations.",
        [
            ("2x (18 shards)", ["v2-scale-2x-000"]),
            ("4x (36 shards)", ["v2-scale-4x-000"]),
            ("8x (73 shards)", ["v2-scale-8x-000"]),
            ("16x (145 shards)", ["v2-scale-16x-000"]),
            ("32x (291 shards)", ["v2-scale-32x-000"]),
            ("full pool, 85x (reference)", ["v2-scale-full-000"]),
            (MAIN + MAIN_NOTE, MAIN_RUN),
        ],
        pending=True,
    ),
    Group(
        "Ablation 8: synthetic-to-real ratio",
        "b series: every real training record (Waymo with the "
        "confidence-weighted pseudo-GT v2 mesh labels, SLOPER4D, 3DPW) drawn "
        "in proportion to the dataset sizes, BEDLAM filling the batch to the "
        "stated share; full schedule with early stop, point anchor.",
        [
            ("90 / 10", ["b-ratio-90-001"]),
            ("80 / 20", ["b-ratio-80-001"]),
            ("70 / 30", ["b-ratio-70-001"]),
            ("60 / 40", ["b-ratio-60-001"]),
            ("50 / 50", ["b-ratio-50-000"]),
            (MAIN + MAIN_NOTE, MAIN_RUN),
        ],
        pending=True,
    ),
]

# methods whose placement is not meaningful on our crops
PLACEMENT_BLANKED = {"human3r"}

# numbers reported by other papers on their own protocol (no code or
# weights to run on our records): SLOPER4D paper, Table 4a, LiDARCap
# trained on SLOPER4D / on LiDARHuman26M + SLOPER4D, tested on SLOPER4D
REPORTED_ROWS = [
    (
        "LiDARCap (SLOPER4D-trained)",
        "LiDAR only, reported in the SLOPER4D paper, own protocol",
        {"S_mpjpe": 86.1, "S_pa_mpjpe": 65.1},
    ),
]


@dataclass
class Row:
    """One table row: label, provenance note and metric values per split."""

    label: str
    note: str
    kind: str  # static | ours
    values: dict[str, float | None] = field(default_factory=dict)
    footnote: str | None = None
    retrain: str | None = None  # why the run has to be trained again
    labels: str | None = None  # Waymo label source: v2 | v1 | lhmr | pedgen
    # | "" (keypoints only); None for static rows


# Waymo shard directory -> pseudo-GT label tag ("" = keypoints only)
WAYMO_LABEL_DIRS = {
    "v1_pseudo2": "v2",
    "v1_pseudo": "v1",
    "v1_lidarhmr": "lhmr",
    "v1_pedgen": "pedgen",
    "v1_simlidar": "v2",  # pseudo-GT v2 labels, simulated returns
    "v1": "",
}


def waymo_labels(root: Path, run: str) -> str | None:
    """Which Waymo label source the run trained on, from its config.json:
    the pseudo-GT tag, "" for keypoints only, None when unknown or when
    Waymo was not a training source."""
    cfg = root / "runs" / run / "config.json"
    if not cfg.exists():
        return None
    try:
        data = json.loads(cfg.read_text())
    except json.JSONDecodeError:
        return None
    for src in data.get("data", {}).get("train", []):
        if src.get("name") != "waymo":
            continue
        tags = {
            WAYMO_LABEL_DIRS.get(Path(d).name, "?") for d in src.get("dirs", [])
        }
        return "+".join(sorted(tags))
    return None


def labels_html(tag: str | None) -> str:
    """The pseudo-GT cell: a tick with the source, a cross for keypoints."""
    if tag is None:
        return '<td class="lab"></td>'
    if tag == "":
        return '<td class="lab no">&#10007;</td>'
    return f'<td class="lab yes">&#10003; ({tag})</td>'


# footnotes of our rows, by label (the recipe stays out of the label)
ROW_FOOTNOTES = {
    "ours · anchor · mix80 (pseudo-GT v2)": (
        "b series: as anchor · mix80, but the Waymo training records carry "
        "the pseudo-GT v2 mesh labels (fit in TokenHMR's tokenizer latent "
        "space from LiDAR-HMR's initialisation, gated; 4,063 of 4,591 "
        "records), each weighted by its confidence in the mesh losses; the "
        "keypoints stay the joint supervision"
    ),
    "ours · main v2": (
        "point-anchored translation, learned gates; every batch 75 % BEDLAM "
        "(full synthetic pool, random main resolution), 10 % Waymo train "
        "(13 keypoints, no mesh label), 10 % SLOPER4D train, 5 % 3DPW "
        "mesh-LiDAR; full schedule with early stop"
    ),
    "ours · main v2 (pseudo-GT LiDAR-HMR)": (
        "as main v2, but the Waymo training records carry LiDAR-HMR's "
        "published SMPL fits as mesh labels (matched per frame by mesh "
        "centre, median 8 cm; 4,586 of 4,591 records), expressed in the "
        "record's camera frame; has_smpl is set, so the full-mesh losses "
        "apply to Waymo rows"
    ),
    "ours · main v2 + pose prior": (
        "as main v2, plus the BEDLAM pose prior on the ankles, feet, head, "
        "wrists and hands of rows without a mesh label (weight 0.05)"
    ),
}

FOOTNOTES = {
    "human3r": (
        "depth not metric on crops, placement columns blanked; "
        "crops without a detection skipped"
    ),
    "sam3d-body": (
        "predicts the MHR body model; SMPL fitted to the MHR mesh with the "
        "conversion tool of the MHR repository (0.65 cm mean vertex "
        "distance on a 12-crop check), then scored like every SMPL row"
    ),
}


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
        "pve": "pve",
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
        done = (root / "runs" / run / "DONE").exists()
        state = "final" if done else "running, full-set eval at"
        return _from_results(
            payload["results"]
        ), f"{state} step {step // 1000}k"
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


def group_rows(
    root: Path, spec: list[tuple[str, list[str]]], pending: bool = False
) -> list[Row]:
    """Rows of one group; runs without any result are left out, or shown
    as queued when ``pending``."""
    rows = []
    for label, runs in spec:
        loaded = [(r, load_run(root, r)) for r in runs]
        found = [(r, x) for r, x in loaded if x is not None]
        reason = next((RETRAIN[r] for r in runs if r in RETRAIN), None)
        tags = {waymo_labels(root, r) for r, _ in found}
        tags.discard(None)
        labels = "/".join(sorted(t for t in tags if t is not None)) or (
            "" if tags else None
        )
        if not found:
            if pending:
                rows.append(
                    Row(
                        label,
                        "queued",
                        "ours",
                        {},
                        ROW_FOOTNOTES.get(label),
                        reason,
                    )
                )
            continue
        values = average([x[0] for _, x in found])
        notes = sorted({x[1] for _, x in found})
        note = "; ".join(notes)
        if len(found) > 1:
            note = f"mean of {len(found)} seeds; " + note
        if len(found) < len(runs):
            note += f" ({len(runs) - len(found)} seed pending)"
        rows.append(
            Row(
                label,
                note,
                "ours",
                values,
                ROW_FOOTNOTES.get(label),
                reason,
                labels,
            )
        )
    return rows


def static_rows(path: Path, spec: list[tuple[str, str, str]]) -> list[Row]:
    """Rows from a score_baselines results file (its siblings fill gaps)."""
    if not path.exists():
        return []
    res = json.loads(path.read_text())
    # siblings fill missing methods and missing splits of known methods
    # (a method's Waymo/SLOPER4D scores and its 3DPW scores may come from
    # different scoring runs); the first file to provide a split wins
    for sibling in sorted(path.parent.glob("results_*.json")):
        if sibling != path:
            for k, v in json.loads(sibling.read_text()).items():
                for split, scores in v.items():
                    res.setdefault(k, {}).setdefault(split, scores)
    rows = []
    for key, label, note in spec:
        if key in res:
            values = _from_results(res[key])
            if key in PLACEMENT_BLANKED:
                for k in values:
                    if k.split("_", 1)[1] in (
                        "abs_mpjpe",
                        "transl_err_m",
                        "map",
                    ):
                        values[k] = None
            rows.append(Row(label, note, "static", values, FOOTNOTES.get(key)))
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
    have = [
        (i, r.values.get(key)) for i, r in enumerate(rows) if not r.retrain
    ]
    valid = [(i, v) for i, v in have if v is not None and np.isfinite(v)]
    valid.sort(key=lambda t: -t[1] if higher else t[1])
    return {i: pos for pos, (i, _) in enumerate(valid[:3])}


def beats_static(rows: list[Row], key: str) -> set[int]:
    """Indices of our rows whose value beats every static (compared) row."""
    higher = key.split("_", 1)[1] in HIGHER_IS_BETTER

    def finite(r: Row) -> float | None:
        v = r.values.get(key)
        return v if v is not None and np.isfinite(v) else None

    ref = [
        v for r in rows if r.kind == "static" and (v := finite(r)) is not None
    ]
    if not ref:
        return set()
    bar = max(ref) if higher else min(ref)
    out = set()
    for i, r in enumerate(rows):
        v = finite(r)
        if r.kind != "ours" or v is None:
            continue
        if (v > bar) if higher else (v < bar):
            out.add(i)
    return out


def _keys() -> list[str]:
    return [f"{tag}_{m}" for tag, _ in SPLITS for m in METRICS]


def table_html(rows: list[Row]) -> str:
    """One ranked table."""
    keys = _keys()
    rank_by_key = {k: ranks(rows, k) for k in keys}
    blue_by_key = {k: beats_static(rows, k) for k in keys}
    cls = ["g", "y", "r"]
    h = [
        '<div class="wrap"><table><thead><tr><th></th><th></th>'
        '<th class="group" colspan="6">Waymo val · 894</th>'
        '<th class="group" colspan="6">SLOPER4D test · 9,904</th>'
        '<th class="group" colspan="6">3DPW test · 6,617</th>'
        '<th class="group"></th></tr><tr><th>pseudo-GT</th><th>method</th>'
    ]
    h.append("".join(f"<th>{lbl}</th>" for _ in SPLITS for lbl in LABELS))
    h.append("<th>retrain</th></tr></thead><tbody>")
    prev = None
    for i, r in enumerate(rows):
        div = " divider" if prev == "static" and r.kind == "ours" else ""
        mark = ""
        if r.footnote:
            n = 1 + sum(1 for x in rows[:i] if x.footnote)
            mark = f"<sup>{n}</sup>"
        grey = " retrain" if r.retrain else ""
        h.append(
            f'<tr class="{r.kind}{div}{grey}">{labels_html(r.labels)}'
            f'<td class="name">{r.label}{mark}<small>{r.note}</small></td>'
        )
        for k in keys:
            rk = rank_by_key[k].get(i)
            c = f" {cls[rk]}" if rk is not None else ""
            if i in blue_by_key[k] and rk is None:
                c = " b"  # beats every compared pipeline but is not ranked
            h.append(f'<td class="num{c}">{fmt(k, r.values.get(k))}</td>')
        h.append(f'<td class="why">{r.retrain or ""}</td></tr>')
        prev = r.kind
    h.append("</tbody></table></div>")
    notes = [(i, r.footnote) for i, r in enumerate(rows) if r.footnote]
    if notes:
        h.append('<ol class="fn">')
        for _, text in notes:
            h.append(f"<li>{text}</li>")
        h.append("</ol>")
    return "".join(h)


STYLE = """
:root{--ground:#F6F8F6;--panel:#FFFFFF;--ink:#1A2421;--muted:#5E6A65;--rule:#D5DCD8;--accent:#0E6B64;
--g-bg:#D9F0DD;--g-ink:#166534;--y-bg:#FBF0C2;--y-ink:#7A5A00;--r-bg:#F9DBD5;--r-ink:#9C3323;--b-bg:#D6E6FA;--b-ink:#1D4E9C;--ours:#EAF3F1;--grey-bg:#ECEDEC;--grey-ink:#7A827E}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ground:#131917;--panel:#1B2320;--ink:#E7ECE9;--muted:#9BA8A2;--rule:#2B3733;--accent:#5FC7BC;
--g-bg:#1D3F28;--g-ink:#8FE0A5;--y-bg:#453A0F;--y-ink:#F1D46B;--r-bg:#4A2019;--r-ink:#F3A08E;--b-bg:#1B3556;--b-ink:#9CC4F5;--ours:#1E2B28;--grey-bg:#202422;--grey-ink:#7E8783}}
:root[data-theme="dark"]{--ground:#131917;--panel:#1B2320;--ink:#E7ECE9;--muted:#9BA8A2;--rule:#2B3733;--accent:#5FC7BC;
--g-bg:#1D3F28;--g-ink:#8FE0A5;--y-bg:#453A0F;--y-ink:#F1D46B;--r-bg:#4A2019;--r-ink:#F3A08E;--b-bg:#1B3556;--b-ink:#9CC4F5;--ours:#1E2B28;--grey-bg:#202422;--grey-ink:#7E8783}
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
.chip.grey{background:var(--grey-bg);color:var(--grey-ink)}.chip.g{background:var(--g-bg);color:var(--g-ink)}.chip.y{background:var(--y-bg);color:var(--y-ink)}.chip.r{background:var(--r-bg);color:var(--r-ink)}.chip.b{background:var(--b-bg);color:var(--b-ink)}
.wrap{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--panel)}
table{border-collapse:collapse;width:100%;min-width:1300px;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap}
th{font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600;background:var(--panel)}
th.group{text-align:center;border-bottom:none;padding-bottom:2px;color:var(--ink);letter-spacing:.14em}
td:first-child,th:first-child{text-align:left}
td.lab{text-align:center;white-space:nowrap;font-family:"IBM Plex Mono",monospace;font-size:.8rem}td.lab.yes{color:var(--g-ink)}td.lab.no{color:var(--r-ink);font-size:1rem}
td.name{font-weight:500}td.name small{display:block;font-weight:400;color:var(--muted);font-size:.76rem}
td.num{font-family:"IBM Plex Mono",monospace;font-size:.88rem}
tr:last-child td{border-bottom:none}tr.ours td{background:var(--ours)}tr.divider td{border-top:2px solid var(--accent)}
tr.retrain td{background:var(--grey-bg)!important;color:var(--grey-ink)}tr.retrain td.num{color:var(--grey-ink)}
td.why{text-align:left;white-space:normal;min-width:260px;max-width:360px;font-size:.76rem;color:var(--muted);line-height:1.3}
table.sched{min-width:0}table.sched td{white-space:normal;text-align:left;vertical-align:top;font-size:.88rem}table.sched td.b{font-weight:600}
td.g{background:var(--g-bg)!important;color:var(--g-ink);font-weight:500}
td.y{background:var(--y-bg)!important;color:var(--y-ink);font-weight:500}
td.b{background:var(--b-bg)!important;color:var(--b-ink);font-weight:600}
td.r{background:var(--r-bg)!important;color:var(--r-ink);font-weight:500}
details{border-top:1px solid var(--rule);padding-top:16px}details>summary{cursor:pointer;font-family:"Fraunces",Georgia,serif;font-weight:600;font-size:1.1rem;color:var(--muted)}details>section{margin-top:28px}
ol.fn{margin:8px 0 0;padding-left:20px;font-size:.8rem;color:var(--muted);max-width:90ch}ol.fn li{margin-bottom:4px}
.notes{display:grid;gap:8px;font-size:.88rem;color:var(--muted);max-width:80ch}.notes b{color:var(--ink);font-weight:600}
@media (max-width:600px){body{padding-inline:16px}h1{font-size:1.6rem}}
"""

FONTS = (
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Sans:wght@400;500;600"
    '&family=IBM+Plex+Mono:wght@400;500&display=swap">'
)


def _eta_text(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    if seconds < 3600:
        return f"~{seconds / 60:.0f} min"
    return f"~{seconds / 3600:.1f} h"


def running_html(squeue: Path | None, root: Path, configs: Path) -> str:
    """The "Currently running" section: what is in the queue right now.

    Mirrors the table `running_table.py` splices into the handoff, so the
    page opens with the live picture and the result tables below it carry
    the same runs as ``running`` rows.
    """
    from lidar_bedlam.scripts.running_table import SOURCES, arrow, collect

    if squeue is None or not squeue.exists():
        return ""
    text = squeue.read_text()
    head = "<h2>Currently running</h2>"
    if not text.strip():
        # a successful squeue always prints its header: empty means the
        # capture failed, which is not the same as an empty queue
        return (
            f"<section>{head}<p class='desc'>Queue state unknown: the "
            "squeue capture came back empty, so the cluster was not "
            "reached. This is not an empty queue.</p></section>"
        )
    runs = collect(text, root / "runs", configs, 1)
    if not runs:
        return (
            f"<section>{head}<p class='desc'>Nothing in the queue.</p>"
            "</section>"
        )
    h = [
        f"<section>{head}<p class='desc'>Regenerated every hourly tick; "
        "arrows compare the latest evaluation with the previous one and "
        "lower is better, so a down arrow is a run still learning. These "
        "runs also appear in their own table below.</p>",
        '<div class="wrap"><table><thead><tr><th>run</th><th>step</th>'
        "<th>of target</th><th>progress</th><th>time left</th>",
    ]
    h.append("".join(f"<th>{lbl} MPJPE</th>" for lbl, _ in SOURCES))
    h.append("</tr></thead><tbody>")
    live = [r for r in runs if not r.queued]
    waiting = [r for r in runs if r.queued]
    for r in live:
        pct = 100.0 * r.step / r.target if r.target else 0.0
        h.append(
            f'<tr class="ours"><td class="name">{r.name}</td>'
            f'<td class="num">{r.step:,}</td>'
            f'<td class="num">{r.target:,} ({r.target_kind})</td>'
            f'<td class="num">{pct:.0f} %</td>'
            f'<td class="num">{_eta_text(r.eta_s)}</td>'
        )
        for label, _ in SOURCES:
            t = r.trends.get(label)
            cell = (
                "–" if t is None else f"{t[0]:.1f} {arrow(t[1])} {t[1]:+.1f} %"
            )
            h.append(f'<td class="num">{cell}</td>')
        h.append("</tr>")
    if waiting:
        h.append(
            f'<tr class="ours"><td class="name">{len(waiting)} queued</td>'
            f'<td class="num" colspan="{4 + len(SOURCES)}">waiting for '
            "nodes; each claims its run name at start</td></tr>"
        )
    h.append("</tbody></table></div></section>")
    return "".join(h)


def build(
    root: Path,
    baselines: Path,
    stamp: str,
    squeue: Path | None = None,
    configs: Path = Path("configs"),
) -> str:
    """The whole page."""
    sections = []
    live = running_html(squeue, root, configs)
    if live:
        sections.append(live)
    archived = []
    for i, g in enumerate(GROUPS):
        rows = group_rows(root, g.rows, g.pending)
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
        html = (
            f"<h2>{g.title}</h2><p class='desc'>{g.desc}</p>{table_html(rows)}"
        )
        if g.archive:
            archived.append(f"<section>{html}</section>")
        else:
            sections.append(f"<section>{html}</section>")
    if archived:
        sections.append(
            "<details><summary>Earlier runs</summary>"
            + "".join(archived)
            + "</details>"
        )
    sections.append(schedule_html())
    body = "\n".join(sections)
    return f"""<title>LiDAR-BEDLAM Scoreboard</title>
{FONTS}
<style>{STYLE}</style>
<main>
<header>
  <div class="eyebrow">{stamp}</div>
  <h1>Placement, box and pose against the published pipelines</h1>
  <p class="lede">Every row is scored on the same records with the same protocol: Waymo val (894 crops, 13 COCO joints, hip-centre placement), the full SLOPER4D test split (9,904 crops, 24 SMPL joints, SMPL translation) and the 3DPW test split (6,617 crops, real image and SMPL label, LiDAR simulated on the labelled mesh; 24 SMPL joints). MPJPE, PA-MPJPE and PVE (per-vertex, SMPL labels only) in mm are root-relative; abs is the mean joint error as predicted, pose and placement together, which exposes the depth ambiguity of image-only methods; placement in metres; mAP over 3D box IoU 0.25/0.5/0.7 (mesh-only methods get a box from the mesh extent, scaled to Waymo's padded box convention).</p>
</header>
<section><div class="legend"><span>Per column, within each table:</span>
<span class="chip g">best</span><span class="chip y">second</span><span class="chip r">third</span>
<span class="chip b">beats every compared pipeline (below the top three)</span>
<span>· lower is better except mAP · shaded rows are our runs · the published pipelines are static</span>
<span class="chip grey">grey</span><span>to be retrained (numbers kept, reason in the last column); grey rows are not ranked</span>
<span>· first column: Waymo label source of the training, &#10003; (v2 | v1 | lhmr | pedgen) = pseudo-GT SMPL mesh, &#10007; = the 3D and 2D keypoints only</span></div></section>
{body}
<section class="notes">
<p><b>The 3D box bug: nearly every row below is superseded.</b> Until
<code>8df62e5</code> the box loss compared raw 3D boxes, but Waymo pedestrian
labels are padded (about 0.90 x 1.78 x 1.00 m) while the box we derive from
the mesh extent is about 0.61 x 1.68 x 0.59 m. The size term therefore
demanded 28 cm more length and 40 cm more width on every Waymo row, and
since the keypoints pin shoulders, elbows and wrists, the only free way to
widen the mesh was to rotate the wrists and hands outward. Measured on two
9k-step tests: wrist bend 31-39 deg with the term on, 6 deg with it off, and
lateral hand reach falling by 10-54 %. Dropping the term moved Waymo val
MPJPE from 85.4 / 86.4 mm to 59.8 mm while SLOPER4D and 3DPW were unchanged
(38.4 vs 38.6 / 37.5 and 55.5 vs 58.1 / 56.1), which is the signature of a
Waymo-label problem rather than a model one. <b>Every run that trained on
Waymo rows is affected, which is all of them except
<code>a-full-synth-only</code></b> (BEDLAM only, so no Waymo rows and no
change). Ablation 8 is the clearest casualty: its finding that less real
data scores better is an artefact, since more Waymo share meant more
corruption (1.5 % Waymo gave 81.7 mm, 6.1 % gave 87.5 mm). The
<code>b-ratio-*-001</code> series plus a new 50/50 is rerunning under the
fix; the retraining schedule lists the rest.</p>
<p><b>Image baselines</b> get our 256 px crop and the crop's true intrinsics; the weak-perspective camera is converted to metric translation with the real principal point. CameraHMR is given the ground-truth intrinsics instead of estimating them.</p>
<p><b>LiDAR-HMR</b> runs the Waymo release weights on 1,024 points centred on the box centre in a z-up frame; SLOPER4D is out of its training domain.</p>
<p><b>Our runs</b>: rows marked final are re-evaluated from <code>last.pt</code> on the full sets; running rows show their latest in-training evaluation; their Waymo placement and abs MPJPE appear once the run is re-evaluated with the current protocol (runs started before the placement fix logged an inflated Waymo placement). Ablations 3 to 5 train 15k steps on the main v2 mixture, ablation 7 trains 60k to 80k steps on it; every ablation table ends with the main run at its final schedule for reference. Runs of the earlier recipe (camera-frame translation, 50/40/10 mixture, the LiDAR-HMR label main, the first pseudo-GT v2 labels with bent wrists) were removed from the board on 2026-09-18; the retraining schedule below lists what replaces them.</p>
</section>
</main>
"""


def schedule_html() -> str:
    """The retraining schedule as a table."""
    h = [
        "<section><h2>Retraining schedule</h2><p class='desc'>Nothing "
        "beyond the short tests starts before the Waymo hand problem is "
        "understood; then the blocks run in this order.</p>"
        '<div class="wrap"><table class="sched"><thead><tr><th>block</th>'
        "<th>runs</th><th>schedule</th><th>why</th></tr></thead><tbody>"
    ]
    for block, runs, sched, why in SCHEDULE:
        h.append(
            f'<tr><td class="b">{block}</td><td>{runs}</td><td>{sched}</td>'
            f"<td>{why}</td></tr>"
        )
    h.append("</tbody></table></div></section>")
    return "".join(h)


def schedule_md() -> str:
    """The retraining schedule as markdown."""
    lines = [
        "# Retraining schedule",
        "",
        "Generated by `lidar_bedlam/scripts/build_scoreboard.py "
        "--schedule-md`; edit `SCHEDULE` and `RETRAIN` there.",
        "",
        "Rule: no full training starts before the Waymo hand problem is "
        "found (block 0). Then the blocks run in order; each run in "
        "`RETRAIN` is greyed on the scoreboard until its replacement exists.",
        "",
        "| block | runs | schedule | why |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {block} | {runs} | {sched} | {why} |"
        for block, runs, sched, why in SCHEDULE
    ]
    lines += ["", "## Greyed rows and their reason", ""]
    lines += [f"- `{run}`: {why}" for run, why in RETRAIN.items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("outputs/helma"))
    ap.add_argument("--squeue-file", type=Path, default=None)
    ap.add_argument("--configs", type=Path, default=Path("configs"))
    ap.add_argument(
        "--baselines", type=Path, default=Path("outputs/baselines")
    )
    ap.add_argument("--stamp", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--schedule-md",
        type=Path,
        default=None,
        help="also write the retraining schedule as markdown (.docs/retrain_schedule.md)",
    )
    args = ap.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        build(
            args.root,
            args.baselines,
            args.stamp,
            args.squeue_file,
            args.configs,
        )
    )
    sys.stdout.write(f"wrote {args.out}\n")
    if args.schedule_md is not None:
        args.schedule_md.write_text(schedule_md())
        sys.stdout.write(f"wrote {args.schedule_md}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
