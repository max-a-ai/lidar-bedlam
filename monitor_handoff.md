# Monitoring handoff (Helma runs and the scoreboard)

<!-- RUNNING:START -->

## Currently running (regenerated every hourly tick)

| run | step | of target | progress | time left | waymo MPJPE | sloper MPJPE | 3dpw MPJPE |
|---|---|---|---|---|---|---|---|
| `m-boxhead-v3-000` | – | – | starting | – | – | – | – |

Arrows compare the latest evaluation with the previous one: `⇘` improving by more than 3 %, `↘` improving by 0.5–3 %, `→` flat within ±0.5 %, `↗` worsening by 0.5–3 %, `⇗` worsening by more than 3 %. Lower MPJPE is better, so a down arrow is a run that is still learning.

<!-- RUNNING:END -->

This file is the single description of the monitoring routine. Whoever
changes the routine (a new run to watch, a new scoreboard option, a new
artifact rule) updates this file in the same change. Never delete it.

It holds what is specific to this project: run names, Helma paths, the
scoreboard builder's options, the arrow thresholds calibrated on these
metrics. The general rules it rests on — Monitor expires after 30 minutes,
`CronCreate` drives the tick, a re-armed monitor replays completions, the
page is one HTML artifact republished in place with the running table first
— live in the `run-monitoring` skill.

This file owns **only** the Helma monitoring routine. The bike nuScenes
data question lives in `data_inspection_handoff.md` and is worked in its own
session, so a tick here never has to touch it.

Start a new Claude Code session in this repo with the model you want and
tell it:

    Read monitor_handoff.md and set up the two monitors and the hourly
    routine exactly as described. Report only what the file says to report.

Last updated 2026-09-18 (running table added at the top; Helma switched from
rsync to a git checkout; the whole b-ratio series and both t-short tests
finished and the queue is empty).

## What is running on Helma
`ssh -4 helma` is flaky: retry up to 3x with a short sleep. Repo
`/hnvme/workspace/v103fe17-lidar-bedlam`, outputs in `outputs/<run>-000/`
(metrics.jsonl, val_*.json, DONE marker), Slurm logs `outputs/slurm-<run>-*.out`.

**All six m-series mains and both t-short tests finished on 2026-09-22; m-boxhead-kp (keypoints only) won Waymo at 60.1 mm.** The live picture is the "Currently
running" table at the top of this file; this section records what each run
is for and what to do when it finishes.

| run | what | on DONE |
|---|---|---|
| b-ratio-90/80/70/60-000 | ratio runs, corrected pseudo-GT v2 labels, full schedule | full-set eval |
| t-short-real-000 | 9k-step hand test, real Waymo returns | nothing (val plots only) |
| t-short-simlidar-000 | 9k-step hand test, Waymo returns simulated on the pseudo-GT mesh | nothing (val plots only) |
| b-ratio-90/80/70/60-001, b-ratio-50-000 | ablation 8 rerun under the box fix (`8df62e5`: box size term dropped on Waymo rows); 50/50 is new (jobs 874734-874738) | full-set eval |
| m-boxhead / m-nobox / m-boxhead-kp / m-gate-none / m-chamfer / m-prior (-000) | jobs 880923-880928; the six 15k mains under the box fix on one split (configs/m_*.yaml); resumable with `RUN_NAME=<run>` and `EXTRA_SET="optim.max_steps=N"`; the board has a group for them | nothing (val plots only); report Waymo MPJPE, transl, mAP |
| m-boxhead-v3-000 | job 883008, running since 2026-09-22; run 7: box head on pseudo-GT v3 labels (`real/v1_pseudo3`, fitter with hip-centre term + joint offsets); same split as the six mains | nothing (val plots only); compare with m-boxhead-000 and m-boxhead-kp-000 |
| d-token-prior-kp / d-token-prior-v3 / d-token-head-kp / d-token-head-v3 (-000) | jobs 883044 (prior chain) and 883045 (head chain), submitted 2026-09-22; debug runs: token-manifold prior (option 1) and token pose head (option 2), each on keypoints-only and on pseudo-GT v3; two chained jobs (CONFIGS= in train.sbatch), board group "Debug" | nothing (val plots only); report Waymo MPJPE, transl, mAP and hands |
| t-short-boxhead-000 | 15k-step test of the padded box head (`d473c55`): Waymo box loss trains a residual head on the detached mesh box; main mixture 75 % BEDLAM + all real; watch Waymo MPJPE, transl and mAP together | nothing (val plots only); report Waymo wrist bend, transl, mAP |
| t-short-nobox-000 | 9k-step hand test, t_short_real recipe with `loss.box3d=0` (Waymo box labels are padded 0.9 x 1.0 m, the mesh box is 0.6 x 0.6 m; suspected cause of the bent hands) | nothing (val plots only); report Waymo wrist bend |

Finished and fully evaluated: `b-full-mix80-000` and all four
`b-ratio-*-000` (every one early-stopped at step 80000 = `min_steps`; evals
verified at that step). Both `t-short-*` finished at step 9000; they are the
block-0 prerequisite of `.docs/retrain_schedule.md`, so nothing beyond them
starts until the Waymo hand problem is settled.

Do not launch, restart or cancel any training without the user's word; the
schedule of what comes next is `.docs/retrain_schedule.md`.

## Monitor 1: b- and t-series validation lines (persistent, polls every 10 min)
Create with the Monitor tool: persistent=true, timeout_ms=3600000,
description "Helma b/t-series runs: latest validation line per run,
failures, completion". The command is the content of `monitor_bseries.sh`
next to this file, passed verbatim. It watches `outputs/b-*-000`,
`outputs/t-*-000`, `outputs/m-*-000` and `outputs/d-*-000`.

It expires after 30 minutes whatever `timeout_ms` says (see the hourly
section below): re-arm it on the expiry notice. Its `seen` cache starts
empty on every arming, so the first event after a re-arm repeats every
`DONE` line — check whether the eval already exists before submitting one,
rather than treating the repeat as a new completion.

On each event: one sentence per run (Waymo MPJPE mm and translation m,
SLOPER4D, 3DPW; the t runs also log `waymo_val_sim`). No push notification
for routine lines.

On a `DONE <run>` line for a b run: submit the full-set eval on Helma.
RUNS must be space-separated and quoted (commas evaluate only the first):

    ssh -4 helma 'cd /hnvme/workspace/v103fe17-lidar-bedlam && sbatch --export=ALL,FORCE=1,RUNS="<run>" lidar_bedlam/slurm/eval_runs.sbatch'

On a `DONE <run>` line for a t run: tell the user the test finished and
that the val plots are on wandb (`image/val_plot_waymo_val` and
`image/val_plot_waymo_val_sim`); no eval job.

On a `FAIL <run> <log>` line: print the last 30 lines of that Slurm log and
notify the user.

## The hourly tick: a cron job, not a monitor (changed 2026-09-18)
**The Monitor tool cannot drive an hourly loop.** Whatever `timeout_ms` or
`persistent` is passed, the harness expires a monitor after 30 minutes, so
`monitor_hourly.sh` — which sleeps 3600 s between passes — emits its first
event on arming and then dies before the second one ever fires. Every
"hourly" tick before this change was really a manual re-arm.

Use `CronCreate` instead, which enqueues a prompt into the running session
so the routine keeps the local repo, the `.venv` and the SSH access it
needs:

    CronCreate(cron="0 * * * *", recurring=true, prompt="Hourly monitoring
    tick. Run the hourly routine exactly as described in monitor_handoff.md
    ... report only what the file says to report.")

Minute 0: the user wants the ticks on the full clock (changed 2026-09-21),
and the report is headed with the full hour whatever minute the scheduler
actually fired at. Know the limits before relying on it:

- **session-only** — the job lives in memory and dies with the Claude
  session; a new session must create it again. Nothing is written to disk.
- **auto-expires after 7 days**, firing one last time.
- fires only while the session is idle, with a few minutes of jitter.

Do not use the `schedule` skill for this: it creates cloud agents, which
have neither this repo nor SSH to Helma.

`monitor_hourly.sh` is kept only for its squeue/traceback snapshot; the
hourly routine below captures squeue itself, so the script is no longer
armed as a monitor.

On each tick run the hourly routine below, then report it in the tick
layout described under "What a tick reports".

## What a tick reports (changed 2026-09-21)

Three parts, in this order, and nothing else. The `tick-it` skill
(`~/.claude/skills/tick-it/SKILL.md`) is the general form of this.

**1. Labelled markdown rows, then the queue.** Plain rows, not a code
block, with the links live so they open from the chat:

    **Tick:** 13:00

    **Online Repo:** https://github.com/max-a-ai/lidar-bedlam
    **Local WS:** ~/Documents/lidar-bedlam
    **Cluster (helma):** `lidar-bedlam` -> /hnvme/workspace/v103fe17-lidar-bedlam
    **wandb:** https://wandb.ai/erik_hm/lidar-bedlam
    **Cluster Queue:**

Straight after the `Cluster Queue:` row, the output of `squeue -u $USER`
in a fenced block, verbatim: no added column, no filtered row, no prose
inside the block; an empty queue is the header line alone. `ssh helma sq`
fails (`command not found`) because `.bash_aliases` is not read by a
non-interactive shell -- send `squeue -u $USER` instead. Job names are all
`lidar-bedlam`; the run-name mapping belongs in the interpretation, not
inside the block.

The clock row is the full hour, not the minute the cron fired.

**2. The scoreboard**, as a clickable link on its own row:
`**Scoreboard:** [LiDAR-BEDLAM Scoreboard](https://claude.ai/artifact/WAL7pLwJJYqjs4KKvUBtnK)`

**3. The interpretation**: what changed since the last tick, what finished,
what failed, what you submitted, what it means. Deltas, not absolutes
(`14 -> 10 pending`). One line if nothing changed.

## Hourly routine (from the repo root /home/max/Documents/lidar-bedlam)

    S=<the session's scratchpad directory>
    for i in 1 2 3; do bash lidar_bedlam/scripts/pull_helma_results.sh >$S/pull.log 2>&1 && break; sleep 5; done
    RUNS=$(for i in 1 2 3; do ssh -4 helma 'cd /hnvme/workspace/v103fe17-lidar-bedlam && ls -d outputs/[btavmd]*-[0-9][0-9][0-9] outputs/v2-*-[0-9][0-9][0-9] 2>/dev/null | xargs -n1 basename' 2>/dev/null && break; sleep 5; done | sort -u)
    for r in $RUNS; do mkdir -p outputs/helma/runs/$r; for i in 1 2 3; do rsync -4 -az helma:/hnvme/workspace/v103fe17-lidar-bedlam/outputs/$r/metrics.jsonl outputs/helma/runs/$r/ 2>/dev/null && break; sleep 3; done; done
    find outputs/helma/runs -type d -empty -delete
    for i in 1 2 3 4 5; do ssh -4 helma 'bash -s' < lidar_bedlam/slurm/capture_squeue.sh > $S/squeue.txt 2>/dev/null; grep -q JOBID $S/squeue.txt && break; sleep 6; done  # appends RUN <jobid> <run-name> lines
    .venv/bin/python lidar_bedlam/scripts/running_table.py --squeue-file $S/squeue.txt --update monitor_handoff.md
    uv run python lidar_bedlam/scripts/build_scoreboard.py --stamp "$(date '+%Y-%m-%d %H:%M')" --out $S/scoreboard.html --schedule-md .docs/retrain_schedule.md --squeue-file $S/squeue.txt

`uv run` exits 120 when it has no TTY; if that happens call
`.venv/bin/python <script>` directly, as the running-table line already does.

The pull script does not pick up `b-*` or `t-*` runs, hence the explicit
rsync; add new run names to that list as they start. Eval results land in
`outputs/helma/eval/<run>-last.json`; check the `step` field is the run's
final step before trusting the scoreboard row.

## The "Currently running" table at the top of this file
`lidar_bedlam/scripts/running_table.py` regenerates it; never hand-edit the
block. It replaces whatever sits between `<!-- RUNNING:START -->` and
`<!-- RUNNING:END -->`, so the rest of this file is untouched. The table is
temporary by design: when the queue is empty it says so, and every hourly
tick overwrites it.

One row per job in `squeue` that has a `metrics.jsonl` under
`outputs/helma/runs/`. Columns:

- **step / of target** — the target is `min_steps` while the run is below it
  (every b-ratio run early-stopped exactly there), otherwise `max_steps`;
  the column names which one it used, so a "53 % of max_steps" row is a run
  that already passed its early-stop floor.
- **time left** — remaining steps times the median `perf/step_s` of the last
  20 training rows, plus `EVAL_OVERHEAD_S` (68 s, from the
  `configs/b_ratio_90.yaml` measurement) for every `eval_every_steps` still
  to come. It is a projection from the current rate, not a promise.
- **one column per validation source** — latest MPJPE, an arrow, and the
  percentage change against the previous evaluation.

Arrows (the metrics are errors, so **down is good** — a down arrow is a run
that is still learning):

| arrow | meaning | change vs previous evaluation |
|---|---|---|
| `⇘` | strongly improving | ≤ −3 % |
| `↘` | slightly improving | −3 % … −0.5 % |
| `→` | stale | within ±0.5 % |
| `↗` | slightly worsening | +0.5 % … +3 % |
| `⇗` | strongly worsening | ≥ +3 % |

The thresholds come from the data: consecutive evaluations of the b-ratio
runs swing about ±4 % on Waymo MPJPE, so ±0.5 % is genuinely flat and
anything past 3 % is a move that outruns the noise. Because a single pair of
evaluations is noisy, a lone `⇗` is not yet a regression — if the arrows
flap, rerun with `--smooth 3`, which averages three evaluations per side
before comparing. Change `STRONG`, `WEAK` or `EVAL_OVERHEAD_S` at the top of
the script, not in this file, and note the change here.

## How the scoreboard is built (keep in sync with the builder)
`lidar_bedlam/scripts/build_scoreboard.py` is the only source of the page.
Since 2026-09-21 its `--out` writes the **compact interactive page**
(`lidar_bedlam/scripts/pocket_board.py`: one metric at a time, groups fold,
a tapped row shows every metric plus validation sparklines, "all metrics
side by side" renders the wide table and switches on by itself from 800 px,
so an iPhone in landscape gets the full table). `--classic <path>` still
writes the old static wide page when someone wants it. Both read the same
loaders, so the numbers are identical. The hourly tick publishes `--out` to
the artifact URL below; the classic page needs no artifact of its own.
Group layout, grey rows and the schedule below apply to both pages.
Current layout (2026-09-18):

- Headline table (mains against the published pipelines), "Main runs",
  ablations 1, 2, 3, 4, 5, 7, 8. Ablation 6 and every camera-frame /
  50-40-10 run were removed on 2026-09-18; do not add them back.
- First column "pseudo-GT": derived from each run's `config.json` (the
  Waymo training source directory; `WAYMO_LABEL_DIRS` maps it to v2 / v1 /
  lhmr / pedgen, `real/v1` = keypoints only = red cross). Runs without a
  config.json show an empty cell; pull the config with the results.
- Every run listed in the `RETRAIN` dict is rendered grey (CSS class
  `retrain`), keeps its numbers, is excluded from the best/second/third
  colours, and shows its reason in the last column "retrain". When the user
  asks to grey or ungrey a row, edit `RETRAIN` (run name -> reason), never
  the HTML.
- The `SCHEDULE` list renders the "Retraining schedule" table at the end of
  the page; `--schedule-md` writes the same list to
  `.docs/retrain_schedule.md`. Edit the list, then rebuild.
- Rows whose run has no result yet show "queued" (groups with
  `pending=True`). The main run `a-full-mix80-000` closes every ablation
  table as the reference.
- The builder must stay ruff/mypy clean; after editing it, run the audit and
  `~/.claude/hooks/mark_audited.sh`, and commit it as
  `refactor: ...` or `minor: ...` in one line.

Then republish the scoreboard artifact in place:

- URL: https://claude.ai/code/artifact/ec267f25-2fcd-4f0b-b3ab-5216656c6550
- A new session must first call the Artifact tool with action "read" and
  that URL once. Afterwards publish `$S/scoreboard.html` with `url` set to
  that URL, a label such as "Hourly tick HH:MM", and no favicon or icon.
  Publishing without `url` creates a separate artifact, which is wrong.
- If the publish is refused because another session published a newer
  version, read the artifact again and publish once more; do not force.

## Deploying code to Helma
**Code goes over git, never rsync** (changed 2026-09-18). The workspace used
to be an rsync copy on top of a stale checkout: it sat 150 commits behind on
`1429370`, the live code was untracked, and every `git pull` failed with
"untracked working tree files would be overwritten". It is now a clean
checkout of `origin/main`.

The division of labour: **Claude pushes only after the user's explicit go
in the chat (rule of 2026-09-21; the git hook then asks for confirmation),
then pulls on Helma and confirms it landed.** Never push unasked, and never
rsync code into the workspace again — that is what broke it.

    # after the user has pushed
    ssh -4 helma 'cd /hnvme/workspace/v103fe17-lidar-bedlam && git pull --ff-only'
    ssh -4 helma 'cd /hnvme/workspace/v103fe17-lidar-bedlam && git log --oneline -1 && git status -sb | head -1'

Confirmation is mandatory: Helma's HEAD must equal local `origin/main` and
`git status -sb` must show no divergence; report both hashes. If the pull
refuses, stop and tell the user — do not clear the way with `checkout -f`,
`reset --hard`, `restore`, `clean`, `rm` or `mv`. The general rule is
`git-it` Flow D.

A pull now also brings `pyproject.toml` and `uv.lock`. Watch for that:
compute
nodes have no internet, and a changed lock makes `uv run` try to install.
New shard directories go to `resources/data/generated/<dir>` on Helma (npz
only) with the `.tokens.npy` sidecars hard-linked from `real/v1`; add the
directory to `STAGE_DIRS` when submitting.

## Standing rules (hooks enforce them)
- Claude pushes only after the user's explicit go in the chat, never on its
  own; force pushes are blocked. Commits are
  one line, `<prefix>: <description>`, one `-m`, at most 72 characters,
  lowercase after the colon, prefix in
  add|bug|minor|refactor|docs|test|config|remove.
- Never launch a training without `--wandb-project` and `--wandb-name`
  (`train.sbatch` sets them). Never train on the workstation (a memory
  guard kills it). Do not resubmit or cancel Helma jobs without the user's
  word.
- After any Python or config edit run the general-codebase audit and then
  `~/.claude/hooks/mark_audited.sh`.
- Keep this file current: any change to the monitors, the hourly routine,
  the scoreboard builder's options or the artifact rules is written here in
  the same change.
