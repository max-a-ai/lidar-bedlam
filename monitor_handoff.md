# Monitoring handoff (Helma runs and the scoreboard)

<!-- RUNNING:START -->

## Currently running (regenerated every hourly tick)

_Nothing in the queue._

<!-- RUNNING:END -->

The monitoring setting for this project moved on 2026-10-15 to the central
store, so it can be pulled on any machine without this repo:

    ~/.claude-skills/monitoring-settings/lidar-bedlam.md

That file is the single source of truth for the routine: cluster paths, the run
classification (mains, ablations, debug, short tests), the evaluation
parameters, the scoreboard options, the artifact URL and the known traps. The
invariant rules it rests on are the `tick-it` and `run-monitoring` skills.

Only the live table above stays here. `lidar_bedlam/scripts/running_table.py`
regenerates it between the `RUNNING` markers every hourly tick — never
hand-edit it, and never delete the markers. It lives in this repo rather than
in the setting so that an hourly rewrite does not churn the skills repo.

History of the routine before the move: `git log -- monitor_handoff.md`.

To pick the routine up in a new session:

    Read ~/.claude-skills/monitoring-settings/lidar-bedlam.md and run the
    hourly routine exactly as described. Report only what it says to report.
