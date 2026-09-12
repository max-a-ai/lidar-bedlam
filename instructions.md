# Agent Coding Standards

Enforce on every change.

## Hard rules

- **UV only.** `uv run`, `uv sync`, `uv add`. Never `pip` directly.
  Reproducibility is paramount.
- **Python version** is pinned in `.python-version` and `pyproject.toml`.
  Don't silently upgrade.
- **Line length: 79.** Ruff enforces. No escape hatches.
- **Functions ≤ ~70 lines.** Split if longer.
- **Type hints everywhere.** `mypy --strict` must pass.
- **Format with Ruff:** `uv run ruff format` + `uv run ruff check --fix`.
- **Slim-first.** Prefer stdlib + subprocess over heavy frameworks. New
  runtime deps need a rationale line in `HANDOFF.md`'s decisions table.

## Workflow

- Keep `.docs/progress.md` updated each session. It has three parts
  sharing the same section names: the gantt timetable at the top, the
  `# Log` in the middle, `# Todos` at the bottom.
- Append log entries as `### YYYY-MM-DD HH:MM — <what> (<commit>)` under
  the section the work belongs to, so any line traces back to a diff.
- Promote finished todos by deleting them; add newly-discovered ones
  under the matching section.
- Keep `HANDOFF.md` current. When you learn something future-you would
  need to pick this up cold, write it down there.
- Every training run writes to `outputs/<run-name>/`. A run worth
  keeping is *copied* into `.docs/runs/<run-name>/` — never moved.

## Verification before saying "done"

1. `uv run ruff check src/` → zero errors
2. `uv run ruff format --check src/` → clean
3. `uv run mypy src/` → zero errors
4. If there's an entry point: `uv run <script-name>` → it launches
   without crashing on the expected platform.

## Directories

`data/`, `checkpoints/` and `outputs/` are gitignored and built per
machine by `python3 scripts/dm_link.py`. Never commit them, never
hand-create a fourth name for the same idea, and never put a `.venv`
inside an HPC workspace — those filesystems are limited by inodes.

## Project-specific

- Training always goes through `main.py` with `--wandb-project` and
  `--wandb-name` (hook-enforced); runs land in `outputs/<run-name>/`
  with `last.pt`, `best.pt`, `val_*.json`, `wandb/` and `DONE`.
- Shards and precomputed ViT tokens live under `data/generated/`; the
  layout is documented in `.docs/data_pipeline.md`.
- The paper is `.docs/latex-draft/main.tex`; build it with `latexmk` in
  that folder. Keep it double-blind (no rig or group names).
- `.docs/progress.md` sections are Data, Model, Experiments, Paper,
  Housekeeping; timetable, log and todos use exactly these names.
