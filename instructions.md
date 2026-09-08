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

- Keep `.vault-<project>/progress.md` updated each session — append a
  dated log entry (instruction, decisions, steps, result).
- Keep `.vault-<project>/todo.md` updated — promote items to done
  (delete), add newly-discovered items at the end.
- Keep `HANDOFF.md` current. When you learn something future-you would
  need to pick this up cold, write it down there.
- When you add a new source module, add a node for it to the Obsidian
  canvas in the vault so the dashboard stays current.

## Verification before saying "done"

1. `uv run ruff check src/` → zero errors
2. `uv run ruff format --check src/` → clean
3. `uv run mypy src/` → zero errors
4. If there's an entry point: `uv run <script-name>` → it launches
   without crashing on the expected platform.

## Obsidian canvas gotcha

Every node and edge in a `.canvas` JSON file MUST include
`"styleAttributes": {}`. Missing this causes silent render failure in
Obsidian ≥1.5. When hand-editing the canvas, preserve this field.
