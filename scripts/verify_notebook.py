"""Execute ``debug/capabilities.ipynb`` and fail on any cell or widget error.

Widget cells (ipywidgets ``Output``) make nbclient wait for its per-cell
timeout after the work is done, so the timeout is kept short; every cell's
real work finishes in well under a minute.

Usage::

    uv run python scripts/verify_notebook.py [--timeout 120]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient

SRC = Path("debug/capabilities.ipynb")
DST = Path("debug/capabilities.executed.ipynb")


def widget_errors(nb: Any) -> list[str]:
    """Errors captured inside ipywidgets Output models."""
    meta = nb.metadata.get("widgets", {})
    state = meta.get("application/vnd.jupyter.widget-state+json", {})
    found: list[str] = []
    for model in state.get("state", {}).values():
        if model.get("model_name") != "OutputModel":
            continue
        for out in model["state"].get("outputs", []):
            if out.get("output_type") == "error":
                found.append(f"widget: {out['ename']}: {out['evalue'][:200]}")
    return found


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args(argv)
    nb = nbformat.read(SRC, as_version=4)
    client = NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(SRC.parent)}},
    )
    t0 = time.time()
    client.execute()
    nbformat.write(nb, DST)
    errors = [
        f"cell {i}: {o['ename']}: {o['evalue'][:200]}"
        for i, c in enumerate(nb.cells)
        if c.cell_type == "code"
        for o in c.outputs
        if o.output_type == "error"
    ] + widget_errors(nb)
    for e in errors:
        sys.stderr.write(e + "\n")
    sys.stdout.write(
        f"{len(nb.cells)} cells in {time.time() - t0:.0f} s, "
        f"{len(errors)} errors, wrote {DST}\n"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
