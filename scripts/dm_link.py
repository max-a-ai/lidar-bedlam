"""Materialise ``data/``, ``checkpoints/`` and ``outputs/`` on this machine.

Reads ``config-global.json``: the ``hosts`` block maps a hostname to the
root paths of that machine, and every dataset / checkpoint entry names its
source with ``{root}`` placeholders. The script creates symlinks
``data/<name>`` and ``checkpoints/<name>`` (replacing existing links) and
the ``outputs/`` directory. Keys starting with ``_`` are documentation.

    python3 scripts/dm_link.py            # link everything for this host
    python3 scripts/dm_link.py --check    # only report what is missing
    python3 scripts/dm_link.py --smoke    # link, then verify the smoke inputs
    python3 scripts/dm_link.py --host helma
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config-global.json"


def entries(section: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The real entries of a section (documentation keys skipped)."""
    return {
        k: v
        for k, v in section.items()
        if not k.startswith("_") and isinstance(v, dict)
    }


def resolve(template: str, roots: dict[str, str]) -> Path | None:
    """Fill ``{root}`` placeholders; None if this host lacks the root."""
    try:
        return Path(template.format(**roots))
    except KeyError:
        return None


def link(target: Path, path: Path, check: bool) -> str:
    """Create or verify one symlink; returns a status word."""
    if not target.exists():
        return "missing"
    if check:
        return "ok" if path.resolve() == target.resolve() else "unlinked"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(target)
    return "linked"


def main() -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(CONFIG.read_text())
    roots: dict[str, str] = cfg["hosts"].get(args.host, {})
    if not roots:
        known = [h for h in cfg["hosts"] if not h.startswith("_")]
        sys.stderr.write(f"host {args.host!r} not in config; known: {known}\n")
        return 2
    problems = 0
    for section, folder in (
        ("datasets", "data"),
        ("checkpoints", "checkpoints"),
    ):
        for name, entry in entries(cfg[section]).items():
            target = resolve(entry["source"], roots)
            if target is None:
                continue  # not available on this host
            if entry.get("create") and not args.check:
                target.mkdir(parents=True, exist_ok=True)
            status = link(target, ROOT / folder / name, args.check)
            problems += status in ("missing", "unlinked")
            print(f"{folder}/{name:18s} {status:9s} {target}")
    (ROOT / "outputs").mkdir(exist_ok=True)
    if args.smoke:
        for rel in cfg["smoke"]["needs"]:
            ok = (ROOT / rel).exists()
            problems += not ok
            print(f"smoke {'ok' if ok else 'missing':9s} {rel}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
