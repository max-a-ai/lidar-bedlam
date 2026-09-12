"""Stream selected BEDLAM frames out of the NAS tar archives.

The archives live on a slow sshfs mount, so they are read exactly once,
sequentially, and only the wanted members are written to the local RAID.

Example (6 fps subset of one group, all sequences)::

    uv run python lidar_bedlam/scripts/extract_bedlam.py \
        --group 20221024_3-10_100_batch01handhair_static_highSchoolGym \
        --frame-stride 5 --modalities png depth masks gt

Layout written: ``<out>/<group>/{png,depth,masks,ground_truth}/...`` as
inside the archives (the leading ``<group>/`` component is kept).
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
from pathlib import Path
from typing import Literal

DEFAULT_SRC = Path("resources/data/bedlam")
DEFAULT_OUT = Path("resources/data/generated/bedlam_raw")
FRAME_RE = re.compile(r"seq_(\d{6})_(\d{4})")


def wanted(name: str, sequences: set[str] | None, stride: int) -> bool:
    """Keep a member if its sequence and frame pass the filters."""
    m = FRAME_RE.search(name)
    if m is None:
        return name.endswith(".csv")  # ground-truth CSVs
    seq, frame = f"seq_{m.group(1)}", int(m.group(2))
    if sequences is not None and seq not in sequences:
        return False
    return frame % stride == 0


def extract(
    archive: Path, out: Path, sequences: set[str] | None, stride: int
) -> int:
    """Stream ``archive`` and write wanted members under ``out``."""
    mode: Literal["r|gz", "r|"] = "r|gz" if archive.suffix == ".gz" else "r|"
    n = 0
    with tarfile.open(archive, mode) as tar:
        for member in tar:
            if not member.isfile() or not wanted(
                member.name, sequences, stride
            ):
                continue
            tar.extract(member, out, filter="data")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", required=True, help="sequence group name")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--sequences", nargs="*", help="e.g. seq_000000")
    ap.add_argument(
        "--modalities", nargs="+", default=["gt", "masks", "png", "depth"]
    )
    args = ap.parse_args(argv)
    seqs = set(args.sequences) if args.sequences else None
    total = 0
    for modality in args.modalities:
        pattern = f"{args.group}_{modality}.*tar*"
        archives = sorted(args.src.glob(pattern))
        if not archives:
            sys.stderr.write(f"no archives for {pattern}\n")
            return 1
        for archive in archives:
            n = extract(archive, args.out, seqs, args.frame_stride)
            total += n
            sys.stdout.write(f"{archive.name}: {n} files\n")
            sys.stdout.flush()
    sys.stdout.write(f"done: {total} files under {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
