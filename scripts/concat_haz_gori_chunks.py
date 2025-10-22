#!/usr/bin/env python
"""
Concatenate per-chunk CLIMADA TC hazard files into a single hazard.

- Discovers input HDF5 chunks by filename pattern (glob) and concatenates them.
- Streams hazards one-by-one using Hazard.append() to keep peak memory down.
- Writes the final combined hazard to HDF5.

Default locations match the Gori et al. (2025) setup you've been using.

Example:
    python scripts/concat_haz_gori_chunks.py \
        --input-dir /home/groups/bakerjw/smeiler/climada_data/data/hazard/tropical_cyclone/gori \
        --pattern 'tc_ncep_reanal_chunk*.hdf5' \
        --output /home/groups/bakerjw/smeiler/climada_data/data/hazard/tropical_cyclone/gori/tc_ncep_reanal.hdf5
"""

from __future__ import annotations
import re
import argparse
import sys
from pathlib import Path
from typing import Iterable, List

from climada.hazard.base import Hazard
from climada.hazard.centroids.centr import Centroids


def _default_paths() -> tuple[Path, str, Path]:
    """Sherlock-friendly defaults based on your repo layout."""
    haz_dir = Path("/home/groups/bakerjw/smeiler/climada_data/data/hazard") \
        / "tropical_cyclone" / "gori"
    input_dir = haz_dir                                     # where chunk files live
    pattern = "tc_ncep_reanal_chunk*.hdf5"                  # chunk filename pattern
    output_path = haz_dir / "tc_ncep_reanal.hdf5"           # final combined file
    return input_dir, pattern, output_path


_CHUNK_NUM = re.compile(r"chunk(\d+)", re.I)

def _sorted_chunk_files(input_dir: Path, pattern: str) -> List[Path]:
    """Return chunk files sorted by the integer after 'chunk' (natural order)."""
    def key(p: Path) -> tuple[int, str]:
        m = _CHUNK_NUM.search(p.name)
        # fallback to 0 if not found to keep deterministic behavior
        n = int(m.group(1)) if m else 0
        return (n, p.name)
    return sorted(input_dir.glob(pattern), key=key)


def concat_hazards(paths: Iterable[Path]) -> Hazard:
    """
    Concatenate hazards read from the given HDF5 paths.

    Uses an accumulator + append loop (lower peak memory than building a huge list).
    """
    acc: Hazard | None = None
    count = 0
    for p in paths:
        h = Hazard.from_hdf5(p)
        if acc is None:
            # create an empty hazard of same subclass & CRS, then append
            acc = h.__class__(centroids=Centroids(lat=[], lon=[], crs=h.centroids.crs))
            # copy simple top-level identifiers so append has consistent context
            acc.haz_type = h.haz_type
            acc.units = h.units
            acc.frequency_unit = h.frequency_unit
            acc.append(h)
        else:
            acc.append(h)
        count += 1
        print(f"[concat] appended: {p.name} (#{count}, events now: {acc.size})", flush=True)
    if acc is None:
        # No files
        return Hazard()
    return acc


def parse_args(argv: list[str]) -> argparse.Namespace:
    d_in, d_pat, d_out = _default_paths()
    ap = argparse.ArgumentParser(description="Concatenate CLIMADA hazard chunks.")
    ap.add_argument("--input-dir", type=Path, default=d_in,
                    help=f"Directory with chunk HDF5 files (default: {d_in})")
    ap.add_argument("--pattern", type=str, default=d_pat,
                    help=f"Glob pattern for chunk files (default: {d_pat})")
    ap.add_argument("--output", type=Path, default=d_out,
                    help=f"Output HDF5 path for combined hazard (default: {d_out})")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional: only combine the first LIMIT files (for testing).")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    args.input_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    files = _sorted_chunk_files(args.input_dir, args.pattern)
    if args.limit:
        files = files[: args.limit]

    if not files:
        raise SystemExit(f"No input files found in {args.input_dir} matching '{args.pattern}'.")

    print(f"[concat] Found {len(files)} files in {args.input_dir} matching '{args.pattern}'.")
    for i, p in enumerate(files[:5], 1):
        print(f"  {i:>3}: {p.name}")
    if len(files) > 5:
        print("  ...")

    haz = concat_hazards(files)

    if haz.size == 0:
        raise SystemExit("Combined hazard has 0 events—nothing to write.")

    print(f"[concat] Writing combined hazard to: {args.output}")
    haz.write_hdf5(args.output)
    print("[concat] Done.")
    return args.output


if __name__ == "__main__":
    main()
