#!/usr/bin/env python3
"""
Build a tropical cyclone Hazard object from Gori et al. (2025) wind-field chunks.

- Process a *chunk* of N events so you can run many chunks in parallel (e.g., SLURM array).
- Per-event frequency is set to (catalog freq) / TOTAL_STORMS, with TOTAL_STORMS hard-coded
  to 5018 so that each subset already carries the final weighting.

CLI examples:
  python scripts/make_haz_gori.py --chunk-size 250 --chunk-id 0
  python scripts/make_haz_gori.py --chunk-size 250 --chunk-id 1
  # If --chunk-id is omitted, falls back to $SLURM_ARRAY_TASK_ID.

Programmatic:
  from scripts.make_haz_gori import main
  haz = main(N=250, chunk_id=0)  # returns the Hazard object
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from modules.hazard_utils import (
    load_tc_hazard_from_wind_mats,
    _read_catalog_freq,   # uses track .mat to get scalar freq
)

# -------------------- paths & constants --------------------
HAZ_DIR = Path("/home/groups/bakerjw/smeiler/climada_data/data/hazard")
MAT_DIR = HAZ_DIR / "tropical_cyclone" / "gori" / "ncep_reanal"
TRACK_MAT_PATH = (
    HAZ_DIR / "tropical_cyclone" / "gori" / "UScoast6_AL_ncep_reanal_roEst1rmEst1_trk100.mat"
)

TOTAL_STORMS = 5018  # hard-coded total across the whole catalog (requested)

OUT_DIR = HAZ_DIR / "tropical_cyclone" / "gori"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# use $SCRATCH if available for TemporaryDirectory
import os as _os
from pathlib import Path as _Path
SCRATCH = _os.environ.get("SCRATCH", "/tmp")



def _pick_chunk(files: list[Path], N: int, chunk_id: int) -> list[Path]:
    start = chunk_id * N
    end = min(start + N, len(files))
    if start >= len(files):
        raise IndexError(
            f"chunk_id {chunk_id} with N={N} starts at {start}, but only {len(files)} files found."
        )
    return files[start:end]


def process_chunk(subset: list[Path], cid: int, N: int, out_dir: Path = OUT_DIR, scratch_dir: str = SCRATCH):
    """Process a single chunk: symlink files into a temp dir, build hazard, set freqs, write HDF5.

    Returns the built Hazard object.
    """
    from tempfile import TemporaryDirectory

    if not subset:
        print(f"[HRP] process_chunk: empty subset for chunk {cid}")
        return None

    # materialize a temp dir containing only this chunk via symlinks
    with TemporaryDirectory(dir=scratch_dir) as tmp:
        tmp_dir = Path(tmp)
        for fp in subset:
            (tmp_dir / fp.name).symlink_to(fp)

        # build hazard for this subset
        haz = load_tc_hazard_from_wind_mats(
            mat_dir=tmp_dir,
            track_mat_path=TRACK_MAT_PATH,
            resolution_deg=0.1,
            pad_deg=1.0,
        )

    # ---- set per-event frequency = freq_scalar / TOTAL_STORMS (hard-coded total) ----
    freq_scalar = _read_catalog_freq(TRACK_MAT_PATH)  # events/year scalar
    per_event = float(freq_scalar) / float(TOTAL_STORMS)
    haz.frequency = np.full(haz.event_id.size, per_event, dtype=float)
    try:
        haz.frequency_unit = "1/year"
    except AttributeError:
        pass

    # tag & write
    try:
        haz.tag = f"gori_ncep_reanal_chunk{cid}_N{N}"
    except Exception:
        pass

    out_path = out_dir / f"tc_ncep_reanal_chunk{cid}_N{N}.hdf5"
    print(f"[HRP] writing HDF5 -> {out_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    haz.write_hdf5(out_path)
    print("[HRP] done.")

    return haz


def main(N: int | None = None, chunk_id: int | None = None, mat_dir: Path | None = None, scratch_dir: str | None = None):
    """Entry point. Can be used programmatically (main(N=250, chunk_id=0)) or as CLI.

    If N is None, parse CLI args.
    """
    if N is None:
        parser = argparse.ArgumentParser(description="Build chunked CLIMADA TC hazard (Gori et al. 2025).")
        parser.add_argument("--chunk-size", type=int, default=250, help="Number of tracks per chunk")
        parser.add_argument("--chunk-id", type=int, default=None, help="If set, only process this chunk id (zero-based)")
        parser.add_argument("--mat-dir", type=str, default=str(MAT_DIR), help="Directory with .mat tracks")
        args = parser.parse_args()
        N = args.chunk_size
        chunk_id = args.chunk_id
        # If running under SLURM array and --chunk-id not provided, use SLURM_ARRAY_TASK_ID
        if chunk_id is None:
            sl = os.environ.get("SLURM_ARRAY_TASK_ID")
            if sl is not None and sl != "":
                try:
                    chunk_id = int(sl)
                except Exception:
                    pass
        mat_dir = Path(args.mat_dir)
    else:
        mat_dir = mat_dir or MAT_DIR
        scratch_dir = scratch_dir or SCRATCH

    # --- build index -> file mapping based on filename numeric id (preserve gaps) ---
    files_by_index = {}
    pat = re.compile(r'(\d+)\.mat$', re.IGNORECASE)
    for p in sorted(mat_dir.glob("*.mat")):
        m = pat.search(p.name)
        if not m:
            continue
        num = int(m.group(1))
        idx = num - 1  # explicit index based on filename number
        files_by_index[idx] = p

    if not files_by_index:
        print(f"[HRP] No .mat files found in {mat_dir}")
        return

    # determine available index range for diagnostics (but keep indexing based on filename numbers)
    min_idx = min(files_by_index.keys())
    max_idx = max(files_by_index.keys())
    total_possible = max_idx + 1
    print(f"[HRP] Found {len(files_by_index)} .mat files with indices {min_idx}..{max_idx} (total slots {total_possible})")

    # helper to get files for a given chunk id by index slots (preserve gaps)
    def files_for_chunk(cid: int):
        start = cid * N
        end = start + N  # exclusive
        selected = [files_by_index[i] for i in range(start, end) if i in files_by_index]
        return selected, start, end - 1

    # If a specific chunk_id requested, process only that chunk
    if chunk_id is not None:
        subset, start_idx, end_idx = files_for_chunk(chunk_id)
        if not subset:
            print(f"[HRP] Chunk {chunk_id} (indices {start_idx}-{end_idx}) contains no existing .mat files -> nothing to do")
            return
        print(f"[HRP] chunk_id={chunk_id} | N={N} | using indices {start_idx}..{end_idx} with {len(subset)} existing files")
        return process_chunk(subset, chunk_id, N, out_dir=OUT_DIR, scratch_dir=(scratch_dir or SCRATCH))

    # Otherwise process all chunks across the full index range (0..max_idx) in sequence
    n_chunks = (max_idx // N) + 1
    for cid in range(n_chunks):
        subset, start_idx, end_idx = files_for_chunk(cid)
        if not subset:
            print(f"[HRP] skipping empty chunk {cid} (indices {start_idx}-{end_idx})")
            continue
        print(f"[HRP] chunk_id={cid} | N={N} | using indices {start_idx}..{end_idx} with {len(subset)} existing files")
        process_chunk(subset, cid, N, out_dir=OUT_DIR, scratch_dir=(scratch_dir or SCRATCH))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build chunked CLIMADA TC hazard (Gori et al. 2025).")
    p.add_argument("--chunk-size", type=int, required=True, help="N events per chunk")
    p.add_argument(
        "--chunk-id",
        type=int,
        default=None,
        help="Zero-based chunk index (defaults to $SLURM_ARRAY_TASK_ID).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(N=args.chunk_size, chunk_id=args.chunk_id)
