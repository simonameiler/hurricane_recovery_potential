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


def _pick_chunk(files: list[Path], N: int, chunk_id: int) -> list[Path]:
    start = chunk_id * N
    end = min(start + N, len(files))
    if start >= len(files):
        raise IndexError(
            f"chunk_id {chunk_id} with N={N} starts at {start}, but only {len(files)} files found."
        )
    return files[start:end]


def main(N: int, chunk_id: int | None = None) -> "Hazard":
    """Build a Hazard object for a chunk of N events; returns the object (and writes HDF5)."""
    if chunk_id is None:
        # allow SLURM array usage without passing explicitly
        sid = os.environ.get("SLURM_ARRAY_TASK_ID")
        if sid is None:
            raise ValueError("chunk_id not provided and SLURM_ARRAY_TASK_ID not set.")
        chunk_id = int(sid)

    files_all = sorted(p for p in MAT_DIR.glob("*.mat") if p.is_file())
    if not files_all:
        raise FileNotFoundError(f"No .mat files in {MAT_DIR}")

    subset = _pick_chunk(files_all, N=N, chunk_id=chunk_id)
    print(
        f"[HRP] chunk_id={chunk_id} | N={N} | using {len(subset)} files: "
        f"{subset[0].name} .. {subset[-1].name}"
    )

    scratch = Path(os.environ.get("SCRATCH", "/tmp"))

    # materialize a temp dir containing only this chunk via symlinks
    with TemporaryDirectory(dir=scratch) as tmp:
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
        haz.tag = f"gori_ncep_reanal_chunk{chunk_id}_N{N}"
    except Exception:
        pass

    out_path = OUT_DIR / f"tc_ncep_reanal_chunk{chunk_id}_N{N}.hdf5"
    print(f"[HRP] writing HDF5 -> {out_path}")
    haz.write_hdf5(out_path)
    print("[HRP] done.")

    return haz


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
