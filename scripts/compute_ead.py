#!/usr/bin/env python3
"""Compute Expected Annual Damage (EAD) per region from per-event aggregated CSVs.

This script multiplies a chosen per-event impact metric (e.g., 'repair_cost_sum_scaled'
or 'units_DS1_raw') by event frequency and sums across events per group (default: `fips`).

By default the script uses a built-in DEFAULT_FREQ = 0.00067334 (≈1/1486) when no
hazard HDF5 is provided. You can optionally pass a Hazard HDF5 (CLIMADA) to obtain
per-event frequencies directly.

Example usages:
  # use default frequency and compute EAD from scaled repair cost
  python3 scripts/compute_ead.py --metric repair_cost_sum_scaled

  # use hazard frequencies and compute EAD for DS1 affected units
  python3 scripts/compute_ead.py --input-dir data/impact/per_event --metric units_DS1_raw \
      --hazard /path/to/tc_ncep_reanal.hdf5 --event-key event_name

"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
IMPACT_DIR = BASE_DIR / "data" / "impact"

DEFAULT_FREQ = 0.00067334


def discover_csvs(input_dir: Path):
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(f"Input directory not found: {p}")
    files = sorted(p.glob("*.csv"))
    return files


def load_combined_csv(files):
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}", file=sys.stderr)
    if not dfs:
        raise SystemExit("No readable CSVs found.")
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    return combined


def load_hazard_freq_map(haz_path: Path, event_key: str = "event_name") -> dict:
    try:
        from climada.hazard.base import Hazard
    except Exception as e:
        raise RuntimeError(f"Failed to import CLIMADA Hazard (needed to read {haz_path}): {e}")
    haz = Hazard.from_hdf5(str(haz_path))
    # hazard.event_name and hazard.event_id and hazard.frequency exist
    if event_key == "event_name":
        keys = list(haz.event_name)
    elif event_key == "event_id":
        keys = [int(x) for x in haz.event_id]
    else:
        raise ValueError("event_key must be 'event_name' or 'event_id'")
    freqs = list(haz.frequency)
    return dict(zip(keys, freqs))


def compute_ead(df: pd.DataFrame, metric: str, freq_map: dict | None, default_freq: float, group_by: list[str], event_key: str):
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found in input CSV columns: {df.columns.tolist()}")

    # ensure numeric
    df = df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)

    # determine event frequency per row
    if freq_map is None:
        df["event_freq"] = float(default_freq)
    else:
        # map by event_key; fall back to default
        if event_key not in df.columns:
            raise KeyError(f"event_key '{event_key}' not found in data columns: {df.columns.tolist()}")
        df["event_freq"] = df[event_key].map(freq_map).astype(float)
        df["event_freq"] = df["event_freq"].fillna(float(default_freq))

    df["ead_contrib"] = df[metric].astype(float) * df["event_freq"].astype(float)

    # group and sum
    grouped = df.groupby(group_by, dropna=False, as_index=False)["ead_contrib"].sum()
    grouped = grouped.rename(columns={"ead_contrib": "EAD_usd_per_year"})
    return grouped


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compute EAD per region from per-event aggregated CSVs")
    p.add_argument("--input-dir", type=Path, default=IMPACT_DIR / "per_event",
                   help="Directory containing per-event aggregated CSVs (default: data/impact/per_event)")
    p.add_argument("--metric", required=True, help="Column name to use as impact metric (e.g. repair_cost_sum_scaled or units_DS1_raw)")
    p.add_argument("--hazard", type=Path, default=None, help="Optional hazard HDF5 to read per-event frequencies from (CLIMADA Hazard)")
    p.add_argument("--default-freq", type=float, default=DEFAULT_FREQ, help=f"Default per-event frequency if hazard not provided (default {DEFAULT_FREQ})")
    p.add_argument("--group-by", type=str, default="fips", help="Comma-separated grouping columns for aggregation (default: fips)")
    p.add_argument("--event-key", type=str, default="event_name", help="Column used to map frequencies from hazard (event_name or event_id). Default 'event_name'.")
    p.add_argument("--out", type=Path, default=None, help="Output CSV path for EAD results (default: <input-dir>/ead_by_<group>.csv)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    files = discover_csvs(args.input_dir)
    print(f"Found {len(files)} CSV files in {args.input_dir}")
    df = load_combined_csv(files)

    freq_map = None
    if args.hazard is not None:
        print(f"Loading hazard frequencies from {args.hazard}")
        freq_map = load_hazard_freq_map(args.hazard, event_key=args.event_key)
        print(f"Loaded {len(freq_map)} event frequencies from hazard.")
    else:
        print(f"No hazard provided. Using DEFAULT_FREQ={args.default_freq} per event.")

    group_by = [g.strip() for g in args.group_by.split(",") if g.strip()]
    ead = compute_ead(df, args.metric, freq_map, args.default_freq, group_by, args.event_key)

    if args.out is None:
        out_name = f"ead_by_{'_'.join(group_by)}.csv"
        out_path = args.input_dir.parent / out_name
    else:
        out_path = args.out

    ead.to_csv(out_path, index=False)
    print(f"Wrote EAD results to: {out_path}  (rows: {len(ead)})")


if __name__ == "__main__":
    main()
