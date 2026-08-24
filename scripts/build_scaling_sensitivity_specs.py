#!/usr/bin/env python3
"""Build the scaling_strength=0.0 (wind-only) per-event damage input used to
reproduce the EARB panel of Figure S9 (scaling_valueadd.png,
notebooks/probabilistic_analysis.ipynb) via
scripts/compute_scaling_valueadd_recovery.py.

g=0.0 is EXACT: the per-event impact CSVs in data/impact/per_event/ already
store the wind-only damage-state bin counts (`_raw`) for the full synthetic
event set, so this script only needs to repackage them into the shadow
per-event schema consumed by scripts/run_scaling_sensitivity_pipeline.py
(and, downstream, the rest of the Stage 5/6 pipeline) -- it does not
recompute or approximate anything.

Output
------
  data/impact_sensitivity/g0.0/per_event/all_events.csv
      Exact copy of the wind-only (`_raw`) damage-state counts, in the
      `_scaled` columns, so the shadow spec can be fed through the
      manuscript's own Stage 5/6 scripts unchanged (see
      scripts/run_scaling_sensitivity_pipeline.py).

Usage:
    python scripts/build_scaling_sensitivity_specs.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PER_EVENT_DEFAULT = REPO / "data" / "impact" / "per_event"
OUT_ROOT = REPO / "data" / "impact_sensitivity"
DS_KEYS = ["DS1", "DS2", "DS3", "DS4"]


def load_per_event(per_event_dir: Path) -> pd.DataFrame:
    files = sorted(Path(per_event_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No per-event CSVs in {per_event_dir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def build_wind_only_spec(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-event-schema DataFrame with units_DS{k}_scaled set equal
    to units_DS{k}_raw (g=0.0 is wind-only by definition)."""
    out = df[["event_index", "event_name", "stcode", "ccode", "fips",
              *[f"units_{k}_raw" for k in DS_KEYS], "repair_cost_sum_raw"]].copy()
    for k in DS_KEYS:
        out[f"units_{k}_scaled"] = out[f"units_{k}_raw"]
    out["repair_cost_sum_scaled"] = out["repair_cost_sum_raw"]
    cols = ["event_index", "event_name", "stcode", "ccode", "fips"]
    for k in DS_KEYS:
        cols += [f"units_{k}_raw", f"units_{k}_scaled"]
    cols += ["repair_cost_sum_raw", "repair_cost_sum_scaled"]
    return out[cols]


def main():
    print(f"Loading per-event impact CSVs from {PER_EVENT_DEFAULT} ...")
    df = load_per_event(PER_EVENT_DEFAULT)
    print(f"  {len(df):,} (event, county) rows")

    out = build_wind_only_spec(df)
    out_dir = OUT_ROOT / "g0.0" / "per_event"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_events.csv"
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")
    print("\nDone. g=1.0 needs no shadow spec: use data/impact/per_event/ directly.")


if __name__ == "__main__":
    main()
