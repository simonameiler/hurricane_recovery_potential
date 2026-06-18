#!/usr/bin/env python3
"""
run_pyrecodes_light.py  (STREAMLINED)

Reads per_event impact CSVs (data/impact/per_event/) DIRECTLY and writes ONE
consolidated recovery table (data/recovery/recovery_potential.csv).

Input : <impact>/per_event/aggregated_*.csv   (county x event; needs columns
         event_name, fips, units_DS1_scaled..units_DS4_scaled)
        permit CSV with FIPS, Average_Building_Permits(12 months)
Output: <recovery>/recovery_potential.csv      (event_name, fips,
         reconstruction_capacity, recovery_potential_months)  -- NaN for zero capacity

recovery = max(floor, demand/capacity);  demand = Σ units_DS{ds}_scaled·τ[ds];
τ = {DS1:1, DS2:1, DS3:3, DS4:6};  floor = longest τ among present damage states.
"""
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

TAU = {1: 1, 2: 1, 3: 3, 4: 6}
PERMIT_COL = "Average_Building_Permits(12 months)"

DEFAULT_PER_EVENT_DIR = BASE_DIR / "data" / "impact" / "per_event"
DEFAULT_PERMIT_FILE = BASE_DIR / "data" / "selected_states_counties_with_permits.csv"
DEFAULT_OUT_CSV = BASE_DIR / "data" / "recovery" / "recovery_potential.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-event-dir", type=Path, default=DEFAULT_PER_EVENT_DIR)
    ap.add_argument("--permit-file", type=Path, default=DEFAULT_PERMIT_FILE)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--capacity-modifier", type=float, default=1.0)
    a = ap.parse_args()

    perm = pd.read_csv(a.permit_file)
    cap = dict(zip(perm["FIPS"].astype(int),
                   pd.to_numeric(perm[PERMIT_COL], errors="coerce").fillna(0.0) * a.capacity_modifier))

    files = sorted(a.per_event_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No per_event CSVs in {a.per_event_dir}")
    print(f"Loading {len(files)} per-event CSV files ...")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    df["fips_i"] = df["fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")
    df["fips_i"] = pd.to_numeric(df["fips_i"], errors="coerce").astype("Int64")
    df["demand"] = sum(df[f"units_DS{k}_scaled"] * TAU[k] for k in TAU)
    sc = [f"units_DS{k}_scaled" for k in TAU]
    df["floor"] = df[sc].gt(0).mul([TAU[k] for k in TAU]).max(axis=1)
    df["cap"] = df["fips_i"].map(cap).fillna(0.0)
    # NaN where capacity is zero/absent — cleaner than inf
    df["recov"] = np.where(df["cap"] <= 0, np.nan,
                           np.maximum(df["floor"], df["demand"] / df["cap"]))

    out = pd.DataFrame({
        "event_name": df["event_name"].astype(str),
        "fips": df["fips_i"].astype("Int64").astype(str),  # unpadded, matches permit/NRI convention
        "reconstruction_capacity": df["cap"].astype(float),
        "recovery_potential_months": df["recov"].astype(float),  # NaN where capacity 0
    }).sort_values(["event_name", "fips"]).reset_index(drop=True)

    out.to_csv(a.out_csv, index=False)
    print(f"Wrote {a.out_csv}  ({len(out):,} county-event rows, "
          f"{out.event_name.nunique()} events, {out.fips.nunique()} counties)")


if __name__ == "__main__":
    main()
