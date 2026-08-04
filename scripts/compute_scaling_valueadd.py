#!/usr/bin/env python3
"""Aggregate the effect of the multi-hazard (rain+surge) scaling on repair demand.

Reads the per-event impact CSVs (which contain both the wind-only `_raw` and the
multi-hazard `_scaled` housing-unit counts per damage state) and writes two small
tables consumed by ``notebooks/probabilistic_analysis.ipynb`` to build the SI
figure ``scaling_valueadd.png``:

  analysis_output/scaling_valueadd_by_county.csv
      fips, wua_wind, wua_all
      Sum over events of weighted units affected (WUA = 1*DS1+1*DS2+3*DS3+6*DS4),
      for wind-only and multi-hazard damage. Their ratio is the per-county
      repair-demand amplification (frequency cancels, so no rate is applied here).

  analysis_output/scaling_valueadd_ds_totals.csv
      ds, units_wind, units_all
      Total housing units in each damage state over the whole event set,
      wind-only vs multi-hazard.

Usage:
    python scripts/compute_scaling_valueadd.py
"""
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PER_EVENT = REPO / "data" / "impact" / "per_event"
OUT = REPO / "analysis_output"
WEIGHTS = {"DS1": 1, "DS2": 1, "DS3": 3, "DS4": 6}


def main():
    files = sorted(PER_EVENT.glob("aggregated_*.csv"))
    if not files:
        raise SystemExit(f"No per-event impact CSVs found in {PER_EVENT}")
    print(f"Aggregating {len(files)} per-event impact files ...")

    by_county = {}
    ds_tot = {ds: [0.0, 0.0] for ds in WEIGHTS}
    for f in files:
        df = pd.read_csv(f)
        df["fips"] = df["fips"].astype(int)
        wua_raw = sum(WEIGHTS[ds] * df[f"units_{ds}_raw"] for ds in WEIGHTS)
        wua_scl = sum(WEIGHTS[ds] * df[f"units_{ds}_scaled"] for ds in WEIGHTS)
        for fips, a, b in zip(df["fips"], wua_raw, wua_scl):
            rec = by_county.setdefault(int(fips), [0.0, 0.0])
            rec[0] += a
            rec[1] += b
        for ds in WEIGHTS:
            ds_tot[ds][0] += df[f"units_{ds}_raw"].sum()
            ds_tot[ds][1] += df[f"units_{ds}_scaled"].sum()

    county = pd.DataFrame(
        [(fips, w, a) for fips, (w, a) in by_county.items()],
        columns=["fips", "wua_wind", "wua_all"],
    ).sort_values("fips")
    county.to_csv(OUT / "scaling_valueadd_by_county.csv", index=False)

    totals = pd.DataFrame(
        [(ds, ds_tot[ds][0], ds_tot[ds][1]) for ds in WEIGHTS],
        columns=["ds", "units_wind", "units_all"],
    )
    totals.to_csv(OUT / "scaling_valueadd_ds_totals.csv", index=False)

    tot_w = county["wua_wind"].sum()
    tot_a = county["wua_all"].sum()
    print(f"  counties: {len(county)}")
    print(f"  total WUA wind-only  : {tot_w:,.0f}")
    print(f"  total WUA all-hazards: {tot_a:,.0f}  ({100*(tot_a-tot_w)/tot_w:+.1f}%)")
    print(f"  wrote scaling_valueadd_by_county.csv, scaling_valueadd_ds_totals.csv")


if __name__ == "__main__":
    main()
