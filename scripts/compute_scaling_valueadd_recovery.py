#!/usr/bin/env python3
"""Aggregate the effect of the multi-hazard (rain+surge) scaling on recovery
burden (EARB) -- the recovery-burden analog of compute_scaling_valueadd.py
(which does the same comparison for repair demand, EARD).

Compares the wind-only (scaling_strength g=0) and default (g=1, manuscript)
EARB per county. Both are already exact, pre-computed outputs of
scripts/compute_recovery_potential.py:
    analysis_output/scaling_sensitivity/g0.0/earp_per_county.csv  (g=0, wind-only)
    analysis_output/earp_per_county.csv                            (g=1, manuscript default)
g=0.0 is exact here too: build_scaling_sensitivity_specs.py's g=0 shadow
per-event spec is an exact copy of the wind-only `_raw` damage-state counts,
not an approximation, so no new per-event aggregation is done in this
script -- it only merges the two already-computed county tables.

Unlike repair demand (a linear sum of weighted damage-state unit counts),
recovery burden B_{e,c} = max(floor, demand/capacity) is nonlinear and
county-specific (capacity varies), so it has no meaningful damage-state
decomposition; this script produces only the county-level comparison used by
the EARB panel of Figure S9
(notebooks/probabilistic_analysis.ipynb, figS9_scaling_valueadd),
not a per-damage-state breakdown.

Output:
  analysis_output/scaling_valueadd_recovery_by_county.csv
      fips, earb_wind, earb_scaled, earb_diff, earb_ratio
      earb_wind/earb_scaled: EARB [months yr⁻¹], wind-only vs multi-hazard.
      earb_diff = earb_scaled - earb_wind.
      earb_ratio = earb_scaled / earb_wind (NaN where earb_wind == 0).

Usage:
    python scripts/compute_scaling_valueadd_recovery.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "analysis_output"
WIND_EARP = OUT / "scaling_sensitivity" / "g0.0" / "earp_per_county.csv"
SCALED_EARP = OUT / "earp_per_county.csv"


def main():
    if not WIND_EARP.exists():
        raise SystemExit(
            f"{WIND_EARP} not found; run scripts/build_scaling_sensitivity_specs.py "
            "and scripts/run_scaling_sensitivity_pipeline.py first"
        )
    wind = pd.read_csv(WIND_EARP, dtype={"fips": str})[["fips", "earp_months_per_year"]]
    wind = wind.rename(columns={"earp_months_per_year": "earb_wind"})
    scaled = pd.read_csv(SCALED_EARP, dtype={"fips": str})[["fips", "earp_months_per_year"]]
    scaled = scaled.rename(columns={"earp_months_per_year": "earb_scaled"})

    county = wind.merge(scaled, on="fips", how="outer").sort_values("fips")
    county["earb_diff"] = county["earb_scaled"] - county["earb_wind"]
    county["earb_ratio"] = np.where(
        county["earb_wind"] > 0, county["earb_scaled"] / county["earb_wind"], np.nan
    )
    county.to_csv(OUT / "scaling_valueadd_recovery_by_county.csv", index=False)

    tw, ta = county["earb_wind"].sum(), county["earb_scaled"].sum()
    n_gray = int(county["earb_ratio"].isna().sum())
    print(f"  counties: {len(county)}")
    print(f"  total EARB wind-only : {tw:,.1f} months/yr")
    print(f"  total EARB scaled    : {ta:,.1f} months/yr  ({100 * (ta - tw) / tw:+.1f}%)")
    print(f"  counties with zero wind-only EARB (ratio undefined): {n_gray}")
    print("  wrote scaling_valueadd_recovery_by_county.csv")


if __name__ == "__main__":
    main()
