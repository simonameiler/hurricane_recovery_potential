"""
Compute Expected Annual Recovery Potential (EARP) per county.

Loads all per-event pyrecodes recovery JSON files, weights each county's
recovery time by the event frequency, and sums across events to produce
EARP (months per year).  Writes analysis_output/earp_per_county.csv for
consumption by create_manuscript_figures.py, create_bivariate_maps.py,
and nri_discriminant_assessment.py.

Run with:
  conda activate climada_env && python scripts/compute_recovery_potential.py

Inputs:
  data/recovery_potential_per_scenario/   pyrecodes per-event JSONs
    Each file: {event_id}_scaled_recovery_potential.json
    Each record: {"fips": ..., "recovery_potential [months]": ...,
                  "reconstruction_capacity": ..., "event": ...}

Output:
  analysis_output/earp_per_county.csv
    fips, earp_months_per_year, num_events,
    total_recovery_months, mean_recovery_per_event, max_recovery
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

RECOVERY_DIR = BASE_DIR / "data" / "recovery_potential_per_scenario"
OUTPUT_FILE = BASE_DIR / "analysis_output" / "earp_per_county.csv"

DEFAULT_FREQ = 0.00067334  # events per year (Poisson rate used throughout)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_recovery_potential_data() -> pd.DataFrame:
    """Load all per-event pyrecodes recovery JSONs into a single DataFrame."""
    recovery_files = sorted(RECOVERY_DIR.glob("*_scaled_recovery_potential.json"))
    if not recovery_files:
        raise FileNotFoundError(
            f"No recovery JSON files found in {RECOVERY_DIR}. "
            "Ensure pyrecodes outputs are present before running this script."
        )
    print(f"Found {len(recovery_files)} recovery JSON files – loading …")

    frames = []
    for idx, f in enumerate(recovery_files, 1):
        if idx % 500 == 0:
            print(f"  {idx}/{len(recovery_files)} …")
        with open(f) as fh:
            frames.append(pd.DataFrame(json.load(fh)))

    df = pd.concat(frames, ignore_index=True)
    df["fips"] = df["fips"].astype(str).str.zfill(5)

    # Infinite recovery times arise when reconstruction_capacity == 0; treat as NaN.
    df["recovery_potential [months]"] = (
        df["recovery_potential [months]"].replace([np.inf, -np.inf], np.nan)
    )

    print(f"Loaded {len(df):,} county-event records "
          f"({df['event'].nunique()} events, {df['fips'].nunique()} counties)")
    return df


# ---------------------------------------------------------------------------
# EARP computation
# ---------------------------------------------------------------------------

def compute_earp(recovery_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Expected Annual Recovery Potential per county.

    EARP = Σ_events (recovery_potential_e * freq)

    where freq = DEFAULT_FREQ (events/year, Poisson rate assumed equal for
    all synthetic events following Gori et al. 2025).

    Returns a DataFrame with one row per county.
    """
    recovery_df = recovery_df.copy()
    recovery_df["weighted_recovery"] = (
        recovery_df["recovery_potential [months]"] * DEFAULT_FREQ
    )

    earp = recovery_df.groupby("fips").agg(
        earp_months_per_year=("weighted_recovery", "sum"),
        num_events=("recovery_potential [months]", "count"),
        total_recovery_months=("recovery_potential [months]", "sum"),
        mean_recovery_per_event=("recovery_potential [months]", "mean"),
        max_recovery=("recovery_potential [months]", "max"),
    ).reset_index()

    earp = earp.replace([np.inf, -np.inf], np.nan)

    finite = earp["earp_months_per_year"].dropna()
    print(f"\nEARP computed for {len(earp)} counties "
          f"({finite.gt(0).sum()} with positive values)")
    print(f"  Mean  EARP : {finite.mean():.4f} months/year")
    print(f"  Median EARP: {finite.median():.4f} months/year")
    print(f"  Max    EARP: {finite.max():.4f} months/year")
    return earp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Compute Expected Annual Recovery Potential (EARP)")
    print("=" * 60)

    recovery_df = load_recovery_potential_data()
    earp_df = compute_earp(recovery_df)

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    earp_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
