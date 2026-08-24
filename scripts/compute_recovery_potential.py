"""
Compute Expected Annual Recovery Burden (EARB) per county.

Manuscript terminology: EARB (expected annual recovery burden,
capacity-equivalent months per year). Legacy identifiers (EARP,
earp_months_per_year, recovery_potential_months) are retained in file and
column names.

Loads all per-event recovery records, weights each county's recovery burden
by the event frequency, and sums across events to produce EARB (months per
year).  Writes analysis_output/earp_per_county.csv for consumption by
notebooks/probabilistic_analysis.ipynb.

Run with:
  conda activate climada_env && python scripts/compute_recovery_potential.py

Inputs:
  data/recovery/recovery_potential.csv
    Columns: event_name, fips, reconstruction_capacity, recovery_potential_months

Output:
  analysis_output/earp_per_county.csv
    fips, earp_months_per_year, num_events,
    total_recovery_months, mean_recovery_per_event, max_recovery
"""

from __future__ import annotations

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

RECOVERY_CSV = BASE_DIR / "data" / "recovery" / "recovery_potential.csv"
OUTPUT_FILE = BASE_DIR / "analysis_output" / "earp_per_county.csv"

DEFAULT_FREQ = 0.00067334  # events per year (Poisson rate used throughout)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_recovery_potential_data(recovery_csv: Path = RECOVERY_CSV) -> pd.DataFrame:
    """Load the consolidated recovery potential CSV."""
    if not recovery_csv.exists():
        raise FileNotFoundError(
            f"Recovery CSV not found: {recovery_csv}. "
            "Run scripts/run_pyrecodes_light.py first."
        )
    print(f"Loading {recovery_csv} …")
    df = pd.read_csv(recovery_csv, dtype={"fips": str})
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    # rename to legacy column name used downstream
    df = df.rename(columns={"recovery_potential_months": "recovery_potential [months]"})
    df["recovery_potential [months]"] = (
        df["recovery_potential [months]"].replace([np.inf, -np.inf], np.nan)
    )
    print(f"Loaded {len(df):,} county-event records "
          f"({df['event_name'].nunique()} events, {df['fips'].nunique()} counties)")
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

def main(recovery_csv: Path = RECOVERY_CSV, out_csv: Path = OUTPUT_FILE):
    """Arguments default to the manuscript (scaling_strength=1.0) paths, so
    `python scripts/compute_recovery_potential.py` with no arguments
    reproduces exactly what it always has. Pass a shadow recovery_csv (see
    scripts/run_pyrecodes_light.py --out-csv) and a separate out_csv to run
    the same analysis for another scaling_strength specification.
    """
    print("=" * 60)
    print("Compute Expected Annual Recovery Potential (EARP)")
    print("=" * 60)

    recovery_df = load_recovery_potential_data(recovery_csv)
    earp_df = compute_earp(recovery_df)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    earp_df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")
    return earp_df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recovery-csv", type=Path, default=RECOVERY_CSV)
    ap.add_argument("--out-csv", type=Path, default=OUTPUT_FILE)
    args = ap.parse_args()
    main(args.recovery_csv, args.out_csv)
