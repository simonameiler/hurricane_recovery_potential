"""Recovery-potential simulation (pyrecodes light).

Analytical stand-in for a full pyrecodes resource-constrained recovery
simulation. For every county-event pair:

    recovery = max(floor, demand / capacity)

where demand is the HAZUS unit-month repair demand summed over damage states
(demand = sum_k units_DS{k}_scaled * tau_k, tau = {DS1: 1, DS2: 1, DS3: 3,
DS4: 6} months), the floor is the longest tau among the damage states present,
and capacity is the county's average monthly building-permit count. Counties
with zero or missing permit capacity get NaN recovery potential.

Used by scripts/run_pyrecodes_light.py (probabilistic event set) and by
notebooks/historical_analysis.ipynb (historical storms).
"""

from pathlib import Path

import numpy as np
import pandas as pd

TAU = {1: 1, 2: 1, 3: 3, 4: 6}  # repair time (months) per damage state
PERMIT_COL = "Average_Building_Permits(12 months)"  # already permits/month


def load_permit_capacity(permit_file, capacity_modifier=1.0):
    """FIPS (int) -> construction capacity (permits/month) from the permit CSV."""
    perm = pd.read_csv(permit_file)
    return dict(
        zip(
            perm["FIPS"].astype(int),
            pd.to_numeric(perm[PERMIT_COL], errors="coerce").fillna(0.0)
            * capacity_modifier,
        )
    )


def compute_recovery_potential(per_event_dir, permit_file, out_csv=None,
                               capacity_modifier=1.0):
    """Compute per-county, per-event recovery potential from impact CSVs.

    Parameters
    ----------
    per_event_dir : directory with aggregated per-event impact CSVs
        (columns: event_name, fips, units_DS1_scaled..units_DS4_scaled).
    permit_file : CSV with FIPS and average monthly building permits.
    out_csv : if given, the consolidated table is written here.
    capacity_modifier : multiplier on permit capacity (sensitivity runs).

    Returns
    -------
    DataFrame with columns event_name, fips (unpadded string),
    reconstruction_capacity, recovery_potential_months (NaN for zero capacity).
    """
    cap = load_permit_capacity(permit_file, capacity_modifier)

    files = sorted(Path(per_event_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No per_event CSVs in {per_event_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    df["fips_i"] = (
        df["fips"].astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")
    )
    df["fips_i"] = pd.to_numeric(df["fips_i"], errors="coerce").astype("Int64")
    df["demand"] = sum(df[f"units_DS{k}_scaled"] * TAU[k] for k in TAU)
    sc = [f"units_DS{k}_scaled" for k in TAU]
    df["floor"] = df[sc].gt(0).mul([TAU[k] for k in TAU]).max(axis=1)
    df["cap"] = df["fips_i"].map(cap).fillna(0.0)
    # NaN where capacity is zero/absent - cleaner than inf
    df["recov"] = np.where(
        df["cap"] <= 0, np.nan, np.maximum(df["floor"], df["demand"] / df["cap"])
    )

    out = (
        pd.DataFrame(
            {
                "event_name": df["event_name"].astype(str),
                # unpadded, matches permit/NRI convention
                "fips": df["fips_i"].astype("Int64").astype(str),
                "reconstruction_capacity": df["cap"].astype(float),
                "recovery_potential_months": df["recov"].astype(float),
            }
        )
        .sort_values(["event_name", "fips"])
        .reset_index(drop=True)
    )

    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        print(
            f"Wrote {out_csv}  ({len(out):,} county-event rows, "
            f"{out.event_name.nunique()} events, {out.fips.nunique()} counties)"
        )
    return out
