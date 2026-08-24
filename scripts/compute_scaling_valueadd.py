#!/usr/bin/env python3
"""Aggregate the effect of the multi-hazard (rain+surge) scaling on repair demand.

Reads the per-event impact CSVs (which contain both the wind-only `_raw` and the
multi-hazard `_scaled` housing-unit counts per damage state) and writes the small
tables consumed by ``notebooks/probabilistic_analysis.ipynb`` to build Figure S9
``scaling_valueadd.png``. This script only ever compares wind-only
(scaling_strength g=0) against the manuscript default (g=1); both are exact
and already fully determined by the cached ``_raw``/``_scaled`` columns, so no
approximation is involved anywhere in this script.

  analysis_output/scaling_valueadd_by_county.csv
      fips, wua_wind, wua_scaled, eard_wind, eard_scaled, eard_diff, eard_ratio
      wua_wind/wua_scaled: sum over events of weighted units affected
      (WUA = 1*DS1+1*DS2+3*DS3+6*DS4), wind-only vs multi-hazard.
      eard_wind/eard_scaled: the same sums annualized (x DEFAULT_FREQ), i.e.
      EARD_wind and EARD_scaled [WUA/yr]; eard_diff = eard_scaled - eard_wind;
      eard_ratio = eard_scaled / eard_wind (NaN where eard_wind == 0).

  analysis_output/scaling_valueadd_ds_totals.csv
      ds, w, units_wind, units_scaled,
      U_wind_per_yr, U_scaled_per_yr, delta_U_per_yr,
      R_wind_per_yr, R_scaled_per_yr, delta_R_per_yr, pct_change
      units_wind/units_scaled: total housing units in each damage state over
      the whole event set, wind-only vs multi-hazard.
      U_*_per_yr = expected annual number of housing-unit DS occurrences,
      U_d(g) = freq * units (frequency cancels for county-level ratios, so
      the original script did not annualize; the manuscript sensitivity
      analysis needs the annualized values for Figure S9 panel b).
      R_*_per_yr = w_d * U_d(g), the damage-state contribution to expected
      annual repair demand (WUA/yr). pct_change = 100*(U_scaled-U_wind)/U_wind.

  analysis_output/scaling_valueadd_transition_matrix_absolute.csv
  analysis_output/scaling_valueadd_transition_matrix_normalized.csv
      Frequency-weighted expected-annual transition matrix from wind-only
      damage state (rows) to default-scaled damage state (columns), and its
      row-normalized (%) version. Exact: see
      modules/scaling_sensitivity.py::transition_matrix_rows for the
      order-statistics argument (no per-building data needed -- the shared
      alpha_{e,c} per county-event makes the transform rank-preserving).

Usage:
    python scripts/compute_scaling_valueadd.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

import sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from modules.scaling_sensitivity import transition_matrix_rows, N_DS

PER_EVENT = REPO / "data" / "impact" / "per_event"
OUT = REPO / "analysis_output"
WEIGHTS = {"DS1": 1, "DS2": 1, "DS3": 3, "DS4": 6}
DS_ORDER = ["DS1", "DS2", "DS3", "DS4"]

# Matches DEFAULT_FREQ in notebooks/probabilistic_analysis.ipynb (Fig. 2/3/4/S9
# EARD definition) and scripts/compare_median_vs_max_events.py; duplicated here
# rather than centralized, following this repo's existing convention.
DEFAULT_FREQ = 0.00067334  # events per year (Poisson rate)


def main():
    files = sorted(PER_EVENT.glob("aggregated_*.csv"))
    if not files:
        raise SystemExit(f"No per-event impact CSVs found in {PER_EVENT}")
    print(f"Aggregating {len(files)} per-event impact files ...")

    by_county = {}
    ds_tot = {ds: [0.0, 0.0] for ds in WEIGHTS}
    trans_total = np.zeros((N_DS, N_DS), dtype=float)

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

        # Exact wind-only -> default-scaled transition matrix (this file's rows)
        raw_units = df[[f"units_{ds}_raw" for ds in DS_ORDER]].to_numpy(dtype=float)
        scaled_units = df[[f"units_{ds}_scaled" for ds in DS_ORDER]].to_numpy(dtype=float)
        # total_units per county-event isn't in this CSV; DS0 count is not
        # needed for the transition matrix among DS1..DS4 themselves, but
        # DS0's *outflow* to DS1..DS4 is. Reconstruct DS0 via total exposed
        # housing units per county (event-invariant), joined below.
        raw5 = np.zeros((len(df), N_DS))
        scaled5 = np.zeros((len(df), N_DS))
        raw5[:, 1:] = raw_units
        scaled5[:, 1:] = scaled_units
        trans_total += _transition_with_ds0(df, raw5, scaled5)

    county = pd.DataFrame(
        [(fips, w, a) for fips, (w, a) in by_county.items()],
        columns=["fips", "wua_wind", "wua_scaled"],
    ).sort_values("fips")
    county["eard_wind"] = county["wua_wind"] * DEFAULT_FREQ
    county["eard_scaled"] = county["wua_scaled"] * DEFAULT_FREQ
    county["eard_diff"] = county["eard_scaled"] - county["eard_wind"]
    county["eard_ratio"] = np.where(
        county["eard_wind"] > 0, county["eard_scaled"] / county["eard_wind"], np.nan
    )
    county.to_csv(OUT / "scaling_valueadd_by_county.csv", index=False)

    totals = pd.DataFrame(
        [(ds, WEIGHTS[ds], ds_tot[ds][0], ds_tot[ds][1]) for ds in WEIGHTS],
        columns=["ds", "w", "units_wind", "units_scaled"],
    )
    totals["U_wind_per_yr"] = totals["units_wind"] * DEFAULT_FREQ
    totals["U_scaled_per_yr"] = totals["units_scaled"] * DEFAULT_FREQ
    totals["delta_U_per_yr"] = totals["U_scaled_per_yr"] - totals["U_wind_per_yr"]
    totals["R_wind_per_yr"] = totals["w"] * totals["U_wind_per_yr"]
    totals["R_scaled_per_yr"] = totals["w"] * totals["U_scaled_per_yr"]
    totals["delta_R_per_yr"] = totals["R_scaled_per_yr"] - totals["R_wind_per_yr"]
    totals["pct_change"] = 100.0 * totals["delta_U_per_yr"] / totals["U_wind_per_yr"]
    totals.to_csv(OUT / "scaling_valueadd_ds_totals.csv", index=False)

    # ---- Transition matrix: annualize (frequency-weighted expected annual counts) ----
    trans_abs = trans_total * DEFAULT_FREQ
    ds_labels = [f"DS{i}" for i in range(N_DS)]
    trans_abs_df = pd.DataFrame(trans_abs, index=ds_labels, columns=ds_labels)
    trans_abs_df.index.name = "wind_only_DS"
    trans_abs_df.to_csv(OUT / "scaling_valueadd_transition_matrix_absolute.csv")

    row_sums = trans_abs.sum(axis=1, keepdims=True)
    trans_norm = np.divide(trans_abs, row_sums, out=np.zeros_like(trans_abs),
                           where=row_sums > 0) * 100.0
    trans_norm_df = pd.DataFrame(trans_norm, index=ds_labels, columns=ds_labels)
    trans_norm_df.index.name = "wind_only_DS"
    trans_norm_df.to_csv(OUT / "scaling_valueadd_transition_matrix_normalized.csv")

    # ---- Diagnostics ----
    tot_w = county["wua_wind"].sum()
    tot_a = county["wua_scaled"].sum()
    print(f"  counties: {len(county)}")
    print(f"  total WUA wind-only  : {tot_w:,.0f}")
    print(f"  total WUA scaled     : {tot_a:,.0f}  ({100*(tot_a-tot_w)/tot_w:+.1f}%)")
    print("  wrote scaling_valueadd_by_county.csv, scaling_valueadd_ds_totals.csv,")
    print("        scaling_valueadd_transition_matrix_{absolute,normalized}.csv")

    print("\n  Ten counties with the largest EARD_scaled/EARD_wind ratio "
          "(diagnostic; large ratios can come from a small denominator):")
    names = _load_county_names()
    top10 = county[county["eard_wind"] > 0].nlargest(10, "eard_ratio")
    for _, r in top10.iterrows():
        nm = names.get(int(r["fips"]), "")
        print(f"    {int(r['fips']):05d} {nm:28s} ratio={r['eard_ratio']:6.2f}  "
              f"EARD_wind={r['eard_wind']:.4f}  EARD_scaled={r['eard_scaled']:.4f}  "
              f"diff={r['eard_diff']:.4f}")

    print("\n  Largest upward damage-state transitions (expected annual housing units):")
    for i in range(N_DS):
        for j in range(i + 1, N_DS):
            print(f"    DS{i}->DS{j}: {trans_abs[i, j]:,.2f} units/yr "
                  f"({trans_norm[i, j]:.2f}% of wind-only DS{i})")
    print("\n  Key adjacent transitions:")
    for i in range(N_DS - 1):
        j = i + 1
        print(f"    DS{i}->DS{j}: {trans_abs[i, j]:,.3f} units/yr "
              f"({trans_norm[i, j]:.3f}% of wind-only DS{i})")


def _transition_with_ds0(df, raw5, scaled5):
    """Fill in the DS0 column (index 0) of raw5/scaled5 using total exposed
    housing units per county (event-invariant), then return this file's
    contribution to the 5x5 transition-count tensor, summed over rows.
    """
    housing = _load_housing_units()
    total_units = df["fips"].map(housing).to_numpy(dtype=float)
    raw_ds1_4 = raw5[:, 1:].sum(axis=1)
    scaled_ds1_4 = scaled5[:, 1:].sum(axis=1)
    raw5[:, 0] = np.clip(total_units - raw_ds1_4, 0.0, None)
    scaled5[:, 0] = np.clip(total_units - scaled_ds1_4, 0.0, None)
    valid = np.isfinite(total_units)
    t = transition_matrix_rows(raw5[valid], scaled5[valid])
    return t.sum(axis=0)


_HOUSING_CACHE = None


def _load_housing_units():
    global _HOUSING_CACHE
    if _HOUSING_CACHE is None:
        h = pd.read_csv(OUT / "county_exposed_housing_units.csv")
        # A handful of rows in this pre-existing committed file have a
        # malformed FIPS (e.g. "22None") from an upstream ccode-join gap;
        # unrelated to the scaling sensitivity analysis, so those counties
        # are dropped here (excluded from the transition matrix via the
        # NaN -> isfinite filter in _transition_with_ds0) rather than fixed.
        fips_num = pd.to_numeric(h["FIPS"], errors="coerce")
        n_bad = int(fips_num.isna().sum())
        if n_bad:
            print(f"  [warn] {n_bad} row(s) with malformed FIPS in "
                  f"county_exposed_housing_units.csv skipped for the transition matrix")
        h = h.loc[fips_num.notna()].copy()
        h["FIPS"] = fips_num.loc[fips_num.notna()].astype(int)
        _HOUSING_CACHE = dict(zip(h["FIPS"], h["exposed_units"]))
    return _HOUSING_CACHE


def _load_county_names():
    try:
        perm = pd.read_csv(REPO / "data" / "selected_states_counties_with_permits_ctfix.csv")
        perm["FIPS"] = perm["FIPS"].astype(int)
        return dict(zip(perm["FIPS"], perm["NAME"] + ", " + perm["STATE_NAME"]))
    except Exception:
        return {}


if __name__ == "__main__":
    main()
