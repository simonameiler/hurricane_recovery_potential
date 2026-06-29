"""
scaling_utils.py
-----------------
All scaling logic for the TC recovery pipeline.

Core formulation (Gori et al. 2025, relative-contribution)
-----------------------------------------------------------
  W_term = beta_w * log(1 + W_{e,c})
  R_term = I_rain * beta_r * log(1 + R_{e,c})
  S_term = I_surge * beta_s * S_{e,c}
  T      = W_term + R_term + S_term

  Scaling_{e,c} = max(1, 1 + Rel_r/Rel_w + Rel_s/Rel_w)

Counties with unknown region or T <= 0 fall back to Scaling = 1.

Sections
--------
1. Coefficients and core formulation  (compute_scaling_relative, apply_bounded)
2. Historical pipeline helpers        (aggregate_hazard_to_counties,
                                       build_RS_from_dmat,
                                       build_historical_scaling_npz)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd


# ---------------------------------------------------------------------------
# 1.  Core formulation
# ---------------------------------------------------------------------------

# Regression coefficients (Gori et al. 2025, Table S1)
# Key: (region, is_coastal)
COEFFS: Dict[tuple, dict] = {
    ("GOM",   False): dict(beta_w=4.15, beta_r=1.13, beta_s=None),
    ("GOM",   True ): dict(beta_w=5.52, beta_r=0.69, beta_s=0.53),
    ("SE",    False): dict(beta_w=4.38, beta_r=0.45, beta_s=None),
    ("SE",    True ): dict(beta_w=5.21, beta_r=0.52, beta_s=0.62),
    ("MA_NE", False): dict(beta_w=1.84, beta_r=0.88, beta_s=None),
    ("MA_NE", True ): dict(beta_w=0.67, beta_r=0.37, beta_s=1.30),
}


def infer_is_coastal(S: np.ndarray) -> np.ndarray:
    """Return bool array (n_counties,): True where surge > 0 for any event."""
    return S.max(axis=0) > 0.0


def compute_scaling_relative(
    W: np.ndarray,
    R: np.ndarray,
    S: np.ndarray,
    county_region: pd.DataFrame,
    rain_2yr: Optional[np.ndarray] = None,
    surge_2yr: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """Compute Scaling[event, county] from relative hazard contributions.

    Parameters
    ----------
    W, R, S        : float arrays of shape (n_events, n_counties).
                     W = county-max wind (m/s), R = total rain (mm),
                     S = surge above MHHW (m).
    county_region  : DataFrame with columns ``county_index`` and ``region``
                     (one of GOM / SE / MA_NE; NA → Scaling = 1).
    rain_2yr       : optional (n_counties,) 2-year return-period rain threshold.
    surge_2yr      : optional (n_counties,) 2-year return-period surge threshold.
    eps            : numerical floor to guard against division by zero.

    Returns
    -------
    dict with keys:
      ``Scaling``      float32 (n_events, n_counties)
      ``is_coastal``   uint8   (n_counties,)
      ``county_index`` int32   (n_counties,)  == 0 .. n_counties-1
    """
    E, C = W.shape
    reg_series = (
        county_region.set_index("county_index")["region"]
        .astype("string")
        .str.upper()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "", regex=False)
    )
    valid        = {"GOM", "SE", "MA_NE"}
    is_region_ok = reg_series.isin(valid).to_numpy()
    is_coastal   = infer_is_coastal(S)

    # inclusion indicators
    if rain_2yr is None:
        I_rain = R > 0.0
    else:
        thr    = np.asarray(rain_2yr, dtype=float)[None, :]
        I_rain = (R > thr) | (np.isnan(thr) & (R > 0.0))

    if surge_2yr is None:
        I_surge = (S > 0.0) & is_coastal[None, :]
    else:
        sth     = np.asarray(surge_2yr, dtype=float)[None, :]
        I_surge = (S > sth) & is_coastal[None, :]

    Scaling   = np.ones((E, C), dtype=float)
    known_idx = np.flatnonzero(is_region_ok)

    if known_idx.size:
        Wterm = np.zeros((E, known_idx.size), dtype=float)
        Rterm = np.zeros_like(Wterm)
        Sterm = np.zeros_like(Wterm)

        for j, idx in enumerate(known_idx):
            region  = reg_series.iloc[idx]
            coastal = bool(is_coastal[idx])
            bet     = COEFFS[(region, coastal)]
            Wterm[:, j] = bet["beta_w"] * np.log1p(np.maximum(W[:, idx], 0.0))
            Rterm[:, j] = (
                I_rain[:, idx].astype(float) * bet["beta_r"]
                * np.log1p(np.maximum(R[:, idx], 0.0))
            )
            if bet["beta_s"] is not None:
                Sterm[:, j] = I_surge[:, idx].astype(float) * bet["beta_s"] * S[:, idx]

        T     = Wterm + Rterm + Sterm
        Rel_w = np.where(T > eps, Wterm / T, 1.0)
        Rel_r = np.where(T > eps, Rterm / T, 0.0)
        Rel_s = np.where(T > eps, Sterm / T, 0.0)

        wind_ok       = Wterm > eps
        Scaling_known = np.where(
            wind_ok,
            np.maximum(1.0, 1.0 + (Rel_r / Rel_w) + (Rel_s / Rel_w)),
            1.0,
        )
        Scaling[:, known_idx] = Scaling_known

    return {
        "Scaling":      Scaling.astype(np.float32),
        "is_coastal":   is_coastal.astype(np.uint8),
        "county_index": np.arange(C, dtype=np.int32),
    }


def apply_bounded(
    d_wind: np.ndarray,
    scaling: np.ndarray,
    k: float = 1.0,
) -> np.ndarray:
    """Apply scaling to fractional wind damage using bounded compounding.

    D_all = 1 - (1 - D_wind)^{Scaling_eff},  Scaling_eff = 1 + k*(Scaling - 1)
    """
    scaling_eff = 1.0 + k * (scaling - 1.0)
    return 1.0 - np.power(1.0 - np.clip(d_wind, 0.0, 1.0), scaling_eff)


# ---------------------------------------------------------------------------
# 2.  Historical pipeline helpers
# ---------------------------------------------------------------------------

def aggregate_hazard_to_counties(
    haz,
    county_region_df: pd.DataFrame,
    counties_shp_path: str | Path,
) -> np.ndarray:
    """Spatial-join hazard centroids to counties; return W[n_events, n_counties].

    Parameters
    ----------
    haz              : CLIMADA Hazard with .centroids.lon/.lat and .intensity (csr)
    county_region_df : loaded county_region.csv (must have county_index, stcode, ccode)
    counties_shp_path: path to US_counties.shp

    Returns
    -------
    W : float64 array (n_events, n_counties) — county max wind (m/s).
        County axis matches county_region_df order (index 0 .. n_counties-1).
    """
    lons        = np.asarray(haz.centroids.lon)
    lats        = np.asarray(haz.centroids.lat)
    n_centroids = len(lons)
    n_events    = haz.intensity.shape[0]
    n_counties  = len(county_region_df)

    cent_gdf = gpd.GeoDataFrame(
        {"centroid_pos": np.arange(n_centroids)},
        geometry=gpd.points_from_xy(lons, lats),
        crs="EPSG:4326",
    )

    counties = gpd.read_file(str(counties_shp_path))[["STATEFP", "COUNTYFP", "geometry"]]
    if counties.crs is None:
        counties = counties.set_crs("EPSG:4326")
    else:
        counties = counties.to_crs("EPSG:4326")
    counties["stcode"] = counties["STATEFP"].astype(int)
    counties["ccode"]  = counties["COUNTYFP"].astype(int)

    joined = gpd.sjoin(
        cent_gdf,
        counties[["stcode", "ccode", "geometry"]],
        how="left",
        predicate="within",
    )

    stcc_to_cidx: Dict[Tuple[int, int], int] = {
        (int(r.stcode), int(r.ccode)): int(r.county_index)
        for _, r in county_region_df.iterrows()
    }

    centroid_county_idx = np.full(n_centroids, -1, dtype=int)
    for _, row in joined.dropna(subset=["stcode", "ccode"]).iterrows():
        ci  = int(row["centroid_pos"])
        key = (int(row["stcode"]), int(row["ccode"]))
        centroid_county_idx[ci] = stcc_to_cidx.get(key, -1)

    valid_mask = centroid_county_idx >= 0
    valid_cpos = np.where(valid_mask)[0]
    valid_cidx = centroid_county_idx[valid_mask]

    print(f"  [aggregate_hazard_to_counties] {valid_mask.sum()}/{n_centroids} centroids "
          f"mapped to {len(np.unique(valid_cidx))} counties")

    intensity_dense = haz.intensity.toarray()  # (n_events, n_centroids)
    W = np.zeros((n_events, n_counties), dtype=float)
    for c_pos, c_idx in zip(valid_cpos, valid_cidx):
        np.maximum(W[:, c_idx], intensity_dense[:, c_pos], out=W[:, c_idx])

    return W


def _match_dmat_for_storm(
    dmat_year: pd.DataFrame,
    stcode: int,
    ccode: int,
    w_val: float,
) -> Tuple[Optional[pd.Series], float, bool]:
    """Return (best_row, residual_m_s, is_ambiguous) for one (county, storm).

    *dmat_year* is pre-filtered to the storm's year.
    Returns (None, 0, False) when no Dmat row exists for this county-year.
    Ambiguous = True when multiple rows exist but their windmax range < 1 m/s.
    """
    grp = dmat_year[(dmat_year["stcode"] == stcode) & (dmat_year["ccode"] == ccode)]
    if grp.empty:
        return None, 0.0, False

    wmax_vals = grp["windmax"].astype(float).values
    residuals = np.abs(wmax_vals - w_val)
    best_pos  = int(np.argmin(residuals))
    best_row  = grp.iloc[best_pos]
    residual  = float(residuals[best_pos])

    ambiguous = len(grp) > 1 and float(wmax_vals.max() - wmax_vals.min()) < 1.0

    return best_row, residual, ambiguous


def build_RS_from_dmat(
    W: np.ndarray,
    event_atcf_ids: List[str],
    event_years: List[int],
    county_region_df: pd.DataFrame,
    dmat_df: pd.DataFrame,
    wind_only_atcf_ids: Optional[List[str]] = None,
    footprint_wind_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build R[n_events, n_counties] and S[n_events, n_counties] from Dmat.

    Parameters
    ----------
    W                       : county max wind from hazard (n_events, n_counties)
    event_atcf_ids          : ATCF IDs in W row order
    event_years             : calendar year for each storm
    county_region_df        : county_region.csv DataFrame
    dmat_df                 : Dmat_region_all.csv DataFrame
    wind_only_atcf_ids      : storms with no Dmat coverage (R/S left 0)
    footprint_wind_threshold: skip counties with W <= this value

    Returns
    -------
    R, S         : rain (mm) and surge MHHW (m) arrays (n_events, n_counties)
    match_report : DataFrame with per-(event, county) match metadata
    """
    if wind_only_atcf_ids is None:
        wind_only_atcf_ids = []

    n_events   = W.shape[0]
    n_counties = W.shape[1]
    R = np.zeros((n_events, n_counties), dtype=float)
    S = np.zeros((n_events, n_counties), dtype=float)

    cidx_to_stcc = {
        int(r.county_index): (int(r.stcode), int(r.ccode))
        for _, r in county_region_df.iterrows()
    }

    rows_report = []

    for e_idx, (atcf_id, year) in enumerate(zip(event_atcf_ids, event_years)):
        if atcf_id in wind_only_atcf_ids:
            continue

        dmat_yr = dmat_df[dmat_df["year"] == year].copy()
        if dmat_yr.empty:
            continue

        dmat_yr = dmat_yr.assign(
            stcode=dmat_yr["stcode"].astype(int),
            ccode=dmat_yr["ccode"].astype(int),
        )

        for c_idx in range(n_counties):
            w_val = float(W[e_idx, c_idx])
            if w_val <= footprint_wind_threshold:
                continue

            stcode, ccode = cidx_to_stcc[c_idx]
            best_row, residual, ambiguous = _match_dmat_for_storm(
                dmat_yr, stcode, ccode, w_val
            )
            if best_row is None:
                continue

            R[e_idx, c_idx] = float(best_row["raintot"])
            S[e_idx, c_idx] = float(best_row["stidemhhw"])
            rows_report.append({
                "atcf_id":      atcf_id,
                "year":         year,
                "county_index": c_idx,
                "stcode":       stcode,
                "ccode":        ccode,
                "W_hazard":     w_val,
                "W_dmat":       float(best_row["windmax"]),
                "residual":     residual,
                "ambiguous":    ambiguous,
                "raintot":      float(best_row["raintot"]),
                "stidemhhw":    float(best_row["stidemhhw"]),
            })

    match_report = pd.DataFrame(rows_report) if rows_report else pd.DataFrame(
        columns=["atcf_id", "year", "county_index", "stcode", "ccode",
                 "W_hazard", "W_dmat", "residual", "ambiguous", "raintot", "stidemhhw"]
    )
    return R, S, match_report


def build_historical_scaling_npz(
    haz,
    event_atcf_ids: List[str],
    event_years: List[int],
    county_region_df: pd.DataFrame,
    dmat_df: pd.DataFrame,
    counties_shp_path: str | Path,
    out_path: str | Path,
    wind_only_atcf_ids: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Compute and save the historical scaling NPZ.

    Steps: hazard → W, Dmat → R/S, compute_scaling_relative, save NPZ.
    Wind-only storms (Ida, Ian) get Scaling = 1 and are flagged.
    Output NPZ includes ``event_names`` so impact_utils can look up rows by
    ATCF ID instead of numeric event index.

    Returns W, R, S, match_report for notebook inspection.
    """
    if wind_only_atcf_ids is None:
        wind_only_atcf_ids = []

    print("Step 3a: Aggregating hazard footprint to county-level max wind ...")
    W = aggregate_hazard_to_counties(haz, county_region_df, counties_shp_path)
    print(f"  W shape: {W.shape}  (non-zero: {(W > 0).sum()})")

    print("Step 3b: Matching Dmat → R, S ...")
    R, S, match_report = build_RS_from_dmat(
        W, event_atcf_ids, event_years,
        county_region_df, dmat_df,
        wind_only_atcf_ids=wind_only_atcf_ids,
    )
    print(f"  R non-zero: {(R > 0).sum()},  S non-zero: {(S > 0).sum()}")
    if not match_report.empty:
        n_amb = int(match_report["ambiguous"].sum())
        n_tot = len(match_report)
        print(f"  Ambiguous matches (windmax gap < 1 m/s): {n_amb}/{n_tot} "
              f"({100*n_amb/max(n_tot, 1):.1f}%)")

    print("Step 3c: Computing Scaling via relative-contribution formulation ...")
    cr = county_region_df.copy()
    if "region" not in cr.columns and "region_str" in cr.columns:
        cr = cr.rename(columns={"region_str": "region"})

    result      = compute_scaling_relative(W, R, S, cr)
    Scaling      = result["Scaling"]
    is_coastal   = result["is_coastal"]
    county_index = result["county_index"]

    wind_only_flags = np.zeros(len(event_atcf_ids), dtype=np.uint8)
    for e_idx, aid in enumerate(event_atcf_ids):
        if aid in wind_only_atcf_ids:
            Scaling[e_idx, :] = 1.0
            wind_only_flags[e_idx] = 1

    out_path = Path(out_path)
    np.savez_compressed(
        out_path,
        Scaling=Scaling,
        is_coastal=is_coastal,
        county_index=county_index,
        event_names=np.array(event_atcf_ids, dtype=object),
        wind_only_flags=wind_only_flags,
    )
    print(f"  Saved → {out_path}  shape={Scaling.shape}")

    return W, R, S, match_report
