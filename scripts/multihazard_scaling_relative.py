
"""
multihazard_scaling_relative.py
--------------------------------

Compute storm–county scaling factors using the **relative-contribution** formulation
based on Gori et al. (2025) county-level regression terms.

Compared to the exponential-ratio version, this module:
- builds per-event–county *terms* (Wind, Rain, Surge) following Gori's form,
- normalizes them to *relative contributions*,
- defines a *scaling* that amplifies wind-only building damages by the missing shares of rain and surge.

Formulation
===========
Let (per event e, county c):
  W_term = beta_w * log(1 + W_{e,c})
  R_term = I_rain * beta_r * log(1 + R_{e,c})
  S_term = I_surge * beta_s * S_{e,c}
where I_rain and I_surge are inclusion indicators (thresholded; S only for coastal counties).

Let T = W_term + R_term + S_term. If T <= 0, fall back to Wind-only:
  Rel_w = 1, Rel_r = 0, Rel_s = 0.
Else:
  Rel_h = term_h / T for h in {w,r,s}.

Define the scaling as:
  Scaling_{e,c} = max(1, 1 + Rel_r/Rel_w + Rel_s/Rel_w).

Neutralization
==============
If a county has no valid region assignment (NA or not in {GOM,SE,MA_NE}), we set Scaling=1 for all events.

Outputs
=======
- NPZ with arrays:
    Scaling : float32 (events x counties)
    county_index : int32 (0..n_counties-1)
    is_coastal : uint8 (1 for coastal, inferred from S>0 ever)
- Optional 'apply' routine to bound scaled building fractions:
    D_all = 1 - (1 - D_wind)^{Scaling_eff}, with Scaling_eff = 1 + k (Scaling - 1), k in [0, +inf).

CLI
===
Example:
  python multihazard_scaling_relative.py \\
    --wind ../data/hazard/maxwindmat_ncep_reanal.mat \\
    --rain ../data/hazard/ptot_rain_county_ncep_reanal.mat \\
    --surge ../data/hazard/maxelev_coastcounty_ncep_reanal.mat \\
    --use-surge-mhhw \\
    --county-region ../data/county_region_for_scaling.csv \\
    --out ../data/scaling_relative.npz

Optional thresholds:
  --county-thresholds ../data/county_thresholds.csv  # columns: county_index, rain_2yr (mm), surge_2yr (m)

"""

from __future__ import annotations
import argparse, os
from typing import Optional, Dict
import numpy as np
import pandas as pd
from scipy.io import loadmat

# Coefficients (Table S1) — we need beta_w, beta_r, beta_s
COEFFS = {
    ("GOM", False): dict(beta_w=4.15, beta_r=1.13, beta_s=None),
    ("GOM", True ): dict(beta_w=5.52, beta_r=0.69, beta_s=0.53),
    ("SE",  False): dict(beta_w=4.38, beta_r=0.45, beta_s=None),
    ("SE",  True ): dict(beta_w=5.21, beta_r=0.52, beta_s=0.62),
    ("MA_NE", False): dict(beta_w=1.84, beta_r=0.88, beta_s=None),
    ("MA_NE", True ): dict(beta_w=0.67, beta_r=0.37, beta_s=1.30),
}

def load_hazards(wind_path: str, rain_path: str, surge_path: str, use_mhhw: bool=True) -> Dict[str, np.ndarray]:
    W = loadmat(wind_path)["maxwindmat"].astype(float)         # (E, C)
    R = loadmat(rain_path)["ptot_mat"].astype(float)           # (C, E) -> T
    S_key = "scounty_mhhw" if use_mhhw else "scounty"
    S = loadmat(surge_path)[S_key].astype(float)               # (C, E) -> T
    R = R.T if R.shape[0] != W.shape[0] else R
    S = S.T if S.shape[0] != W.shape[0] else S
    if not (W.shape == R.shape == S.shape):
        raise ValueError(f"Shape mismatch: W{W.shape}, R{R.shape}, S{S.shape}")
    return {"W": W, "R": R, "S": S}

def infer_is_coastal(S: np.ndarray) -> np.ndarray:
    return (S.max(axis=0) > 0.0)

def compute_scaling_relative(
    W: np.ndarray, R: np.ndarray, S: np.ndarray,
    county_region: pd.DataFrame,
    rain_2yr: Optional[np.ndarray]=None,
    surge_2yr: Optional[np.ndarray]=None,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """
    Compute Scaling_{e,c} from relative contributions. Neutralize (set to 1) for counties with NA region.
    """
    E, C = W.shape
    reg_series = (
        county_region.set_index("county_index")["region"]
        .astype("string").str.upper().str.replace("-", "_", regex=False).str.replace(" ", "", regex=False)
    )
    valid = {"GOM","SE","MA_NE"}
    is_region_ok = reg_series.isin(valid).to_numpy()

    is_coastal = infer_is_coastal(S)

    # Inclusion indicators
    if rain_2yr is None:
        I_rain = (R > 0.0)
    else:
        thr = np.asarray(rain_2yr, dtype=float)[None, :]
        I_rain = (R > thr) | (np.isnan(thr) & (R > 0.0))

    if surge_2yr is None:
        I_surge = (S > 0.0) & is_coastal[None, :] & False  # conservative default if unknown
    else:
        sth = np.asarray(surge_2yr, dtype=float)[None, :]
        I_surge = (S > sth) & is_coastal[None, :]

    # Prepare output
    Scaling = np.ones((E, C), dtype=float)

    # Known region counties: compute terms and scaling
    known_idx = np.flatnonzero(is_region_ok)
    if known_idx.size:
        # Pre-allocate term arrays
        Wterm = np.zeros((E, known_idx.size), dtype=float)
        Rterm = np.zeros_like(Wterm)
        Sterm = np.zeros_like(Wterm)

        # Build terms per county (vector over events)
        for j, idx in enumerate(known_idx):
            region = reg_series.iloc[idx]  # one of GOM/SE/MA_NE
            coastal = bool(is_coastal[idx])
            bet = COEFFS[(region, coastal)]
            # Wind, always included
            Wterm[:, j] = bet["beta_w"] * np.log1p(np.maximum(W[:, idx], 0.0))
            # Rain
            Rterm[:, j] = I_rain[:, idx].astype(float) * bet["beta_r"] * np.log1p(np.maximum(R[:, idx], 0.0))
            # Surge
            if bet["beta_s"] is not None:
                Sterm[:, j] = I_surge[:, idx].astype(float) * bet["beta_s"] * S[:, idx]

        T = Wterm + Rterm + Sterm
        # Handle T<=0 by falling back to Wind-only contributions
        # Else compute relative shares
        Rel_w = np.where(T > eps, Wterm / T, 1.0)
        Rel_r = np.where(T > eps, Rterm / T, 0.0)
        Rel_s = np.where(T > eps, Sterm / T, 0.0)

        # Scaling = max(1, 1 + Rel_r/Rel_w + Rel_s/Rel_w)
        denom = np.maximum(Rel_w, eps)
        Scaling_known = 1.0 + (Rel_r / denom) + (Rel_s / denom)
        Scaling_known = np.maximum(1.0, Scaling_known)

        Scaling[:, known_idx] = Scaling_known

    return {
        "Scaling": Scaling.astype(np.float32),
        "is_coastal": is_coastal.astype(np.uint8),
        "county_index": np.arange(C, dtype=np.int32),
    }

def apply_bounded(
    d_wind: np.ndarray, scaling: np.ndarray, k: float = 1.0
) -> np.ndarray:
    """
    Apply scaling to a fractional wind damage D in [0,1] using bounded compounding:
      D_all = 1 - (1 - D)^{Scaling_eff},  Scaling_eff = 1 + k (Scaling - 1).
    """
    scaling_eff = 1.0 + k * (scaling - 1.0)
    return 1.0 - np.power(1.0 - np.clip(d_wind, 0.0, 1.0), scaling_eff)

def main(argv=None):
    ap = argparse.ArgumentParser(description="Compute single Scaling[event,county] via relative contributions (Gori-consistent).")
    ap.add_argument("--wind", required=True, help="Path to maxwindmat_ncep_reanal.mat")
    ap.add_argument("--rain", required=True, help="Path to ptot_rain_county_ncep_reanal.mat")
    ap.add_argument("--surge", required=True, help="Path to maxelev_coastcounty_ncep_reanal.mat")
    ap.add_argument("--use-surge-mhhw", action="store_true", help="Use scounty_mhhw (recommended).")
    ap.add_argument("--county-region", required=True, help="CSV with county_index,region (GOM/SE/MA_NE).")
    ap.add_argument("--county-thresholds", help="Optional CSV with county_index,rain_2yr,surge_2yr")
    ap.add_argument("--out", required=True, help="Output NPZ path for Scaling.")
    args = ap.parse_args(argv)

    haz = load_hazards(args.wind, args.rain, args.surge, use_mhhw=args.use_surge_mhhw)
    W, R, S = haz["W"], haz["R"], haz["S"]

    cr = pd.read_csv(args.county_region, usecols=["county_index","region"]).sort_values("county_index")
    if cr["county_index"].to_numpy().max() >= W.shape[1]:
        raise ValueError("county_index in county_region exceeds number of counties in hazard matrices.")

    rain_2yr = surge_2yr = None
    if args.county_thresholds:
        thr = pd.read_csv(args.county_thresholds).set_index("county_index").reindex(range(W.shape[1]))
        if "rain_2yr" in thr.columns:
            rain_2yr = pd.to_numeric(thr["rain_2yr"], errors="coerce").to_numpy()
        if "surge_2yr" in thr.columns:
            surge_2yr = pd.to_numeric(thr["surge_2yr"], errors="coerce").to_numpy()

    out = compute_scaling_relative(W, R, S, cr, rain_2yr, surge_2yr)
    np.savez_compressed(args.out, **out)
    print(f"Wrote {args.out} with Scaling shape={out['Scaling'].shape}")

if __name__ == "__main__":
    main()
