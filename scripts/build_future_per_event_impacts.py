"""
Build per-event, per-county impact records for Florida from Gori et al. (2025) future climate synthetic hazard mats.

This script pools 5 GCMs (SSP2-4.5, 2070-2100) to create a future climate dataset.

GCMs:
- canesm_ssp245_2070_2100
- cnrm6_ssp245_2070_2100
- ecearth6_ssp245_2070_2100
- ipsl6_ssp245_2070_2100
- miroc6_ssp245_2070_2100

Inputs (from fl_risk_model/data/hazard/synthetic_hazard_risk_estimates/):
- Wind/maxwindmat_<gcm>_ssp245_2070_2100.mat -> variable: maxwindmat [counties x events], wind speed (m/s)
- Surge/maxelev_coastcounty_<gcm>_ssp245_2070_2100.mat -> variable: scounty_mhhw [counties x events], surge (m MHHW)
- Rain/ptot_rain_county_<gcm>_ssp245_2070_2100.mat -> variable: ptot_mat [counties x events], rainfall total (mm)
- county_region.csv -> county mapping
- GDP data -> county_gdp_1996_2020.mat

Outputs:
- fl_risk_model/data/fl_per_event_impacts_future_ssp245.csv

Notes:
- Uses same impact function as present climate (build_per_event_impacts.py)
- Pools all 5 GCMs together (~31,000 events total)
- Event IDs are offset per GCM to avoid collisions
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.io import loadmat
from pathlib import Path

# ---------------- Configuration ----------------
BASE_DIR = Path("/Users/simonameiler/Documents/work/03_code/repos/systemic_insurance_risk_fl/fl_risk_model/data")
HAZARD_DIR = BASE_DIR / "hazard" / "synthetic_hazard_risk_estimates"

GCMS = [
    "canesm_ssp245cal",
    "cnrm6_ssp245cal",
    "ecearth6_ssp245cal",
    "ipsl6_ssp245cal",
    "miroc6_ssp245cal",
]

COUNTY_REGION_CSV = BASE_DIR / "county_region.csv"
PRESENT_CLIMATE_CSV = BASE_DIR / "fl_per_event_impacts.csv"  # For county names
GDP_MAT_PATH = HAZARD_DIR / "county_gdp_1996_2020.mat"

OUTPUT_CSV = BASE_DIR / "fl_per_event_impacts_future_ssp245.csv"

# Impact function parameters (same as present climate)
B_WIND = 1.0
B_RAIN = 1.0
B_SURGE = 1.0
B_GDP = 1.0
CLIP_WIND_TO = 0.1
EPSILON = 1e-9

# Thresholds
WIND_THRESH = 25.0  # m/s
RAIN_THRESH = 0.0   # mm
SURGE_THRESH = 0.0  # m

FLORIDA_STCODE = 12
USE_SURGE_VAR = "scounty_mhhw"

# ---------------- Helper functions ----------------
def compute_value_and_shares(
    W_ms: np.ndarray, R_mm: np.ndarray, S_m: np.ndarray,
    GDP: np.ndarray, b_wind: float, b_rain: float, b_surge: float, b_gdp: float,
    eps: float, clip_wind_to: float,
    wind_thresh: float, rain_thresh: float, surge_thresh: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Only hazards exceeding their thresholds contribute to log-impact and shares.
    If no hazards exceed thresholds, returns value=0, shares=0.
    """
    mW = W_ms > wind_thresh
    mR = R_mm > rain_thresh
    mS = S_m  > surge_thresh

    W_use = np.maximum(W_ms, clip_wind_to)
    R_use = np.maximum(R_mm, 0.0)
    S_use = np.maximum(S_m, 0.0)

    log_value = b_gdp * np.log(np.maximum(GDP, eps))
    log_value = np.where(mW, log_value + b_wind * np.log(W_use), log_value)
    log_value = np.where(mR, log_value + b_rain * np.log(R_use + 1.0), log_value)
    log_value = np.where(mS, log_value + b_surge * S_use, log_value)

    value = np.exp(log_value)

    c_wind  = np.where(mW, b_wind * np.log(W_use), 0.0)
    c_rain  = np.where(mR, b_rain * np.log(R_use + 1.0), 0.0)
    c_surge = np.where(mS, b_surge * S_use, 0.0)

    c_flood = c_rain + c_surge
    denom = c_wind + c_flood
    with np.errstate(divide="ignore", invalid="ignore"):
        wind_share = np.where(denom > 0, c_wind / denom, 0.0)
        flood_share = np.where(denom > 0, c_flood / denom, 0.0)

    wind_share = np.clip(wind_share, 0.0, 1.0)
    flood_share = np.clip(flood_share, 0.0, 1.0)

    return value, wind_share, flood_share


def load_gcm_data(gcm_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load wind, surge, rain matrices for a single GCM.
    
    Note: Wind files use 'ssp245cal' naming, while Surge/Rain use 'ssp245' naming.
    Also, Wind matrices appear to be transposed relative to Surge/Rain.
    """
    print(f"  Loading {gcm_name}...")
    
    # Extract base GCM name (e.g., "canesm" from "canesm_ssp245cal")
    gcm_base = gcm_name.split('_')[0]
    
    # Load wind (uses ssp245cal naming, transposed)
    wind_path = HAZARD_DIR / "Wind" / f"maxwindmat_{gcm_name}.mat"
    wind_mat = loadmat(str(wind_path))["maxwindmat"]  # (n_events, counties) - TRANSPOSED!
    wind_mat = wind_mat.T  # Convert to (counties, n_events)
    
    # Load surge (uses ssp245 naming)
    surge_gcm_name = f"{gcm_base}_ssp245"
    surge_path = HAZARD_DIR / "Surge" / f"maxelev_coastcounty_{surge_gcm_name}.mat"
    surge_mat = loadmat(str(surge_path))[USE_SURGE_VAR]  # (counties, n_events)
    
    # Load rain (uses ssp245 naming)
    rain_path = HAZARD_DIR / "Rain" / f"ptot_rain_county_{surge_gcm_name}.mat"
    rain_mat = loadmat(str(rain_path))["ptot_mat"]  # (counties, n_events)
    
    print(f"    After transpose - Wind: {wind_mat.shape}, Surge: {surge_mat.shape}, Rain: {rain_mat.shape}")
    
    return wind_mat, surge_mat, rain_mat


def main():
    print("="*80)
    print("Building Future Climate Per-Event Impacts (SSP2-4.5, 2070-2100)")
    print("="*80)
    
    # Load county mapping
    print("\n1. Loading county mapping...")
    reg = pd.read_csv(COUNTY_REGION_CSV)
    fl = reg.loc[reg["stcode"] == FLORIDA_STCODE].copy()
    fl_idx = fl["county_index"].astype(int).values
    countyfp = fl["ccode"].astype(int).values
    print(f"   Found {len(fl_idx)} Florida counties (indices: {fl_idx[0]}-{fl_idx[-1]})")
    
    # Load county names from existing present climate CSV
    print("\n2. Loading county names...")
    present_df = pd.read_csv(PRESENT_CLIMATE_CSV)
    names = present_df[['countyfp', 'county_name']].drop_duplicates().set_index('countyfp')['county_name'].to_dict()
    print(f"   Loaded {len(names)} county names")
    
    # Load GDP data
    print("\n3. Loading GDP data...")
    gdp_mat = loadmat(str(GDP_MAT_PATH))
    county_gdp_all = gdp_mat["county_gdp"]  # (3220, years)
    GDP_arr = county_gdp_all[fl_idx, -1].astype(float)  # Last year, Florida counties
    print(f"   GDP array shape: {GDP_arr.shape}")
    
    # Process each GCM and pool results
    print(f"\n4. Processing {len(GCMS)} GCMs...")
    all_records = []
    event_id_offset = 0
    
    for gcm_idx, gcm_name in enumerate(GCMS):
        print(f"\n   GCM {gcm_idx+1}/{len(GCMS)}: {gcm_name}")
        
        # Load hazard data
        wind_mat, surge_mat, rain_mat = load_gcm_data(gcm_name)
        n_events = wind_mat.shape[1]
        
        # Extract Florida counties
        W_fl = wind_mat[fl_idx, :]  # (67, n_events)
        S_fl = surge_mat[fl_idx, :]
        R_fl = rain_mat[fl_idx, :]
        
        print(f"    Florida subset shape: {W_fl.shape}")
        
        # Create GDP matrix (same for all events)
        GDP_mat = GDP_arr.reshape(-1, 1) * np.ones((1, n_events), dtype=float)
        
        # Compute exceed mask
        exceed_mask = (W_fl > WIND_THRESH) | (R_fl > RAIN_THRESH) | (S_fl > SURGE_THRESH)
        
        # Compute values and shares
        print(f"    Computing impact values and shares...")
        value, wind_share, flood_share = compute_value_and_shares(
            W_fl, R_fl, S_fl, GDP_mat,
            B_WIND, B_RAIN, B_SURGE, B_GDP,
            EPSILON, CLIP_WIND_TO,
            WIND_THRESH, RAIN_THRESH, SURGE_THRESH
        )
        
        # Apply exceed mask
        value = value * exceed_mask
        wind_share = wind_share * exceed_mask
        flood_share = flood_share * exceed_mask
        
        # Build records for this GCM
        print(f"    Building DataFrame...")
        for j in range(n_events):
            ev_id = event_id_offset + j + 1
            
            # Create DataFrame for this event
            df_ev = pd.DataFrame({
                "event_id": ev_id,
                "gcm": gcm_name,
                "countyfp": countyfp,
                "county_name": [names.get(int(c), "") for c in countyfp],
                "value": value[:, j],
                "wind_share": wind_share[:, j],
                "flood_share": flood_share[:, j],
                "W_ms": W_fl[:, j],
                "R_mm": R_fl[:, j],
                "S_m": S_fl[:, j],
            })
            
            # Remove zero-impact rows
            df_ev = df_ev[df_ev["value"] > 0]
            
            if not df_ev.empty:
                all_records.append(df_ev)
        
        # Update offset for next GCM
        event_id_offset += n_events
        print(f"    Added {n_events} events (total offset now: {event_id_offset})")
    
    # Combine all GCMs
    print(f"\n5. Combining all GCMs...")
    combined = pd.concat(all_records, ignore_index=True)
    
    # Save to CSV
    print(f"\n6. Writing output CSV...")
    combined.to_csv(OUTPUT_CSV, index=False)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total events across all GCMs: {event_id_offset}")
    print(f"Total non-zero impact records: {len(combined):,}")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"\nBreakdown by GCM:")
    for gcm_name in GCMS:
        gcm_records = combined[combined["gcm"] == gcm_name]
        print(f"  {gcm_name}: {len(gcm_records):,} records")
    
    print(f"\nSample statistics:")
    print(f"  Mean value: {combined['value'].mean():.2e}")
    print(f"  Mean wind share: {combined['wind_share'].mean():.3f}")
    print(f"  Mean flood share: {combined['flood_share'].mean():.3f}")
    print(f"  Mean wind speed: {combined['W_ms'].mean():.2f} m/s")
    print(f"  Mean rainfall: {combined['R_mm'].mean():.2f} mm")
    print(f"  Mean surge: {combined['S_m'].mean():.2f} m")
    
    print("\n✅ Future climate CSV generation complete!")
    

if __name__ == "__main__":
    main()
