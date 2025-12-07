"""
Hurricane Recovery Potential Analysis - North American Coastal Counties
=========================================================================

This script analyzes the drivers of recovery potential following tropical cyclone
events across North American coastal counties. It examines both annual metrics
(Expected Annual Recovery Potential) and per-event metrics to understand the
relative importance of damage magnitude versus construction capacity.

ANALYSIS COMPONENTS:
-------------------
1. Data Loading & Preparation
2. Expected Annual Damage (EAD) Computation
3. Expected Annual Recovery Potential (EARP) Computation
4. Driver Analysis (Annual & Per-Event)
5. Correlation & Variance Partitioning Analysis
6. County-Level Variance Decomposition
7. Spatial Pattern Analysis
8. Visualization Functions

VISUALIZATIONS GENERATED:
------------------------
- na_coast_ead_by_damage_state.png: 4-panel map of EAD by damage state
- na_coast_earp_metrics.png: 2-panel map of construction capacity and EARP
- na_coast_3panel_ead_capacity_recovery_notitle.png: 3-panel map of EAD, capacity, and EARP
- median_recovery_drivers_scatter.png: 4-panel scatterplot of driver correlations
- variance_partitioning_annual_vs_event.png: Bar charts comparing variance partitioning
- variance_share_annual_vs_event_maps.png: Maps showing variance contribution by county
- variance_share_annual_vs_event.png: Histograms of variance share distributions

Author: Simona Meiler
Date: December 2025
"""

# ============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ============================================================================

import os
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import (LogNorm, LinearSegmentedColormap, BoundaryNorm, 
                                TwoSlopeNorm)
from matplotlib.ticker import NullLocator, LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
import matplotlib.cm as cm

import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from scipy.stats import pearsonr, mannwhitneyu, zscore

from climada.entity.exposures import Exposures

# Configuration
sns.set(style="whitegrid")
warnings.filterwarnings('ignore')

# Constants
DEFAULT_FREQ = 0.00067334  # events/year
RECOVERY_WEIGHTS = {
    'DS1': 1.0,   # 1 month
    'DS2': 1.0,   # 1 month
    'DS3': 3.0,   # 3 months
    'DS4': 6.0    # 6 months
}

# Coastal state FIPS codes
COASTAL_STATE_FIPS = [
    '01',  # Alabama
    '09',  # Connecticut
    '10',  # Delaware
    '12',  # Florida
    '13',  # Georgia
    '22',  # Louisiana
    '23',  # Maine
    '24',  # Maryland
    '25',  # Massachusetts
    '28',  # Mississippi
    '33',  # New Hampshire
    '34',  # New Jersey
    '36',  # New York
    '37',  # North Carolina
    '44',  # Rhode Island
    '45',  # South Carolina
    '48',  # Texas
    '51',  # Virginia
]


# ============================================================================
# SECTION 2: DATA LOADING FUNCTIONS
# ============================================================================

def load_county_boundaries():
    """
    Load US county shapefile and filter to coastal states.
    
    Returns
    -------
    tuple
        (all_counties, coastal_counties) GeoDataFrames
    """
    county_shp_path = Path("..") / "data" / "US_counties.shp"
    
    if not county_shp_path.exists():
        raise FileNotFoundError(f"County shapefile not found at {county_shp_path}")
    
    counties = gpd.read_file(county_shp_path)
    print(f"Loaded {len(counties)} counties")
    
    coastal_counties = counties[counties["STATEFP"].isin(COASTAL_STATE_FIPS)].copy()
    
    # Create GEOID if not present
    if 'GEOID' not in coastal_counties.columns:
        if 'COUNTYFP' in coastal_counties.columns:
            coastal_counties['GEOID'] = coastal_counties['STATEFP'] + coastal_counties['COUNTYFP']
    
    print(f"Filtered to {len(coastal_counties)} coastal counties")
    
    return counties, coastal_counties


def load_construction_capacity():
    """
    Load construction capacity data from building permits.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: fips, construction_capacity
    """
    permits_file = Path("..") / "data" / "selected_states_counties_with_permits.csv"
    
    permits_df = pd.read_csv(permits_file)
    permits_df['fips'] = permits_df['FIPS'].astype(str).str.zfill(5)
    
    capacity_df = permits_df[['fips', 'Average_Building_Permits(12 months)']].copy()
    capacity_df.columns = ['fips', 'construction_capacity']
    
    print(f"Loaded construction capacity for {len(capacity_df)} counties")
    print(f"  Range: {capacity_df['construction_capacity'].min():.1f} - "
          f"{capacity_df['construction_capacity'].max():.1f} permits/month")
    
    return capacity_df


def compute_expected_annual_damage():
    """
    Compute Expected Annual Damage (EAD) per damage state from event files.
    
    Returns
    -------
    tuple
        (ead_df_long, ead_df_wide, units_df)
        - ead_df_long: long format with columns [fips, DS, ead, type]
        - ead_df_wide: wide format with columns [fips, DS1, DS2, DS3, DS4, total_ead]
        - units_df: raw per-event damage data
    """
    print("=== Computing Expected Annual Damage (Units) per Damage State ===\n")
    
    by_event_dir = Path("..") / "impacts_out" / "by_event" / "scaled"
    event_files = sorted(by_event_dir.glob("*_scaled.csv"))
    print(f"Found {len(event_files)} event impact files")
    
    # Load and combine all files
    all_units = []
    for f in event_files:
        df = pd.read_csv(f)
        df_units = df[['event_name', 'fips', 'units_DS1_scaled', 'units_DS2_scaled', 
                        'units_DS3_scaled', 'units_DS4_scaled']].copy()
        all_units.append(df_units)
    
    units_df = pd.concat(all_units, ignore_index=True)
    print(f"Loaded {len(units_df)} county-event pairs")
    
    # Ensure FIPS is 5-digit zero-padded string
    units_df['fips'] = units_df['fips'].astype(str).str.zfill(5)
    
    # Compute EAD by multiplying units by event frequency
    for ds in ['DS1', 'DS2', 'DS3', 'DS4']:
        units_df[f'weighted_{ds}'] = units_df[f'units_{ds}_scaled'] * DEFAULT_FREQ
    
    # Sum across events per county
    ead_computed = units_df.groupby('fips').agg({
        'weighted_DS1': 'sum',
        'weighted_DS2': 'sum',
        'weighted_DS3': 'sum',
        'weighted_DS4': 'sum',
        'event_name': 'count'
    }).reset_index()
    
    ead_computed.columns = ['fips', 'DS1', 'DS2', 'DS3', 'DS4', 'num_events']
    
    print(f"\n=== Expected Annual Damage Statistics ===")
    print(f"Counties with data: {len(ead_computed)}")
    for ds in ['DS1', 'DS2', 'DS3', 'DS4']:
        total = ead_computed[ds].sum()
        print(f"  {ds}: Total={total:,.1f} units/yr")
    
    # Convert to long format
    ead_df = pd.melt(ead_computed, id_vars=['fips'], 
                     value_vars=['DS1', 'DS2', 'DS3', 'DS4'],
                     var_name='DS', value_name='ead')
    ead_df['type'] = 'scaled'
    
    # Also create wide format for easy analysis
    ead_wide = ead_df.pivot(index='fips', columns='DS', values='ead').reset_index()
    ead_wide.columns.name = None
    ead_wide['fips'] = ead_wide['fips'].astype(str).str.zfill(5)
    ead_wide['total_ead'] = ead_wide[['DS1', 'DS2', 'DS3', 'DS4']].sum(axis=1)
    
    return ead_df, ead_wide, units_df


def load_recovery_potential_data():
    """
    Load per-event recovery potential data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: event, fips, recovery_potential [months], etc.
    """
    print("=== Loading Recovery Potential Data ===\n")
    
    recovery_per_event_dir = Path("..") / "data" / "recovery_potential_per_scenario"
    recovery_files = list(recovery_per_event_dir.glob("*_scaled_recovery_potential.json"))
    
    print(f"Found {len(recovery_files)} recovery potential files")
    print("Loading all files (this may take a moment)...")
    
    all_recovery_events = []
    for idx, f in enumerate(recovery_files):
        if (idx + 1) % 500 == 0:
            print(f"  Loaded {idx + 1}/{len(recovery_files)} files...")
        
        with open(f, 'r') as file:
            data = json.load(file)
            df = pd.DataFrame(data)
            all_recovery_events.append(df)
    
    recovery_all_events = pd.concat(all_recovery_events, ignore_index=True)
    print(f"\nLoaded {len(recovery_all_events)} rows from "
          f"{recovery_all_events['event'].nunique()} unique events")
    
    # Ensure FIPS is 5-digit string
    recovery_all_events['fips'] = recovery_all_events['fips'].astype(str).str.zfill(5)
    
    # Handle infinity values (capacity = 0)
    recovery_all_events['recovery_potential [months]'] = (
        recovery_all_events['recovery_potential [months]']
        .replace([np.inf, -np.inf], np.nan)
    )
    
    return recovery_all_events


def compute_expected_annual_recovery_potential(recovery_all_events):
    """
    Compute Expected Annual Recovery Potential (EARP) per county.
    
    Parameters
    ----------
    recovery_all_events : pd.DataFrame
        Per-event recovery potential data
    
    Returns
    -------
    pd.DataFrame
        DataFrame with EARP metrics per county
    """
    print("\n=== Computing EARP (Expected Annual Recovery Potential) ===\n")
    
    # Multiply recovery potential by event frequency
    recovery_all_events['weighted_recovery'] = (
        recovery_all_events['recovery_potential [months]'] * DEFAULT_FREQ
    )
    
    # Sum across events per county
    earp_df = recovery_all_events.groupby('fips').agg({
        'weighted_recovery': 'sum',
        'recovery_potential [months]': ['count', 'sum', 'mean', 'max']
    }).reset_index()
    
    earp_df.columns = ['fips', 'earp_months_per_year', 'num_events', 
                       'total_recovery_months', 'mean_recovery_per_event', 'max_recovery']
    
    # Replace infinities with NaN
    earp_df = earp_df.replace([np.inf, -np.inf], np.nan)
    
    print(f"Computed EARP for {len(earp_df)} counties")
    earp_finite = earp_df['earp_months_per_year'].dropna()
    if len(earp_finite) > 0:
        print(f"  Mean EARP: {earp_finite.mean():.4f} months/year")
        print(f"  Median EARP: {earp_finite.median():.4f} months/year")
    
    return earp_df


# ============================================================================
# SECTION 3: DATA PREPARATION FOR ANALYSIS
# ============================================================================

def prepare_annual_driver_analysis(earp_df, ead_wide, capacity_df):
    """
    Prepare dataset for annual driver analysis.
    
    Merges EARP, EAD, and construction capacity data, filters invalid values,
    and computes log-transformed variables.
    
    Parameters
    ----------
    earp_df : pd.DataFrame
        Expected Annual Recovery Potential data
    ead_wide : pd.DataFrame
        Expected Annual Damage data (wide format)
    capacity_df : pd.DataFrame
        Construction capacity data
    
    Returns
    -------
    pd.DataFrame
        Prepared dataset with log-transformed variables
    """
    print("\n=== Preparing Data for Annual Driver Analysis ===\n")
    
    # Merge EARP, EAD, and capacity
    driver_analysis = earp_df[['fips', 'earp_months_per_year']].copy()
    driver_analysis = driver_analysis.merge(ead_wide[['fips', 'total_ead']], 
                                           on='fips', how='inner')
    driver_analysis = driver_analysis.merge(capacity_df, on='fips', how='inner')
    
    # Merge individual damage states for weighted damage computation
    driver_analysis = driver_analysis.merge(
        ead_wide[['fips', 'DS1', 'DS2', 'DS3', 'DS4']], 
        on='fips', 
        how='left'
    )
    
    # Remove invalid values
    driver_analysis = driver_analysis[
        (driver_analysis['earp_months_per_year'] > 0) & 
        (driver_analysis['total_ead'] > 0) & 
        (driver_analysis['construction_capacity'] > 0)
    ]
    
    print(f"Counties with complete data: {len(driver_analysis)}")
    
    # Compute weighted EAD (recovery-time-weighted damage)
    driver_analysis['weighted_ead'] = (
        driver_analysis['DS1'] * RECOVERY_WEIGHTS['DS1'] +
        driver_analysis['DS2'] * RECOVERY_WEIGHTS['DS2'] +
        driver_analysis['DS3'] * RECOVERY_WEIGHTS['DS3'] +
        driver_analysis['DS4'] * RECOVERY_WEIGHTS['DS4']
    )
    
    # Log-transform for correlations
    driver_analysis['log_earp'] = np.log10(driver_analysis['earp_months_per_year'])
    driver_analysis['log_risk'] = np.log10(driver_analysis['total_ead'])
    driver_analysis['log_weighted_risk'] = np.log10(driver_analysis['weighted_ead'])
    driver_analysis['log_capacity'] = np.log10(driver_analysis['construction_capacity'])
    
    return driver_analysis


def prepare_per_event_analysis(recovery_all_events, units_df, capacity_df):
    """
    Prepare dataset for per-event analysis with mean, median, and max values.
    
    Parameters
    ----------
    recovery_all_events : pd.DataFrame
        Per-event recovery potential data
    units_df : pd.DataFrame
        Per-event damage data
    capacity_df : pd.DataFrame
        Construction capacity data
    
    Returns
    -------
    pd.DataFrame
        Full dataset with all per-event metrics (mean, median, max)
    """
    print("\n=== Preparing Per-Event Metrics ===\n")
    
    # Mean, median, and max recovery time per event for each county
    per_event_recovery = recovery_all_events.groupby('fips')['recovery_potential [months]'].agg(
        ['mean', 'median', 'max', 'count']
    ).reset_index()
    per_event_recovery.columns = ['fips', 'mean_recovery_months', 
                                   'median_recovery_months', 'max_recovery_months', 'num_events']
    
    # Mean, median, and max damage per event for each county
    per_event_damage = units_df.groupby('fips')[[
        'units_DS1_scaled', 'units_DS2_scaled', 
        'units_DS3_scaled', 'units_DS4_scaled'
    ]].agg(['mean', 'median', 'max']).reset_index()
    
    # Flatten column names
    per_event_damage.columns = ['fips', 'DS1_mean', 'DS1_median', 'DS1_max',
                                 'DS2_mean', 'DS2_median', 'DS2_max',
                                 'DS3_mean', 'DS3_median', 'DS3_max',
                                 'DS4_mean', 'DS4_median', 'DS4_max']
    
    # Calculate total damage (mean, median, max)
    per_event_damage['mean_damage_units'] = (
        per_event_damage[['DS1_mean', 'DS2_mean', 'DS3_mean', 'DS4_mean']].sum(axis=1)
    )
    per_event_damage['median_damage_units'] = (
        per_event_damage[['DS1_median', 'DS2_median', 'DS3_median', 'DS4_median']].sum(axis=1)
    )
    per_event_damage['max_damage_units'] = (
        per_event_damage[['DS1_max', 'DS2_max', 'DS3_max', 'DS4_max']].sum(axis=1)
    )
    
    # Calculate weighted damage (recovery-time-weighted)
    per_event_damage['median_weighted_damage'] = (
        per_event_damage['DS1_median'] * RECOVERY_WEIGHTS['DS1'] +
        per_event_damage['DS2_median'] * RECOVERY_WEIGHTS['DS2'] +
        per_event_damage['DS3_median'] * RECOVERY_WEIGHTS['DS3'] +
        per_event_damage['DS4_median'] * RECOVERY_WEIGHTS['DS4']
    )
    per_event_damage['max_weighted_damage'] = (
        per_event_damage['DS1_max'] * RECOVERY_WEIGHTS['DS1'] +
        per_event_damage['DS2_max'] * RECOVERY_WEIGHTS['DS2'] +
        per_event_damage['DS3_max'] * RECOVERY_WEIGHTS['DS3'] +
        per_event_damage['DS4_max'] * RECOVERY_WEIGHTS['DS4']
    )
    
    # Merge with capacity
    per_event_analysis = per_event_recovery.merge(per_event_damage, on='fips', how='inner')
    per_event_analysis = per_event_analysis.merge(capacity_df, on='fips', how='inner')
    
    print(f"Prepared per-event data for {len(per_event_analysis)} counties")
    
    return per_event_analysis


def prepare_per_event_analysis_median(per_event_analysis):
    """
    Filter and prepare median per-event analysis.
    
    Parameters
    ----------
    per_event_analysis : pd.DataFrame
        Full per-event dataset
    
    Returns
    -------
    pd.DataFrame
        Filtered dataset with log-transformed median metrics
    """
    # Filter to valid median data
    per_event_analysis_median = per_event_analysis[
        (per_event_analysis['median_recovery_months'] > 0) & 
        (per_event_analysis['median_damage_units'] > 0) & 
        (per_event_analysis['construction_capacity'] > 0)
    ].copy()
    
    print(f"Counties with complete per-event MEDIAN data: {len(per_event_analysis_median)}")
    
    # Log-transform
    per_event_analysis_median['log_recovery'] = np.log10(
        per_event_analysis_median['median_recovery_months']
    )
    per_event_analysis_median['log_damage'] = np.log10(
        per_event_analysis_median['median_damage_units']
    )
    per_event_analysis_median['log_weighted_damage'] = np.log10(
        per_event_analysis_median['median_weighted_damage']
    )
    per_event_analysis_median['log_capacity'] = np.log10(
        per_event_analysis_median['construction_capacity']
    )
    
    return per_event_analysis_median


def prepare_per_event_analysis_maximum(per_event_analysis):
    """
    Filter and prepare maximum per-event analysis.
    
    Parameters
    ----------
    per_event_analysis : pd.DataFrame
        Full per-event dataset
    
    Returns
    -------
    pd.DataFrame
        Filtered dataset with log-transformed maximum metrics
    """
    # Filter to valid maximum data
    per_event_analysis_maximum = per_event_analysis[
        (per_event_analysis['max_recovery_months'] > 0) & 
        (per_event_analysis['max_damage_units'] > 0) & 
        (per_event_analysis['construction_capacity'] > 0)
    ].copy()
    
    print(f"Counties with complete per-event MAXIMUM data: {len(per_event_analysis_maximum)}")
    
    # Log-transform
    per_event_analysis_maximum['log_recovery'] = np.log10(
        per_event_analysis_maximum['max_recovery_months']
    )
    per_event_analysis_maximum['log_damage'] = np.log10(
        per_event_analysis_maximum['max_damage_units']
    )
    per_event_analysis_maximum['log_weighted_damage'] = np.log10(
        per_event_analysis_maximum['max_weighted_damage']
    )
    per_event_analysis_maximum['log_capacity'] = np.log10(
        per_event_analysis_maximum['construction_capacity']
    )
    
    return per_event_analysis_maximum


# ============================================================================
# SECTION 4: CORRELATION ANALYSIS
# ============================================================================

def compute_correlations_annual(driver_analysis):
    """
    Compute correlations for annual metrics.
    
    Parameters
    ----------
    driver_analysis : pd.DataFrame
        Annual driver analysis dataset
    
    Returns
    -------
    dict
        Correlation coefficients and p-values
    """
    print("\n=== Correlation Analysis: Annual Metrics ===\n")
    
    corr_earp_risk, p_earp_risk = pearsonr(
        driver_analysis['log_earp'], 
        driver_analysis['log_risk']
    )
    corr_earp_capacity, p_earp_capacity = pearsonr(
        driver_analysis['log_earp'], 
        driver_analysis['log_capacity']
    )
    
    print(f"EARP vs Risk (EAD):      {corr_earp_risk:+.3f} (p={p_earp_risk:.2e})")
    print(f"EARP vs Capacity:        {corr_earp_capacity:+.3f} (p={p_earp_capacity:.2e})")
    
    return {
        'corr_risk': corr_earp_risk,
        'p_risk': p_earp_risk,
        'corr_capacity': corr_earp_capacity,
        'p_capacity': p_earp_capacity
    }


def compute_correlations_per_event(per_event_analysis_median):
    """
    Compute correlations for per-event metrics.
    
    Parameters
    ----------
    per_event_analysis_median : pd.DataFrame
        Per-event analysis dataset
    
    Returns
    -------
    dict
        Correlation coefficients and p-values
    """
    print("\n=== Correlation Analysis: Per-Event Metrics ===\n")
    
    corr_recovery_damage, p_recovery_damage = pearsonr(
        per_event_analysis_median['log_recovery'], 
        per_event_analysis_median['log_damage']
    )
    corr_recovery_capacity, p_recovery_capacity = pearsonr(
        per_event_analysis_median['log_recovery'], 
        per_event_analysis_median['log_capacity']
    )
    
    print(f"Recovery vs Damage:   {corr_recovery_damage:+.3f} (p={p_recovery_damage:.2e})")
    print(f"Recovery vs Capacity: {corr_recovery_capacity:+.3f} (p={p_recovery_capacity:.2e})")
    
    return {
        'corr_damage': corr_recovery_damage,
        'p_damage': p_recovery_damage,
        'corr_capacity': corr_recovery_capacity,
        'p_capacity': p_recovery_capacity
    }


# ============================================================================
# SECTION 5: VARIANCE PARTITIONING ANALYSIS
# ============================================================================

def variance_partitioning(y, X1, X2, var1_name='Variable 1', var2_name='Variable 2'):
    """
    Partition variance explained by two predictors into unique and shared components.
    
    Uses hierarchical variance partitioning to decompose R² into:
    - Unique contribution of predictor 1
    - Unique contribution of predictor 2
    - Shared contribution
    - Unexplained variance
    
    Parameters
    ----------
    y : array-like
        Response variable (e.g., log recovery time)
    X1 : array-like
        First predictor (e.g., log damage)
    X2 : array-like
        Second predictor (e.g., log capacity)
    var1_name : str
        Name of first variable
    var2_name : str
        Name of second variable
    
    Returns
    -------
    dict
        Variance components and model information
    """
    # Ensure 2D arrays
    X1 = np.array(X1).reshape(-1, 1)
    X2 = np.array(X2).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    
    # Fit individual models
    model_1 = LinearRegression().fit(X1, y)
    model_2 = LinearRegression().fit(X2, y)
    
    # Fit combined model
    X_both = np.hstack([X1, X2])
    model_both = LinearRegression().fit(X_both, y)
    
    # Calculate R² values
    r2_1 = r2_score(y, model_1.predict(X1))
    r2_2 = r2_score(y, model_2.predict(X2))
    r2_both = r2_score(y, model_both.predict(X_both))
    
    # Variance partitioning
    unique_1 = r2_both - r2_2
    unique_2 = r2_both - r2_1
    shared = r2_1 + r2_2 - r2_both
    unexplained = 1 - r2_both
    
    # Get standardized coefficients
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_both_std = scaler_X.fit_transform(X_both)
    y_std = scaler_y.fit_transform(y)
    model_both_std = LinearRegression().fit(X_both_std, y_std)
    
    return {
        'r2_var1': r2_1,
        'r2_var2': r2_2,
        'r2_combined': r2_both,
        'unique_var1': unique_1,
        'unique_var2': unique_2,
        'shared': shared,
        'unexplained': unexplained,
        'beta_var1': model_both_std.coef_[0][0],
        'beta_var2': model_both_std.coef_[0][1],
        'model_both': model_both,
        'var1_name': var1_name,
        'var2_name': var2_name
    }


def perform_variance_partitioning_analysis(driver_analysis, per_event_analysis_median, 
                                          per_event_analysis_maximum):
    """
    Perform variance partitioning for annual, per-event median, and per-event maximum metrics.
    
    Parameters
    ----------
    driver_analysis : pd.DataFrame
        Annual metrics dataset
    per_event_analysis_median : pd.DataFrame
        Per-event median metrics dataset
    per_event_analysis_maximum : pd.DataFrame
        Per-event maximum metrics dataset
    
    Returns
    -------
    tuple
        (vp_annual, vp_event_median, vp_event_maximum) variance partitioning results
    """
    print("\n" + "="*80)
    print("VARIANCE PARTITIONING ANALYSIS")
    print("="*80)
    
    # Annual metrics
    print("\n--- ANNUAL METRICS (EARP) ---")
    y_annual = driver_analysis['log_earp'].values
    X_damage_annual = driver_analysis['log_risk'].values
    X_capacity_annual = driver_analysis['log_capacity'].values
    
    vp_annual = variance_partitioning(
        y_annual, X_damage_annual, X_capacity_annual,
        var1_name='EAD (Annual Damage)', var2_name='Construction Capacity'
    )
    
    print(f"R² Combined: {vp_annual['r2_combined']:.4f}")
    print(f"  Unique to Damage:   {vp_annual['unique_var1']*100:.1f}%")
    print(f"  Unique to Capacity: {vp_annual['unique_var2']*100:.1f}%")
    print(f"  Shared:             {vp_annual['shared']*100:.1f}%")
    
    # Per-event MEDIAN metrics
    print("\n--- PER-EVENT METRICS (Median) ---")
    y_event = per_event_analysis_median['log_recovery'].values
    X_damage_event = per_event_analysis_median['log_damage'].values
    X_capacity_event = per_event_analysis_median['log_capacity'].values
    
    vp_event_median = variance_partitioning(
        y_event, X_damage_event, X_capacity_event,
        var1_name='Median Event Damage', var2_name='Construction Capacity'
    )
    
    print(f"R² Combined: {vp_event_median['r2_combined']:.4f}")
    print(f"  Unique to Damage:   {vp_event_median['unique_var1']*100:.1f}%")
    print(f"  Unique to Capacity: {vp_event_median['unique_var2']*100:.1f}%")
    print(f"  Shared:             {vp_event_median['shared']*100:.1f}%")
    
    # Per-event MAXIMUM metrics
    print("\n--- PER-EVENT METRICS (Maximum) ---")
    y_event_max = per_event_analysis_maximum['log_recovery'].values
    X_damage_max = per_event_analysis_maximum['log_damage'].values
    X_capacity_max = per_event_analysis_maximum['log_capacity'].values
    
    vp_event_maximum = variance_partitioning(
        y_event_max, X_damage_max, X_capacity_max,
        var1_name='Maximum Event Damage', var2_name='Construction Capacity'
    )
    
    print(f"R² Combined: {vp_event_maximum['r2_combined']:.4f}")
    print(f"  Unique to Damage:   {vp_event_maximum['unique_var1']*100:.1f}%")
    print(f"  Unique to Capacity: {vp_event_maximum['unique_var2']*100:.1f}%")
    print(f"  Shared:             {vp_event_maximum['shared']*100:.1f}%")
    
    return vp_annual, vp_event_median, vp_event_maximum


# ============================================================================
# SECTION 6: COUNTY-LEVEL VARIANCE DECOMPOSITION
# ============================================================================

def compute_county_variance_contributions_per_event(per_event_analysis_median):
    """
    Compute county-level variance contributions from damage and capacity.
    
    Uses a global regression model to partition variance contributions at the
    county level based on squared predictor contributions weighted by their
    regression coefficients.
    
    Parameters
    ----------
    per_event_analysis_median : pd.DataFrame
        Per-event median metrics per county
    
    Returns
    -------
    tuple
        (enhanced dataframe with variance shares, model coefficients dict)
    """
    print("="*80)
    print("COUNTY-LEVEL VARIANCE DECOMPOSITION: PER-EVENT")
    print("="*80)
    
    # Filter to valid data
    mask = (
        (per_event_analysis_median['median_damage_units'] > 0) & 
        (per_event_analysis_median['construction_capacity'] > 0) &
        (per_event_analysis_median['median_recovery_months'] > 0)
    )
    
    data = per_event_analysis_median[mask].copy()
    
    # Log-transform
    data['log_recovery'] = np.log10(data['median_recovery_months'])
    data['log_damage'] = np.log10(data['median_damage_units'])
    data['log_capacity'] = np.log10(data['construction_capacity'])
    
    print(f"\nSample size: n = {len(data)} counties")
    
    # Fit global regression model
    X = data[['log_damage', 'log_capacity']].values
    y = data['log_recovery'].values
    
    model = LinearRegression().fit(X, y)
    
    beta_0 = model.intercept_
    beta_D = model.coef_[0]  # Damage coefficient
    beta_C = model.coef_[1]  # Capacity coefficient
    r2 = model.score(X, y)
    
    print("\n" + "="*80)
    print("GLOBAL REGRESSION MODEL")
    print("="*80)
    print(f"\nlog(Recovery) = {beta_0:.3f} + {beta_D:+.3f}·log(Damage) "
          f"+ {beta_C:+.3f}·log(Capacity)")
    print(f"\nR² = {r2:.3f}")
    print(f"\nInterpretation:")
    print(f"  • β_D = {beta_D:+.3f}: "
          f"{'Positive' if beta_D > 0 else 'Negative'} effect of damage")
    print(f"  • β_C = {beta_C:+.3f}: "
          f"{'Positive' if beta_C > 0 else 'Negative'} effect of capacity")
    
    # Compute county-level contributions
    print("\n" + "="*80)
    print("COUNTY-LEVEL VARIANCE CONTRIBUTIONS")
    print("="*80)
    
    data['contribution_damage'] = (beta_D * data['log_damage']) ** 2
    data['contribution_capacity'] = (beta_C * data['log_capacity']) ** 2
    data['contribution_total'] = (
        data['contribution_damage'] + data['contribution_capacity']
    )
    
    # Normalize to get shares (0-1)
    data['share_damage'] = (
        data['contribution_damage'] / data['contribution_total']
    )
    data['share_capacity'] = (
        data['contribution_capacity'] / data['contribution_total']
    )
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"\nDamage Share:")
    print(f"  Mean: {data['share_damage'].mean():.3f}")
    print(f"  Median: {data['share_damage'].median():.3f}")
    print(f"  Std: {data['share_damage'].std():.3f}")
    print(f"  Range: [{data['share_damage'].min():.3f}, "
          f"{data['share_damage'].max():.3f}]")
    
    print(f"\nCapacity Share:")
    print(f"  Mean: {data['share_capacity'].mean():.3f}")
    print(f"  Median: {data['share_capacity'].median():.3f}")
    print(f"  Std: {data['share_capacity'].std():.3f}")
    print(f"  Range: [{data['share_capacity'].min():.3f}, "
          f"{data['share_capacity'].max():.3f}]")
    
    # Classify dominant driver
    data['dominant_driver_variance'] = data.apply(
        lambda row: 'Damage' if row['share_damage'] > row['share_capacity'] 
        else 'Capacity',
        axis=1
    )
    
    driver_counts = data['dominant_driver_variance'].value_counts()
    print(f"\n=== Dominant Driver (by variance share) ===")
    for driver, count in driver_counts.items():
        pct = 100 * count / len(data)
        print(f"  {driver}-driven: {count} counties ({pct:.1f}%)")
    
    # ==============================================================
    # SENSITIVITY ANALYSIS: Marginal effects of interventions
    # ==============================================================
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: Intervention Effects (Per-Event Median)")
    print("="*80)
    
    # Calculate marginal effects for standardized interventions
    log_doubling = np.log10(2)
    log_halving = np.log10(0.5)
    
    data['sensitivity_capacity_2x'] = abs(beta_C) * log_doubling
    data['sensitivity_damage_50pct'] = abs(beta_D) * abs(log_halving)
    data['sensitivity_ratio'] = (
        data['sensitivity_capacity_2x'] / data['sensitivity_damage_50pct']
    )
    
    print(f"\n=== Marginal Effects (Per-Event Median) ===")
    print(f"\nEffect of DOUBLING capacity:")
    print(f"  Mean recovery reduction: {data['sensitivity_capacity_2x'].mean():.3f} log10(months)")
    
    print(f"\nEffect of 50% damage reduction:")
    print(f"  Mean recovery reduction: {data['sensitivity_damage_50pct'].mean():.3f} log10(months)")
    
    print(f"\nSensitivity Ratio (capacity effect / damage effect):")
    print(f"  Mean: {data['sensitivity_ratio'].mean():.3f}")
    print(f"  Median: {data['sensitivity_ratio'].median():.3f}")
    print(f"  Counties where capacity 2x > damage 50%: "
          f"{(data['sensitivity_ratio'] > 1).sum()} ({(data['sensitivity_ratio'] > 1).sum()/len(data)*100:.1f}%)")
    
    # Hybrid classification
    capacity_low_threshold = data['construction_capacity'].quantile(0.25)
    damage_high_threshold = data['median_damage_units'].quantile(0.75)
    
    def classify_intervention_priority_event(row):
        """Classify counties by intervention priority using variance + conditions."""
        if row['construction_capacity'] < capacity_low_threshold and row['share_capacity'] < 0.4:
            return 'Critical Capacity Bottleneck'
        elif row['share_capacity'] > 0.5 and row['construction_capacity'] < capacity_low_threshold:
            return 'Capacity Building Priority'
        elif row['share_damage'] > 0.5 and row['median_damage_units'] > damage_high_threshold:
            return 'Damage Mitigation Priority'
        else:
            return 'Mixed Strategy'
    
    data['intervention_priority'] = data.apply(
        classify_intervention_priority_event, axis=1
    )
    
    priority_counts = data['intervention_priority'].value_counts()
    print(f"\n=== Intervention Priority Classification (Per-Event) ===")
    for priority, count in priority_counts.items():
        pct = 100 * count / len(data)
        print(f"  {priority}: {count} counties ({pct:.1f}%)")
    
    # Merge back into original dataframe
    result_df = per_event_analysis_median.merge(
        data[['fips', 'share_damage', 'share_capacity', 'dominant_driver_variance',
              'contribution_damage', 'contribution_capacity',
              'sensitivity_capacity_2x', 'sensitivity_damage_50pct',
              'sensitivity_ratio', 'intervention_priority']],
        on='fips',
        how='left'
    )
    
    print(f"\n✓ County-level variance shares and sensitivity metrics calculated and merged")
    
    model_info = {
        'beta_0': beta_0,
        'beta_D': beta_D,
        'beta_C': beta_C,
        'r2': r2
    }
    
    return result_df, model_info


def compute_county_variance_contributions_per_event_maximum(per_event_analysis_maximum):
    """
    Compute county-level variance contributions from damage and capacity for maximum events.
    
    Uses a global regression model to partition variance contributions at the
    county level based on squared predictor contributions weighted by their
    regression coefficients.
    
    Parameters
    ----------
    per_event_analysis_maximum : pd.DataFrame
        Per-event maximum metrics per county
    
    Returns
    -------
    tuple
        (enhanced dataframe with variance shares, model coefficients dict)
    """
    print("="*80)
    print("COUNTY-LEVEL VARIANCE DECOMPOSITION: PER-EVENT MAXIMUM")
    print("="*80)
    
    # Filter to valid data
    mask = (
        (per_event_analysis_maximum['max_damage_units'] > 0) & 
        (per_event_analysis_maximum['construction_capacity'] > 0) &
        (per_event_analysis_maximum['max_recovery_months'] > 0)
    )
    
    data = per_event_analysis_maximum[mask].copy()
    
    # Log-transform
    data['log_recovery'] = np.log10(data['max_recovery_months'])
    data['log_damage'] = np.log10(data['max_damage_units'])
    data['log_capacity'] = np.log10(data['construction_capacity'])
    
    print(f"\nSample size: n = {len(data)} counties")
    
    # Fit global regression model
    X = data[['log_damage', 'log_capacity']].values
    y = data['log_recovery'].values
    
    model = LinearRegression().fit(X, y)
    
    beta_0 = model.intercept_
    beta_D = model.coef_[0]  # Damage coefficient
    beta_C = model.coef_[1]  # Capacity coefficient
    r2 = model.score(X, y)
    
    print("\n" + "="*80)
    print("GLOBAL REGRESSION MODEL (Maximum Events)")
    print("="*80)
    print(f"\nlog(Recovery) = {beta_0:.3f} + {beta_D:+.3f}·log(MaxDamage) "
          f"+ {beta_C:+.3f}·log(Capacity)")
    print(f"\nR² = {r2:.3f}")
    print(f"\nInterpretation:")
    print(f"  • β_D = {beta_D:+.3f}: "
          f"{'Positive' if beta_D > 0 else 'Negative'} effect of max damage")
    print(f"  • β_C = {beta_C:+.3f}: "
          f"{'Positive' if beta_C > 0 else 'Negative'} effect of capacity")
    
    # Compute county-level contributions
    print("\n" + "="*80)
    print("COUNTY-LEVEL VARIANCE CONTRIBUTIONS")
    print("="*80)
    
    data['contribution_damage'] = (beta_D * data['log_damage']) ** 2
    data['contribution_capacity'] = (beta_C * data['log_capacity']) ** 2
    data['contribution_total'] = (
        data['contribution_damage'] + data['contribution_capacity']
    )
    
    # Normalize to get shares (0-1)
    data['share_damage'] = (
        data['contribution_damage'] / data['contribution_total']
    )
    data['share_capacity'] = (
        data['contribution_capacity'] / data['contribution_total']
    )
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"\nDamage Share:")
    print(f"  Mean: {data['share_damage'].mean():.3f}")
    print(f"  Median: {data['share_damage'].median():.3f}")
    print(f"  Std: {data['share_damage'].std():.3f}")
    print(f"  Range: [{data['share_damage'].min():.3f}, "
          f"{data['share_damage'].max():.3f}]")
    
    print(f"\nCapacity Share:")
    print(f"  Mean: {data['share_capacity'].mean():.3f}")
    print(f"  Median: {data['share_capacity'].median():.3f}")
    print(f"  Std: {data['share_capacity'].std():.3f}")
    print(f"  Range: [{data['share_capacity'].min():.3f}, "
          f"{data['share_capacity'].max():.3f}]")
    
    # Classify dominant driver
    data['dominant_driver_variance'] = data.apply(
        lambda row: 'Damage' if row['share_damage'] > row['share_capacity'] 
        else 'Capacity',
        axis=1
    )
    
    driver_counts = data['dominant_driver_variance'].value_counts()
    print(f"\n=== Dominant Driver (by variance share) ===")
    for driver, count in driver_counts.items():
        pct = 100 * count / len(data)
        print(f"  {driver}-driven: {count} counties ({pct:.1f}%)")
    
    # ==============================================================
    # SENSITIVITY ANALYSIS: Marginal effects of interventions
    # ==============================================================
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: Intervention Effects (Per-Event Maximum)")
    print("="*80)
    
    # Calculate marginal effects for standardized interventions
    log_doubling = np.log10(2)
    log_halving = np.log10(0.5)
    
    data['sensitivity_capacity_2x'] = abs(beta_C) * log_doubling
    data['sensitivity_damage_50pct'] = abs(beta_D) * abs(log_halving)
    data['sensitivity_ratio'] = (
        data['sensitivity_capacity_2x'] / data['sensitivity_damage_50pct']
    )
    
    print(f"\n=== Marginal Effects (Per-Event Maximum) ===")
    print(f"\nEffect of DOUBLING capacity:")
    print(f"  Mean recovery reduction: {data['sensitivity_capacity_2x'].mean():.3f} log10(months)")
    
    print(f"\nEffect of 50% damage reduction:")
    print(f"  Mean recovery reduction: {data['sensitivity_damage_50pct'].mean():.3f} log10(months)")
    
    print(f"\nSensitivity Ratio (capacity effect / damage effect):")
    print(f"  Mean: {data['sensitivity_ratio'].mean():.3f}")
    print(f"  Median: {data['sensitivity_ratio'].median():.3f}")
    print(f"  Counties where capacity 2x > damage 50%: "
          f"{(data['sensitivity_ratio'] > 1).sum()} ({(data['sensitivity_ratio'] > 1).sum()/len(data)*100:.1f}%)")
    
    # Hybrid classification
    capacity_low_threshold = data['construction_capacity'].quantile(0.25)
    damage_high_threshold = data['max_damage_units'].quantile(0.75)
    
    def classify_intervention_priority_max(row):
        """Classify counties by intervention priority using variance + conditions."""
        if row['construction_capacity'] < capacity_low_threshold and row['share_capacity'] < 0.4:
            return 'Critical Capacity Bottleneck'
        elif row['share_capacity'] > 0.5 and row['construction_capacity'] < capacity_low_threshold:
            return 'Capacity Building Priority'
        elif row['share_damage'] > 0.5 and row['max_damage_units'] > damage_high_threshold:
            return 'Damage Mitigation Priority'
        else:
            return 'Mixed Strategy'
    
    data['intervention_priority'] = data.apply(
        classify_intervention_priority_max, axis=1
    )
    
    priority_counts = data['intervention_priority'].value_counts()
    print(f"\n=== Intervention Priority Classification (Per-Event Maximum) ===")
    for priority, count in priority_counts.items():
        pct = 100 * count / len(data)
        print(f"  {priority}: {count} counties ({pct:.1f}%)")
    
    # Merge back into original dataframe
    result_df = per_event_analysis_maximum.merge(
        data[['fips', 'share_damage', 'share_capacity', 'dominant_driver_variance',
              'contribution_damage', 'contribution_capacity',
              'sensitivity_capacity_2x', 'sensitivity_damage_50pct',
              'sensitivity_ratio', 'intervention_priority']],
        on='fips',
        how='left'
    )
    
    print(f"\n✓ County-level variance shares and sensitivity metrics calculated and merged")
    
    model_info = {
        'beta_0': beta_0,
        'beta_D': beta_D,
        'beta_C': beta_C,
        'r2': r2
    }
    
    return result_df, model_info


def compute_county_variance_contributions_annual(driver_analysis):
    """
    Compute county-level variance contributions using annual metrics.
    
    Parameters
    ----------
    driver_analysis : pd.DataFrame
        DataFrame with annual metrics (EARP, EAD, capacity)
    
    Returns
    -------
    tuple
        (enhanced dataframe with variance shares, model coefficients dict)
    """
    print("="*80)
    print("COUNTY-LEVEL VARIANCE DECOMPOSITION: ANNUAL METRICS")
    print("="*80)
    
    # Filter to valid data
    mask_annual = (
        (driver_analysis['total_ead'] > 0) & 
        (driver_analysis['construction_capacity'] > 0) &
        (driver_analysis['earp_months_per_year'] > 0)
    )
    
    data_annual = driver_analysis[mask_annual].copy()
    
    # Log-transform
    data_annual['log_earp'] = np.log10(data_annual['earp_months_per_year'])
    data_annual['log_ead'] = np.log10(data_annual['total_ead'])
    data_annual['log_capacity'] = np.log10(data_annual['construction_capacity'])
    
    print(f"\nSample size: n = {len(data_annual)} counties")
    
    # Fit global regression model
    X_annual = data_annual[['log_ead', 'log_capacity']].values
    y_annual = data_annual['log_earp'].values
    
    model_annual = LinearRegression().fit(X_annual, y_annual)
    
    beta_0_annual = model_annual.intercept_
    beta_D_annual = model_annual.coef_[0]  # Damage (EAD) coefficient
    beta_C_annual = model_annual.coef_[1]  # Capacity coefficient
    r2_annual = model_annual.score(X_annual, y_annual)
    
    print("\n" + "="*80)
    print("GLOBAL REGRESSION MODEL (ANNUAL)")
    print("="*80)
    print(f"\nlog(EARP) = {beta_0_annual:.3f} + {beta_D_annual:+.3f}·log(EAD) "
          f"+ {beta_C_annual:+.3f}·log(Capacity)")
    print(f"\nR² = {r2_annual:.3f}")
    print(f"\nInterpretation:")
    print(f"  • β_D = {beta_D_annual:+.3f}: "
          f"{'Positive' if beta_D_annual > 0 else 'Negative'} effect of annual damage")
    print(f"  • β_C = {beta_C_annual:+.3f}: "
          f"{'Positive' if beta_C_annual > 0 else 'Negative'} effect of capacity")
    
    # Compute county-level contributions
    print("\n" + "="*80)
    print("COUNTY-LEVEL VARIANCE CONTRIBUTIONS (ANNUAL)")
    print("="*80)
    
    data_annual['contribution_damage'] = (
        beta_D_annual * data_annual['log_ead']
    ) ** 2
    data_annual['contribution_capacity'] = (
        beta_C_annual * data_annual['log_capacity']
    ) ** 2
    data_annual['contribution_total'] = (
        data_annual['contribution_damage'] + data_annual['contribution_capacity']
    )
    
    # Normalize to get shares (0-1)
    data_annual['share_damage'] = (
        data_annual['contribution_damage'] / data_annual['contribution_total']
    )
    data_annual['share_capacity'] = (
        data_annual['contribution_capacity'] / data_annual['contribution_total']
    )
    
    # Print summary statistics
    print(f"\n=== Summary Statistics (Annual) ===")
    print(f"\nDamage (EAD) Share:")
    print(f"  Mean: {data_annual['share_damage'].mean():.3f}")
    print(f"  Median: {data_annual['share_damage'].median():.3f}")
    print(f"  Std: {data_annual['share_damage'].std():.3f}")
    print(f"  Range: [{data_annual['share_damage'].min():.3f}, "
          f"{data_annual['share_damage'].max():.3f}]")
    
    print(f"\nCapacity Share:")
    print(f"  Mean: {data_annual['share_capacity'].mean():.3f}")
    print(f"  Median: {data_annual['share_capacity'].median():.3f}")
    print(f"  Std: {data_annual['share_capacity'].std():.3f}")
    print(f"  Range: [{data_annual['share_capacity'].min():.3f}, "
          f"{data_annual['share_capacity'].max():.3f}]")
    
    # Classify dominant driver
    data_annual['dominant_driver_variance_annual'] = data_annual.apply(
        lambda row: 'Damage' if row['share_damage'] > row['share_capacity'] 
        else 'Capacity',
        axis=1
    )
    
    driver_counts_annual = data_annual['dominant_driver_variance_annual'].value_counts()
    print(f"\n=== Dominant Driver (Annual, by variance share) ===")
    for driver, count in driver_counts_annual.items():
        pct = 100 * count / len(data_annual)
        print(f"  {driver}-driven: {count} counties ({pct:.1f}%)")
    
    # ==============================================================
    # SENSITIVITY ANALYSIS: Marginal effects of interventions
    # ==============================================================
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: Intervention Effects")
    print("="*80)
    
    # Calculate marginal effects for standardized interventions
    # Effect of doubling capacity (log10(2) ≈ 0.301)
    log_doubling = np.log10(2)
    
    # Effect of 50% damage reduction (log10(0.5) ≈ -0.301)
    log_halving = np.log10(0.5)
    
    data_annual['sensitivity_capacity_2x'] = abs(beta_C_annual) * log_doubling
    data_annual['sensitivity_damage_50pct'] = abs(beta_D_annual) * abs(log_halving)
    
    # Calculate relative sensitivity
    data_annual['sensitivity_ratio'] = (
        data_annual['sensitivity_capacity_2x'] / data_annual['sensitivity_damage_50pct']
    )
    
    print(f"\n=== Marginal Effects (Annual) ===")
    print(f"\nEffect of DOUBLING capacity:")
    print(f"  Mean recovery reduction: {data_annual['sensitivity_capacity_2x'].mean():.3f} log10(months)")
    print(f"  In real terms: ~{(10**(data_annual['sensitivity_capacity_2x'].mean()) - 1)*100:.1f}% change")
    
    print(f"\nEffect of 50% damage reduction:")
    print(f"  Mean recovery reduction: {data_annual['sensitivity_damage_50pct'].mean():.3f} log10(months)")
    print(f"  In real terms: ~{(1 - 10**(-data_annual['sensitivity_damage_50pct'].mean()))*100:.1f}% reduction")
    
    print(f"\nSensitivity Ratio (capacity effect / damage effect):")
    print(f"  Mean: {data_annual['sensitivity_ratio'].mean():.3f}")
    print(f"  Median: {data_annual['sensitivity_ratio'].median():.3f}")
    print(f"  Counties where capacity 2x > damage 50%: "
          f"{(data_annual['sensitivity_ratio'] > 1).sum()} ({(data_annual['sensitivity_ratio'] > 1).sum()/len(data_annual)*100:.1f}%)")
    
    # ==============================================================
    # HYBRID CLASSIFICATION: Variance share + Absolute conditions
    # ==============================================================
    print("\n" + "="*80)
    print("HYBRID INTERVENTION CLASSIFICATION")
    print("="*80)
    
    # Define thresholds
    capacity_low_threshold = data_annual['construction_capacity'].quantile(0.25)
    damage_high_threshold = data_annual['total_ead'].quantile(0.75)
    variance_threshold = 0.5  # Balanced if both within 0.4-0.6 range
    
    # Classification logic
    def classify_intervention_priority(row):
        """Classify counties by intervention priority using variance + conditions."""
        # Critical capacity bottleneck: extremely low capacity regardless of variance
        if row['construction_capacity'] < capacity_low_threshold and row['share_capacity'] < 0.4:
            return 'Critical Capacity Bottleneck'
        
        # Capacity-driven: high capacity variance share + low absolute capacity
        elif row['share_capacity'] > 0.5 and row['construction_capacity'] < capacity_low_threshold:
            return 'Capacity Building Priority'
        
        # Damage-driven: high damage variance share + high absolute damage + adequate capacity
        elif row['share_damage'] > 0.5 and row['total_ead'] > damage_high_threshold:
            return 'Damage Mitigation Priority'
        
        # Mixed/balanced
        else:
            return 'Mixed Strategy'
    
    data_annual['intervention_priority'] = data_annual.apply(
        classify_intervention_priority, axis=1
    )
    
    priority_counts = data_annual['intervention_priority'].value_counts()
    print(f"\n=== Intervention Priority Classification ===")
    for priority, count in priority_counts.items():
        pct = 100 * count / len(data_annual)
        print(f"  {priority}: {count} counties ({pct:.1f}%)")
    
    print(f"\nKey Insight:")
    print(f"  'Critical Capacity Bottleneck' counties are damage-driven by variance")
    print(f"  (damage explains the pattern better), BUT capacity is so low that")
    print(f"  it's a universal constraint requiring immediate attention regardless")
    print(f"  of what the variance decomposition shows.")
    
    # Merge back into original dataframe
    result_df = driver_analysis.merge(
        data_annual[['fips', 'share_damage', 'share_capacity', 
                      'dominant_driver_variance_annual',
                      'contribution_damage', 'contribution_capacity',
                      'sensitivity_capacity_2x', 'sensitivity_damage_50pct',
                      'sensitivity_ratio', 'intervention_priority']],
        on='fips',
        how='left',
        suffixes=('', '_annual')
    )
    
    print(f"\n✓ Annual variance shares and sensitivity metrics calculated and merged")
    
    model_info = {
        'beta_0': beta_0_annual,
        'beta_D': beta_D_annual,
        'beta_C': beta_C_annual,
        'r2': r2_annual
    }
    
    return result_df, model_info


# ============================================================================
# SECTION 7: SPATIAL PATTERN ANALYSIS
# ============================================================================

def analyze_spatial_patterns(driver_analysis, per_event_analysis_median, coastal_counties):
    """
    Quantify spatial patterns in driver dominance (coastal gradient effect).
    
    Parameters
    ----------
    driver_analysis : pd.DataFrame
        Annual metrics with variance shares
    per_event_analysis_median : pd.DataFrame
        Per-event metrics with variance shares
    coastal_counties : gpd.GeoDataFrame
        County geometries
    
    Returns
    -------
    tuple
        (comparison_data, gdf_change) with change analysis
    """
    print("="*80)
    print("SPATIAL PATTERN ANALYSIS: Coastal Gradient Effect")
    print("="*80)
    
    # Merge annual and per-event data for comparison
    comparison_data = driver_analysis[['fips', 'share_capacity']].merge(
        per_event_analysis_median[['fips', 'share_capacity']],
        on='fips',
        suffixes=('_annual', '_event'),
        how='inner'
    )
    
    # Calculate difference: Annual capacity share - Event capacity share
    comparison_data['capacity_share_change'] = (
        comparison_data['share_capacity_annual'] - comparison_data['share_capacity_event']
    )
    
    print(f"\nCapacity Share Change (Annual - Per-Event):")
    print(f"  Mean: {comparison_data['capacity_share_change'].mean():+.3f}")
    print(f"  Median: {comparison_data['capacity_share_change'].median():+.3f}")
    print(f"  Std: {comparison_data['capacity_share_change'].std():.3f}")
    
    # Merge with geodataframe for mapping
    gdf_change = coastal_counties.merge(
        comparison_data[['fips', 'capacity_share_change', 
                         'share_capacity_annual', 'share_capacity_event']],
        left_on='GEOID',
        right_on='fips',
        how='left'
    )
    
    # Geographic correlation analysis
    gdf_change['centroid_lon'] = gdf_change.geometry.centroid.x
    gdf_change['centroid_lat'] = gdf_change.geometry.centroid.y
    
    comparison_geo = comparison_data.merge(
        gdf_change[['fips', 'centroid_lon', 'centroid_lat']],
        on='fips',
        how='left'
    )
    
    # Correlate capacity share change with geographic position
    corr_lat, p_lat = pearsonr(
        comparison_geo.dropna()['centroid_lat'],
        comparison_geo.dropna()['capacity_share_change']
    )
    
    print(f"\n=== Geographic Correlation Analysis ===")
    print(f"\nCapacity Share Change vs. Latitude:")
    print(f"  r = {corr_lat:+.3f}, p = {p_lat:.4f}")
    
    return comparison_data, gdf_change


# ============================================================================
# SECTION 8: VISUALIZATION FUNCTIONS
# ============================================================================

def create_ead_damage_state_maps(ead_wide, coastal_counties, output_dir="../analysis_output"):
    """
    Create 4-panel visualization of EAD per damage state.
    
    Parameters
    ----------
    ead_wide : pd.DataFrame
        EAD data in wide format
    coastal_counties : gpd.GeoDataFrame
        County geometries
    output_dir : str
        Output directory for figures
    """
    # Merge with coastal county geometries
    merged_ead = coastal_counties.merge(ead_wide, left_on='GEOID', right_on='fips', how='left')
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    damage_states = [
        ('DS1', 'cividis', 'Damage State 1: Slight\n(2-5% damage)'),
        ('DS2', 'cividis', 'Damage State 2: Moderate\n(5-10% damage)'),
        ('DS3', 'cividis', 'Damage State 3: Extensive\n(10-50% damage)'),
        ('DS4', 'cividis', 'Damage State 4: Complete\n(>50% damage)')
    ]
    
    # Create plotting copy with zeros as NaN
    merged_ead_plot = merged_ead.copy()
    for ds, _, _ in damage_states:
        merged_ead_plot.loc[merged_ead_plot[ds] <= 0, ds] = np.nan
    
    # Determine shared colorbar range
    all_positive_values = []
    for ds, _, _ in damage_states:
        data_positive = merged_ead_plot[ds].dropna()
        if not data_positive.empty:
            all_positive_values.extend(data_positive.values)
    
    if all_positive_values:
        vmin = max(min(all_positive_values) * 0.01, 0.01)
        vmax = max(all_positive_values)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None
    
    # Plot each damage state
    for idx, (ax, (ds, cmap, title)) in enumerate(zip(axes, damage_states)):
        merged_ead_plot.plot(
            column=ds,
            cmap=cmap,
            norm=norm,
            linewidth=0.1,
            edgecolor="0.5",
            legend=False,
            ax=ax,
            missing_kwds={'color': 'white', 'label': 'No data / Zero'}
        )
        ax.set_title(title, fontsize=12, pad=-1)
        ax.axis("off")
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.88, hspace=-0.05, wspace=0.0)
    cbar_ax = fig.add_axes([0.90, 0.3, 0.015, 0.4])
    sm = plt.cm.ScalarMappable(cmap='cividis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('# units', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/na_coast_ead_by_damage_state.png", 
                dpi=300, bbox_inches="tight")
    plt.show()
    
    print("✓ EAD damage state maps created")


def create_variance_share_maps(coastal_counties, driver_analysis, 
                               per_event_analysis_median, model_annual, model_event,
                               output_dir="../analysis_output"):
    """
    Create side-by-side maps of variance shares for annual vs per-event metrics.
    
    Parameters
    ----------
    coastal_counties : gpd.GeoDataFrame
        County boundaries
    driver_analysis : pd.DataFrame
        Annual metrics with variance shares
    per_event_analysis_median : pd.DataFrame
        Per-event metrics with variance shares
    model_annual : dict
        Annual model coefficients
    model_event : dict
        Per-event model coefficients
    output_dir : str
        Output directory for figures
    """
    print("Creating spatial map of variance shares...")
    
    # Merge with geodataframe - Annual
    gdf_variance_annual = coastal_counties.merge(
        driver_analysis[['fips', 'share_damage', 'share_capacity', 
                         'dominant_driver_variance_annual']],
        left_on='GEOID',
        right_on='fips',
        how='left'
    )
    
    # Merge with geodataframe - Per-Event
    gdf_variance_event = coastal_counties.merge(
        per_event_analysis_median[['fips', 'share_damage', 'share_capacity', 
                                    'dominant_driver_variance']],
        left_on='GEOID',
        right_on='fips',
        how='left'
    )
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Adjust subplot positioning to ensure consistent title placement
    fig.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.95, wspace=0.05)
    
    # Diverging colormap (red=damage, blue=capacity)
    colors_diverging = [
        '#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffff',
        '#e0f3f8', '#abd9e9', '#74add1', '#4575b4'
    ]
    cmap = LinearSegmentedColormap.from_list(
        'damage_capacity', colors_diverging, N=256
    )
    
    # Left panel: Annual (EARP)
    ax1 = axes[0]
    gdf_variance_annual.plot(
        column='share_capacity',
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidth=0.1,
        edgecolor='0.5',
        legend=False,
        ax=ax1,
        missing_kwds={'color': '#dfdcdc', 'label': 'No data'}
    )
    ax1.set_title('Expected Annual Recovery Potential', fontsize=12, y=1.02)
    ax1.axis('off')
    
    # Right panel: Per-Event (Median)
    ax2 = axes[1]
    gdf_variance_event.plot(
        column='share_capacity',
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidth=0.1,
        edgecolor='0.5',
        legend=False,
        ax=ax2,
        missing_kwds={'color': '#dfdcdc', 'label': 'No data'}
    )
    ax2.set_title('Median Event Recovery Potential', fontsize=12, y=1.02)
    ax2.axis('off')
    
    # Add shared colorbar using figure-level positioning (doesn't resize ax2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.95, 0.2, 0.015, 0.6])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Capacity Variance Share\n(0=Damage, 1=Capacity)', 
                   fontsize=10)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    cbar.ax.tick_params(labelsize=10)
    
    plt.savefig(f"{output_dir}/variance_share_annual_vs_event_maps.png", 
                dpi=300, bbox_inches="tight")
    plt.show()
    
    print("✓ Variance share maps created")


def create_variance_share_maps_three_panel(coastal_counties, driver_analysis, 
                                           per_event_analysis_median, per_event_analysis_maximum,
                                           model_annual, model_event_median, model_event_maximum,
                                           output_dir="../analysis_output"):
    """
    Create 3-panel maps of variance shares: annual, median event, and maximum event.
    
    Parameters
    ----------
    coastal_counties : gpd.GeoDataFrame
        County boundaries
    driver_analysis : pd.DataFrame
        Annual metrics with variance shares
    per_event_analysis_median : pd.DataFrame
        Per-event median metrics with variance shares
    per_event_analysis_maximum : pd.DataFrame
        Per-event maximum metrics with variance shares
    model_annual : dict
        Annual model coefficients
    model_event_median : dict
        Per-event median model coefficients
    model_event_maximum : dict
        Per-event maximum model coefficients
    output_dir : str
        Output directory for figures
    """
    print("Creating 3-panel spatial map of variance shares...")
    
    # Merge with geodataframe - Annual
    gdf_variance_annual = coastal_counties.merge(
        driver_analysis[['fips', 'share_damage', 'share_capacity', 
                         'dominant_driver_variance_annual']],
        left_on='GEOID',
        right_on='fips',
        how='left'
    )
    
    # Merge with geodataframe - Per-Event Median
    gdf_variance_median = coastal_counties.merge(
        per_event_analysis_median[['fips', 'share_damage', 'share_capacity', 
                                    'dominant_driver_variance']],
        left_on='GEOID',
        right_on='fips',
        how='left'
    )
    
    # Merge with geodataframe - Per-Event Maximum
    gdf_variance_maximum = coastal_counties.merge(
        per_event_analysis_maximum[['fips', 'share_damage', 'share_capacity', 
                                     'dominant_driver_variance']],
        left_on='GEOID',
        right_on='fips',
        how='left'
    )
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Adjust subplot positioning to ensure consistent title placement
    fig.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.92, wspace=0.05)
    
    # Diverging colormap (red=damage, blue=capacity)
    colors_diverging = [
        '#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffff',
        '#e0f3f8', '#abd9e9', '#74add1', '#4575b4'
    ]
    cmap = LinearSegmentedColormap.from_list(
        'damage_capacity', colors_diverging, N=256
    )
    
    # Panel 1: Annual (EARP)
    ax1 = axes[0]
    gdf_variance_annual.plot(
        column='share_capacity',
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidth=0.1,
        edgecolor='0.5',
        legend=False,
        ax=ax1,
        missing_kwds={'color': '#dfdcdc', 'label': 'No data'}
    )
    ax1.set_title('Annual EARP', fontsize=12, y=1.02)
    ax1.axis('off')
    
    # Panel 2: Per-Event Median
    ax2 = axes[1]
    gdf_variance_median.plot(
        column='share_capacity',
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidth=0.1,
        edgecolor='0.5',
        legend=False,
        ax=ax2,
        missing_kwds={'color': '#dfdcdc', 'label': 'No data'}
    )
    ax2.set_title('Median Event', fontsize=12, y=1.02)
    ax2.axis('off')
    
    # Panel 3: Per-Event Maximum
    ax3 = axes[2]
    gdf_variance_maximum.plot(
        column='share_capacity',
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidth=0.1,
        edgecolor='0.5',
        legend=False,
        ax=ax3,
        missing_kwds={'color': '#dfdcdc', 'label': 'No data'}
    )
    ax3.set_title('Maximum Event', fontsize=12, y=1.02)
    ax3.axis('off')
    
    # Add shared colorbar using figure-level positioning
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.93, 0.2, 0.012, 0.6])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Capacity Variance Share\n(0=Damage, 1=Capacity)', 
                   fontsize=10)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    cbar.ax.tick_params(labelsize=10)
    
    plt.savefig(f"{output_dir}/variance_share_three_panel_maps.png", 
                dpi=300, bbox_inches="tight")
    plt.show()
    
    print("✓ 3-panel variance share maps created")


def create_variance_context_indices(coastal_counties, driver_analysis, per_event_analysis_median,
                                     capacity_df, ead_wide, earp_df,
                                     output_dir="../analysis_output"):
    """
    Create maps showing compound indices that combine variance drivers with actual conditions.
    
    Four indices reveal policy-relevant patterns:
    1. Capacity Bottleneck Index: capacity_share × (1/capacity) 
       → High where recovery is capacity-driven AND capacity is low
    2. Damage Exposure Index: damage_share × damage
       → High where recovery is damage-driven AND damage is high  
    3. Recovery Burden Index: EARP weighted by dominant driver
    4. Mismatch Index: Driver doesn't match actual condition
    
    Parameters
    ----------
    coastal_counties : gpd.GeoDataFrame
        County boundaries
    driver_analysis : pd.DataFrame
        Annual metrics with variance shares
    per_event_analysis_median : pd.DataFrame
        Per-event metrics (not used currently, for future expansion)
    capacity_df : pd.DataFrame
        Construction capacity data
    ead_wide : pd.DataFrame
        Expected Annual Damage data
    earp_df : pd.DataFrame
        Expected Annual Recovery Potential data
    output_dir : str
        Output directory
    """
    print("\nCreating variance context index maps...")
    
    # Merge all data
    analysis_df = driver_analysis[['fips', 'share_damage', 'share_capacity']].copy()
    analysis_df = analysis_df.merge(capacity_df[['fips', 'construction_capacity']], on='fips', how='left')
    analysis_df = analysis_df.merge(earp_df[['fips', 'earp_months_per_year']], on='fips', how='left')
    analysis_df = analysis_df.merge(ead_wide[['fips', 'total_ead']], on='fips', how='left')
    
    # Calculate compound indices
    
    # 1. Capacity Bottleneck Index: high when capacity-driven AND capacity is low
    capacity_inverse = 1 / (analysis_df['construction_capacity'] + 10)
    capacity_inverse_norm = (capacity_inverse - capacity_inverse.min()) / (capacity_inverse.max() - capacity_inverse.min())
    analysis_df['capacity_bottleneck_index'] = analysis_df['share_capacity'] * capacity_inverse_norm
    
    # 2. Damage Exposure Index: high when damage-driven AND damage is high
    ead_norm = (analysis_df['total_ead'] - analysis_df['total_ead'].min()) / (analysis_df['total_ead'].max() - analysis_df['total_ead'].min())
    analysis_df['damage_exposure_index'] = analysis_df['share_damage'] * ead_norm
    
    # 3. Recovery Burden Index: EARP weighted by whichever driver dominates
    earp_norm = (analysis_df['earp_months_per_year'] - analysis_df['earp_months_per_year'].min()) / \
                (analysis_df['earp_months_per_year'].max() - analysis_df['earp_months_per_year'].min())
    dominant_share = analysis_df[['share_damage', 'share_capacity']].max(axis=1)
    analysis_df['recovery_burden_index'] = earp_norm * dominant_share
    
    # 4. Mismatch Index: identifies puzzling cases
    capacity_norm = (analysis_df['construction_capacity'] - analysis_df['construction_capacity'].min()) / \
                    (analysis_df['construction_capacity'].max() - analysis_df['construction_capacity'].min())
    analysis_df['mismatch_capacity'] = analysis_df['share_capacity'] * capacity_norm
    analysis_df['mismatch_damage'] = analysis_df['share_damage'] * (1 - ead_norm)
    analysis_df['mismatch_index'] = analysis_df['mismatch_capacity'] + analysis_df['mismatch_damage']
    
    # Merge with geodataframe
    gdf = coastal_counties.merge(
        analysis_df[['fips', 'capacity_bottleneck_index', 'damage_exposure_index', 
                     'recovery_burden_index', 'mismatch_index']],
        left_on='GEOID',
        right_on='fips',
        how='left'
    )
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.92, wspace=0.1, hspace=0.15)
    
    # Panel 1: Capacity Bottleneck Index
    ax1 = axes[0, 0]
    gdf.plot(
        column='capacity_bottleneck_index',
        cmap='Blues',
        linewidth=0.1,
        edgecolor='0.5',
        legend=True,
        ax=ax1,
        legend_kwds={'label': 'Index Value', 'shrink': 0.6},
        missing_kwds={'color': '#f0f0f0'}
    )
    ax1.set_title('Capacity Bottleneck Index\n(Capacity-driven × Low Capacity)\nPriority for capacity building', 
                  fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Damage Exposure Index
    ax2 = axes[0, 1]
    gdf.plot(
        column='damage_exposure_index',
        cmap='Reds',
        linewidth=0.1,
        edgecolor='0.5',
        legend=True,
        ax=ax2,
        legend_kwds={'label': 'Index Value', 'shrink': 0.6},
        missing_kwds={'color': '#f0f0f0'}
    )
    ax2.set_title('Damage Exposure Index\n(Damage-driven × High Damage)\nPriority for damage mitigation', 
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Panel 3: Recovery Burden Index
    ax3 = axes[1, 0]
    gdf.plot(
        column='recovery_burden_index',
        cmap='YlOrRd',
        linewidth=0.1,
        edgecolor='0.5',
        legend=True,
        ax=ax3,
        legend_kwds={'label': 'Index Value', 'shrink': 0.6},
        missing_kwds={'color': '#f0f0f0'}
    )
    ax3.set_title('Recovery Burden Index\n(EARP × Dominant Driver)\nOverall priority', 
                  fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Panel 4: Mismatch Index
    ax4 = axes[1, 1]
    gdf.plot(
        column='mismatch_index',
        cmap='Purples',
        linewidth=0.1,
        edgecolor='0.5',
        legend=True,
        ax=ax4,
        legend_kwds={'label': 'Index Value', 'shrink': 0.6},
        missing_kwds={'color': '#f0f0f0'}
    )
    ax4.set_title('Mismatch Index\n(Driver doesn\'t match condition)\nCurious cases', 
                  fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    plt.savefig(f"{output_dir}/variance_context_indices.png", 
                dpi=300, bbox_inches="tight")
    plt.show()
    
    # Save indices to CSV
    output_csv = Path(output_dir) / "variance_context_indices.csv"
    analysis_df.to_csv(output_csv, index=False)
    
    # Print summary
    print("\n=== Capacity Bottleneck Index (Top 10) ===")
    top_bottleneck = analysis_df.nlargest(10, 'capacity_bottleneck_index')[
        ['fips', 'capacity_bottleneck_index', 'share_capacity', 'construction_capacity']
    ]
    print(top_bottleneck.to_string(index=False))
    
    print("\n=== Damage Exposure Index (Top 10) ===")
    top_damage = analysis_df.nlargest(10, 'damage_exposure_index')[
        ['fips', 'damage_exposure_index', 'share_damage', 'total_ead']
    ]
    print(top_damage.to_string(index=False))
    
    print(f"\n✓ Variance context indices saved to: {output_csv}")
    print("✓ Variance context index maps created")


def create_per_event_bottleneck_indices(coastal_counties, per_event_analysis_median,
                                         capacity_df, output_dir="../analysis_output"):
    """
    Create bottleneck analysis for per-event median metrics.
    
    Identifies priority counties for intervention based on typical event characteristics:
    1. Capacity Bottleneck: capacity-driven × low capacity
       → Build capacity to improve typical event recovery
    2. Damage Exposure: damage-driven × high median damage
       → Reduce damage exposure to improve typical events
    3. Recovery Burden: high median recovery × strong driver dominance
       → Overall priority counties for typical events
    
    Parameters
    ----------
    coastal_counties : gpd.GeoDataFrame
        County boundaries
    per_event_analysis_median : pd.DataFrame
        Per-event median metrics with variance shares
    capacity_df : pd.DataFrame
        Construction capacity data
    output_dir : str
        Output directory
    
    Returns
    -------
    pd.DataFrame
        Per-event bottleneck indices with county identifiers
    """
    print("\nCreating per-event bottleneck indices...")
    
    # Merge all data
    analysis_df = per_event_analysis_median[['fips', 'share_damage', 'share_capacity',
                                               'median_damage_units', 'median_recovery_months']].copy()
    analysis_df = analysis_df.merge(capacity_df[['fips', 'construction_capacity']], on='fips', how='left')
    
    # Calculate compound indices
    
    # 1. Capacity Bottleneck Index: high when capacity-driven AND capacity is low
    capacity_inverse = 1 / (analysis_df['construction_capacity'] + 10)
    capacity_inverse_norm = (capacity_inverse - capacity_inverse.min()) / (capacity_inverse.max() - capacity_inverse.min())
    analysis_df['capacity_bottleneck_index'] = analysis_df['share_capacity'] * capacity_inverse_norm
    
    # 2. Damage Exposure Index: high when damage-driven AND median damage is high
    damage_norm = (analysis_df['median_damage_units'] - analysis_df['median_damage_units'].min()) / \
                  (analysis_df['median_damage_units'].max() - analysis_df['median_damage_units'].min())
    analysis_df['damage_exposure_index'] = analysis_df['share_damage'] * damage_norm
    
    # 3. Recovery Burden Index: median recovery weighted by whichever driver dominates
    recovery_norm = (analysis_df['median_recovery_months'] - analysis_df['median_recovery_months'].min()) / \
                    (analysis_df['median_recovery_months'].max() - analysis_df['median_recovery_months'].min())
    dominant_share = analysis_df[['share_damage', 'share_capacity']].max(axis=1)
    analysis_df['recovery_burden_index'] = recovery_norm * dominant_share
    
    # Merge with geodataframe
    gdf = coastal_counties.merge(
        analysis_df[['fips', 'capacity_bottleneck_index', 'damage_exposure_index', 
                     'recovery_burden_index']],
        left_on='GEOID',
        right_on='fips',
        how='left'
    )
    
    # Create 1x3 figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95, wspace=0.1)
    
    # Panel 1: Capacity Bottleneck Index
    ax1 = axes[0]
    gdf.plot(
        column='capacity_bottleneck_index',
        cmap='Blues',
        linewidth=0.1,
        edgecolor='0.5',
        legend=True,
        ax=ax1,
        legend_kwds={'label': 'Index Value', 'shrink': 0.7},
        missing_kwds={'color': '#f0f0f0'}
    )
    ax1.set_title('Capacity Bottleneck\n(Median Event)\nCapacity-driven × Low Capacity', 
                  fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Damage Exposure Index
    ax2 = axes[1]
    gdf.plot(
        column='damage_exposure_index',
        cmap='Reds',
        linewidth=0.1,
        edgecolor='0.5',
        legend=True,
        ax=ax2,
        legend_kwds={'label': 'Index Value', 'shrink': 0.7},
        missing_kwds={'color': '#f0f0f0'}
    )
    ax2.set_title('Damage Exposure\n(Median Event)\nDamage-driven × High Damage', 
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Panel 3: Recovery Burden Index
    ax3 = axes[2]
    gdf.plot(
        column='recovery_burden_index',
        cmap='YlOrRd',
        linewidth=0.1,
        edgecolor='0.5',
        legend=True,
        ax=ax3,
        legend_kwds={'label': 'Index Value', 'shrink': 0.7},
        missing_kwds={'color': '#f0f0f0'}
    )
    ax3.set_title('Recovery Burden\n(Median Event)\nHigh Recovery × Strong Driver', 
                  fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    plt.savefig(f"{output_dir}/per_event_bottleneck_indices.png", 
                dpi=300, bbox_inches="tight")
    plt.show()
    
    # Save indices to CSV
    output_csv = Path(output_dir) / "per_event_bottleneck_indices.csv"
    analysis_df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Per-event bottleneck indices saved to: {output_csv}")
    print("✓ Per-event bottleneck index maps created")
    
    return analysis_df


def create_intervention_priority_maps(driver_analysis, per_event_analysis_median,
                                     coastal_counties, output_dir="../analysis_output"):
    """
    Create 1x2 map visualization showing hybrid intervention priorities.
    
    Hybrid classification combines:
    - Variance decomposition (which factors explain patterns)
    - Absolute conditions (capacity/damage thresholds)
    
    Parameters
    ----------
    driver_analysis : pd.DataFrame
        Annual metrics with intervention priority
    per_event_analysis_median : pd.DataFrame
        Per-event metrics with intervention priority
    coastal_counties : gpd.GeoDataFrame
        County boundaries
    output_dir : str
        Output directory
    """
    print("\n" + "="*80)
    print("CREATING INTERVENTION PRIORITY MAPS")
    print("="*80)
    
    # Merge annual data with geometries
    gdf_annual = coastal_counties.merge(
        driver_analysis[['fips', 'intervention_priority', 
                        'share_capacity', 'construction_capacity']],
        left_on='GEOID',
        right_on='fips',
        how='left'
    )
    
    # Merge per-event data with geometries
    gdf_event = coastal_counties.merge(
        per_event_analysis_median[['fips', 'intervention_priority',
                                   'share_capacity', 'construction_capacity']],
        left_on='GEOID',
        right_on='fips',
        how='left'
    )
    
    # Create figure with 1x2 layout
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Define categorical colormap
    priority_colors = {
        'Critical Capacity Bottleneck': '#d7191c',  # Dark red
        'Capacity Building Priority': '#fdae61',    # Orange
        'Damage Mitigation Priority': '#abd9e9',    # Light blue
        'Mixed Strategy': '#2c7bb6'                  # Dark blue
    }
    
    # ========================================================================
    # Panel 1: Annual Hybrid Classification
    # ========================================================================
    ax1 = axes[0]
    
    gdf_annual['priority_numeric'] = gdf_annual['intervention_priority'].map({
        'Critical Capacity Bottleneck': 1,
        'Capacity Building Priority': 2,
        'Damage Mitigation Priority': 3,
        'Mixed Strategy': 4
    })
    
    gdf_annual.plot(
        column='priority_numeric',
        categorical=True,
        cmap='RdYlBu',
        linewidth=0.1,
        edgecolor='0.5',
        legend=False,
        ax=ax1,
        missing_kwds={'color': '#f0f0f0'}
    )
    
    ax1.set_title('Annual Perspective', 
                  fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # ========================================================================
    # Panel 2: Per-Event Hybrid Classification
    # ========================================================================
    ax2 = axes[1]
    
    gdf_event['priority_numeric'] = gdf_event['intervention_priority'].map({
        'Critical Capacity Bottleneck': 1,
        'Capacity Building Priority': 2,
        'Damage Mitigation Priority': 3,
        'Mixed Strategy': 4
    })
    
    gdf_event.plot(
        column='priority_numeric',
        categorical=True,
        cmap='RdYlBu',
        linewidth=0.1,
        edgecolor='0.5',
        legend=False,
        ax=ax2,
        missing_kwds={'color': '#f0f0f0'}
    )
    
    ax2.set_title('Per-Event Median Perspective', 
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Shared legend centered below both panels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d7191c', label='Critical Capacity Bottleneck'),
        Patch(facecolor='#fdae61', label='Capacity Building Priority'),
        Patch(facecolor='#abd9e9', label='Damage Mitigation Priority'),
        Patch(facecolor='#2c7bb6', label='Mixed Strategy')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/intervention_priority_maps.png", 
                dpi=300, bbox_inches="tight")
    plt.show()
    
    print(f"✓ Saved: intervention_priority_maps.png")
    
    # Print category counts
    print("\n" + "="*80)
    print("INTERVENTION PRIORITY DISTRIBUTION")
    print("="*80)
    
    print("\n=== ANNUAL PERSPECTIVE ===")
    priority_counts_annual = gdf_annual['intervention_priority'].value_counts()
    for priority, count in priority_counts_annual.items():
        pct = 100 * count / len(gdf_annual.dropna(subset=['intervention_priority']))
        print(f"  {priority}: {count} counties ({pct:.1f}%)")
    
    print("\n=== PER-EVENT PERSPECTIVE ===")
    priority_counts_event = gdf_event['intervention_priority'].value_counts()
    for priority, count in priority_counts_event.items():
        pct = 100 * count / len(gdf_event.dropna(subset=['intervention_priority']))
        print(f"  {priority}: {count} counties ({pct:.1f}%)")


# ============================================================================
# SECTION 7B: PER-COUNTY SENSITIVITY METRICS
# ============================================================================

def compute_county_sensitivity_metrics(recovery_all_events, units_df, capacity_df, 
                                      output_dir="../analysis_output"):
    """
    Compute sensitivity metrics for each county based on their full event portfolio.
    
    For each county, computes:
    1. Capacity Saturation Factor: Ratio of actual recovery to theoretical minimum (damage/capacity)
    2. Damage State Profile: Which damage state (DS1-DS4) dominates recovery burden
    3. Damage Quantiles: Where county ranks in damage distribution (tests if high saturation = extreme damage)
    
    This approach extracts patterns directly from ~27,000 event-county pairs without
    overlaying statistical models. Each county's response to multiple events reveals
    whether they respond linearly, have thresholds, or are chronically constrained.
    
    Parameters
    ----------
    recovery_all_events : DataFrame
        Event-county recovery times with columns: event, fips, recovery_potential [months]
    units_df : DataFrame
        Event-county damage by DS with columns: event_name, fips, units_DS1-4_scaled
    capacity_df : DataFrame
        County capacity with columns: fips, construction_capacity
    output_dir : str
        Directory to save CSV output
        
    Returns
    -------
    sensitivity_df : DataFrame
        Metrics per county with columns:
        - fips: county identifier
        - num_events: number of events county experienced
        - recovery_elasticity: correlation(damage, recovery) across events
        - saturation_factor_mean: mean(actual_recovery / theoretical_minimum)
        - saturation_factor_std: std deviation of saturation factor
        - dominant_ds: which DS (1-4) contributes most to total damage
        - total_damage_mean: mean total damage across events
        - recovery_mean: mean recovery time across events
    """
    print("\n" + "="*80)
    print("COMPUTING COUNTY SENSITIVITY METRICS")
    print("="*80)
    
    # Merge recovery and damage data
    # Match on event and fips
    recovery_events = recovery_all_events.copy()
    units_df_copy = units_df.copy()
    
    # Ensure event column names match
    if 'event' in recovery_events.columns and 'event_name' in units_df_copy.columns:
        recovery_events = recovery_events.rename(columns={'event': 'event_name'})
    elif 'event_name' in recovery_events.columns and 'event' in units_df_copy.columns:
        units_df_copy = units_df_copy.rename(columns={'event': 'event_name'})
    
    # Ensure consistent data types for merge keys
    # Convert event_name to string in both DataFrames
    if 'event_name' in recovery_events.columns:
        recovery_events['event_name'] = recovery_events['event_name'].astype(str)
    if 'event_name' in units_df_copy.columns:
        units_df_copy['event_name'] = units_df_copy['event_name'].astype(str)
    
    # Ensure fips is string in both DataFrames
    recovery_events['fips'] = recovery_events['fips'].astype(str)
    units_df_copy['fips'] = units_df_copy['fips'].astype(str)
    
    # Compute total damage per event-county
    ds_cols = [c for c in units_df_copy.columns if 'units_DS' in c and 'scaled' in c]
    units_df_copy['total_damage'] = units_df_copy[ds_cols].sum(axis=1)
    
    # DS-specific repair times (months per unit) for weighted damage
    ds_repair_times = {
        'units_DS1_scaled': 1.0,
        'units_DS2_scaled': 1.0,
        'units_DS3_scaled': 3.0,
        'units_DS4_scaled': 6.0
    }
    
    # Compute weighted damage: sum(DS_i_units * repair_time_per_unit)
    units_df_copy['weighted_damage'] = sum(
        units_df_copy[ds_col] * ds_repair_times[ds_col] 
        for ds_col in ds_cols if ds_col in ds_repair_times
    )
    
    # Merge recovery and damage
    merged = recovery_events.merge(
        units_df_copy[['event_name', 'fips', 'total_damage', 'weighted_damage'] + ds_cols],
        on=['event_name', 'fips'],
        how='inner'
    )
    
    # Merge with capacity (ensure fips is string)
    capacity_df_copy = capacity_df.copy()
    capacity_df_copy['fips'] = capacity_df_copy['fips'].astype(str)
    merged = merged.merge(
        capacity_df_copy[['fips', 'construction_capacity']],
        on='fips',
        how='left'
    )
    
    print(f"\nMerged data: {len(merged):,} event-county pairs")
    print(f"Counties with data: {merged['fips'].nunique()}")
    print(f"Counties with capacity data: {merged['construction_capacity'].notna().sum():,}")
    
    # Initialize results list
    results = []
    
    # Minimum events required for reliable correlation
    MIN_EVENTS_FOR_ELASTICITY = 5  # Need sufficient data for meaningful correlation
    
    # Process each county
    counties = merged['fips'].unique()
    print(f"\nProcessing {len(counties)} counties...")
    print(f"Minimum events required for elasticity calculation: {MIN_EVENTS_FOR_ELASTICITY}")
    
    for fips in counties:
        county_data = merged[merged['fips'] == fips].copy()
        
        # Count events
        num_events = len(county_data)
        
        # Extract capacity (should be constant per county)
        capacity = county_data['construction_capacity'].iloc[0]
        
        # 1. CAPACITY SATURATION FACTOR
        # actual_recovery / theoretical_minimum where theoretical_minimum accounts for 
        # DS-specific recovery times: DS1=1mo, DS2=1mo, DS3=3mo, DS4=6mo
        # Note: weighted_damage already computed above when merging data
        if pd.notna(capacity) and capacity > 0:
            # Theoretical minimum = weighted_damage / capacity
            county_data['theoretical_min'] = county_data['weighted_damage'] / capacity
            county_data['saturation_factor'] = (
                county_data['recovery_potential [months]'] / county_data['theoretical_min']
            )
            # Handle edge cases
            county_data['saturation_factor'] = county_data['saturation_factor'].replace(
                [np.inf, -np.inf], np.nan
            )
            saturation_mean = county_data['saturation_factor'].mean()
            saturation_std = county_data['saturation_factor'].std()
        else:
            saturation_mean = np.nan
            saturation_std = np.nan
        
        # 2. DOMINANT DAMAGE STATE
        # Which DS contributes most to total damage across all events
        ds_totals = county_data[ds_cols].sum()
        if ds_totals.sum() > 0:
            dominant_ds_col = ds_totals.idxmax()
            # Extract DS number from column name (e.g., 'units_DS3_scaled' -> 3)
            dominant_ds = int(dominant_ds_col.split('_DS')[1].split('_')[0])
        else:
            dominant_ds = np.nan
        
        # 3. DAMAGE STATISTICS
        # Store mean and max damage for quantile calculation later
        total_damage_mean = county_data['weighted_damage'].mean()
        total_damage_max = county_data['weighted_damage'].max()
        recovery_mean = county_data['recovery_potential [months]'].mean()
        
        results.append({
            'fips': fips,
            'num_events': num_events,
            'saturation_factor_mean': saturation_mean,
            'saturation_factor_std': saturation_std,
            'dominant_ds': dominant_ds,
            'weighted_damage_mean': total_damage_mean,
            'weighted_damage_max': total_damage_max,
            'recovery_mean': recovery_mean,
            'construction_capacity': capacity
        })
    
    sensitivity_df = pd.DataFrame(results)
    
    # Compute damage quantiles for each county
    # This identifies whether high-saturation counties face extreme damage or are chronically constrained
    sensitivity_df['damage_quantile_mean'] = sensitivity_df['weighted_damage_mean'].rank(pct=True)
    sensitivity_df['damage_quantile_max'] = sensitivity_df['weighted_damage_max'].rank(pct=True)
    
    # Compute capacity quantiles
    # This identifies whether high-saturation counties have low capacity or are simply overwhelmed
    sensitivity_df['capacity_quantile'] = sensitivity_df['construction_capacity'].rank(pct=True)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SENSITIVITY METRICS SUMMARY")
    print("="*80)
    print(f"\nTotal counties analyzed: {len(sensitivity_df)}")
    print(f"Counties with saturation data: {sensitivity_df['saturation_factor_mean'].notna().sum()}")
    
    # Event count distribution
    print(f"\nEvent count per county:")
    print(f"  Mean: {sensitivity_df['num_events'].mean():.1f}")
    print(f"  Median: {sensitivity_df['num_events'].median():.0f}")
    print(f"  Min: {sensitivity_df['num_events'].min():.0f}")
    print(f"  Max: {sensitivity_df['num_events'].max():.0f}")
    
    print("\nCapacity Saturation Factor Distribution:")
    print(sensitivity_df['saturation_factor_mean'].describe())
    
    print("\nDamage Quantile Distribution (Mean):")
    print(sensitivity_df['damage_quantile_mean'].describe())
    
    print("\nDominant Damage State Counts:")
    print(sensitivity_df['dominant_ds'].value_counts().sort_index())
    
    # Test hypothesis: Are high-saturation counties damage-bottlenecked?
    high_sat = sensitivity_df[sensitivity_df['saturation_factor_mean'] > 5]
    if len(high_sat) > 0:
        print(f"\nHigh-Saturation Counties (>5x):")
        print(f"  Count: {len(high_sat)}")
        print(f"  Mean damage quantile: {high_sat['damage_quantile_mean'].mean():.2f}")
        print(f"  Median damage quantile: {high_sat['damage_quantile_mean'].median():.2f}")
        print(f"  Counties in top 25% damage: {(high_sat['damage_quantile_mean'] > 0.75).sum()}")
        print(f"  Counties in bottom 50% damage: {(high_sat['damage_quantile_mean'] <= 0.5).sum()}")
    
    # Save to CSV
    output_path = Path(output_dir) / "county_sensitivity_metrics.csv"
    sensitivity_df.to_csv(output_path, index=False)
    print(f"\nSaved sensitivity metrics to: {output_path}")
    
    return sensitivity_df


def create_sensitivity_metrics_maps(sensitivity_df, coastal_counties, 
                                   output_dir="../analysis_output"):
    """
    Create 3-panel spatial visualization of county sensitivity metrics.
    
    Panel 1: Damage State Profile
        - Categorical map showing which DS (1-4) drives recovery burden
        - DS4 (Complete): Most severe damage dominates
        - DS3 (Extensive): Major damage drives recovery
        - DS1-2: Minor damage drives recovery (unusual, suggests resilient county)
    
    Panel 2: Capacity Saturation Factor
        - Values ~1.0: Pure permit constraint, recovery = damage / capacity
        - Values 2-5: Moderate saturation, other factors contribute
        - Values >5: Severe saturation, non-permit bottlenecks dominate
        
    Panel 3: Damage Quantile (Max)
        - Shows where county ranks in maximum damage distribution (0=lowest, 1=highest)
        - High quantile = Counties that experienced extreme damage events
        - Low quantile = Counties with relatively moderate maximum damage
    
    Parameters
    ----------
    sensitivity_df : DataFrame
        County metrics from compute_county_sensitivity_metrics()
    coastal_counties : GeoDataFrame
        County geometries for mapping
    output_dir : str
        Directory to save figure
    """
    print("\n" + "="*80)
    print("CREATING SENSITIVITY METRICS MAPS")
    print("="*80)
    
    # Merge sensitivity metrics with geometries
    gdf = coastal_counties.merge(
        sensitivity_df,
        left_on='GEOID',
        right_on='fips',
        how='left'
    )
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={'wspace': 0.3})
    
    # ========== PANEL 1: DAMAGE STATE PROFILE ==========
    ax = axes[0]
    
    gdf_with_data_ds = gdf[gdf['dominant_ds'].notna()]
    
    if len(gdf_with_data_ds) > 0:
        # Create categorical colormap
        ds_colors = {
            1: '#d4e6f1',  # Light blue - minor damage
            2: '#f9e79f',  # Light yellow - moderate damage
            3: '#f8b878',  # Orange - extensive damage
            4: '#e74c3c'   # Red - complete damage
        }
        
        # Plot each damage state separately
        for ds in sorted(gdf_with_data_ds['dominant_ds'].unique()):
            gdf_ds = gdf_with_data_ds[gdf_with_data_ds['dominant_ds'] == ds]
            gdf_ds.plot(
                ax=ax,
                color=ds_colors.get(ds, 'gray'),
                edgecolor='0.5',
                linewidth=0.1
            )
        
        # Create manual legend with proper colors
        legend_elements = [Patch(facecolor=ds_colors[ds], edgecolor='0.5', label=f'DS{int(ds)}') 
                          for ds in sorted(gdf_with_data_ds['dominant_ds'].unique())]
        ax.legend(
            handles=legend_elements,
            title='Dominant\nDamage State',
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fancybox=False,
            shadow=False,
            title_fontsize=9,
            fontsize=9
        )
    
    gdf[gdf['dominant_ds'].isna()].plot(
        ax=ax,
        color='white',
        edgecolor='0.5',
        linewidth=0.1
    )
    
    ax.set_title('Damage State Profile', fontsize=12, pad=10)
    ax.axis('off')
    
    # ========== PANEL 2: CAPACITY SATURATION FACTOR ==========
    ax = axes[1]
    
    gdf_with_data = gdf[gdf['saturation_factor_mean'].notna()]
    
    if len(gdf_with_data) > 0:
        # Cap at reasonable upper limit for visualization
        gdf_plot = gdf_with_data.copy()
        gdf_plot['saturation_capped'] = gdf_plot['saturation_factor_mean'].clip(upper=10)
        
        vmin = 1.0  # Theoretical minimum
        vmax = gdf_plot['saturation_capped'].quantile(0.95)
        
        cax2 = ax.figure.add_axes([0.62, 0.30, 0.01, 0.35])
        gdf_plot.plot(
            column='saturation_capped',
            ax=ax,
            cmap='plasma',
            vmin=vmin,
            vmax=vmax,
            edgecolor='0.5',
            linewidth=0.1,
            legend=True,
            legend_kwds={
                'label': 'Saturation Factor',
                'shrink': 0.8
            },
            cax=cax2
        )
        # Make colorbar outline black
        for spine in cax2.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
    
    gdf[gdf['saturation_factor_mean'].isna()].plot(
        ax=ax,
        color='white',
        edgecolor='0.5',
        linewidth=0.1
    )
    
    ax.set_title('Capacity Saturation Factor', fontsize=12, pad=10)
    ax.axis('off')
    
    # ========== PANEL 3: DAMAGE QUANTILE (MAX) ==========
    ax = axes[2]
    
    gdf_with_data_dq = gdf[gdf['damage_quantile_max'].notna()]
    
    if len(gdf_with_data_dq) > 0:
        gdf_with_data_dq.plot(
            column='damage_quantile_max',
            ax=ax,
            cmap='YlOrRd',
            edgecolor='black',
            linewidth=0.3,
            legend=False,
            vmin=0,
            vmax=1
        )
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(
            cmap='YlOrRd',
            norm=plt.Normalize(vmin=0, vmax=1)
        )
        sm._A = []
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('Damage Quantile (Max)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(1)
    
    ax.set_title('Maximum Damage Quantile', fontsize=12, pad=10)
    ax.axis('off')
    
    # Adjust layout
    #plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / "county_sensitivity_metrics_maps.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved sensitivity metrics maps to: {output_path}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SPATIAL DISTRIBUTION SUMMARY")
    print("="*80)
    
    print(f"\nCounties mapped: {len(gdf_with_data_ds)}")
    print(f"Counties with DS profile: {gdf['dominant_ds'].notna().sum()}")
    print(f"Counties with saturation: {gdf['saturation_factor_mean'].notna().sum()}")
    print(f"Counties with damage quantile (max): {gdf['damage_quantile_max'].notna().sum()}")


def create_saturation_damage_scatterplot(sensitivity_df, output_dir="../analysis_output"):
    """
    Create scatterplot with marginal distributions showing relationship between
    capacity saturation and damage quantile.
    
    Tests hypothesis: Are high-saturation counties facing extreme damage (overwhelmed)
    or moderate damage (chronically constrained)?
    
    Parameters
    ----------
    sensitivity_df : DataFrame
        County metrics from compute_county_sensitivity_metrics()
    output_dir : str
        Directory to save figure
    """
    print("\n" + "="*80)
    print("CREATING SATURATION-DAMAGE SCATTERPLOT WITH MARGINALS")
    print("="*80)
    
    # Prepare data
    plot_df = sensitivity_df[
        sensitivity_df['saturation_factor_mean'].notna() & 
        sensitivity_df['damage_quantile_mean'].notna()
    ].copy()
    
    if len(plot_df) == 0:
        print("No data available for scatterplot")
        return
    
    # Define colors for damage states
    ds_colors = {
        1: '#d4e6f1',  # Light blue
        2: '#f9e79f',  # Light yellow
        3: '#f8b878',  # Orange
        4: '#e74c3c'   # Red
    }
    
    # Cap saturation for visualization
    plot_df['saturation_capped'] = plot_df['saturation_factor_mean'].clip(upper=20)
    
    # Create figure with gridspec for marginal plots
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.05, wspace=0.05, 
                          height_ratios=[1, 4, 0.2], width_ratios=[4, 1, 0.2])
    
    # Main scatterplot
    ax_main = fig.add_subplot(gs[1, 0])
    
    # Plot all counties by DS
    for ds in sorted(plot_df['dominant_ds'].dropna().unique()):
        ds_data = plot_df[plot_df['dominant_ds'] == ds]
        ax_main.scatter(
            ds_data['damage_quantile_mean'],
            ds_data['saturation_capped'],
            c=ds_colors.get(ds, 'gray'),
            s=40,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            label=f'DS{int(ds)}'
        )
    
    # Highlight high-saturation counties (>5)
    high_sat = plot_df[plot_df['saturation_factor_mean'] > 5]
    if len(high_sat) > 0:
        ax_main.scatter(
            high_sat['damage_quantile_mean'],
            high_sat['saturation_capped'],
            s=100,
            facecolors='none',
            edgecolors='red',
            linewidth=2.5,
            label='High Saturation (>5x)',
            zorder=10
        )
    
    # Add reference lines
    ax_main.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, 
                    label='Theoretical Min', zorder=0)
    ax_main.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    ax_main.axhline(y=5.0, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    
    ax_main.set_xlabel('Damage Quantile (Mean)', fontsize=11)
    ax_main.set_ylabel('Capacity Saturation Factor', fontsize=11)
    ax_main.set_xlim(-0.05, 1.05)
    ax_main.set_ylim(0.7, min(20, plot_df['saturation_capped'].max() * 1.1))
    ax_main.set_yscale('log')
    ax_main.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax_main.legend(loc='upper left', fontsize=9, frameon=True, fancybox=False, 
                   framealpha=0.9, edgecolor='black')
    
    # Add quadrant labels
    ax_main.text(0.25, 12, 'Chronically\nConstrained', fontsize=9, alpha=0.4, 
                ha='center', va='center', style='italic')
    ax_main.text(0.75, 12, 'Overwhelmed\nby Magnitude', fontsize=9, alpha=0.4, 
                ha='center', va='center', style='italic')
    
    # Top marginal: histogram of damage quantiles
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_top.hist(plot_df['damage_quantile_mean'], bins=30, color='steelblue', 
                alpha=0.6, edgecolor='black', linewidth=0.5)
    ax_top.set_ylabel('Count', fontsize=9)
    ax_top.tick_params(labelbottom=False, labelsize=8)
    ax_top.grid(True, alpha=0.2, axis='y')
    
    # Right marginal: histogram of saturation (log scale)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_right.hist(np.log10(plot_df['saturation_capped']), bins=30, 
                  orientation='horizontal', color='coral', alpha=0.6, 
                  edgecolor='black', linewidth=0.5)
    ax_right.set_xlabel('Count', fontsize=9)
    ax_right.tick_params(labelleft=False, labelsize=8)
    ax_right.grid(True, alpha=0.2, axis='x')
    
    # Save figure
    output_path = Path(output_dir) / "saturation_damage_scatterplot_marginals.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved scatterplot with marginals to: {output_path}")
    plt.close()


def create_saturation_damage_violinplot(sensitivity_df, output_dir="../analysis_output"):
    """
    Create violin plot showing distribution of saturation factors across damage quantile bins.
    
    Shows how saturation distribution changes with damage severity and reveals
    bimodal patterns (chronically constrained vs overwhelmed).
    
    Parameters
    ----------
    sensitivity_df : DataFrame
        County metrics from compute_county_sensitivity_metrics()
    output_dir : str
        Directory to save figure
    """
    print("\n" + "="*80)
    print("CREATING SATURATION DISTRIBUTION BY DAMAGE BINS")
    print("="*80)
    
    # Prepare data
    plot_df = sensitivity_df[
        sensitivity_df['saturation_factor_mean'].notna() & 
        sensitivity_df['damage_quantile_mean'].notna()
    ].copy()
    
    if len(plot_df) == 0:
        print("No data available for violin plot")
        return
    
    # Create damage bins
    plot_df['damage_bin'] = pd.cut(
        plot_df['damage_quantile_mean'],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['0-25%', '25-50%', '50-75%', '75-100%'],
        include_lowest=True
    )
    
    # Cap saturation for visualization
    plot_df['saturation_capped'] = plot_df['saturation_factor_mean'].clip(upper=20)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Prepare data for violin plot
    data_by_bin = [plot_df[plot_df['damage_bin'] == bin_label]['saturation_capped'].dropna().values
                   for bin_label in ['0-25%', '25-50%', '50-75%', '75-100%']]
    
    # Create violin plot
    parts = ax.violinplot(
        data_by_bin,
        positions=[1, 2, 3, 4],
        widths=0.7,
        showmeans=True,
        showmedians=True,
        showextrema=True
    )
    
    # Customize violin colors
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(['#d4e6f1', '#f9e79f', '#f8b878', '#e74c3c'][i])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    # Customize other elements
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(1.5)
    
    # Add reference line at saturation = 1
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, 
               label='Theoretical Minimum', zorder=0)
    ax.axhline(y=5.0, color='red', linestyle=':', linewidth=1.5, alpha=0.5, 
               label='High Saturation Threshold', zorder=0)
    
    ax.set_xlabel('Damage Quantile Bin', fontsize=12)
    ax.set_ylabel('Capacity Saturation Factor', fontsize=12)
    ax.set_title('Distribution of Saturation Across Damage Severity', fontsize=13, pad=15)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['Low\n(0-25%)', 'Moderate\n(25-50%)', 'High\n(50-75%)', 'Extreme\n(75-100%)'])
    ax.set_yscale('log')
    ax.set_ylim(0.7, 20)
    ax.grid(True, alpha=0.2, axis='y')
    ax.legend(loc='upper left', fontsize=10, frameon=True, fancybox=False, 
              framealpha=0.9, edgecolor='black')
    
    # Add sample size annotations
    for i, bin_label in enumerate(['0-25%', '25-50%', '50-75%', '75-100%']):
        n = len(plot_df[plot_df['damage_bin'] == bin_label])
        ax.text(i+1, 0.75, f'n={n}', ha='center', fontsize=8, style='italic')
    
    # Save figure
    output_path = Path(output_dir) / "saturation_damage_violinplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved violin plot to: {output_path}")
    plt.close()


def create_saturation_comparison_scatterplots(sensitivity_df, output_dir="../analysis_output"):
    """
    Create side-by-side scatterplots showing saturation against damage and capacity quantiles.
    
    Panel 1: Saturation vs Damage Quantile - Tests if high saturation is due to extreme damage
    Panel 2: Saturation vs Capacity Quantile - Tests if high saturation is due to low capacity
    
    Parameters
    ----------
    sensitivity_df : DataFrame
        County metrics from compute_county_sensitivity_metrics()
    output_dir : str
        Directory to save figure
    """
    print("\n" + "="*80)
    print("CREATING SATURATION COMPARISON SCATTERPLOTS")
    print("="*80)
    
    # Prepare data
    plot_df = sensitivity_df[
        sensitivity_df['saturation_factor_mean'].notna() & 
        sensitivity_df['damage_quantile_mean'].notna() &
        sensitivity_df['capacity_quantile'].notna()
    ].copy()
    
    if len(plot_df) == 0:
        print("No data available for comparison scatterplots")
        return
    
    # Define colors for damage states
    ds_colors = {
        1: '#d4e6f1',  # Light blue
        2: '#f9e79f',  # Light yellow
        3: '#f8b878',  # Orange
        4: '#e74c3c'   # Red
    }
    
    # Cap saturation for visualization
    plot_df['saturation_capped'] = plot_df['saturation_factor_mean'].clip(upper=20)
    
    # Create figure with 2 panels side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========== PANEL 1: SATURATION VS DAMAGE QUANTILE ==========
    ax = axes[0]
    
    # Plot all counties by DS
    for ds in sorted(plot_df['dominant_ds'].dropna().unique()):
        ds_data = plot_df[plot_df['dominant_ds'] == ds]
        ax.scatter(
            ds_data['damage_quantile_mean'],
            ds_data['saturation_capped'],
            c=ds_colors.get(ds, 'gray'),
            s=40,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            label=f'DS{int(ds)}'
        )
    
    # Highlight high-saturation counties (>5)
    high_sat = plot_df[plot_df['saturation_factor_mean'] > 5]
    if len(high_sat) > 0:
        ax.scatter(
            high_sat['damage_quantile_mean'],
            high_sat['saturation_capped'],
            s=100,
            facecolors='none',
            edgecolors='red',
            linewidth=2.5,
            label='High Saturation (>5x)',
            zorder=10
        )
    
    # Add reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, 
               label='Theoretical Min', zorder=0)
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    ax.axhline(y=5.0, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    
    ax.set_xlabel('Damage Quantile (Mean)', fontsize=12)
    ax.set_ylabel('Capacity Saturation Factor', fontsize=12)
    ax.set_title('(A) Saturation vs. Damage Severity', fontsize=13, pad=15, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.7, min(20, plot_df['saturation_capped'].max() * 1.1))
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=9, frameon=True, fancybox=False, 
              framealpha=0.9, edgecolor='black')
    
    # Add quadrant labels
    ax.text(0.25, 12, 'Chronically\nConstrained', fontsize=9, alpha=0.4, 
           ha='center', va='center', style='italic')
    ax.text(0.75, 12, 'Overwhelmed\nby Damage', fontsize=9, alpha=0.4, 
           ha='center', va='center', style='italic')
    
    # ========== PANEL 2: SATURATION VS CAPACITY QUANTILE ==========
    ax = axes[1]
    
    # Plot all counties by DS
    for ds in sorted(plot_df['dominant_ds'].dropna().unique()):
        ds_data = plot_df[plot_df['dominant_ds'] == ds]
        ax.scatter(
            ds_data['capacity_quantile'],
            ds_data['saturation_capped'],
            c=ds_colors.get(ds, 'gray'),
            s=40,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            label=f'DS{int(ds)}'
        )
    
    # Highlight high-saturation counties (>5)
    if len(high_sat) > 0:
        ax.scatter(
            high_sat['capacity_quantile'],
            high_sat['saturation_capped'],
            s=100,
            facecolors='none',
            edgecolors='red',
            linewidth=2.5,
            label='High Saturation (>5x)',
            zorder=10
        )
    
    # Add reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, 
               label='Theoretical Min', zorder=0)
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    ax.axhline(y=5.0, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    
    ax.set_xlabel('Capacity Quantile (Permit Capacity)', fontsize=12)
    ax.set_ylabel('Capacity Saturation Factor', fontsize=12)
    ax.set_title('(B) Saturation vs. Capacity Level', fontsize=13, pad=15, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.7, min(20, plot_df['saturation_capped'].max() * 1.1))
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=False, 
              framealpha=0.9, edgecolor='black')
    
    # Add quadrant labels
    ax.text(0.25, 12, 'Low Capacity\nSaturated', fontsize=9, alpha=0.4, 
           ha='center', va='center', style='italic')
    ax.text(0.75, 12, 'High Capacity\nSaturated', fontsize=9, alpha=0.4, 
           ha='center', va='center', style='italic')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / "saturation_comparison_scatterplots.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved comparison scatterplots to: {output_path}")
    plt.close()


def create_elasticity_comparison_maps(sensitivity_df, coastal_counties, output_dir="../analysis_output"):
    """
    Create 3-panel comparison of elasticity metrics to visualize impact of DS-weighting.
    
    Compares unweighted vs. weighted elasticity to show how much damage state composition
    explains apparent non-linearity in recovery responses.
    
    Panel 1: Unweighted Elasticity
        - correlation(total_units, recovery_time)
        - Counterfactual: treats all DS equally
    
    Panel 2: Weighted Elasticity  
        - correlation(DS-normalized_damage, recovery_time)
        - Aligned with how recovery was computed
    
    Panel 3: Elasticity Improvement
        - Difference: weighted - unweighted
        - Shows where DS composition was masking linearity
    
    Parameters
    ----------
    sensitivity_df : DataFrame
        County metrics from compute_county_sensitivity_metrics()
    coastal_counties : GeoDataFrame
        County geometries for mapping
    output_dir : str
        Directory to save figure
    """
    print("\n" + "="*80)
    print("CREATING ELASTICITY COMPARISON MAPS")
    print("="*80)
    
    # Merge sensitivity metrics with geometries
    gdf = coastal_counties.merge(
        sensitivity_df,
        left_on='GEOID',
        right_on='fips',
        how='left'
    )
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={'wspace': 0.3})
    
    # ========== PANEL 1: UNWEIGHTED ELASTICITY ==========
    ax = axes[0]
    
    gdf_with_data = gdf[gdf['recovery_elasticity_unweighted'].notna()]
    
    if len(gdf_with_data) > 0:
        vmin, vmax = 0.0, 1.0
        
        cax1 = ax.figure.add_axes([0.29, 0.30, 0.01, 0.35])
        gdf_with_data.plot(
            column='recovery_elasticity_unweighted',
            ax=ax,
            cmap='RdYlGn_r',
            vmin=vmin,
            vmax=vmax,
            edgecolor='0.5',
            linewidth=0.1,
            legend=True,
            legend_kwds={'label': 'Elasticity (Unweighted)', 'shrink': 0.8},
            cax=cax1
        )
        for spine in cax1.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
    
    gdf[gdf['recovery_elasticity_unweighted'].isna()].plot(
        ax=ax, color='white', edgecolor='0.5', linewidth=0.1
    )
    
    ax.set_title('Unweighted Elasticity\n(All DS Equal)', fontsize=12, pad=10)
    ax.axis('off')
    
    # ========== PANEL 2: WEIGHTED ELASTICITY ==========
    ax = axes[1]
    
    gdf_with_data = gdf[gdf['recovery_elasticity_weighted'].notna()]
    
    if len(gdf_with_data) > 0:
        vmin, vmax = 0.0, 1.0
        
        cax2 = ax.figure.add_axes([0.62, 0.30, 0.01, 0.35])
        gdf_with_data.plot(
            column='recovery_elasticity_weighted',
            ax=ax,
            cmap='RdYlGn_r',
            vmin=vmin,
            vmax=vmax,
            edgecolor='0.5',
            linewidth=0.1,
            legend=True,
            legend_kwds={'label': 'Elasticity (Weighted)', 'shrink': 0.8},
            cax=cax2
        )
        for spine in cax2.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
    
    gdf[gdf['recovery_elasticity_weighted'].isna()].plot(
        ax=ax, color='white', edgecolor='0.5', linewidth=0.1
    )
    
    ax.set_title('Weighted Elasticity\n(DS-Specific Times)', fontsize=12, pad=10)
    ax.axis('off')
    
    # ========== PANEL 3: ELASTICITY IMPROVEMENT ==========
    ax = axes[2]
    
    gdf_with_data = gdf[gdf['elasticity_improvement'].notna()]
    
    if len(gdf_with_data) > 0:
        # Use diverging colormap centered at 0
        abs_max = gdf_with_data['elasticity_improvement'].abs().max()
        vmin, vmax = -abs_max, abs_max
        
        cax3 = ax.figure.add_axes([0.95, 0.30, 0.01, 0.35])
        gdf_with_data.plot(
            column='elasticity_improvement',
            ax=ax,
            cmap='RdBu',  # Red = negative (weighted worse), Blue = positive (weighted better)
            vmin=vmin,
            vmax=vmax,
            edgecolor='0.5',
            linewidth=0.1,
            legend=True,
            legend_kwds={'label': 'Improvement (Weighted - Unweighted)', 'shrink': 0.8},
            cax=cax3
        )
        for spine in cax3.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
    
    gdf[gdf['elasticity_improvement'].isna()].plot(
        ax=ax, color='white', edgecolor='0.5', linewidth=0.1
    )
    
    ax.set_title('Elasticity Improvement\n(Weighted - Unweighted)', fontsize=12, pad=10)
    ax.axis('off')
    
    # Save figure
    output_path = Path(output_dir) / "elasticity_comparison_maps.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved elasticity comparison maps to: {output_path}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("ELASTICITY COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\nCounties with both metrics: {gdf['elasticity_improvement'].notna().sum()}")
    print(f"Mean unweighted elasticity: {gdf['recovery_elasticity_unweighted'].mean():.4f}")
    print(f"Mean weighted elasticity: {gdf['recovery_elasticity_weighted'].mean():.4f}")
    print(f"Mean improvement: {gdf['elasticity_improvement'].mean():.4f}")
    print(f"\nCounties with positive improvement: {(gdf['elasticity_improvement'] > 0).sum()}")
    print(f"Counties with negative improvement: {(gdf['elasticity_improvement'] < 0).sum()}")


def compute_event_response_curves(recovery_all_events, units_df, capacity_df, output_dir="../analysis_output"):
    """
    Analyze county-level event response curves to classify recovery typologies.
    
    For each county, examines how recovery time responds to damage across all events
    to classify into typologies based on response patterns.
    
    Parameters
    ----------
    recovery_all_events : DataFrame
        Event-county recovery times with columns: event, fips, recovery_potential [months]
    units_df : DataFrame
        Event-county damage by DS with columns: event_name, fips, units_DS1-4_scaled
    capacity_df : DataFrame
        County capacity with columns: fips, construction_capacity
    output_dir : str
        Directory to save outputs
        
    Returns
    -------
    typology_df : DataFrame
        County typologies with response curve characteristics
    """
    print("\n" + "="*80)
    print("COMPUTING EVENT RESPONSE CURVES AND COUNTY TYPOLOGIES")
    print("="*80)
    
    # Filter to counties with both metrics
    plot_df = sensitivity_df[
        sensitivity_df['recovery_elasticity'].notna() & 
        sensitivity_df['saturation_factor_mean'].notna()
    ].copy()
    
    # Cap saturation at 50 for visualization
    plot_df['saturation_capped'] = plot_df['saturation_factor_mean'].clip(upper=50)
    
    # Calculate median values for quadrant division
    elasticity_median = plot_df['recovery_elasticity'].median()
    saturation_median = plot_df['saturation_capped'].median()
    
    print(f"\nMedian elasticity: {elasticity_median:.3f}")
    print(f"Median saturation: {saturation_median:.3f}")
    
    # Identify quadrants
    plot_df['quadrant'] = 'Other'
    plot_df.loc[
        (plot_df['recovery_elasticity'] >= elasticity_median) & 
        (plot_df['saturation_capped'] >= saturation_median), 'quadrant'
    ] = 'Q1: High-High'
    plot_df.loc[
        (plot_df['recovery_elasticity'] < elasticity_median) & 
        (plot_df['saturation_capped'] >= saturation_median), 'quadrant'
    ] = 'Q2: Low-High'
    plot_df.loc[
        (plot_df['recovery_elasticity'] < elasticity_median) & 
        (plot_df['saturation_capped'] < saturation_median), 'quadrant'
    ] = 'Q3: Low-Low'
    plot_df.loc[
        (plot_df['recovery_elasticity'] >= elasticity_median) & 
        (plot_df['saturation_capped'] < saturation_median), 'quadrant'
    ] = 'Q4: High-Low'
    
    # Select 3 example counties per quadrant (extreme values)
    examples = []
    
    # Q1: Highest elasticity + saturation
    q1 = plot_df[plot_df['quadrant'] == 'Q1: High-High'].copy()
    q1['distance'] = (q1['recovery_elasticity'] - q1['recovery_elasticity'].max())**2 + \
                     (q1['saturation_capped'] - q1['saturation_capped'].max())**2
    examples.extend(q1.nsmallest(3, 'distance')['fips'].tolist())
    
    # Q2: Lowest elasticity + highest saturation
    q2 = plot_df[plot_df['quadrant'] == 'Q2: Low-High'].copy()
    q2['distance'] = (q2['recovery_elasticity'] - q2['recovery_elasticity'].min())**2 + \
                     (q2['saturation_capped'] - q2['saturation_capped'].max())**2
    examples.extend(q2.nsmallest(3, 'distance')['fips'].tolist())
    
    # Q3: Lowest elasticity + saturation
    q3 = plot_df[plot_df['quadrant'] == 'Q3: Low-Low'].copy()
    q3['distance'] = (q3['recovery_elasticity'] - q3['recovery_elasticity'].min())**2 + \
                     (q3['saturation_capped'] - q3['saturation_capped'].min())**2
    examples.extend(q3.nsmallest(3, 'distance')['fips'].tolist())
    
    # Q4: Highest elasticity + lowest saturation
    q4 = plot_df[plot_df['quadrant'] == 'Q4: High-Low'].copy()
    q4['distance'] = (q4['recovery_elasticity'] - q4['recovery_elasticity'].max())**2 + \
                     (q4['saturation_capped'] - q4['saturation_capped'].min())**2
    examples.extend(q4.nsmallest(3, 'distance')['fips'].tolist())
    
    plot_df['is_example'] = plot_df['fips'].isin(examples)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # DS color mapping
    ds_colors = {
        1: '#d4e6f1',  # Light blue
        2: '#f9e79f',  # Light yellow
        3: '#f8b878',  # Orange
        4: '#e74c3c'   # Red
    }
    
    # Plot all counties
    for ds in sorted(plot_df['dominant_ds'].dropna().unique()):
        ds_data = plot_df[plot_df['dominant_ds'] == ds]
        ax.scatter(
            ds_data['recovery_elasticity'],
            ds_data['saturation_capped'],
            c=ds_colors.get(ds, 'gray'),
            s=20,
            alpha=0.6,
            edgecolors='0.5',
            linewidth=0.5,
            label=f'DS{int(ds)}'
        )
    
    # Highlight example counties
    examples_df = plot_df[plot_df['is_example']]
    ax.scatter(
        examples_df['recovery_elasticity'],
        examples_df['saturation_capped'],
        s=80,
        facecolors='none',
        edgecolors='black',
        linewidth=1.5,
        zorder=10
    )
    
    # Add labels for examples
    for _, row in examples_df.iterrows():
        ax.annotate(
            row['fips'],
            (row['recovery_elasticity'], row['saturation_capped']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=6,
            alpha=0.8
        )
    
    # Add median lines
    ax.axhline(saturation_median, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(elasticity_median, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Add quadrant labels
    ax.text(0.95, 0.95, 'High-High\n(Crisis)', transform=ax.transAxes,
            ha='right', va='top', fontsize=7, alpha=0.6)
    ax.text(0.05, 0.95, 'Low-High\n(Constrained)', transform=ax.transAxes,
            ha='left', va='top', fontsize=7, alpha=0.6)
    ax.text(0.05, 0.05, 'Low-Low\n(Manageable)', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=7, alpha=0.6)
    ax.text(0.95, 0.05, 'High-Low\n(At Capacity)', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7, alpha=0.6)
    
    ax.set_xlabel('Recovery Elasticity', fontsize=10)
    ax.set_ylabel('Capacity Saturation Factor', fontsize=10)
    ax.set_title('County Recovery Profiles', fontsize=11, pad=10)
    ax.legend(loc='upper left', fontsize=7, frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.8, min(50, plot_df['saturation_capped'].quantile(0.98)))
    
    # Save
    output_path = Path(output_dir) / "elasticity_saturation_scatterplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved scatterplot to: {output_path}")
    
    plt.close()
    
    # Print example counties
    print("\n" + "="*80)
    print("EXAMPLE COUNTIES BY QUADRANT")
    print("="*80)
    for quadrant in ['Q1: High-High', 'Q2: Low-High', 'Q3: Low-Low', 'Q4: High-Low']:
        print(f"\n{quadrant}:")
        quad_examples = plot_df[plot_df['is_example'] & (plot_df['quadrant'] == quadrant)]
        for _, row in quad_examples.iterrows():
            print(f"  {row['fips']}: Elasticity={row['recovery_elasticity']:.3f}, "
                  f"Saturation={row['saturation_factor_mean']:.2f}, DS={int(row['dominant_ds'])}")


# ============================================================================
# SECTION 8: SINGLE-EVENT ANALYSIS
# ============================================================================

def analyze_single_event(event_name, recovery_all_events, units_df, capacity_df, 
                         coastal_counties, output_dir="../analysis_output", event_mapping=None):
    """
    Analyze recovery drivers for a specific historical hurricane event.
    
    Performs county-level driver analysis for an individual event to identify:
    - Which counties were most affected
    - Whether recovery was capacity-driven or damage-driven per county
    - Priority counties for intervention specific to this event type
    
    Parameters
    ----------
    event_name : str
        Event identifier from recovery dataset (e.g., '2558')
    recovery_all_events : pd.DataFrame
        All event recovery data
    units_df : pd.DataFrame
        All event damage data
    capacity_df : pd.DataFrame
        Construction capacity data
    coastal_counties : gpd.GeoDataFrame
        County boundaries
    output_dir : str
        Output directory
    event_mapping : dict, optional
        Mapping from recovery event names to damage event names
    
    Returns
    -------
    pd.DataFrame
        Single-event analysis with variance decomposition
    """
    print("\n" + "="*80)
    print(f"SINGLE EVENT ANALYSIS: {event_name}")
    print("="*80)
    
    # Extract event data from recovery
    event_recovery = recovery_all_events[recovery_all_events['event'] == event_name].copy()
    
    # Get corresponding damage event name using mapping
    if event_mapping is not None and event_name in event_mapping:
        damage_event_name = event_mapping[event_name]
        print(f"Mapped to damage event: {damage_event_name}")
    else:
        damage_event_name = event_name
        
    event_damage = units_df[units_df['event_name'] == damage_event_name].copy()
    
    if len(event_recovery) == 0:
        print(f"ERROR: No recovery data found for event {event_name}")
        return None
    if len(event_damage) == 0:
        print(f"ERROR: No damage data found for event {event_name}")
        return None
    
    print(f"\nEvent affects {len(event_recovery)} counties")
    
    # Merge damage and recovery
    event_analysis = event_recovery[['fips', 'recovery_potential [months]']].merge(
        event_damage[['fips', 'units_DS1_scaled', 'units_DS2_scaled', 
                      'units_DS3_scaled', 'units_DS4_scaled']], 
        on='fips', how='inner'
    )
    
    # Calculate total damage
    event_analysis['total_damage_units'] = (
        event_analysis['units_DS1_scaled'] + event_analysis['units_DS2_scaled'] + 
        event_analysis['units_DS3_scaled'] + event_analysis['units_DS4_scaled']
    )
    
    # Merge with capacity
    event_analysis = event_analysis.merge(capacity_df, on='fips', how='inner')
    
    # Filter to counties with actual impact
    event_analysis = event_analysis[
        (event_analysis['recovery_potential [months]'] > 0) & 
        (event_analysis['total_damage_units'] > 0) & 
        (event_analysis['construction_capacity'] > 0)
    ].copy()
    
    print(f"Counties with measurable impact: {len(event_analysis)}")
    
    if len(event_analysis) < 3:
        print(f"WARNING: Too few counties ({len(event_analysis)}) for analysis")
        return None
    
    # Fit regression model to get capacity coefficient for sensitivity analysis
    # (only if enough counties for a meaningful regression)
    if len(event_analysis) >= 10:
        event_analysis['log_recovery'] = np.log10(event_analysis['recovery_potential [months]'])
        event_analysis['log_damage'] = np.log10(event_analysis['total_damage_units'])
        event_analysis['log_capacity'] = np.log10(event_analysis['construction_capacity'])
        
        X = event_analysis[['log_damage', 'log_capacity']].values
        y = event_analysis['log_recovery'].values
        
        model = LinearRegression().fit(X, y)
        
        beta_C = model.coef_[1]  # Capacity coefficient
        r2 = model.score(X, y)
        
        print("\n" + "="*80)
        print("REGRESSION MODEL")
        print("="*80)
        print(f"log(Recovery) = {model.intercept_:.3f} + {model.coef_[0]:+.3f}·log(Damage) + {beta_C:+.3f}·log(Capacity)")
        print(f"R² = {r2:.3f}")
    else:
        # For small events, use a default capacity elasticity
        beta_C = -0.5  # Typical value from larger analysis
        print("\n" + "="*80)
        print(f"NOTE: Too few counties ({len(event_analysis)}) for regression - using default capacity elasticity")
        print("="*80)
    
    # Summary statistics
    print(f"\n=== Impact Summary ===")
    print(f"  Counties affected: {len(event_analysis)}")
    print(f"  Total damage (units): {event_analysis['total_damage_units'].sum():,.0f}")
    print(f"  Mean recovery time: {event_analysis['recovery_potential [months]'].mean():.1f} months")
    print(f"  Median recovery time: {event_analysis['recovery_potential [months]'].median():.1f} months")
    print(f"  Max recovery time: {event_analysis['recovery_potential [months]'].max():.1f} months")
    
    # ========================================================================
    # ABSOLUTE CAPACITY CONSTRAINT ANALYSIS
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("ABSOLUTE CAPACITY CONSTRAINT ANALYSIS")
    print(f"{'='*80}")
    
    # 1. Damage-to-Capacity Ratio (workload per unit capacity)
    event_analysis['damage_capacity_ratio'] = (
        event_analysis['total_damage_units'] / event_analysis['construction_capacity']
    )
    
    # 2. Capacity stress levels
    ratio_75th = event_analysis['damage_capacity_ratio'].quantile(0.75)
    ratio_90th = event_analysis['damage_capacity_ratio'].quantile(0.90)
    
    event_analysis['capacity_stress'] = 'Low'
    event_analysis.loc[
        event_analysis['damage_capacity_ratio'] > ratio_75th, 'capacity_stress'
    ] = 'High'
    event_analysis.loc[
        event_analysis['damage_capacity_ratio'] > ratio_90th, 'capacity_stress'
    ] = 'Severe'
    
    stress_counts = event_analysis['capacity_stress'].value_counts()
    print(f"\nCapacity Stress Levels (by damage/capacity ratio):")
    for level in ['Low', 'High', 'Severe']:
        if level in stress_counts:
            count = stress_counts[level]
            pct = 100 * count / len(event_analysis)
            print(f"  {level}: {count} counties ({pct:.1f}%)")
    
    # 3. Capacity sensitivity: How much would doubling capacity reduce recovery?
    capacity_multiplier = 2 ** beta_C
    event_analysis['recovery_if_double_capacity'] = (
        event_analysis['recovery_potential [months]'] * capacity_multiplier
    )
    event_analysis['months_saved_if_double_capacity'] = (
        event_analysis['recovery_potential [months]'] - 
        event_analysis['recovery_if_double_capacity']
    )
    
    total_months_saved = event_analysis['months_saved_if_double_capacity'].sum()
    mean_months_saved = event_analysis['months_saved_if_double_capacity'].mean()
    
    print(f"\nCapacity Intervention Sensitivity (if all capacity doubled):")
    print(f"  Total months saved across all counties: {total_months_saved:,.0f}")
    print(f"  Mean months saved per county: {mean_months_saved:.1f}")
    print(f"  Capacity elasticity (β_C): {beta_C:.3f}")
    
    # 4. Intervention priority score
    event_analysis['intervention_priority'] = (
        event_analysis['damage_capacity_ratio'] * 
        event_analysis['recovery_potential [months]']
    )
    
    top_priority = event_analysis.nlargest(10, 'intervention_priority')
    print(f"\n=== TOP 10 COUNTIES FOR CAPACITY INTERVENTION ===")
    print("(Ranked by damage/capacity ratio × recovery time)")
    print(top_priority[['fips', 'total_damage_units', 'construction_capacity',
                        'damage_capacity_ratio', 'recovery_potential [months]',
                        'months_saved_if_double_capacity']].to_string(index=False))
    
    return event_analysis


def create_single_event_maps(event_name, event_analysis, coastal_counties, 
                             output_dir="../analysis_output"):
    """
    Create spatial visualizations for single-event analysis.
    
    Generates:
    1. 3-panel map: Damage + Capacity + Recovery Potential
    2. 3-panel map: Capacity Constraints
    
    Parameters
    ----------
    event_name : str
        Event identifier
    event_analysis : pd.DataFrame
        Single-event analysis results
    coastal_counties : gpd.GeoDataFrame
        County boundaries
    output_dir : str
        Output directory
    """
    print(f"\nCreating visualizations for {event_name}...")
    
    # Sanitize event name for filename
    safe_name = event_name.replace('/', '_').replace(' ', '_')
    
    # Merge with geodataframe
    gdf = coastal_counties.merge(
        event_analysis[['fips', 'total_damage_units', 'construction_capacity', 
                       'recovery_potential [months]', 'damage_capacity_ratio', 
                       'capacity_stress', 'months_saved_if_double_capacity', 
                       'intervention_priority']],
        left_on='GEOID',
        right_on='fips',
        how='left'
    )
    
    # ========================================================================
    # 1. Three-panel map: Damage + Capacity + Recovery
    # ========================================================================
    from matplotlib.colors import LogNorm
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Hurricane Event Analysis: {event_name}', fontsize=14, fontweight='bold')
    
    # Panel 1: Damage (log scale)
    ax1 = axes[0]
    damage_data = gdf['total_damage_units'].dropna()
    if len(damage_data) > 0 and damage_data.max() > 0:
        vmin_damage = max(damage_data[damage_data > 0].min(), 0.1)
        vmax_damage = damage_data.max()
        gdf.plot(
            column='total_damage_units',
            cmap='YlOrRd',
            linewidth=0.1,
            edgecolor='0.5',
            legend=True,
            ax=ax1,
            norm=LogNorm(vmin=vmin_damage, vmax=vmax_damage),
            legend_kwds={'label': 'Units damaged (log scale)', 'shrink': 0.7},
            missing_kwds={'color': '#f0f0f0'}
        )
    ax1.set_title('Total Damage', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Capacity (log scale)
    ax2 = axes[1]
    capacity_data = gdf['construction_capacity'].dropna()
    if len(capacity_data) > 0 and capacity_data.max() > 0:
        vmin_capacity = max(capacity_data[capacity_data > 0].min(), 0.1)
        vmax_capacity = capacity_data.max()
        gdf.plot(
            column='construction_capacity',
            cmap='Greens',
            linewidth=0.1,
            edgecolor='0.5',
            legend=True,
            ax=ax2,
            norm=LogNorm(vmin=vmin_capacity, vmax=vmax_capacity),
            legend_kwds={'label': 'Permits/month (log scale)', 'shrink': 0.7},
            missing_kwds={'color': '#f0f0f0'}
        )
    ax2.set_title('Construction Capacity', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Panel 3: Recovery Potential (log scale)
    ax3 = axes[2]
    recovery_data = gdf['recovery_potential [months]'].dropna()
    if len(recovery_data) > 0 and recovery_data.max() > 0:
        vmin_recovery = max(recovery_data[recovery_data > 0].min(), 0.1)
        vmax_recovery = recovery_data.max()
        gdf.plot(
            column='recovery_potential [months]',
            cmap='Purples',
            linewidth=0.1,
            edgecolor='0.5',
            legend=True,
            ax=ax3,
            norm=LogNorm(vmin=vmin_recovery, vmax=vmax_recovery),
            legend_kwds={'label': 'Months (log scale)', 'shrink': 0.7},
            missing_kwds={'color': '#f0f0f0'}
        )
    ax3.set_title('Recovery Potential', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/single_event_{safe_name}_3panel.png", 
                dpi=300, bbox_inches="tight")
    plt.show()
    
    # ========================================================================
    # 2. Capacity Constraint Maps
    # ========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f'Capacity Constraints: {event_name}', 
                 fontsize=14, fontweight='bold')
    
    # Panel 1: Damage/Capacity Ratio
    ax1 = axes[0]
    ratio_data = gdf['damage_capacity_ratio'].dropna()
    if len(ratio_data) > 0 and ratio_data.max() > 0:
        gdf.plot(
            column='damage_capacity_ratio',
            cmap='YlOrRd',
            linewidth=0.1,
            edgecolor='0.5',
            legend=True,
            ax=ax1,
            norm=LogNorm(vmin=max(ratio_data[ratio_data > 0].min(), 0.1), 
                        vmax=ratio_data.max()),
            legend_kwds={'label': 'Damage/Capacity\n(log scale)', 'shrink': 0.7},
            missing_kwds={'color': '#f0f0f0'}
        )
    ax1.set_title('Damage/Capacity Ratio\n(Capacity Workload)', 
                  fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Months saved if capacity doubled
    ax2 = axes[1]
    gdf.plot(
        column='months_saved_if_double_capacity',
        cmap='Purples',
        linewidth=0.1,
        edgecolor='0.5',
        legend=True,
        ax=ax2,
        legend_kwds={'label': 'Months saved', 'shrink': 0.7},
        missing_kwds={'color': '#f0f0f0'}
    )
    ax2.set_title('Recovery Reduction\nif Capacity Doubled', 
                  fontsize=11, fontweight='bold')
    ax2.axis('off')
    
    # Panel 3: Intervention Priority Score
    ax3 = axes[2]
    priority_data = gdf['intervention_priority'].dropna()
    if len(priority_data) > 0 and priority_data.max() > 0:
        gdf.plot(
            column='intervention_priority',
            cmap='RdPu',
            linewidth=0.1,
            edgecolor='0.5',
            legend=True,
            ax=ax3,
            norm=LogNorm(vmin=max(priority_data[priority_data > 0].min(), 0.1),
                        vmax=priority_data.max()),
            legend_kwds={'label': 'Priority score\n(log scale)', 'shrink': 0.7},
            missing_kwds={'color': '#f0f0f0'}
        )
    ax3.set_title('Capacity Intervention Priority\n(Ratio × Recovery Time)', 
                  fontsize=11, fontweight='bold')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/single_event_{safe_name}_capacity_constraints.png", 
                dpi=300, bbox_inches="tight")
    plt.show()
    
    print(f"\n✓ Visualizations saved for {event_name}")


def list_available_events(recovery_all_events, units_df, top_n=20):
    """
    List available events sorted by total impact.
    
    Parameters
    ----------
    recovery_all_events : pd.DataFrame
        All event recovery data
    units_df : pd.DataFrame
        All event damage data
    top_n : int
        Number of top events to display
    
    Returns
    -------
    pd.DataFrame
        Event summary statistics
    """
    print("\n" + "="*80)
    print("AVAILABLE EVENTS FOR SINGLE-EVENT ANALYSIS")
    print("="*80)
    
    # Get unique events from both datasets
    recovery_events = set(recovery_all_events['event'].unique())
    damage_events = set(units_df['event_name'].unique())
    
    print(f"\nEvents in recovery data: {len(recovery_events)}")
    print(f"Events in damage data: {len(damage_events)}")
    
    # Show sample event names to understand format differences
    print(f"\nSample recovery event names: {sorted(list(recovery_events))[:5]}")
    print(f"Sample damage event names: {sorted(list(damage_events))[:5]}")
    
    # Try to find common events (exact match)
    common_events = recovery_events.intersection(damage_events)
    print(f"\nEvents with EXACT match: {len(common_events)}")
    
    # Create mapping: recovery event -> damage event
    # Recovery events often have format like '2558' while damage has '2005236N14283'
    # Let's create a mapping based on event ID substring
    event_mapping = {}
    for rec_event in recovery_events:
        # Try to find damage events that contain this recovery event ID
        rec_event_str = str(rec_event)
        matches = [d_event for d_event in damage_events if rec_event_str in str(d_event)]
        if len(matches) == 1:
            event_mapping[rec_event] = matches[0]
        elif len(matches) > 1:
            # Multiple matches - take first one but warn
            event_mapping[rec_event] = matches[0]
    
    print(f"\nEvents that can be mapped: {len(event_mapping)}")
    
    if len(event_mapping) == 0:
        print("\nWARNING: No matching events found between datasets!")
        return None
    
    # Show some mappings
    print(f"\nSample event mappings (recovery -> damage):")
    for i, (rec, dam) in enumerate(list(event_mapping.items())[:5]):
        print(f"  '{rec}' -> '{dam}'")
    
    # Aggregate by event (only mapped events)
    event_summary = recovery_all_events[
        recovery_all_events['event'].isin(event_mapping.keys())
    ].groupby('event').agg({
        'recovery_potential [months]': ['count', 'sum', 'mean', 'max'],
        'fips': 'nunique'
    }).reset_index()
    
    event_summary.columns = ['event', 'num_records', 'total_recovery', 
                             'mean_recovery', 'max_recovery', 'num_counties']
    
    # Add corresponding damage event name
    event_summary['damage_event'] = event_summary['event'].map(event_mapping)
    
    # Sort by total impact
    event_summary = event_summary.sort_values('total_recovery', ascending=False)
    
    print(f"\n\nTop {top_n} events by total recovery burden:")
    print(event_summary[['event', 'damage_event', 'num_counties', 'total_recovery', 
                         'mean_recovery', 'max_recovery']].head(top_n).to_string(index=False))
    
    print(f"\n\nTotal analyzable events: {len(event_summary)}")
    print("\nNOTE: Use the 'event' column value (not 'damage_event') when calling analyze_single_event()")
    
    return event_summary


def screen_events_by_driver(recovery_all_events, units_df, capacity_df, event_mapping,
                             min_counties=30, top_n=10):
    """
    Screen events to find those with different driver profiles.
    
    Quickly computes variance decomposition for all events to identify:
    - Damage-driven events (typical)
    - Capacity-driven events (interesting cases)
    - Balanced events
    
    Parameters
    ----------
    recovery_all_events : pd.DataFrame
        All event recovery data
    units_df : pd.DataFrame
        All event damage data
    capacity_df : pd.DataFrame
        County capacity data
    event_mapping : dict
        Mapping from recovery event names to damage event names
    min_counties : int
        Minimum counties required for analysis
    top_n : int
        Number of top events to show per category
    
    Returns
    -------
    pd.DataFrame
        Event screening results with variance shares
    """
    print("\n" + "="*80)
    print("SCREENING EVENTS BY DRIVER PROFILE")
    print("="*80)
    
    screening_results = []
    
    # Get analyzable events
    analyzable_events = [e for e in event_mapping.keys() 
                        if e in recovery_all_events['event'].unique()]
    
    print(f"\nScreening {len(analyzable_events)} events...")
    
    for event_name in analyzable_events:
        # Get event data
        event_recovery = recovery_all_events[
            recovery_all_events['event'] == event_name
        ].copy()
        
        damage_event_name = event_mapping[event_name]
        event_damage = units_df[units_df['event_name'] == damage_event_name].copy()
        
        # Merge and filter
        event_analysis = event_recovery[['fips', 'recovery_potential [months]']].merge(
            event_damage[['fips', 'units_DS1_scaled', 'units_DS2_scaled', 
                          'units_DS3_scaled', 'units_DS4_scaled']], 
            on='fips', how='inner'
        )
        
        event_analysis['total_damage_units'] = (
            event_analysis['units_DS1_scaled'] + event_analysis['units_DS2_scaled'] + 
            event_analysis['units_DS3_scaled'] + event_analysis['units_DS4_scaled']
        )
        
        event_analysis = event_analysis.merge(capacity_df, on='fips', how='inner')
        
        event_analysis = event_analysis[
            (event_analysis['recovery_potential [months]'] > 0) & 
            (event_analysis['total_damage_units'] > 0) & 
            (event_analysis['construction_capacity'] > 0)
        ].copy()
        
        # Skip if too few counties
        if len(event_analysis) < min_counties:
            continue
        
        # Quick regression
        event_analysis['log_recovery'] = np.log10(event_analysis['recovery_potential [months]'])
        event_analysis['log_damage'] = np.log10(event_analysis['total_damage_units'])
        event_analysis['log_capacity'] = np.log10(event_analysis['construction_capacity'])
        
        X = event_analysis[['log_damage', 'log_capacity']].values
        y = event_analysis['log_recovery'].values
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Global R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_global = 1 - (ss_res / ss_tot)
        
        # Variance decomposition
        X_damage_only = X[:, [0]]
        X_capacity_only = X[:, [1]]
        
        model_damage = LinearRegression().fit(X_damage_only, y)
        model_capacity = LinearRegression().fit(X_capacity_only, y)
        
        y_pred_damage = model_damage.predict(X_damage_only)
        y_pred_capacity = model_capacity.predict(X_capacity_only)
        
        r2_damage = 1 - (np.sum((y - y_pred_damage) ** 2) / ss_tot)
        r2_capacity = 1 - (np.sum((y - y_pred_capacity) ** 2) / ss_tot)
        
        # Unique contributions
        unique_damage = r2_global - r2_capacity
        unique_capacity = r2_global - r2_damage
        shared = r2_damage + r2_capacity - r2_global
        
        # Share of explained variance
        if r2_global > 0:
            share_damage = unique_damage / r2_global
            share_capacity = unique_capacity / r2_global
            share_shared = shared / r2_global
        else:
            share_damage = share_capacity = share_shared = 0
        
        # Store results
        screening_results.append({
            'event': event_name,
            'damage_event': damage_event_name,
            'num_counties': len(event_analysis),
            'total_recovery': event_analysis['recovery_potential [months]'].sum(),
            'mean_recovery': event_analysis['recovery_potential [months]'].mean(),
            'r2_global': r2_global,
            'share_damage': share_damage,
            'share_capacity': share_capacity,
            'share_shared': share_shared,
            'beta_damage': model.coef_[0],
            'beta_capacity': model.coef_[1]
        })
    
    # Convert to DataFrame
    screening_df = pd.DataFrame(screening_results)
    
    # Categorize events
    screening_df['driver_category'] = 'Balanced'
    screening_df.loc[screening_df['share_capacity'] > 0.4, 'driver_category'] = 'Capacity-driven'
    screening_df.loc[screening_df['share_damage'] > 0.6, 'driver_category'] = 'Damage-driven'
    
    print(f"\nSuccessfully screened {len(screening_df)} events")
    print(f"\nDriver Categories:")
    print(screening_df['driver_category'].value_counts())
    
    # Show top capacity-driven events
    print(f"\n{'='*80}")
    print(f"TOP {top_n} CAPACITY-DRIVEN EVENTS (highest capacity variance share)")
    print(f"{'='*80}")
    capacity_driven = screening_df.nlargest(top_n, 'share_capacity')
    print(capacity_driven[['event', 'num_counties', 'mean_recovery', 'r2_global',
                           'share_damage', 'share_capacity', 'share_shared']].to_string(index=False))
    
    # Show top damage-driven events
    print(f"\n{'='*80}")
    print(f"TOP {top_n} DAMAGE-DRIVEN EVENTS (highest damage variance share)")
    print(f"{'='*80}")
    damage_driven = screening_df.nlargest(top_n, 'share_damage')
    print(damage_driven[['event', 'num_counties', 'mean_recovery', 'r2_global',
                         'share_damage', 'share_capacity', 'share_shared']].to_string(index=False))
    
    # Show most balanced events
    print(f"\n{'='*80}")
    print(f"MOST BALANCED EVENTS")
    print(f"{'='*80}")
    screening_df['balance_score'] = 1 - abs(screening_df['share_damage'] - screening_df['share_capacity'])
    balanced = screening_df.nlargest(top_n, 'balance_score')
    print(balanced[['event', 'num_counties', 'mean_recovery', 'r2_global',
                    'share_damage', 'share_capacity', 'share_shared']].to_string(index=False))
    
    return screening_df


def create_variance_distribution_plots(driver_analysis, per_event_analysis_median,
                                       output_dir="../analysis_output"):
    """
    Identify counties where interventions (capacity building or damage reduction) 
    would have the greatest impact on recovery times.
    
    This analysis identifies:
    1. Capacity-limited regions: High capacity-driven variance + low capacity
       → Capacity building would greatly reduce recovery times
    2. Damage-limited regions: High damage-driven variance + high damage
       → Damage mitigation would be more effective than capacity building
    
    Parameters
    ----------
    driver_analysis : pd.DataFrame
        Annual metrics with variance shares
    per_event_analysis_median : pd.DataFrame
        Per-event metrics with variance shares
    capacity_df : pd.DataFrame
        Construction capacity data
    earp_df : pd.DataFrame
        EARP metrics
    ead_wide : pd.DataFrame
        Expected Annual Damage data
    output_dir : str
        Output directory
    
    Returns
    -------
    pd.DataFrame
        Strategic intervention recommendations per county
    """
    print("\n" + "="*80)
    print("STRATEGIC INTERVENTION ANALYSIS")
    print("="*80)
    
    # Merge all relevant data for annual analysis
    strategic_df = driver_analysis[['fips', 'share_damage', 'share_capacity', 'log_earp', 'log_risk', 'log_capacity']].copy()
    strategic_df = strategic_df.merge(capacity_df[['fips', 'construction_capacity']], on='fips', how='left')
    strategic_df = strategic_df.merge(earp_df[['fips', 'earp_months_per_year']], on='fips', how='left')
    strategic_df = strategic_df.merge(ead_wide[['fips', 'total_ead']], on='fips', how='left')
    
    # Define thresholds
    capacity_driven_threshold = 0.6  # >60% capacity contribution
    damage_driven_threshold = 0.6    # >60% damage contribution
    
    # Percentile thresholds for severity
    low_capacity_pct = 25  # Bottom quartile
    high_damage_pct = 75   # Top quartile
    high_earp_pct = 75     # Top quartile (long recovery times)
    
    # Calculate percentiles
    capacity_low_threshold = strategic_df['construction_capacity'].quantile(low_capacity_pct/100)
    damage_high_threshold = strategic_df['total_ead'].quantile(high_damage_pct/100)
    earp_high_threshold = strategic_df['earp_months_per_year'].quantile(high_earp_pct/100)
    
    # Identify strategic regions
    strategic_df['is_capacity_constrained'] = strategic_df['construction_capacity'] < capacity_low_threshold
    strategic_df['is_high_damage'] = strategic_df['total_ead'] > damage_high_threshold
    strategic_df['is_high_earp'] = strategic_df['earp_months_per_year'] > earp_high_threshold
    strategic_df['is_capacity_driven'] = strategic_df['share_capacity'] > capacity_driven_threshold
    strategic_df['is_damage_driven'] = strategic_df['share_damage'] > damage_driven_threshold
    
    # Priority 1: Capacity-constrained + capacity-driven + high EARP
    # → Building capacity would dramatically reduce recovery times
    strategic_df['priority_capacity_building'] = (
        strategic_df['is_capacity_constrained'] & 
        strategic_df['is_capacity_driven'] & 
        strategic_df['is_high_earp']
    )
    
    # Priority 2: High damage + damage-driven + high EARP + NOT capacity constrained
    # → Damage mitigation (structural hardening, etc.) would be more effective
    # → Explicitly ensure these are NOT capacity limited (capacity is adequate)
    strategic_df['priority_damage_mitigation'] = (
        strategic_df['is_high_damage'] & 
        strategic_df['is_damage_driven'] & 
        strategic_df['is_high_earp'] & 
        ~strategic_df['is_capacity_constrained']  # NOT capacity limited
    )
    
    # Calculate "leverage scores" - how much improvement is possible
    # For capacity building: EARP × capacity_share (higher = more benefit from capacity increase)
    strategic_df['capacity_leverage'] = (
        strategic_df['earp_months_per_year'] * strategic_df['share_capacity']
    )
    
    # For damage mitigation: EARP × damage_share (higher = more benefit from damage reduction)
    strategic_df['damage_leverage'] = (
        strategic_df['earp_months_per_year'] * strategic_df['share_damage']
    )
    
    # Assign primary recommendation
    strategic_df['primary_intervention'] = 'Mixed'
    strategic_df.loc[strategic_df['priority_capacity_building'], 'primary_intervention'] = 'Capacity Building'
    strategic_df.loc[strategic_df['priority_damage_mitigation'], 'primary_intervention'] = 'Damage Mitigation'
    
    # Priority tier (1=highest priority)
    strategic_df['priority_tier'] = 3  # Default: low priority
    strategic_df.loc[strategic_df['priority_capacity_building'], 'priority_tier'] = 1
    strategic_df.loc[strategic_df['priority_damage_mitigation'], 'priority_tier'] = 1
    strategic_df.loc[strategic_df['is_high_earp'] & (strategic_df['priority_tier'] == 3), 'priority_tier'] = 2
    
    # Report findings
    print(f"\n=== STRATEGIC INTERVENTION OPPORTUNITIES ===\n")
    
    n_capacity = strategic_df['priority_capacity_building'].sum()
    n_damage = strategic_df['priority_damage_mitigation'].sum()
    
    print(f"Priority Capacity Building Counties: {n_capacity}")
    print(f"  → High capacity-driven recovery (>{capacity_driven_threshold*100:.0f}% variance)")
    print(f"  → Low construction capacity (<{capacity_low_threshold:.0f} units/month)")
    print(f"  → High recovery burden (>{earp_high_threshold:.2f} months/year)")
    print(f"  → Interpretation: Increasing capacity would greatly reduce recovery times\n")
    
    print(f"Priority Damage Mitigation Counties: {n_damage}")
    print(f"  → High damage-driven recovery (>{damage_driven_threshold*100:.0f}% variance)")
    print(f"  → High expected damage (>{damage_high_threshold:.0f} units/year)")
    print(f"  → High recovery burden (>{earp_high_threshold:.2f} months/year)")
    print(f"  → NOT capacity constrained (capacity is adequate)")
    print(f"  → Interpretation: Damage reduction would be more effective than capacity building\n")
    
    # Top 10 capacity building opportunities
    top_capacity = strategic_df[strategic_df['priority_capacity_building']].nlargest(10, 'capacity_leverage')
    if len(top_capacity) > 0:
        print("Top 10 Capacity Building Opportunities:")
        print(f"{'FIPS':<8} {'Capacity':<12} {'EARP (mo/yr)':<14} {'Cap Share':<12} {'Leverage':<10}")
        print("-" * 70)
        for _, row in top_capacity.iterrows():
            print(f"{row['fips']:<8} {row['construction_capacity']:<12.0f} "
                  f"{row['earp_months_per_year']:<14.2f} {row['share_capacity']:<12.2f} "
                  f"{row['capacity_leverage']:<10.2f}")
    
    # Top 10 damage mitigation opportunities
    top_damage = strategic_df[strategic_df['priority_damage_mitigation']].nlargest(10, 'damage_leverage')
    if len(top_damage) > 0:
        print(f"\nTop 10 Damage Mitigation Opportunities:")
        print(f"{'FIPS':<8} {'EAD (units)':<14} {'EARP (mo/yr)':<14} {'Dmg Share':<12} {'Leverage':<10}")
        print("-" * 70)
        for _, row in top_damage.iterrows():
            print(f"{row['fips']:<8} {row['total_ead']:<14.1f} "
                  f"{row['earp_months_per_year']:<14.2f} {row['share_damage']:<12.2f} "
                  f"{row['damage_leverage']:<10.2f}")
    
    # Save detailed results
    output_path = Path(output_dir) / "strategic_intervention_analysis.csv"
    strategic_df.to_csv(output_path, index=False)
    print(f"\n✓ Strategic intervention analysis saved to: {output_path}")
    
    return strategic_df


def create_earp_capacity_maps(earp_df, capacity_df, coastal_counties, output_dir="../analysis_output"):
    """
    Create 2-panel plot: Construction Capacity and EARP.
    
    Args:
        earp_df: DataFrame with EARP per county
        capacity_df: DataFrame with construction capacity per county
        coastal_counties: GeoDataFrame of coastal counties
        output_dir: Directory to save outputs
    """
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import NullLocator
    
    # Merge data
    merged_earp_capacity = coastal_counties.copy()
    merged_earp_capacity = merged_earp_capacity.merge(capacity_df, left_on='GEOID', right_on='fips', how='left')
    merged_earp_capacity = merged_earp_capacity.drop(columns=['fips'])
    merged_earp_capacity = merged_earp_capacity.merge(
        earp_df[['fips', 'earp_months_per_year']], left_on='GEOID', right_on='fips', how='left'
    )
    merged_earp_capacity = merged_earp_capacity.drop(columns=['fips'])
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes = axes.flatten()
    
    recovery_metrics = [
        ('construction_capacity', 'Greens', 'Construction Capacity'),
        ('earp_months_per_year', 'Purples_r', 'Expected Annual Recovery Potential')
    ]
    
    merged_earp_plot = merged_earp_capacity.copy()
    for metric, _, _ in recovery_metrics:
        merged_earp_plot.loc[merged_earp_plot[metric] <= 0, metric] = np.nan
    
    for idx, (ax, (metric, cmap, title)) in enumerate(zip(axes, recovery_metrics)):
        data_positive = merged_earp_plot[metric].dropna()
        
        if not data_positive.empty and len(data_positive) > 0:
            vmin = data_positive.min()
            vmax = data_positive.max()
            
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0 or vmax <= 0:
                norm = None
            else:
                log_vmin = vmin / 2
                norm = LogNorm(vmin=log_vmin, vmax=vmax)
        else:
            norm = None
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        merged_earp_plot.plot(
            column=metric, cmap=cmap, norm=norm,
            linewidth=0.1, edgecolor="0.5",
            legend=True, ax=ax, cax=cax,
            missing_kwds={"color": "white", "label": "No data", "edgecolor": "0.5"}
        )
        
        ax.set_title(title, fontsize=12, pad=2)
        ax.axis("off")
        
        if metric == 'construction_capacity':
            cax.set_ylabel('permits/month', fontsize=10)
            cax.tick_params(labelsize=10)
            cax.tick_params(which='minor', length=0)
        elif metric == 'earp_months_per_year':
            cax.invert_yaxis()
            cax.yaxis.set_major_locator(NullLocator())
            cax.yaxis.set_minor_locator(NullLocator())
            cax.tick_params(which='both', left=False, right=False, labelleft=False)
            cax.text(1.5, 0.95, 'high', transform=cax.transAxes, 
                    fontsize=10, va='top', ha='left')
            cax.text(1.5, 0.05, 'low', transform=cax.transAxes, 
                    fontsize=10, va='bottom', ha='left')
            cax.set_ylabel('recovery potential', fontsize=10)
        
        for spine in cax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/na_coast_earp_metrics.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved: na_coast_earp_metrics.png")


def create_three_panel_maps(ead_wide, capacity_df, earp_df, coastal_counties, output_dir="../analysis_output"):
    """
    Create 3-panel plot: Total EAD + Construction Capacity + Recovery Potential.
    
    Args:
        ead_wide: DataFrame with EAD by damage state per county
        capacity_df: DataFrame with construction capacity per county
        earp_df: DataFrame with EARP per county
        coastal_counties: GeoDataFrame of coastal counties
        output_dir: Directory to save outputs
    """
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import NullLocator
    
    # Compute total EAD
    ead_wide = ead_wide.copy()
    ead_wide['total_ead'] = ead_wide[['DS1', 'DS2', 'DS3', 'DS4']].sum(axis=1)
    
    # Merge all data
    merged_all_metrics = coastal_counties.copy()
    merged_all_metrics = merged_all_metrics.merge(
        ead_wide[['fips', 'total_ead']], left_on='GEOID', right_on='fips', how='left'
    )
    merged_all_metrics = merged_all_metrics.drop(columns=['fips'])
    merged_all_metrics = merged_all_metrics.merge(capacity_df, left_on='GEOID', right_on='fips', how='left')
    merged_all_metrics = merged_all_metrics.drop(columns=['fips'])
    merged_all_metrics = merged_all_metrics.merge(
        earp_df[['fips', 'earp_months_per_year']], left_on='GEOID', right_on='fips', how='left'
    )
    merged_all_metrics = merged_all_metrics.drop(columns=['fips'])
    
    merged_all_plot = merged_all_metrics.copy()
    for col in ['total_ead', 'construction_capacity', 'earp_months_per_year']:
        merged_all_plot.loc[merged_all_plot[col] <= 0, col] = np.nan
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes = axes.flatten()
    
    metrics = [
        ('total_ead', 'cividis', 'Expected Annual Units Affected', '# units'),
        ('construction_capacity', 'Greens', 'Construction Capacity', 'permits/month'),
        ('earp_months_per_year', 'Purples_r', 'Expected Annual Recovery Potential', 'recovery potential')
    ]
    
    subplot_labels = ['a', 'b', 'c']
    
    for idx, (ax, (metric, cmap, title, ylabel)) in enumerate(zip(axes, metrics)):
        data_positive = merged_all_plot[metric].dropna()
        
        if not data_positive.empty and len(data_positive) > 0:
            vmin = data_positive.min()
            vmax = data_positive.max()
            
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0 or vmax <= 0:
                norm = None
            else:
                log_vmin = vmin / 2
                norm = LogNorm(vmin=log_vmin, vmax=vmax)
        else:
            norm = None
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        merged_all_plot.plot(
            column=metric, cmap=cmap, norm=norm,
            linewidth=0.1, edgecolor="0.5",
            legend=True, ax=ax, cax=cax,
            missing_kwds={"color": "white", "label": "No data", "edgecolor": "0.5"}
        )
        
        ax.axis("off")
        ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left')
        
        if metric == 'earp_months_per_year':
            cax.invert_yaxis()
            cax.yaxis.set_major_locator(NullLocator())
            cax.yaxis.set_minor_locator(NullLocator())
            cax.tick_params(which='both', left=False, right=False, labelleft=False)
            cax.text(1.5, 0.95, 'high', transform=cax.transAxes, 
                    fontsize=10, va='top', ha='left')
            cax.text(1.5, 0.05, 'low', transform=cax.transAxes, 
                    fontsize=10, va='bottom', ha='left')
            cax.set_ylabel(ylabel, fontsize=10)
        else:
            cax.set_ylabel(ylabel, fontsize=10)
            cax.tick_params(labelsize=10)
            cax.tick_params(which='minor', length=0)
        
        for spine in cax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/na_coast_3panel_ead_capacity_recovery_notitle.png", 
                dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved: na_coast_3panel_ead_capacity_recovery_notitle.png")


def create_three_panel_maps_per_event(per_event_analysis_median, capacity_df, coastal_counties, 
                                       output_dir="../analysis_output"):
    """
    Create 3-panel plot: Median Event Damage + Construction Capacity + Median Recovery Potential.
    
    Per-event equivalent of the annual 3-panel map, showing typical event characteristics.
    
    Args:
        per_event_analysis_median: DataFrame with median per-event metrics per county
        capacity_df: DataFrame with construction capacity per county
        coastal_counties: GeoDataFrame of coastal counties
        output_dir: Directory to save outputs
    """
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import NullLocator
    
    # Merge all data
    merged_all_metrics = coastal_counties.copy()
    merged_all_metrics = merged_all_metrics.merge(
        per_event_analysis_median[['fips', 'median_damage_units', 'median_recovery_months']], 
        left_on='GEOID', right_on='fips', how='left'
    )
    merged_all_metrics = merged_all_metrics.drop(columns=['fips'])
    merged_all_metrics = merged_all_metrics.merge(
        capacity_df[['fips', 'construction_capacity']], 
        left_on='GEOID', right_on='fips', how='left'
    )
    merged_all_metrics = merged_all_metrics.drop(columns=['fips'])
    
    # Replace non-positive values with NaN for plotting
    merged_all_plot = merged_all_metrics.copy()
    for col in ['median_damage_units', 'construction_capacity', 'median_recovery_months']:
        merged_all_plot.loc[merged_all_plot[col] <= 0, col] = np.nan
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes = axes.flatten()
    
    metrics = [
        ('median_damage_units', 'cividis', 'Median Event Units Affected', '# units'),
        ('construction_capacity', 'Greens', 'Construction Capacity', 'permits/month'),
        ('median_recovery_months', 'Purples_r', 'Median Event Recovery Potential', 'months')
    ]
    
    subplot_labels = ['a', 'b', 'c']
    
    for idx, (ax, (metric, cmap, title, ylabel)) in enumerate(zip(axes, metrics)):
        data_positive = merged_all_plot[metric].dropna()
        
        if not data_positive.empty and len(data_positive) > 0:
            vmin = data_positive.min()
            vmax = data_positive.max()
            
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0 or vmax <= 0:
                norm = None
            else:
                log_vmin = vmin / 2
                norm = LogNorm(vmin=log_vmin, vmax=vmax)
        else:
            norm = None
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        merged_all_plot.plot(
            column=metric, cmap=cmap, norm=norm,
            linewidth=0.1, edgecolor="0.5",
            legend=True, ax=ax, cax=cax,
            missing_kwds={"color": "white", "label": "No data", "edgecolor": "0.5"}
        )
        
        ax.axis("off")
        ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left')
        
        if metric == 'median_recovery_months':
            cax.invert_yaxis()
            cax.yaxis.set_major_locator(NullLocator())
            cax.yaxis.set_minor_locator(NullLocator())
            cax.tick_params(which='both', left=False, right=False, labelleft=False)
            cax.text(1.5, 0.95, 'high', transform=cax.transAxes, 
                    fontsize=10, va='top', ha='left')
            cax.text(1.5, 0.05, 'low', transform=cax.transAxes, 
                    fontsize=10, va='bottom', ha='left')
            cax.set_ylabel(ylabel, fontsize=10)
        else:
            cax.set_ylabel(ylabel, fontsize=10)
            cax.tick_params(labelsize=10)
            cax.tick_params(which='minor', length=0)
        
        for spine in cax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/na_coast_3panel_median_event_damage_capacity_recovery_notitle.png", 
                dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved: na_coast_3panel_median_event_damage_capacity_recovery_notitle.png")


def create_three_panel_maps_per_event_median_titled(per_event_analysis_median, capacity_df, coastal_counties, 
                                                     output_dir="../analysis_output"):
    """
    Create 3-panel plot with titles: Median Event Damage + Construction Capacity + Median Recovery Potential.
    
    Same as create_three_panel_maps_per_event but with titles instead of subplot labels.
    
    Args:
        per_event_analysis_median: DataFrame with median per-event metrics per county
        capacity_df: DataFrame with construction capacity per county
        coastal_counties: GeoDataFrame of coastal counties
        output_dir: Directory to save outputs
    """
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import NullLocator
    
    # Merge all data
    merged_all_metrics = coastal_counties.copy()
    merged_all_metrics = merged_all_metrics.merge(
        per_event_analysis_median[['fips', 'median_damage_units', 'median_recovery_months']], 
        left_on='GEOID', right_on='fips', how='left'
    )
    merged_all_metrics = merged_all_metrics.drop(columns=['fips'])
    merged_all_metrics = merged_all_metrics.merge(
        capacity_df[['fips', 'construction_capacity']], 
        left_on='GEOID', right_on='fips', how='left'
    )
    merged_all_metrics = merged_all_metrics.drop(columns=['fips'])
    
    # Replace non-positive values with NaN for plotting
    merged_all_plot = merged_all_metrics.copy()
    for col in ['median_damage_units', 'construction_capacity', 'median_recovery_months']:
        merged_all_plot.loc[merged_all_plot[col] <= 0, col] = np.nan
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes = axes.flatten()
    
    metrics = [
        ('median_damage_units', 'cividis', 'Median Event Damage', '# units'),
        ('construction_capacity', 'Greens', 'Construction Capacity', 'permits/month'),
        ('median_recovery_months', 'Purples_r', 'Median Recovery Potential', 'recovery potential')
    ]
    
    for idx, (ax, (metric, cmap, title, ylabel)) in enumerate(zip(axes, metrics)):
        data_positive = merged_all_plot[metric].dropna()
        
        if not data_positive.empty and len(data_positive) > 0:
            vmin = data_positive.min()
            vmax = data_positive.max()
            
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0 or vmax <= 0:
                norm = None
            else:
                log_vmin = vmin / 2
                norm = LogNorm(vmin=log_vmin, vmax=vmax)
        else:
            norm = None
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        merged_all_plot.plot(
            column=metric, cmap=cmap, norm=norm,
            linewidth=0.1, edgecolor="0.5",
            legend=True, ax=ax, cax=cax,
            missing_kwds={"color": "white", "label": "No data", "edgecolor": "0.5"}
        )
        
        # Add title instead of subplot label
        ax.set_title(title, fontsize=12, pad=2)
        ax.axis("off")
        
        if metric == 'median_recovery_months':
            cax.invert_yaxis()
            cax.yaxis.set_major_locator(NullLocator())
            cax.yaxis.set_minor_locator(NullLocator())
            cax.tick_params(which='both', left=False, right=False, labelleft=False)
            cax.text(1.5, 0.95, 'high', transform=cax.transAxes, 
                    fontsize=10, va='top', ha='left')
            cax.text(1.5, 0.05, 'low', transform=cax.transAxes, 
                    fontsize=10, va='bottom', ha='left')
            cax.set_ylabel(ylabel, fontsize=10)
        else:
            cax.set_ylabel(ylabel, fontsize=10)
            cax.tick_params(labelsize=10)
            cax.tick_params(which='minor', length=0)
        
        for spine in cax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/na_coast_3panel_median_event_damage_capacity_recovery_titled.png", 
                dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved: na_coast_3panel_median_event_damage_capacity_recovery_titled.png")


def create_three_panel_maps_per_event_maximum(per_event_analysis_maximum, capacity_df, coastal_counties, 
                                               output_dir="../analysis_output"):
    """
    Create 3-panel plot: Maximum Event Damage + Construction Capacity + Maximum Recovery Potential.
    
    Shows worst-case event characteristics for emergency preparedness planning.
    
    Args:
        per_event_analysis_maximum: DataFrame with maximum per-event metrics per county
        capacity_df: DataFrame with construction capacity per county
        coastal_counties: GeoDataFrame of coastal counties
        output_dir: Directory to save outputs
    """
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import NullLocator
    
    # Merge all data
    merged_all_metrics = coastal_counties.copy()
    merged_all_metrics = merged_all_metrics.merge(
        per_event_analysis_maximum[['fips', 'max_damage_units', 'max_recovery_months']], 
        left_on='GEOID', right_on='fips', how='left'
    )
    merged_all_metrics = merged_all_metrics.drop(columns=['fips'])
    merged_all_metrics = merged_all_metrics.merge(
        capacity_df[['fips', 'construction_capacity']], 
        left_on='GEOID', right_on='fips', how='left'
    )
    merged_all_metrics = merged_all_metrics.drop(columns=['fips'])
    
    # Replace non-positive values with NaN for plotting
    merged_all_plot = merged_all_metrics.copy()
    for col in ['max_damage_units', 'construction_capacity', 'max_recovery_months']:
        merged_all_plot.loc[merged_all_plot[col] <= 0, col] = np.nan
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes = axes.flatten()
    
    metrics = [
        ('max_damage_units', 'cividis', 'Maximum Event Units Affected', '# units'),
        ('construction_capacity', 'Greens', 'Construction Capacity', 'permits/month'),
        ('max_recovery_months', 'Purples_r', 'Maximum Event Recovery Potential', 'months')
    ]
    
    subplot_labels = ['a', 'b', 'c']
    
    for idx, (ax, (metric, cmap, title, ylabel)) in enumerate(zip(axes, metrics)):
        data_positive = merged_all_plot[metric].dropna()
        
        if not data_positive.empty and len(data_positive) > 0:
            vmin = data_positive.min()
            vmax = data_positive.max()
            
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0 or vmax <= 0:
                norm = None
            else:
                log_vmin = vmin / 2
                norm = LogNorm(vmin=log_vmin, vmax=vmax)
        else:
            norm = None
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        merged_all_plot.plot(
            column=metric, cmap=cmap, norm=norm,
            linewidth=0.1, edgecolor="0.5",
            legend=True, ax=ax, cax=cax,
            missing_kwds={"color": "white", "label": "No data", "edgecolor": "0.5"}
        )
        
        ax.axis("off")
        ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left')
        
        if metric == 'max_recovery_months':
            cax.invert_yaxis()
            cax.yaxis.set_major_locator(NullLocator())
            cax.yaxis.set_minor_locator(NullLocator())
            cax.tick_params(which='both', left=False, right=False, labelleft=False)
            cax.text(1.5, 0.95, 'high', transform=cax.transAxes, 
                    fontsize=10, va='top', ha='left')
            cax.text(1.5, 0.05, 'low', transform=cax.transAxes, 
                    fontsize=10, va='bottom', ha='left')
            cax.set_ylabel(ylabel, fontsize=10)
        else:
            cax.set_ylabel(ylabel, fontsize=10)
            cax.tick_params(labelsize=10)
            cax.tick_params(which='minor', length=0)
        
        for spine in cax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/na_coast_3panel_maximum_event_damage_capacity_recovery_notitle.png", 
                dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved: na_coast_3panel_maximum_event_damage_capacity_recovery_notitle.png")


def create_driver_scatterplots(driver_analysis, per_event_analysis_median, 
                                corr_annual, corr_event, output_dir="../analysis_output"):
    """
    Create 2x2 scatterplot showing annual and per-event driver relationships.
    
    Args:
        driver_analysis: DataFrame with annual metrics
        per_event_analysis_median: DataFrame with per-event metrics
        corr_annual: Dict with annual correlations
        corr_event: Dict with per-event correlations
        output_dir: Directory to save outputs
    """
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogLocator
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    
    label_fs = 12
    tick_fs = 9
    cbar_label_fs = 11
    
    # Top left: EARP vs Risk (colored by capacity)
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(
        driver_analysis['total_ead'], 
        driver_analysis['earp_months_per_year'],
        c=driver_analysis['construction_capacity'],
        cmap='viridis', alpha=0.6, s=30, norm=LogNorm()
    )
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.invert_yaxis()
    ax1.set_xlabel('EAUA (# units)', fontsize=label_fs)
    ax1.set_ylabel('EARP (low–high)', fontsize=label_fs)
    ax1.grid(False)
    
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('CC (permits/month)', fontsize=cbar_label_fs)
    cbar1.ax.tick_params(which='both', labelsize=tick_fs)
    cbar1.ax.tick_params(which='minor', length=0)
    
    ax1.text(0.05, 0.02, f'r = {corr_annual["corr_risk"]:+.3f}\nn = {len(driver_analysis):,}', 
             transform=ax1.transAxes, fontsize=10, va='bottom')
    
    # Top right: EARP vs Capacity (colored by risk)
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(
        driver_analysis['construction_capacity'], 
        driver_analysis['earp_months_per_year'],
        c=driver_analysis['total_ead'],
        cmap='plasma', alpha=0.6, s=30, norm=LogNorm()
    )
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.invert_yaxis()
    ax2.set_xlabel('CC (permits/month)', fontsize=label_fs)
    ax2.set_ylabel('EARP (low–high)', fontsize=label_fs)
    ax2.grid(False)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('EAUA (# units)', fontsize=cbar_label_fs)
    cbar2.ax.tick_params(which='both', labelsize=tick_fs)
    cbar2.ax.tick_params(which='minor', length=0)
    
    ax2.text(0.68, 0.02, f'r = {corr_annual["corr_capacity"]:+.3f}\nn = {len(driver_analysis):,}', 
             transform=ax2.transAxes, fontsize=10, va='bottom')
    
    # Bottom left: Recovery vs Damage (colored by capacity)
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(
        per_event_analysis_median['median_damage_units'], 
        per_event_analysis_median['median_recovery_months'],
        c=per_event_analysis_median['construction_capacity'],
        cmap='viridis', alpha=0.6, s=30, norm=LogNorm()
    )
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.invert_yaxis()
    ax3.set_xlabel('MUA (# units)', fontsize=label_fs)
    ax3.set_ylabel('MRP (low–high)', fontsize=label_fs)
    ax3.grid(False)
    
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('CC (permits/month)', fontsize=cbar_label_fs)
    cbar3.ax.tick_params(which='both', labelsize=tick_fs)
    cbar3.ax.tick_params(which='minor', length=0)
    
    ax3.text(0.05, 0.02, f'r = {corr_event["corr_damage"]:+.3f}\nn = {len(per_event_analysis_median):,}', 
             transform=ax3.transAxes, fontsize=10, va='bottom')
    
    # Bottom right: Recovery vs Capacity (colored by damage)
    ax4 = axes[1, 1]
    scatter4 = ax4.scatter(
        per_event_analysis_median['construction_capacity'], 
        per_event_analysis_median['median_recovery_months'],
        c=per_event_analysis_median['median_damage_units'],
        cmap='plasma', alpha=0.6, s=30, norm=LogNorm()
    )
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.invert_yaxis()
    ax4.set_xlabel('CC (permits/month)', fontsize=label_fs)
    ax4.set_ylabel('MRP (low–high)', fontsize=label_fs)
    ax4.grid(False)
    
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label('MUA (# units)', fontsize=cbar_label_fs)
    cbar4.ax.tick_params(which='both', labelsize=tick_fs)
    cbar4.ax.tick_params(which='minor', length=0)
    
    ax4.text(0.68, 0.02, f'r = {corr_event["corr_capacity"]:+.3f}\nn = {len(per_event_analysis_median):,}', 
             transform=ax4.transAxes, fontsize=10, va='bottom')
    
    # Colorbar borders
    for cbar in [cbar1, cbar2, cbar3, cbar4]:
        for spine in cbar.ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
    
    # Panel borders
    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_edgecolor('0.4')
            spine.set_linewidth(0.8)
        ax.tick_params(color='0.4', labelcolor='0.2')
    
    # Panel labels
    panel_labels = ['a', 'b', 'c', 'd']
    for label, ax in zip(panel_labels, [ax1, ax2, ax3, ax4]):
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left')
    
    # Axis ticks
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.tick_params(axis='x', which='major', bottom=True, top=False,
                       labelbottom=True, labelsize=tick_fs)
        ax.tick_params(axis='y', which='both', left=False, right=False,
                       labelleft=True, labelsize=tick_fs)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/median_recovery_drivers_scatter.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved: median_recovery_drivers_scatter.png")


def create_variance_partitioning_plots(vp_annual, vp_event, output_dir="../analysis_output"):
    """
    Create variance partitioning comparison plots.
    
    Args:
        vp_annual: Dict with annual variance partitioning results
        vp_event: Dict with per-event variance partitioning results
        output_dir: Directory to save outputs
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#e41a1c', '#377eb8', '#984ea3', '#cccccc']
    
    # Left: Annual metrics
    ax1 = axes[0]
    variance_components_annual = [
        vp_annual['unique_var1'] * 100,
        vp_annual['unique_var2'] * 100,
        vp_annual['shared'] * 100,
        vp_annual['unexplained'] * 100
    ]
    
    labels = [
        f"Unique\nDamage\n({vp_annual['unique_var1']*100:.1f}%)",
        f"Unique\nCapacity\n({vp_annual['unique_var2']*100:.1f}%)",
        f"Shared\n(Coupled)\n({vp_annual['shared']*100:.1f}%)",
        f"Unexplained\n({vp_annual['unexplained']*100:.1f}%)"
    ]
    
    bars1 = ax1.bar(range(4), variance_components_annual, color=colors, 
                    edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Annual Metrics (EARP)\nVariance Partitioning', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(variance_components_annual) * 1.15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='black', linewidth=0.8)
    
    ax1.text(0.98, 0.98, f"Total R² = {vp_annual['r2_combined']:.3f}", 
             transform=ax1.transAxes, fontsize=11, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    # Right: Per-event metrics
    ax2 = axes[1]
    variance_components_event = [
        vp_event['unique_var1'] * 100,
        vp_event['unique_var2'] * 100,
        vp_event['shared'] * 100,
        vp_event['unexplained'] * 100
    ]
    
    labels = [
        f"Unique\nDamage\n({vp_event['unique_var1']*100:.1f}%)",
        f"Unique\nCapacity\n({vp_event['unique_var2']*100:.1f}%)",
        f"Shared\n(Coupled)\n({vp_event['shared']*100:.1f}%)",
        f"Unexplained\n({vp_event['unexplained']*100:.1f}%)"
    ]
    
    bars2 = ax2.bar(range(4), variance_components_event, color=colors, 
                    edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Event Metrics (Median Recovery)\nVariance Partitioning', 
                  fontsize=13, fontweight='bold')
    ax2.set_ylim(0, max(variance_components_event) * 1.15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='black', linewidth=0.8)
    
    ax2.text(0.98, 0.98, f"Total R² = {vp_event['r2_combined']:.3f}", 
             transform=ax2.transAxes, fontsize=11, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/variance_partitioning_annual_vs_event.png", 
                dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved: variance_partitioning_annual_vs_event.png")


# ============================================================================
# SECTION 9: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function to run complete analysis pipeline.
    
    Returns
    -------
    dict
        Dictionary containing all analysis results
    """
    print("="*80)
    print("HURRICANE RECOVERY POTENTIAL ANALYSIS")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Load all data
    # ========================================================================
    print("\n### STEP 1: LOADING DATA ###")
    counties, coastal_counties = load_county_boundaries()
    capacity_df = load_construction_capacity()
    ead_df, ead_wide, units_df = compute_expected_annual_damage()
    recovery_all_events = load_recovery_potential_data()
    earp_df = compute_expected_annual_recovery_potential(recovery_all_events)
    
    # ========================================================================
    # STEP 2: Prepare analysis datasets
    # ========================================================================
    print("\n### STEP 2: PREPARING ANALYSIS DATA ###")
    driver_analysis = prepare_annual_driver_analysis(earp_df, ead_wide, capacity_df)
    per_event_analysis = prepare_per_event_analysis(
        recovery_all_events, units_df, capacity_df
    )
    per_event_analysis_median = prepare_per_event_analysis_median(
        per_event_analysis
    )
    per_event_analysis_maximum = prepare_per_event_analysis_maximum(
        per_event_analysis
    )
    
    # ========================================================================
    # STEP 3: Correlation analysis
    # ========================================================================
    # print("\n### STEP 3: CORRELATION ANALYSIS ###")
    # corr_annual = compute_correlations_annual(driver_analysis)
    # corr_event = compute_correlations_per_event(per_event_analysis_median)
    
    # ========================================================================
    # STEP 4: Variance partitioning
    # ========================================================================
    # print("\n### STEP 4: VARIANCE PARTITIONING ###")
    # vp_annual, vp_event_median, vp_event_maximum = perform_variance_partitioning_analysis(
    #     driver_analysis, per_event_analysis_median, per_event_analysis_maximum
    # )
    
    # ========================================================================
    # STEP 5: County-level variance decomposition
    # ========================================================================
    # print("\n### STEP 5: COUNTY-LEVEL VARIANCE DECOMPOSITION ###")
    # per_event_analysis_median, model_event_median = compute_county_variance_contributions_per_event(
    #     per_event_analysis_median
    # )
    # per_event_analysis_maximum, model_event_maximum = compute_county_variance_contributions_per_event_maximum(
    #     per_event_analysis_maximum
    # )
    # driver_analysis, model_annual = compute_county_variance_contributions_annual(
    #     driver_analysis
    # )
    
    # ========================================================================
    # STEP 6: Spatial pattern analysis
    # ========================================================================
    # print("\n### STEP 6: SPATIAL PATTERN ANALYSIS ###")
    # comparison_data, gdf_change = analyze_spatial_patterns(
    #     driver_analysis, per_event_analysis_median, coastal_counties
    # )
    
    # ========================================================================
    # STEP 7: Create visualizations
    # ========================================================================
    print("\n### STEP 7: CREATING VISUALIZATIONS ###")
    
    # Create output directory if it doesn't exist
    output_dir = Path("../analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    # EAD damage state maps
    # create_ead_damage_state_maps(ead_wide, coastal_counties, str(output_dir))
    
    # EARP and capacity maps
    # create_earp_capacity_maps(earp_df, capacity_df, coastal_counties, str(output_dir))
    
    # Three-panel maps (EAD + Capacity + EARP)
    # create_three_panel_maps(ead_wide, capacity_df, earp_df, coastal_counties, str(output_dir))
    
    # Three-panel maps - Per-Event (Median Damage + Capacity + Median Recovery)
    # create_three_panel_maps_per_event(per_event_analysis_median, capacity_df, coastal_counties, str(output_dir))
    
    # Three-panel maps - Per-Event (Median, with titles instead of labels)
    # create_three_panel_maps_per_event_median_titled(per_event_analysis_median, capacity_df, coastal_counties, str(output_dir))
    
    # Three-panel maps - Per-Event (Maximum Damage + Capacity + Maximum Recovery)
    # create_three_panel_maps_per_event_maximum(per_event_analysis_maximum, capacity_df, coastal_counties, str(output_dir))
    
    # Driver scatterplots
    # create_driver_scatterplots(
    #     driver_analysis, per_event_analysis_median, 
    #     corr_annual, corr_event, str(output_dir)
    # )
    
    # Variance partitioning plots (using median for now - can be updated later)
    # create_variance_partitioning_plots(vp_annual, vp_event_median, str(output_dir))
    
    # Variance share maps (2-panel: annual vs median)
    # create_variance_share_maps(
    #     coastal_counties, driver_analysis, per_event_analysis_median,
    #     model_annual, model_event_median, str(output_dir)
    # )
    
    # Variance share maps (3-panel: annual, median, maximum)
    # create_variance_share_maps_three_panel(
    #     coastal_counties, driver_analysis, per_event_analysis_median, per_event_analysis_maximum,
    #     model_annual, model_event_median, model_event_maximum, str(output_dir)
    # )
    
    # Variance context indices (combines variance drivers with actual conditions)
    # create_variance_context_indices(
    #     coastal_counties, driver_analysis, per_event_analysis_median,
    #     capacity_df, ead_wide, earp_df, str(output_dir)
    # )
    
    # Per-event bottleneck indices
    # per_event_bottleneck_df = create_per_event_bottleneck_indices(
    #     coastal_counties, per_event_analysis_median, capacity_df, str(output_dir)
    # )
    
    # Intervention priority maps
    # create_intervention_priority_maps(
    #     driver_analysis, per_event_analysis_median, coastal_counties, str(output_dir)
    # )
    
    # County sensitivity metrics (new event-level analysis)
    sensitivity_df = compute_county_sensitivity_metrics(
        recovery_all_events, units_df, capacity_df, str(output_dir)
    )
    create_sensitivity_metrics_maps(
        sensitivity_df, coastal_counties, str(output_dir)
    )
    
    # Event response curves and typologies
    # typology_df = compute_event_response_curves(
    #     recovery_all_events, units_df, capacity_df, str(output_dir)
    # )
    
    # Print final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey findings:")
    # print(f"  • {len(driver_analysis)} counties analyzed (annual)")
    # print(f"  • {len(per_event_analysis_median)} counties analyzed (per-event median)")
    # print(f"  • {len(per_event_analysis_maximum)} counties analyzed (per-event maximum)")
    # print(f"  • Annual R²: {model_annual['r2']:.3f}")
    # print(f"  • Per-event Median R²: {model_event_median['r2']:.3f}")
    # print(f"  • Per-event Maximum R²: {model_event_maximum['r2']:.3f}")
    print(f"  • County sensitivity metrics computed for {len(sensitivity_df)} counties")
    
    print(f"\nVisualizations generated:")
    # print(f"  1. na_coast_ead_by_damage_state.png")
    # print(f"  2. na_coast_earp_metrics.png")
    # print(f"  3. na_coast_3panel_ead_capacity_recovery_notitle.png (Annual)")
    # print(f"  4. na_coast_3panel_median_event_damage_capacity_recovery_notitle.png")
    # print(f"  5. na_coast_3panel_maximum_event_damage_capacity_recovery_notitle.png")
    # print(f"  6. median_recovery_drivers_scatter.png")
    # print(f"  7. variance_partitioning_annual_vs_event.png")
    # print(f"  8. variance_share_annual_vs_event_maps.png (2-panel: annual vs median)")
    # print(f"  9. variance_share_three_panel_maps.png (3-panel: annual/median/maximum)")
    # print(f"  10. variance_share_annual_vs_event.png (distribution)")
    # print(f"  11. variance_context_indices.png (Annual bottleneck analysis)")
    # print(f"  12. per_event_bottleneck_indices.png (Median event bottleneck analysis)")
    # print(f"  13. top_bottleneck_counties_map.png (Spatial distribution of top 10)")
    # print(f"  14. top_bottleneck_counties_profiles.png (Damage/capacity/recovery profiles)")
    # print(f"  15. intervention_priority_maps.png (Strategic intervention priorities)")
    print(f"  1. county_sensitivity_metrics_maps.png (Per-county event-level analysis)")
    print(f"      - Panel 1: Recovery elasticity to damage")
    print(f"      - Panel 2: Capacity saturation factors (DS-weighted)")
    print(f"      - Panel 3: Damage state profiles")
    
    # print(f"\nVariance Partitioning Summary:")
    # print(f"  Annual (EARP):")
    # print(f"    - Unique damage:   {vp_annual['unique_var1']*100:.1f}%")
    # print(f"    - Unique capacity: {vp_annual['unique_var2']*100:.1f}%")
    # print(f"    - Shared:          {vp_annual['shared']*100:.1f}%")
    # print(f"    - Unexplained:     {vp_annual['unexplained']*100:.1f}%")
    # print(f"  Per-Event Median:")
    # print(f"    - Unique damage:   {vp_event_median['unique_var1']*100:.1f}%")
    # print(f"    - Unique capacity: {vp_event_median['unique_var2']*100:.1f}%")
    # print(f"    - Shared:          {vp_event_median['shared']*100:.1f}%")
    # print(f"    - Unexplained:     {vp_event_median['unexplained']*100:.1f}%")
    # print(f"  Per-Event Maximum:")
    # print(f"    - Unique damage:   {vp_event_maximum['unique_var1']*100:.1f}%")
    # print(f"    - Unique capacity: {vp_event_maximum['unique_var2']*100:.1f}%")
    # print(f"    - Shared:          {vp_event_maximum['shared']*100:.1f}%")
    # print(f"    - Unexplained:     {vp_event_maximum['unexplained']*100:.1f}%")
    
    return {
        'counties': counties,
        'coastal_counties': coastal_counties,
        'capacity_df': capacity_df,
        'ead_wide': ead_wide,
        'driver_analysis': driver_analysis,
        'per_event_analysis_median': per_event_analysis_median,
        'per_event_analysis_maximum': per_event_analysis_maximum,
        # 'corr_annual': corr_annual,
        # 'corr_event': corr_event,
        # 'vp_annual': vp_annual,
        # 'vp_event_median': vp_event_median,
        # 'vp_event_maximum': vp_event_maximum,
        # 'model_annual': model_annual,
        # 'model_event_median': model_event_median,
        # 'model_event_maximum': model_event_maximum,
        # 'comparison_data': comparison_data,
        # 'gdf_change': gdf_change,
        'recovery_all_events': recovery_all_events,
        'units_df': units_df,
        'sensitivity_df': sensitivity_df
    }


if __name__ == "__main__":
    results = main()
    
    # ========================================================================
    # OPTIONAL: SINGLE-EVENT ANALYSIS
    # ========================================================================
    # Uncomment to analyze specific events
    
    # List top events by impact
    print("\n\n" + "="*80)
    print("SINGLE-EVENT ANALYSIS TOOLS")
    print("="*80)
    event_summary = list_available_events(
        results['recovery_all_events'], 
        results['units_df'],
        top_n=30
    )
    
    # Extract event mapping from summary
    if event_summary is not None:
        event_mapping = dict(zip(event_summary['event'], event_summary['damage_event']))
        results['event_mapping'] = event_mapping
        results['event_summary'] = event_summary
    else:
        event_mapping = None
    
    # Screen events by driver profile to find interesting cases
    print("\n" + "="*80)
    print("STEP 16: Screen Events by Driver Profile")
    print("="*80)
    
    screening_results = screen_events_by_driver(
        results['recovery_all_events'],
        results['units_df'],
        results['capacity_df'],
        event_mapping,
        min_counties=30,
        top_n=10
    )
    results['screening_results'] = screening_results
    
    # Example: Analyze a specific event
    # Uncomment and modify event name as needed:
    
    # Choose an event - try a capacity-driven one!
    # capacity_driven = screening_results.nlargest(1, 'share_capacity')
    # event_to_analyze = capacity_driven.iloc[0]['event']
    
    event_to_analyze = '3936'  # Use event ID from recovery data
    event_analysis = analyze_single_event(
        event_to_analyze,
        results['recovery_all_events'],
        results['units_df'],
        results['capacity_df'],
        results['coastal_counties'],
        "../analysis_output",
        event_mapping=event_mapping
    )
    
    if event_analysis is not None:
        create_single_event_maps(
            event_to_analyze,
            event_analysis,
            results['coastal_counties'],
            "../analysis_output"
        )
    
    print("\n\nTo analyze a specific event, uncomment the example code above")
    print("and set event_to_analyze to the desired event name from the list.")
    print("\nTIP: Use screening_results to find capacity-driven events:")
    print("  capacity_driven = screening_results.nlargest(1, 'share_capacity')")
    print("  event_to_analyze = capacity_driven.iloc[0]['event']")

