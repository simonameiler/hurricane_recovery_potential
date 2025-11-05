#!/usr/bin/env python3
"""
Analyze and compare raw vs scaled impacts from per-event aggregated outputs.

Produces:
1. Summary statistics: total repair costs (raw vs scaled), scaling factors
2. Histograms: distribution of units affected per damage state
3. County-level maps: EAD per damage state

Usage:
    python scripts/analyze_impacts.py \
        --events-dir /path/to/by_event \
        --output-dir /path/to/analysis_output
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
from matplotlib.colors import LogNorm, Normalize
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr

# Set style
sns.set_style("whitegrid")
mpl.rcParams['figure.dpi'] = 150

DEFAULT_FREQ = 0.00067334  # events/year (same as compute_ead.py)


def load_all_events(events_dir: Path, subdir: str = "raw"):
    """Load and concatenate all per-event CSVs from raw or scaled subdirectory."""
    csv_dir = events_dir / subdir
    if not csv_dir.exists():
        raise FileNotFoundError(f"Directory not found: {csv_dir}")
    
    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    
    print(f"Loading {len(csv_files)} files from {csv_dir}...")
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        # infer event_name from filename if missing
        if "event_name" not in df.columns:
            event_name = f.stem.replace("_raw", "").replace("_scaled", "")
            df["event_name"] = event_name
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(combined)} rows, {combined['event_name'].nunique()} unique events")
    return combined


def compare_repair_costs(raw_df, scaled_df, output_dir: Path):
    """Compare total repair costs between raw and scaled."""
    print("\n=== Repair Cost Comparison ===")
    
    # Aggregate by event
    raw_by_event = raw_df.groupby("event_name")["repair_cost_sum_raw"].sum()
    scaled_by_event = scaled_df.groupby("event_name")["repair_cost_sum_scaled"].sum()
    
    comparison = pd.DataFrame({
        "raw_total": raw_by_event,
        "scaled_total": scaled_by_event,
    }).fillna(0)
    comparison["scaling_factor"] = comparison["scaled_total"] / comparison["raw_total"].replace(0, np.nan)
    comparison["absolute_change"] = comparison["scaled_total"] - comparison["raw_total"]
    comparison["percent_change"] = 100 * (comparison["scaled_total"] - comparison["raw_total"]) / comparison["raw_total"].replace(0, np.nan)
    
    # Summary stats
    print(f"\nTotal repair cost (raw):    ${comparison['raw_total'].sum():,.0f}")
    print(f"Total repair cost (scaled): ${comparison['scaled_total'].sum():,.0f}")
    print(f"Overall scaling factor:     {comparison['scaled_total'].sum() / comparison['raw_total'].sum():.3f}")
    print(f"\nPer-event scaling factor:")
    print(f"  Mean:   {comparison['scaling_factor'].mean():.3f}")
    print(f"  Median: {comparison['scaling_factor'].median():.3f}")
    print(f"  Std:    {comparison['scaling_factor'].std():.3f}")
    print(f"  Min:    {comparison['scaling_factor'].min():.3f}")
    print(f"  Max:    {comparison['scaling_factor'].max():.3f}")
    
    # Save summary
    summary_path = output_dir / "repair_cost_summary.csv"
    comparison.to_csv(summary_path)
    print(f"\nSaved summary to: {summary_path}")
    
    # Plot: raw vs scaled scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(comparison["raw_total"], comparison["scaled_total"], alpha=0.5, s=20)
    max_val = max(comparison["raw_total"].max(), comparison["scaled_total"].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label="1:1 line")
    ax.set_xlabel("Raw Repair Cost ($)")
    ax.set_ylabel("Scaled Repair Cost ($)")
    ax.set_title("Raw vs Scaled Repair Costs (per event)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Histogram of scaling factors
    ax = axes[1]
    ax.hist(comparison["scaling_factor"].dropna(), bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label="No scaling (1.0)")
    ax.axvline(comparison["scaling_factor"].median(), color='green', linestyle='--', linewidth=2, 
               label=f"Median ({comparison['scaling_factor'].median():.2f})")
    ax.set_xlabel("Scaling Factor (scaled / raw)")
    ax.set_ylabel("Number of Events")
    ax.set_title("Distribution of Per-Event Scaling Factors")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "repair_cost_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    plt.close()
    
    return comparison


def plot_ds_distributions(raw_df, scaled_df, output_dir: Path):
    """Plot histograms of units affected per damage state."""
    print("\n=== Damage State Distribution Analysis ===")
    
    # Melt dataframes to long format for DS columns
    ds_cols_raw = [c for c in raw_df.columns if c.startswith("units_DS") and c.endswith("_raw")]
    ds_cols_scaled = [c for c in scaled_df.columns if c.startswith("units_DS") and c.endswith("_scaled")]
    
    # Aggregate across all events and counties
    raw_totals = {col.replace("units_", "").replace("_raw", ""): raw_df[col].sum() for col in ds_cols_raw}
    scaled_totals = {col.replace("units_", "").replace("_scaled", ""): scaled_df[col].sum() for col in ds_cols_scaled}
    
    print("\nTotal units affected per DS:")
    for ds in ["DS1", "DS2", "DS3", "DS4"]:
        raw_val = raw_totals.get(ds, 0)
        scaled_val = scaled_totals.get(ds, 0)
        change = scaled_val - raw_val
        pct = 100 * change / raw_val if raw_val > 0 else 0
        print(f"  {ds}: raw={raw_val:,}, scaled={scaled_val:,}, change={change:+,} ({pct:+.1f}%)")
    
    # Create bar plot comparing raw vs scaled totals
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ds_labels = ["DS1: Slight", "DS2: Moderate", "DS3: Extensive", "DS4: Complete"]
    ds_keys = ["DS1", "DS2", "DS3", "DS4"]
    x = np.arange(len(ds_keys))
    width = 0.35
    
    raw_vals = [raw_totals.get(ds, 0) for ds in ds_keys]
    scaled_vals = [scaled_totals.get(ds, 0) for ds in ds_keys]
    
    ax.bar(x - width/2, raw_vals, width, label='Raw', alpha=0.8, edgecolor='black')
    ax.bar(x + width/2, scaled_vals, width, label='Scaled', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Damage State')
    ax.set_ylabel('Total Units Affected')
    ax.set_title('Total Units Affected per Damage State (All Events & Counties)')
    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "ds_distribution_totals.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    plt.close()
    
    # Histogram: per-county per-event distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, ds in enumerate(ds_keys):
        ax = axes[i]
        raw_col = f"units_{ds}_raw"
        scaled_col = f"units_{ds}_scaled"
        
        if raw_col in raw_df.columns and scaled_col in scaled_df.columns:
            # Filter to non-zero values for better visualization
            raw_nonzero = raw_df[raw_df[raw_col] > 0][raw_col]
            scaled_nonzero = scaled_df[scaled_df[scaled_col] > 0][scaled_col]
            
            bins = np.logspace(0, np.log10(max(raw_nonzero.max(), scaled_nonzero.max()) + 1), 50)
            ax.hist(raw_nonzero, bins=bins, alpha=0.5, label='Raw', edgecolor='black')
            ax.hist(scaled_nonzero, bins=bins, alpha=0.5, label='Scaled', edgecolor='black')
            ax.set_xscale('log')
            ax.set_xlabel('Units Affected (log scale)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{ds_labels[i]}')
            ax.legend()
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "ds_distribution_histograms.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    plt.close()


def compute_and_map_ead_per_ds(raw_df, scaled_df, output_dir: Path, shapefile_path: Path, freq: float = DEFAULT_FREQ):
    """Compute EAD per damage state per county and create maps."""
    print("\n=== Computing County-Level EAD per Damage State ===")
    
    # For simplicity, assume uniform frequency across events (or you can load from hazard)
    print(f"Using frequency: {freq} events/year")
    
    ds_keys = ["DS1", "DS2", "DS3", "DS4"]
    ds_labels = ["DS1: Slight", "DS2: Moderate", "DS3: Extensive", "DS4: Complete"]
    
    # Compute EAD per DS per county
    def compute_ead(df, suffix):
        results = []
        for ds in ds_keys:
            col = f"units_{ds}_{suffix}"
            if col not in df.columns:
                continue
            # Group by fips and sum units, then multiply by frequency
            ead_df = df.groupby("fips")[col].sum().reset_index()
            ead_df["fips"] = ead_df["fips"].astype(str).str.zfill(5)  # Ensure fips is string with zero-padding
            ead_df["ead"] = ead_df[col] * freq
            ead_df["DS"] = ds
            ead_df["type"] = suffix
            results.append(ead_df[["fips", "DS", "type", "ead"]])
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    ead_raw = compute_ead(raw_df, "raw")
    ead_scaled = compute_ead(scaled_df, "scaled")
    
    # Load county shapefile
    if not shapefile_path.exists():
        print(f"Warning: Shapefile not found at {shapefile_path}. Skipping maps.")
        return
    
    counties = gpd.read_file(shapefile_path)
    
    # Create fips column from STATEFP and COUNTYFP
    if "STATEFP" in counties.columns and "COUNTYFP" in counties.columns:
        counties["fips"] = (counties["STATEFP"].astype(str) + counties["COUNTYFP"].astype(str)).str.zfill(5)
    elif "FIPS" in counties.columns:
        counties["fips"] = counties["FIPS"].astype(str).str.zfill(5)
    elif "fips" in counties.columns:
        counties["fips"] = counties["fips"].astype(str).str.zfill(5)
    else:
        print("Warning: Cannot create FIPS column from shapefile. Cannot create maps.")
        return
    
    # Filter to only states in our study area
    study_states = ['01', '09', '10', '12', '13', '22', '23', '24', '25', '28', 
                    '33', '34', '36', '37', '42', '44', '45', '48', '51']
    counties = counties[counties["STATEFP"].isin(study_states)].copy()
    print(f"Filtered to {len(counties)} counties in study area ({len(study_states)} states)")
    
    # Merge EAD data with geometries and plot with log scale
    for ds_idx, ds in enumerate(ds_keys):
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        for ax, ead_df, label in zip(axes, [ead_raw, ead_scaled], ["Raw", "Scaled"]):
            ds_ead = ead_df[ead_df["DS"] == ds].copy()
            merged = counties.merge(ds_ead, on="fips", how="left").fillna(0)
            
            # Use log scale, handling zeros by adding small value
            merged['ead_log'] = merged['ead'].replace(0, np.nan)
            vmin = merged['ead_log'].min()
            vmax = merged['ead_log'].max()
            
            # Plot with log scale
            merged.plot(column="ead_log", ax=ax, legend=True, cmap="YlOrRd",
                       norm=LogNorm(vmin=max(vmin, 0.01), vmax=vmax) if vmax > 0 else None,
                       legend_kwds={'label': f"EAD (units/year, log scale)", 'shrink': 0.8},
                       edgecolor='black', linewidth=0.1, missing_kwds={'color': 'lightgrey'})
            ax.set_title(f"{ds_labels[ds_idx]} - {label}")
            ax.axis('off')
        
        plt.tight_layout()
        plot_path = output_dir / f"ead_map_{ds}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved EAD map for {ds}: {plot_path}")
        plt.close()
    
    # Save EAD summary CSVs
    ead_raw.to_csv(output_dir / "ead_per_county_per_ds_raw.csv", index=False)
    ead_scaled.to_csv(output_dir / "ead_per_county_per_ds_scaled.csv", index=False)
    print(f"\nSaved EAD summary CSVs to {output_dir}")


def map_total_repair_cost_ead(raw_df, scaled_df, output_dir: Path, shapefile_path: Path, freq: float = DEFAULT_FREQ):
    """Create maps of total repair cost EAD per county."""
    print("\n=== Mapping Total Repair Cost EAD per County ===")
    
    # Compute total repair cost EAD per county
    raw_ead = raw_df.groupby('fips')['repair_cost_sum_raw'].sum().reset_index()
    raw_ead['fips'] = raw_ead['fips'].astype(str).str.zfill(5)
    raw_ead['ead'] = raw_ead['repair_cost_sum_raw'] * freq
    
    scaled_ead = scaled_df.groupby('fips')['repair_cost_sum_scaled'].sum().reset_index()
    scaled_ead['fips'] = scaled_ead['fips'].astype(str).str.zfill(5)
    scaled_ead['ead'] = scaled_ead['repair_cost_sum_scaled'] * freq
    
    # Load and filter shapefile
    if not shapefile_path.exists():
        print(f"Warning: Shapefile not found at {shapefile_path}. Skipping maps.")
        return
    
    counties = gpd.read_file(shapefile_path)
    
    if "STATEFP" in counties.columns and "COUNTYFP" in counties.columns:
        counties["fips"] = (counties["STATEFP"].astype(str) + counties["COUNTYFP"].astype(str)).str.zfill(5)
    else:
        print("Warning: Cannot create FIPS column from shapefile.")
        return
    
    # Filter to study states
    study_states = ['01', '09', '10', '12', '13', '22', '23', '24', '25', '28', 
                    '33', '34', '36', '37', '42', '44', '45', '48', '51']
    counties = counties[counties["STATEFP"].isin(study_states)].copy()
    
    # Create side-by-side maps
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    for ax, ead_df, label in zip(axes, [raw_ead, scaled_ead], ["Raw", "Scaled"]):
        merged = counties.merge(ead_df[['fips', 'ead']], on="fips", how="left").fillna(0)
        
        # Use log scale
        merged['ead_log'] = merged['ead'].replace(0, np.nan)
        vmin = merged['ead_log'].min()
        vmax = merged['ead_log'].max()
        
        if vmax > 0:
            merged.plot(column="ead_log", ax=ax, legend=True, cmap="YlOrRd",
                       norm=LogNorm(vmin=max(vmin, 1), vmax=vmax),
                       legend_kwds={'label': f"Repair Cost EAD ($/year, log scale)", 'shrink': 0.8},
                       edgecolor='black', linewidth=0.1, missing_kwds={'color': 'lightgrey'})
        ax.set_title(f"Total Repair Cost EAD - {label}")
        ax.axis('off')
    
    plt.tight_layout()
    plot_path = output_dir / "ead_map_total_repair_cost.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved total repair cost EAD map to: {plot_path}")
    plt.close()


def compare_with_aal_ncep(raw_df, scaled_df, output_dir: Path, aal_path: Path, county_mapping_path: Path, freq: float = DEFAULT_FREQ):
    """Compare simulation EAD with AAL from ncep_reanal.mat file."""
    print("\n=== Comparing Simulation EAD with AAL (ncep_reanal) ===")
    
    # Load AAL data
    if not aal_path.exists():
        print(f"Warning: AAL file not found at {aal_path}. Skipping comparison.")
        return
    
    aal_data = loadmat(aal_path)
    aal_values = aal_data['AAL'].flatten()  # shape (3220,)
    aal_total = aal_data['AAL_tot'][0, 0]
    
    print(f"AAL total from .mat file: ${aal_total:,.0f}")
    print(f"AAL has {len(aal_values)} county values")
    
    # Load county mapping to get fips for each county_index
    county_map = pd.read_csv(county_mapping_path)
    county_map['county_index'] = county_map.index  # Ensure index is explicit
    county_map['fips'] = county_map['fips'].astype(str).str.zfill(5)
    
    # Create AAL dataframe with fips
    aal_df = pd.DataFrame({
        'county_index': range(len(aal_values)),
        'aal_ncep': aal_values
    })
    aal_df = aal_df.merge(county_map[['county_index', 'fips']], on='county_index', how='left')
    
    # Compute total EAD from simulation (sum across all DS)
    # For raw
    raw_ead_total = raw_df.groupby('fips').agg({
        'units_DS1_raw': 'sum',
        'units_DS2_raw': 'sum',
        'units_DS3_raw': 'sum',
        'units_DS4_raw': 'sum',
        'repair_cost_sum_raw': 'sum'
    }).reset_index()
    raw_ead_total['fips'] = raw_ead_total['fips'].astype(str).str.zfill(5)
    raw_ead_total['total_units_ead'] = (
        raw_ead_total['units_DS1_raw'] + 
        raw_ead_total['units_DS2_raw'] + 
        raw_ead_total['units_DS3_raw'] + 
        raw_ead_total['units_DS4_raw']
    ) * freq
    raw_ead_total['repair_cost_ead'] = raw_ead_total['repair_cost_sum_raw'] * freq
    
    # For scaled
    scaled_ead_total = scaled_df.groupby('fips').agg({
        'units_DS1_scaled': 'sum',
        'units_DS2_scaled': 'sum',
        'units_DS3_scaled': 'sum',
        'units_DS4_scaled': 'sum',
        'repair_cost_sum_scaled': 'sum'
    }).reset_index()
    scaled_ead_total['fips'] = scaled_ead_total['fips'].astype(str).str.zfill(5)
    scaled_ead_total['total_units_ead'] = (
        scaled_ead_total['units_DS1_scaled'] + 
        scaled_ead_total['units_DS2_scaled'] + 
        scaled_ead_total['units_DS3_scaled'] + 
        scaled_ead_total['units_DS4_scaled']
    ) * freq
    scaled_ead_total['repair_cost_ead'] = scaled_ead_total['repair_cost_sum_scaled'] * freq
    
    # Merge with AAL
    comparison = aal_df.merge(
        raw_ead_total[['fips', 'total_units_ead', 'repair_cost_ead']], 
        on='fips', 
        how='left',
        suffixes=('', '_raw')
    ).rename(columns={'total_units_ead': 'sim_units_ead_raw', 'repair_cost_ead': 'sim_repair_ead_raw'})
    
    comparison = comparison.merge(
        scaled_ead_total[['fips', 'total_units_ead', 'repair_cost_ead']], 
        on='fips', 
        how='left'
    ).rename(columns={'total_units_ead': 'sim_units_ead_scaled', 'repair_cost_ead': 'sim_repair_ead_scaled'})
    
    # Fill NaN with 0 for counties with no simulation data
    comparison = comparison.fillna(0)
    
    # Summary statistics
    print(f"\nSimulation total repair cost EAD (raw):    ${comparison['sim_repair_ead_raw'].sum():,.0f}")
    print(f"Simulation total repair cost EAD (scaled): ${comparison['sim_repair_ead_scaled'].sum():,.0f}")
    print(f"AAL ncep_reanal total:                      ${aal_total:,.0f}")
    print(f"\nRatio (sim_raw / AAL):    {comparison['sim_repair_ead_raw'].sum() / aal_total:.3f}")
    print(f"Ratio (sim_scaled / AAL): {comparison['sim_repair_ead_scaled'].sum() / aal_total:.3f}")
    
    # Correlation analysis (only for counties with non-zero values in both)
    mask_raw = (comparison['aal_ncep'] > 0) & (comparison['sim_repair_ead_raw'] > 0)
    mask_scaled = (comparison['aal_ncep'] > 0) & (comparison['sim_repair_ead_scaled'] > 0)
    
    if mask_raw.sum() > 0:
        pearson_r_raw, pearson_p_raw = pearsonr(
            np.log10(comparison.loc[mask_raw, 'aal_ncep']), 
            np.log10(comparison.loc[mask_raw, 'sim_repair_ead_raw'])
        )
        spearman_r_raw, spearman_p_raw = spearmanr(
            comparison.loc[mask_raw, 'aal_ncep'], 
            comparison.loc[mask_raw, 'sim_repair_ead_raw']
        )
        print(f"\nCorrelation (Raw vs AAL, n={mask_raw.sum()} counties with both >0):")
        print(f"  Pearson r (log-log):  {pearson_r_raw:.3f} (p={pearson_p_raw:.3e})")
        print(f"  Spearman r:           {spearman_r_raw:.3f} (p={spearman_p_raw:.3e})")
    
    if mask_scaled.sum() > 0:
        pearson_r_scaled, pearson_p_scaled = pearsonr(
            np.log10(comparison.loc[mask_scaled, 'aal_ncep']), 
            np.log10(comparison.loc[mask_scaled, 'sim_repair_ead_scaled'])
        )
        spearman_r_scaled, spearman_p_scaled = spearmanr(
            comparison.loc[mask_scaled, 'aal_ncep'], 
            comparison.loc[mask_scaled, 'sim_repair_ead_scaled']
        )
        print(f"\nCorrelation (Scaled vs AAL, n={mask_scaled.sum()} counties with both >0):")
        print(f"  Pearson r (log-log):  {pearson_r_scaled:.3f} (p={pearson_p_scaled:.3e})")
        print(f"  Spearman r:           {spearman_r_scaled:.3f} (p={spearman_p_scaled:.3e})")
    
    # Save comparison data
    comparison.to_csv(output_dir / "aal_comparison.csv", index=False)
    print(f"\nSaved comparison data to: {output_dir / 'aal_comparison.csv'}")
    
    # Analyze systematic bias in the scatter plot
    print("\n=== Analyzing Systematic Bias ===")
    mask = (comparison['aal_ncep'] > 0) & (comparison['sim_repair_ead_scaled'] > 0)
    if mask.sum() > 0:
        # Compute ratio (simulation / AAL) as function of AAL magnitude
        comparison.loc[mask, 'ratio_scaled'] = comparison.loc[mask, 'sim_repair_ead_scaled'] / comparison.loc[mask, 'aal_ncep']
        
        # Bin by AAL magnitude
        comparison.loc[mask, 'aal_log_bin'] = pd.cut(
            np.log10(comparison.loc[mask, 'aal_ncep']), 
            bins=10, 
            labels=False
        )
        
        bias_stats = comparison[mask].groupby('aal_log_bin')['ratio_scaled'].agg(['mean', 'median', 'count'])
        print("\nRatio (Simulation/AAL) by AAL magnitude bins:")
        print(bias_stats)
        print("\nObservation: Lower AAL values tend to have ratio > 1 (sim overestimates),")
        print("while higher AAL values may have ratio < 1 (sim underestimates).")
        print("This could suggest:")
        print("  - AAL has a minimum threshold/cutoff for small impacts")
        print("  - Simulation captures more small events")
        print("  - Different impact functions for extreme events")
    
    # Create scatter plots with additional analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Top row: scatter plots (as before)
    for ax, sim_col, title in zip(axes[0], 
                                   ['sim_repair_ead_raw', 'sim_repair_ead_scaled'],
                                   ['Raw Simulation', 'Scaled Simulation']):
        mask = (comparison['aal_ncep'] > 0) & (comparison[sim_col] > 0)
        if mask.sum() > 0:
            ax.scatter(comparison.loc[mask, 'aal_ncep'], 
                      comparison.loc[mask, sim_col], 
                      alpha=0.5, s=30, edgecolor='black', linewidth=0.5)
            
            # Add 1:1 line
            min_val = min(comparison.loc[mask, 'aal_ncep'].min(), comparison.loc[mask, sim_col].min())
            max_val = max(comparison.loc[mask, 'aal_ncep'].max(), comparison.loc[mask, sim_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=2, label='1:1 line')
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('AAL ncep_reanal ($/year)')
            ax.set_ylabel(f'Simulation EAD ($/year)')
            ax.set_title(f'{title} vs AAL\n(n={mask.sum()} counties)')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    # Bottom row: ratio analysis
    for ax, sim_col, title in zip(axes[1],
                                   ['sim_repair_ead_raw', 'sim_repair_ead_scaled'],
                                   ['Raw', 'Scaled']):
        mask = (comparison['aal_ncep'] > 0) & (comparison[sim_col] > 0)
        if mask.sum() > 0:
            ratio = comparison.loc[mask, sim_col] / comparison.loc[mask, 'aal_ncep']
            ax.scatter(comparison.loc[mask, 'aal_ncep'], ratio,
                      alpha=0.5, s=30, edgecolor='black', linewidth=0.5)
            ax.axhline(y=1, color='r', linestyle='--', linewidth=2, label='1:1 (perfect match)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('AAL ncep_reanal ($/year)')
            ax.set_ylabel('Ratio (Simulation/AAL)')
            ax.set_title(f'{title}: Ratio vs AAL Magnitude')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    plot_path = output_dir / "aal_comparison_scatter.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved scatter plot to: {plot_path}")
    plt.close()
    
    # Create normalized spatial comparison maps
    print("\nCreating normalized spatial comparison maps...")
    create_normalized_spatial_comparison(comparison, output_dir)
    
    return comparison


def create_normalized_spatial_comparison(comparison, output_dir: Path):
    """Create maps comparing normalized AAL and EAD to focus on spatial patterns."""
    print("\n=== Creating Normalized Spatial Comparison ===")
    
    # Load shapefile
    shapefile_path = Path("data/US_counties.shp")
    if not shapefile_path.exists():
        print(f"Warning: Shapefile not found. Skipping normalized maps.")
        return
    
    counties = gpd.read_file(shapefile_path)
    if "STATEFP" in counties.columns and "COUNTYFP" in counties.columns:
        counties["fips"] = (counties["STATEFP"].astype(str) + counties["COUNTYFP"].astype(str)).str.zfill(5)
    
    # Filter to study states
    study_states = ['01', '09', '10', '12', '13', '22', '23', '24', '25', '28', 
                    '33', '34', '36', '37', '42', '44', '45', '48', '51']
    counties = counties[counties["STATEFP"].isin(study_states)].copy()
    
    # Normalize values (0-1 scale within each dataset)
    comparison['aal_normalized'] = comparison['aal_ncep'] / comparison['aal_ncep'].max()
    comparison['sim_raw_normalized'] = comparison['sim_repair_ead_raw'] / comparison['sim_repair_ead_raw'].max()
    comparison['sim_scaled_normalized'] = comparison['sim_repair_ead_scaled'] / comparison['sim_repair_ead_scaled'].max()
    
    # Compute difference in normalized values
    comparison['diff_raw'] = comparison['sim_raw_normalized'] - comparison['aal_normalized']
    comparison['diff_scaled'] = comparison['sim_scaled_normalized'] - comparison['aal_normalized']
    
    # Create 3-panel figure: AAL, Sim Scaled, Difference
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    # Determine common color scale for first two panels
    vmin_common = min(
        comparison['aal_normalized'][comparison['aal_normalized'] > 0].min(),
        comparison['sim_scaled_normalized'][comparison['sim_scaled_normalized'] > 0].min()
    )
    vmax_common = 1.0  # Both are normalized to max of 1
    
    # Panel 1: AAL normalized
    merged = counties.merge(comparison[['fips', 'aal_normalized']], on='fips', how='left').fillna(0)
    merged['aal_normalized_nz'] = merged['aal_normalized'].replace(0, np.nan)
    if merged['aal_normalized_nz'].max() > 0:
        merged.plot(column='aal_normalized_nz', ax=axes[0], legend=True, cmap='YlOrRd',
                   norm=LogNorm(vmin=vmin_common, vmax=vmax_common),
                   legend_kwds={'label': 'Normalized value (log scale)', 'shrink': 0.8},
                   edgecolor='black', linewidth=0.1, missing_kwds={'color': 'lightgrey'})
    axes[0].set_title('AAL ncep_reanal (Normalized)')
    axes[0].axis('off')
    
    # Panel 2: Simulation scaled normalized
    merged = counties.merge(comparison[['fips', 'sim_scaled_normalized']], on='fips', how='left').fillna(0)
    merged['sim_scaled_normalized_nz'] = merged['sim_scaled_normalized'].replace(0, np.nan)
    if merged['sim_scaled_normalized_nz'].max() > 0:
        merged.plot(column='sim_scaled_normalized_nz', ax=axes[1], legend=True, cmap='YlOrRd',
                   norm=LogNorm(vmin=vmin_common, vmax=vmax_common),
                   legend_kwds={'label': 'Normalized value (log scale)', 'shrink': 0.8},
                   edgecolor='black', linewidth=0.1, missing_kwds={'color': 'lightgrey'})
    axes[1].set_title('Simulation EAD Scaled (Normalized)')
    axes[1].axis('off')
    
    # Panel 3: Difference (Sim - AAL)
    merged = counties.merge(comparison[['fips', 'diff_scaled']], on='fips', how='left').fillna(0)
    vmax_diff = max(abs(merged['diff_scaled'].min()), abs(merged['diff_scaled'].max()))
    merged.plot(column='diff_scaled', ax=axes[2], legend=True, cmap='RdBu_r',
               vmin=-vmax_diff, vmax=vmax_diff,
               legend_kwds={'label': 'Difference (Sim - AAL)', 'shrink': 0.8},
               edgecolor='black', linewidth=0.1, missing_kwds={'color': 'lightgrey'})
    axes[2].set_title('Difference (Simulation - AAL)\nRed: Sim > AAL, Blue: AAL > Sim')
    axes[2].axis('off')
    
    plt.tight_layout()
    plot_path = output_dir / "normalized_spatial_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved normalized spatial comparison to: {plot_path}")
    plt.close()
    
    # Compute spatial correlation on normalized values
    mask = (comparison['aal_normalized'] > 0) & (comparison['sim_scaled_normalized'] > 0)
    if mask.sum() > 0:
        pearson_r, pearson_p = pearsonr(
            comparison.loc[mask, 'aal_normalized'],
            comparison.loc[mask, 'sim_scaled_normalized']
        )
        spearman_r, spearman_p = spearmanr(
            comparison.loc[mask, 'aal_normalized'],
            comparison.loc[mask, 'sim_scaled_normalized']
        )
        print(f"\nCorrelation on normalized values (n={mask.sum()}):")
        print(f"  Pearson r:  {pearson_r:.3f} (p={pearson_p:.3e})")
        print(f"  Spearman r: {spearman_r:.3f} (p={spearman_p:.3e})")


def main():
    parser = argparse.ArgumentParser(description="Analyze and compare raw vs scaled impacts")
    parser.add_argument("--events-dir", type=Path, 
                       default="impacts_out/by_event",
                       help="Directory containing raw/ and scaled/ subdirectories")
    parser.add_argument("--output-dir", type=Path, default="./analysis_output",
                       help="Output directory for plots and summaries")
    parser.add_argument("--shapefile", type=Path,
                       default="data/US_counties.shp",
                       help="Path to US counties shapefile for mapping")
    parser.add_argument("--frequency", type=float, default=DEFAULT_FREQ,
                       help="Event frequency (events/year) for EAD calculation")
    parser.add_argument("--aal-file", type=Path, default="impacts_out/AAL_ncep_reanal.mat",
                       help="Path to AAL .mat file for comparison")
    parser.add_argument("--county-mapping", type=Path, default="data/county_region.csv",
                       help="Path to county mapping CSV with county_index and fips")
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading raw and scaled event data...")
    raw_df = load_all_events(args.events_dir, "raw")
    scaled_df = load_all_events(args.events_dir, "scaled")
    
    # 1. Compare repair costs
    compare_repair_costs(raw_df, scaled_df, args.output_dir)
    
    # 2. Plot DS distributions
    plot_ds_distributions(raw_df, scaled_df, args.output_dir)
    
    # 3. Compute and map EAD per DS
    compute_and_map_ead_per_ds(raw_df, scaled_df, args.output_dir, args.shapefile, args.frequency)
    
    # 4. Map total repair cost EAD
    map_total_repair_cost_ead(raw_df, scaled_df, args.output_dir, args.shapefile, args.frequency)
    
    # 5. Compare with AAL ncep_reanal
    compare_with_aal_ncep(raw_df, scaled_df, args.output_dir, args.aal_file, args.county_mapping, args.frequency)
    
    print(f"\n=== Analysis complete! ===")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
