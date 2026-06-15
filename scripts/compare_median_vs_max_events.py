#!/usr/bin/env python3
"""
Compare Median vs Maximum Event Impacts per County

Creates two 3-panel maps comparing:
- Median event: typical weighted damage, capacity, recovery time
- Max event: maximum weighted damage event, capacity, recovery time for that event

Author: Simona Meiler
Date: January 2026
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import json

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

# Configuration
RECOVERY_WEIGHTS = {
    'DS1': 1.0,
    'DS2': 1.0,
    'DS3': 3.0,
    'DS4': 6.0
}

COASTAL_STATE_FIPS = ['01', '09', '10', '12', '13', '22', '23', '24', '25', '28',
                      '33', '34', '36', '37', '42', '44', '45', '48', '51']

DEFAULT_FREQ = 0.00067334  # events/year


def load_event_data():
    """Load all per-event impact data."""
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    print("\n1. Loading per-event impact data...")
    
    by_event_dir = BASE_DIR / "impacts_out" / "by_event" / "scaled"
    event_files = sorted(by_event_dir.glob("*_scaled.csv"))
    
    print(f"   Found {len(event_files)} event files")
    
    all_events = []
    for f in event_files:
        df = pd.read_csv(f)
        df['event_name'] = f.stem.replace('_scaled', '')
        all_events.append(df)
    
    events_df = pd.concat(all_events, ignore_index=True)
    events_df['fips'] = events_df['fips'].astype(str).str.zfill(5)
    
    print(f"   Loaded {len(events_df)} county-event records")
    print(f"   Unique events: {events_df['event_name'].nunique()}")
    print(f"   Unique counties: {events_df['fips'].nunique()}")
    
    return events_df


def load_recovery_data():
    """Load per-event recovery potential data."""
    print("\n2. Loading recovery potential data...")
    
    recovery_dir = BASE_DIR / "data" / "recovery_potential_per_scenario"
    recovery_files = list(recovery_dir.glob("*_scaled_recovery_potential.json"))
    
    print(f"   Found {len(recovery_files)} recovery files")
    
    all_recovery = []
    for idx, f in enumerate(recovery_files):
        if (idx + 1) % 500 == 0:
            print(f"   Loaded {idx + 1}/{len(recovery_files)} files...")
        
        with open(f, 'r') as file:
            data = json.load(file)
            df = pd.DataFrame(data)
            all_recovery.append(df)
    
    recovery_df = pd.concat(all_recovery, ignore_index=True)
    recovery_df['fips'] = recovery_df['fips'].astype(str).str.zfill(5)
    recovery_df['recovery_potential [months]'] = (
        recovery_df['recovery_potential [months]'].replace([np.inf, -np.inf], np.nan)
    )
    
    print(f"   Loaded {len(recovery_df)} recovery records")
    
    return recovery_df


def load_capacity_data():
    """Load construction capacity data."""
    print("\n3. Loading construction capacity data...")
    
    permits_file = BASE_DIR / "data" / "selected_states_counties_with_permits.csv"
    permits_df = pd.read_csv(permits_file)
    permits_df['fips'] = permits_df['FIPS'].astype(str).str.zfill(5)
    
    capacity_df = permits_df[['fips', 'Average_Building_Permits(12 months)']].copy()
    capacity_df.columns = ['fips', 'construction_capacity']
    
    print(f"   Loaded capacity for {len(capacity_df)} counties")
    print(f"   Range: {capacity_df['construction_capacity'].min():.1f} - {capacity_df['construction_capacity'].max():.1f} permits/month")
    
    return capacity_df


def load_spatial_data():
    """Load county spatial boundaries."""
    print("\n4. Loading spatial data...")
    
    counties = gpd.read_file(BASE_DIR / "data" / "US_counties.shp")
    
    # Ensure GEOID exists
    if 'GEOID' not in counties.columns:
        if 'STATEFP' in counties.columns and 'COUNTYFP' in counties.columns:
            counties['GEOID'] = counties['STATEFP'] + counties['COUNTYFP']
        elif 'STATE_FIPS' in counties.columns and 'CNTY_FIPS' in counties.columns:
            counties['GEOID'] = counties['STATE_FIPS'].astype(str).str.zfill(2) + counties['CNTY_FIPS'].astype(str).str.zfill(3)
    
    # Filter to coastal states
    if 'STATEFP' in counties.columns:
        state_col = 'STATEFP'
    elif 'STATE_FIPS' in counties.columns:
        counties['STATEFP'] = counties['STATE_FIPS'].astype(str).str.zfill(2)
        state_col = 'STATEFP'
    else:
        # Extract from GEOID
        counties['STATEFP'] = counties['GEOID'].astype(str).str[:2]
        state_col = 'STATEFP'
    
    coastal_counties = counties[counties[state_col].isin(COASTAL_STATE_FIPS)].copy()
    print(f"   Loaded {len(coastal_counties)} coastal counties")
    
    return coastal_counties


def compute_median_event_metrics(events_df, recovery_df, capacity_df):
    """Compute median event metrics per county."""
    print("\n" + "="*80)
    print("COMPUTING MEDIAN EVENT METRICS")
    print("="*80)
    
    # Calculate weighted damage per event-county
    events_df['weighted_damage'] = (
        events_df['units_DS1_scaled'] * RECOVERY_WEIGHTS['DS1'] +
        events_df['units_DS2_scaled'] * RECOVERY_WEIGHTS['DS2'] +
        events_df['units_DS3_scaled'] * RECOVERY_WEIGHTS['DS3'] +
        events_df['units_DS4_scaled'] * RECOVERY_WEIGHTS['DS4']
    )
    
    # Compute median weighted damage per county
    median_damage = events_df.groupby('fips')['weighted_damage'].median().reset_index()
    median_damage.columns = ['fips', 'median_weighted_damage']
    
    # Compute median recovery time per county
    median_recovery = recovery_df.groupby('fips')['recovery_potential [months]'].median().reset_index()
    median_recovery.columns = ['fips', 'median_recovery_months']
    
    # Merge with capacity
    median_metrics = median_damage.merge(median_recovery, on='fips', how='inner')
    median_metrics = median_metrics.merge(capacity_df, on='fips', how='inner')
    
    # Filter valid data
    median_metrics = median_metrics[
        (median_metrics['median_weighted_damage'] > 0) &
        (median_metrics['median_recovery_months'] > 0) &
        (median_metrics['construction_capacity'] > 0)
    ]
    
    print(f"Counties with median event data: {len(median_metrics)}")
    print(f"  Median weighted damage range: {median_metrics['median_weighted_damage'].min():.1f} to {median_metrics['median_weighted_damage'].max():.1f}")
    print(f"  Median recovery range: {median_metrics['median_recovery_months'].min():.2f} to {median_metrics['median_recovery_months'].max():.1f} months")
    
    return median_metrics


def compute_max_event_metrics(events_df, recovery_df, capacity_df):
    """Compute maximum event metrics per county."""
    print("\n" + "="*80)
    print("COMPUTING MAXIMUM EVENT METRICS")
    print("="*80)
    
    # Calculate weighted damage per event-county
    events_df['weighted_damage'] = (
        events_df['units_DS1_scaled'] * RECOVERY_WEIGHTS['DS1'] +
        events_df['units_DS2_scaled'] * RECOVERY_WEIGHTS['DS2'] +
        events_df['units_DS3_scaled'] * RECOVERY_WEIGHTS['DS3'] +
        events_df['units_DS4_scaled'] * RECOVERY_WEIGHTS['DS4']
    )
    
    # Merge damage and recovery data (need to match by fips AND event)
    # First, get event names aligned
    recovery_renamed = recovery_df.rename(columns={'event': 'event_name'})
    
    merged = events_df[['fips', 'event_name', 'weighted_damage']].merge(
        recovery_renamed[['fips', 'event_name', 'recovery_potential [months]']],
        on=['fips', 'event_name'],
        how='inner'
    )
    
    print(f"Merged {len(merged)} event-county records with both damage and recovery")
    
    # Find index of max weighted damage event per county
    idx_max = merged.groupby('fips')['weighted_damage'].idxmax()
    print(f"Counties with valid max event: {idx_max.notna().sum()}")
    print(f"Counties filtered out (no events): {idx_max.isna().sum()}")
    
    # Extract max event data per county
    max_metrics = merged.loc[idx_max.dropna()].reset_index(drop=True)
    max_metrics = max_metrics.rename(columns={
        'weighted_damage': 'max_weighted_damage',
        'recovery_potential [months]': 'max_recovery_months'
    })
    
    # Add capacity
    max_metrics = max_metrics.merge(capacity_df, on='fips', how='inner')
    
    # Filter valid data
    max_metrics = max_metrics[
        (max_metrics['max_weighted_damage'] > 0) &
        (max_metrics['max_recovery_months'] > 0) &
        (max_metrics['construction_capacity'] > 0)
    ]
    
    print(f"Counties with max event data: {len(max_metrics)}")
    print(f"  Max weighted damage range: {max_metrics['max_weighted_damage'].min():.1f} to {max_metrics['max_weighted_damage'].max():.1f}")
    print(f"  Max recovery range: {max_metrics['max_recovery_months'].min():.2f} to {max_metrics['max_recovery_months'].max():.1f} months")
    
    return max_metrics


def compute_annual_metrics(events_df, recovery_df, capacity_df):
    """Compute expected annual metrics (EAD weighted, EARP)."""
    print("\n" + "="*80)
    print("COMPUTING EXPECTED ANNUAL METRICS")
    print("="*80)
    
    # Calculate weighted damage per event-county
    events_df['weighted_damage'] = (
        events_df['units_DS1_scaled'] * RECOVERY_WEIGHTS['DS1'] +
        events_df['units_DS2_scaled'] * RECOVERY_WEIGHTS['DS2'] +
        events_df['units_DS3_scaled'] * RECOVERY_WEIGHTS['DS3'] +
        events_df['units_DS4_scaled'] * RECOVERY_WEIGHTS['DS4']
    )
    
    # Compute Expected Annual Weighted Damage (sum across events × frequency)
    ead_weighted = events_df.groupby('fips')['weighted_damage'].sum().reset_index()
    ead_weighted['ead_weighted'] = ead_weighted['weighted_damage'] * DEFAULT_FREQ
    ead_weighted = ead_weighted[['fips', 'ead_weighted']]
    
    # Compute Expected Annual Recovery Potential (sum across events × frequency)
    earp = recovery_df.groupby('fips')['recovery_potential [months]'].sum().reset_index()
    earp['earp_months_per_year'] = earp['recovery_potential [months]'] * DEFAULT_FREQ
    earp = earp[['fips', 'earp_months_per_year']]
    
    # Merge with capacity
    annual_metrics = ead_weighted.merge(earp, on='fips', how='inner')
    annual_metrics = annual_metrics.merge(capacity_df, on='fips', how='inner')
    
    # Filter valid data
    annual_metrics = annual_metrics[
        (annual_metrics['ead_weighted'] > 0) &
        (annual_metrics['earp_months_per_year'] > 0) &
        (annual_metrics['construction_capacity'] > 0)
    ]
    
    print(f"Counties with annual metrics: {len(annual_metrics)}")
    print(f"  EAD (weighted) range: {annual_metrics['ead_weighted'].min():.2f} to {annual_metrics['ead_weighted'].max():.2f}")
    print(f"  EARP range: {annual_metrics['earp_months_per_year'].min():.2f} to {annual_metrics['earp_months_per_year'].max():.2f} months/year")
    
    return annual_metrics


def create_scatter_comparison(annual_metrics, median_metrics, max_metrics, output_file):
    """Create 3x2 scatter plot comparing annual vs median vs max metrics."""
    print(f"\nCreating scatter comparison plot...")
    
    from matplotlib.ticker import LogLocator
    from scipy.stats import pearsonr
    
    fig, axes = plt.subplots(3, 2, figsize=(8, 9))
    
    label_fs = 12
    tick_fs = 9
    cbar_label_fs = 11
    
    # ---------------- TOP LEFT: Annual damage vs EARP ----------------
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(
        annual_metrics['ead_weighted'], 
        annual_metrics['earp_months_per_year'],
        c=annual_metrics['construction_capacity'],
        cmap='viridis',
        alpha=0.6,
        s=30,
        norm=LogNorm()
    )
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.invert_yaxis()
    ax1.set_xlabel('EAWUA (weighted units)', fontsize=label_fs)
    ax1.set_ylabel('EARP (low–high)', fontsize=label_fs)
    ax1.grid(False)
    
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('CC (permits/month)', fontsize=cbar_label_fs)
    cbar1.ax.tick_params(which='both', labelsize=tick_fs)
    cbar1.ax.tick_params(which='minor', length=0)
    
    corr_annual_risk, _ = pearsonr(np.log10(annual_metrics['earp_months_per_year']), 
                                    np.log10(annual_metrics['ead_weighted']))
    ax1.text(0.05, 0.02, f'r = {corr_annual_risk:+.3f}\nn = {len(annual_metrics):,}', 
             transform=ax1.transAxes, fontsize=10, va='bottom')
    
    # ---------------- TOP RIGHT: Capacity vs EARP ----------------
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(
        annual_metrics['construction_capacity'], 
        annual_metrics['earp_months_per_year'],
        c=annual_metrics['ead_weighted'],
        cmap='plasma',
        alpha=0.6,
        s=30,
        norm=LogNorm()
    )
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.invert_yaxis()
    ax2.set_xlabel('CC (permits/month)', fontsize=label_fs)
    ax2.set_ylabel('EARP (low–high)', fontsize=label_fs)
    ax2.grid(False)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('EAWUA (weighted units)', fontsize=cbar_label_fs)
    cbar2.ax.tick_params(which='both', labelsize=tick_fs)
    cbar2.ax.tick_params(which='minor', length=0)
    
    corr_annual_capacity, _ = pearsonr(np.log10(annual_metrics['earp_months_per_year']), 
                                        np.log10(annual_metrics['construction_capacity']))
    ax2.text(0.68, 0.02, f'r = {corr_annual_capacity:+.3f}\nn = {len(annual_metrics):,}', 
             transform=ax2.transAxes, fontsize=10, va='bottom')
    
    # ---------------- BOTTOM LEFT: Median damage vs recovery ----------------
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(
        median_metrics['median_weighted_damage'], 
        median_metrics['median_recovery_months'],
        c=median_metrics['construction_capacity'],
        cmap='viridis',
        alpha=0.6,
        s=30,
        norm=LogNorm()
    )
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.invert_yaxis()
    ax3.set_xlabel('MWUA (weighted units)', fontsize=label_fs)
    ax3.set_ylabel('MRP (low–high)', fontsize=label_fs)
    ax3.grid(False)
    
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('CC (permits/month)', fontsize=cbar_label_fs)
    cbar3.ax.tick_params(which='both', labelsize=tick_fs)
    cbar3.ax.tick_params(which='minor', length=0)
    
    corr_median_damage, _ = pearsonr(np.log10(median_metrics['median_recovery_months']), 
                                      np.log10(median_metrics['median_weighted_damage']))
    ax3.text(0.05, 0.02, f'r = {corr_median_damage:+.3f}\nn = {len(median_metrics):,}', 
             transform=ax3.transAxes, fontsize=10, va='bottom')
    
    # ---------------- BOTTOM RIGHT: Capacity vs Median recovery ----------------
    ax4 = axes[1, 1]
    scatter4 = ax4.scatter(
        median_metrics['construction_capacity'], 
        median_metrics['median_recovery_months'],
        c=median_metrics['median_weighted_damage'],
        cmap='plasma',
        alpha=0.6,
        s=30,
        norm=LogNorm()
    )
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.invert_yaxis()
    ax4.set_xlabel('CC (permits/month)', fontsize=label_fs)
    ax4.set_ylabel('MRP (low–high)', fontsize=label_fs)
    ax4.grid(False)
    
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label('MWUA (weighted units)', fontsize=cbar_label_fs)
    cbar4.ax.tick_params(which='both', labelsize=tick_fs)
    cbar4.ax.tick_params(which='minor', length=0)
    
    corr_median_capacity, _ = pearsonr(np.log10(median_metrics['median_recovery_months']), 
                                        np.log10(median_metrics['construction_capacity']))
    ax4.text(0.68, 0.02, f'r = {corr_median_capacity:+.3f}\nn = {len(median_metrics):,}', 
             transform=ax4.transAxes, fontsize=10, va='bottom')
    
    # ---------------- BOTTOM LEFT: Max damage vs recovery ----------------
    ax5 = axes[2, 0]
    scatter5 = ax5.scatter(
        max_metrics['max_weighted_damage'], 
        max_metrics['max_recovery_months'],
        c=max_metrics['construction_capacity'],
        cmap='viridis',
        alpha=0.6,
        s=30,
        norm=LogNorm()
    )
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.invert_yaxis()
    ax5.set_xlabel('Max WUA (weighted units)', fontsize=label_fs)
    ax5.set_ylabel('Max RP (low–high)', fontsize=label_fs)
    ax5.grid(False)
    
    cbar5 = plt.colorbar(scatter5, ax=ax5)
    cbar5.set_label('CC (permits/month)', fontsize=cbar_label_fs)
    cbar5.ax.tick_params(which='both', labelsize=tick_fs)
    cbar5.ax.tick_params(which='minor', length=0)
    
    corr_max_damage, _ = pearsonr(np.log10(max_metrics['max_recovery_months']), 
                                   np.log10(max_metrics['max_weighted_damage']))
    ax5.text(0.05, 0.02, f'r = {corr_max_damage:+.3f}\nn = {len(max_metrics):,}', 
             transform=ax5.transAxes, fontsize=10, va='bottom')
    
    # ---------------- BOTTOM RIGHT: Capacity vs Max recovery ----------------
    ax6 = axes[2, 1]
    scatter6 = ax6.scatter(
        max_metrics['construction_capacity'], 
        max_metrics['max_recovery_months'],
        c=max_metrics['max_weighted_damage'],
        cmap='plasma',
        alpha=0.6,
        s=30,
        norm=LogNorm()
    )
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6.invert_yaxis()
    ax6.set_xlabel('CC (permits/month)', fontsize=label_fs)
    ax6.set_ylabel('Max RP (low–high)', fontsize=label_fs)
    ax6.grid(False)
    
    cbar6 = plt.colorbar(scatter6, ax=ax6)
    cbar6.set_label('Max WUA (weighted units)', fontsize=cbar_label_fs)
    cbar6.ax.tick_params(which='both', labelsize=tick_fs)
    cbar6.ax.tick_params(which='minor', length=0)
    
    corr_max_capacity, _ = pearsonr(np.log10(max_metrics['max_recovery_months']), 
                                     np.log10(max_metrics['construction_capacity']))
    ax6.text(0.68, 0.02, f'r = {corr_max_capacity:+.3f}\nn = {len(max_metrics):,}', 
             transform=ax6.transAxes, fontsize=10, va='bottom')
    
    # Colorbar borders
    for cbar in [cbar1, cbar2, cbar3, cbar4, cbar5, cbar6]:
        for spine in cbar.ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
    
    # Panel borders
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        for spine in ax.spines.values():
            spine.set_edgecolor('0.4')
            spine.set_linewidth(0.8)
        ax.tick_params(color='0.4', labelcolor='0.2')
    
    # Panel labels
    panel_labels = ['a', 'b', 'c', 'd', 'e', 'f']
    for label, ax in zip(panel_labels, [ax1, ax2, ax3, ax4, ax5, ax6]):
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left')
    
    # Axis ticks
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.tick_params(axis='x', which='major', bottom=True, top=False,
                       labelbottom=True, labelsize=tick_fs)
        ax.tick_params(axis='y', which='both', left=False, right=False,
                       labelleft=False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"   Saved: {output_file}")
    plt.close()


def compute_ds_percentages(events_df):
    """Compute percentage of damage in each DS per county."""
    print("\n" + "="*80)
    print("COMPUTING DAMAGE STATE PERCENTAGES")
    print("="*80)
    
    # Sum total units per DS per county across all events
    ds_totals = events_df.groupby('fips').agg({
        'units_DS1_scaled': 'sum',
        'units_DS2_scaled': 'sum',
        'units_DS3_scaled': 'sum',
        'units_DS4_scaled': 'sum'
    }).reset_index()
    
    # Calculate total damage and percentages
    ds_totals['total_units'] = (
        ds_totals['units_DS1_scaled'] + 
        ds_totals['units_DS2_scaled'] + 
        ds_totals['units_DS3_scaled'] + 
        ds_totals['units_DS4_scaled']
    )
    
    # Calculate percentages
    ds_totals['pct_DS1'] = 100 * ds_totals['units_DS1_scaled'] / ds_totals['total_units']
    ds_totals['pct_DS2'] = 100 * ds_totals['units_DS2_scaled'] / ds_totals['total_units']
    ds_totals['pct_DS3'] = 100 * ds_totals['units_DS3_scaled'] / ds_totals['total_units']
    ds_totals['pct_DS4'] = 100 * ds_totals['units_DS4_scaled'] / ds_totals['total_units']
    
    # Filter to counties with actual damage
    ds_totals = ds_totals[ds_totals['total_units'] > 0]
    
    print(f"Counties with DS percentage data: {len(ds_totals)}")
    print(f"\nOverall DS distribution:")
    total_all = ds_totals[['units_DS1_scaled', 'units_DS2_scaled', 
                            'units_DS3_scaled', 'units_DS4_scaled']].sum()
    total_sum = total_all.sum()
    print(f"  DS1: {100*total_all['units_DS1_scaled']/total_sum:.1f}%")
    print(f"  DS2: {100*total_all['units_DS2_scaled']/total_sum:.1f}%")
    print(f"  DS3: {100*total_all['units_DS3_scaled']/total_sum:.1f}%")
    print(f"  DS4: {100*total_all['units_DS4_scaled']/total_sum:.1f}%")
    
    return ds_totals[['fips', 'pct_DS1', 'pct_DS2', 'pct_DS3', 'pct_DS4', 'total_units']]


def create_ds_percentage_map(coastal_counties, ds_pct_df, output_file):
    """Create 2x2 map showing DS percentage distribution per county."""
    print(f"\nCreating DS percentage map...")
    
    # Merge with spatial data
    merged = coastal_counties.merge(ds_pct_df, left_on='GEOID', right_on='fips', how='left')
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    ds_configs = [
        ('pct_DS1', 'DS1 (Slight)', 'Greens'),
        ('pct_DS2', 'DS2 (Moderate)', 'YlOrBr'),
        ('pct_DS3', 'DS3 (Extensive)', 'Oranges'),
        ('pct_DS4', 'DS4 (Complete)', 'Reds')
    ]
    
    for idx, (ax, (metric, title, cmap)) in enumerate(zip(axes, ds_configs)):
        # Create divider for colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        # Plot with fixed scale 0-100%
        merged.plot(
            column=metric,
            cmap=cmap,
            vmin=0,
            vmax=100,
            linewidth=0.1,
            edgecolor="0.5",
            legend=True,
            ax=ax,
            cax=cax,
            missing_kwds={
                "color": "white",
                "label": "No data",
                "edgecolor": "0.5"
            }
        )
        
        # Title and axis
        ax.set_title(f'{title}\n% of Total Damage', fontsize=12, pad=5)
        ax.axis("off")
        
        # Colorbar formatting
        cax.set_ylabel('% of damage', fontsize=10)
        cax.tick_params(labelsize=9)
        
        for spine in cax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"   Saved: {output_file}")
    plt.close()


def create_3panel_map(coastal_counties, metrics_df, metric_cols, title_prefix, output_file):
    """Create 3-panel map with specified metrics."""
    print(f"\nCreating 3-panel map: {title_prefix}...")
    
    # Merge metrics with spatial data
    merged = coastal_counties.merge(metrics_df, left_on='GEOID', right_on='fips', how='left')
    
    # Create plotting copy with zeros as NaN
    merged_plot = merged.copy()
    for col in metric_cols:
        merged_plot.loc[merged_plot[col] <= 0, col] = np.nan
    
    # Define titles based on metric type
    if 'ead_weighted' in metric_cols:
        damage_title = f'{title_prefix}\nWeighted Units Affected'
        damage_ylabel = 'weighted units'
        recovery_title = f'{title_prefix}\nRecovery Potential'
        recovery_ylabel = 'months/year'
    elif 'median' in metric_cols[0]:
        damage_title = f'{title_prefix}\nWeighted Damage'
        damage_ylabel = 'weighted units'
        recovery_title = f'{title_prefix}\nRecovery Time'
        recovery_ylabel = 'months'
    else:  # max
        damage_title = f'{title_prefix}\nWeighted Damage'
        damage_ylabel = 'weighted units'
        recovery_title = f'{title_prefix}\nRecovery Time'
        recovery_ylabel = 'months'
    
    # Define metrics for each panel
    metrics_config = [
        (metric_cols[0], 'cividis', damage_title, damage_ylabel),
        (metric_cols[1], 'Greens', f'{title_prefix}\nConstruction Capacity', 'permits/month'),
        (metric_cols[2], 'Purples_r', recovery_title, recovery_ylabel)
    ]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes = axes.flatten()
    
    for idx, (ax, (metric, cmap, title, ylabel)) in enumerate(zip(axes, metrics_config)):
        # Get valid data for this metric
        data_positive = merged_plot[metric].dropna()
        
        if not data_positive.empty and len(data_positive) > 0:
            vmin = data_positive.min()
            vmax = data_positive.max()
            
            if np.isfinite(vmin) and np.isfinite(vmax) and vmin > 0 and vmax > 0:
                log_vmin = vmin / 2
                norm = LogNorm(vmin=log_vmin, vmax=vmax)
            else:
                norm = None
        else:
            norm = None
        
        # Create divider for colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        # Plot
        merged_plot.plot(
            column=metric,
            cmap=cmap,
            norm=norm,
            linewidth=0.1,
            edgecolor="0.5",
            legend=True,
            ax=ax,
            cax=cax,
            missing_kwds={
                "color": "white",
                "label": "No data",
                "edgecolor": "0.5"
            }
        )
        
        # Title and axis
        ax.set_title(title, fontsize=12, pad=2)
        ax.axis("off")
        
        # Colorbar formatting
        if 'Recovery' in title or 'recovery' in ylabel:
            # Invert for recovery (higher = worse)
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
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"   Saved: {output_file}")
    plt.close()


def print_comparison_statistics(median_metrics, max_metrics):
    """Print comparison statistics between median and max events."""
    print("\n" + "="*80)
    print("MEDIAN VS MAX EVENT COMPARISON")
    print("="*80)
    
    # Merge on common counties
    comparison = median_metrics[['fips', 'median_weighted_damage', 'median_recovery_months']].merge(
        max_metrics[['fips', 'max_weighted_damage', 'max_recovery_months']],
        on='fips',
        how='inner'
    )
    
    print(f"\nCounties with both median and max data: {len(comparison)}")
    
    # Calculate ratios
    comparison['damage_ratio'] = comparison['max_weighted_damage'] / comparison['median_weighted_damage']
    comparison['recovery_ratio'] = comparison['max_recovery_months'] / comparison['median_recovery_months']
    
    print("\nDamage Ratio (Max / Median):")
    print(f"  Mean: {comparison['damage_ratio'].mean():.2f}x")
    print(f"  Median: {comparison['damage_ratio'].median():.2f}x")
    print(f"  Min: {comparison['damage_ratio'].min():.2f}x")
    print(f"  Max: {comparison['damage_ratio'].max():.2f}x")
    
    print("\nRecovery Ratio (Max / Median):")
    print(f"  Mean: {comparison['recovery_ratio'].mean():.2f}x")
    print(f"  Median: {comparison['recovery_ratio'].median():.2f}x")
    print(f"  Min: {comparison['recovery_ratio'].min():.2f}x")
    print(f"  Max: {comparison['recovery_ratio'].max():.2f}x")
    
    # Save comparison
    output_file = BASE_DIR / "analysis_output" / "median_vs_max_event_comparison.csv"
    comparison.to_csv(output_file, index=False)
    print(f"\nComparison data saved to: {output_file}")
    
    return comparison


def main():
    """Main analysis pipeline."""
    # Load data
    events_df = load_event_data()
    recovery_df = load_recovery_data()
    capacity_df = load_capacity_data()
    coastal_counties = load_spatial_data()
    
    # Compute metrics
    annual_metrics = compute_annual_metrics(events_df, recovery_df, capacity_df)
    median_metrics = compute_median_event_metrics(events_df, recovery_df, capacity_df)
    max_metrics = compute_max_event_metrics(events_df, recovery_df, capacity_df)
    ds_pct = compute_ds_percentages(events_df)
    
    # Print comparison statistics
    comparison = print_comparison_statistics(median_metrics, max_metrics)
    
    # Create maps
    print("\n" + "="*80)
    print("CREATING MAPS")
    print("="*80)
    
    output_dir = BASE_DIR / "analysis_output"
    
    create_3panel_map(
        coastal_counties,
        annual_metrics,
        ['ead_weighted', 'construction_capacity', 'earp_months_per_year'],
        'Expected Annual',
        output_dir / "annual_3panel.png"
    )
    
    create_3panel_map(
        coastal_counties,
        median_metrics,
        ['median_weighted_damage', 'construction_capacity', 'median_recovery_months'],
        'Median Event',
        output_dir / "median_event_3panel.png"
    )
    
    create_3panel_map(
        coastal_counties,
        max_metrics,
        ['max_weighted_damage', 'construction_capacity', 'max_recovery_months'],
        'Maximum Event',
        output_dir / "max_event_3panel.png"
    )
    
    create_ds_percentage_map(
        coastal_counties,
        ds_pct,
        output_dir / "ds_percentage_distribution_map.png"
    )
    
    # Create scatter comparison
    print("\n" + "="*80)
    print("CREATING SCATTER PLOTS")
    print("="*80)
    
    create_scatter_comparison(
        annual_metrics,
        median_metrics,
        max_metrics,
        output_dir / "recovery_drivers_annual_vs_median_vs_max.png"
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
