#!/usr/bin/env python3
"""
Compare median vs maximum event impacts per county.

Computes, for every county, the median and the maximum event (by repair
demand in weighted units affected, WUA) across the probabilistic event set,
together with the recovery burden of those events, and writes the comparison
table

Following the manuscript definition, event-level (median) metrics are computed
over damaging events only, i.e. events with non-zero repair demand
(D_{e,c} > 0). Zero-demand footprint events are excluded before taking
medians.

    analysis_output/median_vs_max_event_comparison.csv

consumed by notebooks/probabilistic_analysis.ipynb (median/max event triptych
maps and the recovery-driver scatterplots).

Author: Simona Meiler
Date: January 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

# Configuration
RECOVERY_WEIGHTS = {
    'DS1': 1.0,
    'DS2': 1.0,
    'DS3': 3.0,
    'DS4': 6.0
}

DEFAULT_FREQ = 0.00067334  # events/year

def load_event_data():
    """Load all per-event impact data."""
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    print("\n1. Loading per-event impact data...")

    per_event_dir = BASE_DIR / "data" / "impact" / "per_event"
    event_files = sorted(per_event_dir.glob("*.csv"))

    print(f"   Found {len(event_files)} event files")

    all_events = []
    for f in event_files:
        df = pd.read_csv(f)
        all_events.append(df)

    events_df = pd.concat(all_events, ignore_index=True)
    events_df['fips'] = events_df['fips'].astype(str).str.zfill(5)
    
    print(f"   Loaded {len(events_df)} county-event records")
    print(f"   Unique events: {events_df['event_name'].nunique()}")
    print(f"   Unique counties: {events_df['fips'].nunique()}")
    
    return events_df


def load_recovery_data():
    """Load per-event recovery burden data (column names retain the legacy identifier recovery_potential_months)."""
    print("\n2. Loading recovery burden data...")

    recovery_csv = BASE_DIR / "data" / "recovery" / "recovery_potential.csv"
    recovery_df = pd.read_csv(recovery_csv, dtype={"fips": str})
    recovery_df['fips'] = recovery_df['fips'].astype(str).str.zfill(5)
    recovery_df['recovery_potential [months]'] = (
        pd.to_numeric(recovery_df['recovery_potential_months'], errors='coerce')
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
    
    # Damaging events only (manuscript definition: D_{e,c} > 0);
    # zero-demand footprint events are excluded before taking medians.
    damaging = events_df[events_df['weighted_damage'] > 0]
    n_dropped = len(events_df) - len(damaging)
    print(f"Restricting to damaging events (D > 0): "
          f"{len(damaging)} of {len(events_df)} county-event records kept "
          f"({n_dropped} zero-demand records excluded)")

    # Compute median repair demand (WUA) per county over damaging events
    median_damage = damaging.groupby('fips')['weighted_damage'].median().reset_index()
    median_damage.columns = ['fips', 'median_weighted_damage']

    # Compute median recovery burden per county over damaging events.
    # Recovery burden is 0 if and only if repair demand is 0 (the lower bound
    # in Eq. 4 guarantees B >= 1 month whenever any units are damaged), so
    # filtering on burden > 0 is equivalent to D_{e,c} > 0.
    recovery_damaging = recovery_df[recovery_df['recovery_potential [months]'] > 0]
    median_recovery = recovery_damaging.groupby('fips')['recovery_potential [months]'].median().reset_index()
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
    print(f"  Median recovery-burden range: {median_metrics['median_recovery_months'].min():.2f} to {median_metrics['median_recovery_months'].max():.1f} months")
    
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
    print(f"  Max-event recovery-burden range: {max_metrics['max_recovery_months'].min():.2f} to {max_metrics['max_recovery_months'].max():.1f} months")
    
    return max_metrics


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
    """Compute the median-vs-max event comparison table."""
    events_df = load_event_data()
    recovery_df = load_recovery_data()
    capacity_df = load_capacity_data()

    median_metrics = compute_median_event_metrics(events_df, recovery_df, capacity_df)
    max_metrics = compute_max_event_metrics(events_df, recovery_df, capacity_df)

    print_comparison_statistics(median_metrics, max_metrics)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
