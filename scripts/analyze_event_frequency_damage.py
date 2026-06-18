#!/usr/bin/env python3
"""
Analyze event frequency and damage distribution across counties.

Computes:
1. Total events affecting each county
2. Events causing non-zero damage per county
3. Weighted damage units relative to total housing stock

Author: Simona Meiler
Date: January 2026
"""

import pandas as pd
import numpy as np
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

def load_event_data():
    """Load all per-event scaled impact data."""
    print("Loading per-event impact data...")
    
    per_event_dir = BASE_DIR / "data" / "impact" / "per_event"
    event_files = sorted(per_event_dir.glob("*.csv"))

    print(f"Found {len(event_files)} event files")

    all_events = []
    for f in event_files:
        df = pd.read_csv(f)
        all_events.append(df)
    
    events_df = pd.concat(all_events, ignore_index=True)
    
    # Ensure FIPS is 5-digit string
    events_df['fips'] = events_df['fips'].astype(str).str.zfill(5)
    
    print(f"Loaded {len(events_df)} county-event records")
    print(f"  Unique events: {events_df['event_name'].nunique()}")
    print(f"  Unique counties: {events_df['fips'].nunique()}")
    
    return events_df


def load_housing_stock():
    """Load total housing units per county from exposure data."""
    print("\nLoading housing stock data...")
    
    # Load from county exposure summary
    county_file = BASE_DIR / "analysis_output" / "county_exposed_housing_units.csv"
    
    if county_file.exists():
        housing_df = pd.read_csv(county_file)
        # Rename columns to match
        housing_df = housing_df.rename(columns={'FIPS': 'fips', 'exposed_units': 'total_units'})
        housing_df['fips'] = housing_df['fips'].astype(str).str.zfill(5)
        print(f"Loaded housing stock for {len(housing_df)} counties")
        return housing_df[['fips', 'total_units']].copy()
    else:
        print("Warning: county_exposed_housing_units.csv not found")
        print("Computing from event data (may underestimate total units)...")
        return None


def compute_event_frequency_metrics(events_df):
    """Compute event frequency metrics per county."""
    print("\n" + "="*80)
    print("COMPUTING EVENT FREQUENCY METRICS")
    print("="*80)
    
    # Calculate total damage per event-county
    events_df['total_damage_units'] = (
        events_df['units_DS1_scaled'] + 
        events_df['units_DS2_scaled'] + 
        events_df['units_DS3_scaled'] + 
        events_df['units_DS4_scaled']
    )
    
    # Calculate weighted damage (recovery time weighted)
    events_df['weighted_damage_units'] = (
        events_df['units_DS1_scaled'] * RECOVERY_WEIGHTS['DS1'] +
        events_df['units_DS2_scaled'] * RECOVERY_WEIGHTS['DS2'] +
        events_df['units_DS3_scaled'] * RECOVERY_WEIGHTS['DS3'] +
        events_df['units_DS4_scaled'] * RECOVERY_WEIGHTS['DS4']
    )
    
    # Group by county
    county_metrics = events_df.groupby('fips').agg({
        'event_name': 'count',  # Total events
        'total_damage_units': [
            lambda x: (x > 0).sum(),  # Events with damage
            'sum'  # Total damage across all events
        ],
        'weighted_damage_units': 'sum',  # Total weighted damage
        'units_DS1_scaled': 'sum',
        'units_DS2_scaled': 'sum',
        'units_DS3_scaled': 'sum',
        'units_DS4_scaled': 'sum'
    }).reset_index()
    
    # Flatten column names
    county_metrics.columns = [
        'fips',
        'total_events',
        'events_with_damage',
        'total_damage_units',
        'total_weighted_damage_units',
        'total_DS1_units',
        'total_DS2_units',
        'total_DS3_units',
        'total_DS4_units'
    ]
    
    # Calculate percentage of events causing damage
    county_metrics['pct_events_with_damage'] = (
        100 * county_metrics['events_with_damage'] / county_metrics['total_events']
    )
    
    print(f"\nProcessed {len(county_metrics)} counties")
    
    return county_metrics


def add_housing_stock_metrics(county_metrics, housing_df):
    """Add housing stock and calculate normalized metrics."""
    print("\nAdding housing stock normalization...")
    
    if housing_df is not None:
        county_metrics = county_metrics.merge(housing_df, on='fips', how='left')
    else:
        # Fallback: estimate from max damaged units (underestimate)
        print("  Using fallback: estimating total units from max damage")
        county_metrics['total_units'] = county_metrics['total_damage_units'] * 2
    
    # Calculate normalized metrics (damage relative to housing stock)
    county_metrics['damage_units_per_total'] = (
        county_metrics['total_damage_units'] / county_metrics['total_units']
    )
    
    county_metrics['weighted_damage_per_total'] = (
        county_metrics['total_weighted_damage_units'] / county_metrics['total_units']
    )
    
    county_metrics['pct_housing_affected'] = (
        100 * county_metrics['damage_units_per_total']
    )
    
    return county_metrics


def print_summary_statistics(county_metrics):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print("\nEvent Frequency:")
    print(f"  Mean events per county: {county_metrics['total_events'].mean():.1f}")
    print(f"  Median events per county: {county_metrics['total_events'].median():.0f}")
    print(f"  Max events per county: {county_metrics['total_events'].max():.0f}")
    print(f"  Min events per county: {county_metrics['total_events'].min():.0f}")
    
    print("\nEvents Causing Damage:")
    print(f"  Mean events with damage per county: {county_metrics['events_with_damage'].mean():.1f}")
    print(f"  Median events with damage: {county_metrics['events_with_damage'].median():.0f}")
    print(f"  Mean % of events causing damage: {county_metrics['pct_events_with_damage'].mean():.1f}%")
    
    print("\nDamage Units (Total across all events):")
    print(f"  Mean total damage units per county: {county_metrics['total_damage_units'].mean():.0f}")
    print(f"  Median total damage units: {county_metrics['total_damage_units'].median():.0f}")
    print(f"  Max total damage units: {county_metrics['total_damage_units'].max():.0f}")
    
    print("\nWeighted Damage Units (Total across all events):")
    print(f"  Mean weighted damage per county: {county_metrics['total_weighted_damage_units'].mean():.0f}")
    print(f"  Median weighted damage: {county_metrics['total_weighted_damage_units'].median():.0f}")
    
    if 'total_units' in county_metrics.columns:
        print("\nDamage Relative to Housing Stock:")
        print(f"  Mean % of housing affected (total): {county_metrics['pct_housing_affected'].mean():.2f}%")
        print(f"  Median % of housing affected: {county_metrics['pct_housing_affected'].median():.2f}%")
        print(f"  Max % of housing affected: {county_metrics['pct_housing_affected'].max():.2f}%")
    
    print("\nDamage State Distribution (Total Units):")
    total_all = county_metrics[['total_DS1_units', 'total_DS2_units', 
                                  'total_DS3_units', 'total_DS4_units']].sum()
    total_sum = total_all.sum()
    print(f"  DS1: {total_all['total_DS1_units']:,.0f} ({100*total_all['total_DS1_units']/total_sum:.1f}%)")
    print(f"  DS2: {total_all['total_DS2_units']:,.0f} ({100*total_all['total_DS2_units']/total_sum:.1f}%)")
    print(f"  DS3: {total_all['total_DS3_units']:,.0f} ({100*total_all['total_DS3_units']/total_sum:.1f}%)")
    print(f"  DS4: {total_all['total_DS4_units']:,.0f} ({100*total_all['total_DS4_units']/total_sum:.1f}%)")


def print_top_counties(county_metrics, n=10):
    """Print top counties by various metrics."""
    print("\n" + "="*80)
    print(f"TOP {n} COUNTIES BY METRICS")
    print("="*80)
    
    print(f"\nMost Events:")
    top = county_metrics.nlargest(n, 'total_events')[['fips', 'total_events', 'events_with_damage', 'pct_events_with_damage']]
    for _, row in top.iterrows():
        print(f"  {row['fips']}: {row['total_events']:.0f} events ({row['events_with_damage']:.0f} with damage, {row['pct_events_with_damage']:.1f}%)")
    
    print(f"\nMost Events with Damage:")
    top = county_metrics.nlargest(n, 'events_with_damage')[['fips', 'events_with_damage', 'total_events']]
    for _, row in top.iterrows():
        print(f"  {row['fips']}: {row['events_with_damage']:.0f} events with damage (of {row['total_events']:.0f} total)")
    
    print(f"\nHighest Total Damage (Units):")
    top = county_metrics.nlargest(n, 'total_damage_units')[['fips', 'total_damage_units', 'events_with_damage']]
    for _, row in top.iterrows():
        print(f"  {row['fips']}: {row['total_damage_units']:,.0f} units ({row['events_with_damage']:.0f} events)")
    
    print(f"\nHighest Weighted Damage (Units × Recovery Time):")
    top = county_metrics.nlargest(n, 'total_weighted_damage_units')[['fips', 'total_weighted_damage_units', 'total_damage_units']]
    for _, row in top.iterrows():
        print(f"  {row['fips']}: {row['total_weighted_damage_units']:,.0f} weighted units ({row['total_damage_units']:,.0f} units)")
    
    if 'pct_housing_affected' in county_metrics.columns:
        print(f"\nHighest % of Housing Stock Affected:")
        top = county_metrics.nlargest(n, 'pct_housing_affected')[['fips', 'pct_housing_affected', 'total_units']]
        for _, row in top.iterrows():
            if pd.notna(row['total_units']):
                print(f"  {row['fips']}: {row['pct_housing_affected']:.2f}% of {row['total_units']:,.0f} units")


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("EVENT FREQUENCY AND DAMAGE ANALYSIS")
    print("="*80)
    
    # Load data
    events_df = load_event_data()
    housing_df = load_housing_stock()
    
    # Compute metrics
    county_metrics = compute_event_frequency_metrics(events_df)
    county_metrics = add_housing_stock_metrics(county_metrics, housing_df)
    
    # Print summaries
    print_summary_statistics(county_metrics)
    print_top_counties(county_metrics, n=10)
    
    # Save results
    output_file = BASE_DIR / "analysis_output" / "county_event_frequency_damage_metrics.csv"
    county_metrics.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
