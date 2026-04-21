"""
Damage-Recovery Quadrant Analysis (NORMALIZED BY TOTAL HOUSING UNITS)

Uses repair-time weighted damage units NORMALIZED by total county housing units
and construction capacity. This reveals the proportion of a county's housing
stock damaged, controlling for county size.

Weights: DS1/DS2 = 1 month, DS3 = 3 months, DS4 = 6 months per unit.
Normalization: weighted_damage_units / total_housing_units_in_county
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import geopandas as gpd
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"
DATA_DIR = BASE_DIR / "data"
IMPACTS_DIR = BASE_DIR / "impacts_out" / "by_event" / "scaled"
RECOVERY_DIR = DATA_DIR / "recovery_potential_per_scenario"
EXPOSURE_DIR = DATA_DIR / "exposure"

# Color palette for quadrants
QUADRANT_COLORS = {
    'High Damage / High Capacity': '#2c7bb6',
    'High Damage / Low Capacity': '#d7191c',
    'Low Damage / High Capacity': '#1a9850',
    'Low Damage / Low Capacity': '#fdae61'
}

# Repair time weights per damage state (months per unit)
DS_REPAIR_TIMES = {
    'units_DS1_scaled': 1.0,
    'units_DS2_scaled': 1.0,
    'units_DS3_scaled': 3.0,
    'units_DS4_scaled': 6.0
}


def load_housing_units():
    """
    Load total housing units per county from Census data.
    Returns a dict mapping FIPS -> total housing units (2024 estimate).
    """
    print("Loading housing units data...")
    
    # Load the housing units file
    hu_df = pd.read_excel(
        EXPOSURE_DIR / 'CO-EST2024-HU.xlsx',
        skiprows=2  # Skip header rows
    )
    
    # Keep only county rows (remove state/national totals)
    # County rows start with a period
    hu_df = hu_df[hu_df['Geographic Area'].str.startswith('.', na=False)].copy()
    
    # Use 2024 estimates FIRST (last column is 'Unnamed: 6' which is 2024 data)
    # The last column is already numeric
    hu_df['total_units'] = pd.to_numeric(hu_df.iloc[:, -1], errors='coerce')
    
    # Extract county name and state from Geographic Area  
    hu_df['county_state'] = hu_df['Geographic Area'].str.replace('.', '', 1)
    
    # Load FIPS lookup
    fips_df = pd.read_csv(DATA_DIR / 'selected_states_counties.csv')
    
    # Create county-state name for matching
    # Louisiana uses "Parish" instead of "County"
    fips_df['county_state'] = fips_df.apply(
        lambda row: f"{row['NAME'].title()} Parish, {row['STATE_NAME']}" 
        if row['STATE_NAME'] == 'Louisiana' 
        else f"{row['NAME'].title()} County, {row['STATE_NAME']}", 
        axis=1
    )
    
    # Merge to get FIPS
    merged = hu_df.merge(fips_df[['FIPS', 'county_state']], on='county_state', how='inner')
    
    # Create FIPS -> housing units mapping
    housing_units = dict(zip(
        merged['FIPS'].astype(str).str.zfill(5),
        merged['total_units']  # Already numeric
    ))
    
    print(f"Loaded housing units for {len(housing_units):,} counties")
    print(f"Example: FIPS 01001 (Autauga, AL) = {housing_units.get('01001', 'N/A'):,.0f} units")
    
    return housing_units


def load_all_event_county_pairs():
    """
    Load all event-county pairs with weighted damage units.
    """
    print("Loading all event-county pairs...")
    
    # Load all recovery potential files
    recovery_data = []
    recovery_files = sorted(RECOVERY_DIR.glob("*_scaled_recovery_potential.json"))
    
    print(f"Found {len(recovery_files)} recovery potential files")
    
    for i, filepath in enumerate(recovery_files):
        if i % 1000 == 0:
            print(f"  Processing file {i}/{len(recovery_files)}")
        
        with open(filepath, 'r') as f:
            event_data = json.load(f)
            recovery_data.extend(event_data)
    
    recovery_df = pd.DataFrame(recovery_data)
    recovery_df['event'] = recovery_df['event'].astype(str)
    recovery_df['fips'] = recovery_df['fips'].astype(str).str.zfill(5)
    recovery_df['reconstruction_capacity'] = recovery_df['reconstruction_capacity'].astype(float)
    recovery_df['recovery_potential [months]'] = recovery_df['recovery_potential [months]'].astype(float)
    
    print(f"Loaded {len(recovery_df):,} recovery potential records")
    
    # Load all damage impact files
    damage_data = []
    impact_files = sorted(IMPACTS_DIR.glob("*_scaled.csv"))
    
    print(f"Found {len(impact_files)} impact files")
    
    for i, filepath in enumerate(impact_files):
        if i % 1000 == 0:
            print(f"  Processing file {i}/{len(impact_files)}")
        
        df = pd.read_csv(filepath)
        df['event'] = filepath.stem.replace('_scaled', '')
        damage_data.append(df)
    
    damage_df = pd.concat(damage_data, ignore_index=True)
    damage_df['fips'] = damage_df['fips'].astype(str).str.zfill(5)
    
    # Calculate weighted damage units (repair-time weighted)
    damage_df['weighted_damage_units'] = sum(
        damage_df[col].fillna(0) * DS_REPAIR_TIMES[col]
        for col in DS_REPAIR_TIMES.keys()
    )
    
    # Calculate total units (unweighted) and repair cost for reference
    damage_df['total_damage_units'] = sum(
        damage_df[col].fillna(0)
        for col in DS_REPAIR_TIMES.keys()
    )
    
    damage_df['repair_cost_total'] = damage_df['repair_cost_sum_scaled'].fillna(0)
    
    print(f"Loaded {len(damage_df):,} damage records")
    
    # Merge damage and recovery
    merged_df = recovery_df.merge(
        damage_df[['event', 'fips', 'weighted_damage_units', 'total_damage_units', 'repair_cost_total']],
        on=['event', 'fips'],
        how='inner'
    )
    
    print(f"Merged dataset: {len(merged_df):,} event-county pairs")
    
    # Filter to valid cases
    valid_df = merged_df[
        (merged_df['weighted_damage_units'] > 0) &
        (merged_df['recovery_potential [months]'] > 0) &
        (merged_df['reconstruction_capacity'] > 0)
    ].copy()
    
    print(f"Valid pairs (weighted damage > 0, recovery > 0): {len(valid_df):,}")
    
    # Rename columns for clarity
    valid_df = valid_df.rename(columns={
        'reconstruction_capacity': 'construction_capacity',
        'recovery_potential [months]': 'recovery_months'
    })
    
    return valid_df[['event', 'fips', 'construction_capacity', 'recovery_months', 
                     'weighted_damage_units', 'total_damage_units', 'repair_cost_total']]


def normalize_by_housing_units(df, housing_units):
    """
    Add normalized damage metric: weighted damage units / total housing units.
    This gives the proportion of county housing stock damaged (weighted by repair time).
    """
    print("\nNormalizing damage by total housing units...")
    
    # Add total housing units for each county
    df['total_housing_units'] = df['fips'].map(housing_units)
    
    # Calculate normalized damage (as percentage)
    df['pct_housing_damaged'] = (df['weighted_damage_units'] / df['total_housing_units']) * 100
    
    # Remove counties without housing data
    before = len(df)
    df = df[df['total_housing_units'].notna() & (df['total_housing_units'] > 0)].copy()
    after = len(df)
    
    if before > after:
        print(f"  Removed {before - after:,} pairs with missing housing unit data")
    
    print(f"  Final dataset: {len(df):,} event-county pairs")
    print(f"  Normalized damage range: {df['pct_housing_damaged'].min():.4f}% to {df['pct_housing_damaged'].max():.2f}%")
    print(f"  Median normalized damage: {df['pct_housing_damaged'].median():.4f}%")
    
    return df


def assign_quadrants(df, damage_threshold=None, capacity_threshold=None):
    """Assign quadrants based on normalized damage (% housing damaged) and capacity"""
    df = df.copy()
    
    # Use median splits if no thresholds provided
    if damage_threshold is None:
        damage_threshold = df['pct_housing_damaged'].median()
    if capacity_threshold is None:
        capacity_threshold = df['construction_capacity'].median()
    
    print(f"\nQuadrant thresholds:")
    print(f"  % Housing damaged: {damage_threshold:.4f}%")
    print(f"  Capacity: {capacity_threshold:.2f} permits/month")
    
    # Assign quadrants
    df['quadrant'] = 'Unknown'
    
    high_damage = df['pct_housing_damaged'] >= damage_threshold
    high_capacity = df['construction_capacity'] >= capacity_threshold
    
    df.loc[high_damage & high_capacity, 'quadrant'] = 'High Damage / High Capacity'
    df.loc[high_damage & ~high_capacity, 'quadrant'] = 'High Damage / Low Capacity'
    df.loc[~high_damage & high_capacity, 'quadrant'] = 'Low Damage / High Capacity'
    df.loc[~high_damage & ~high_capacity, 'quadrant'] = 'Low Damage / Low Capacity'
    
    return df, damage_threshold, capacity_threshold


def create_quadrant_scatter(df, damage_threshold, capacity_threshold, output_file):
    """
    Create scatter plot showing normalized damage vs capacity colored by quadrant.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    for quadrant, color in QUADRANT_COLORS.items():
        subset = df[df['quadrant'] == quadrant]
        ax.scatter(subset['pct_housing_damaged'], subset['construction_capacity'],
                   alpha=0.4, s=25, c=color, label=quadrant, edgecolors='none')
    
    # Add threshold lines
    ax.axvline(damage_threshold, color='black', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Median damage: {damage_threshold:.4f}%')
    ax.axhline(capacity_threshold, color='black', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Median capacity: {capacity_threshold:.1f}')
    
    # Set log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Labels and formatting
    ax.set_xlabel('% of County Housing Stock Damaged (weighted)', fontsize=13)
    ax.set_ylabel('Construction Capacity (permits/month)', fontsize=13)
    ax.set_title('Damage-Recovery Quadrant Analysis (Normalized by Housing Stock)', 
                 fontsize=14, pad=15)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def main():
    print("=" * 80)
    print("DAMAGE-RECOVERY QUADRANT ANALYSIS (NORMALIZED BY HOUSING STOCK)")
    print("=" * 80)
    
    # Load housing units data
    housing_units = load_housing_units()
    
    # Load damage/recovery data
    df = load_all_event_county_pairs()
    
    # Normalize by housing units
    df = normalize_by_housing_units(df, housing_units)
    
    # Assign quadrants based on normalized damage
    df, damage_threshold, capacity_threshold = assign_quadrants(df)
    
    # Summary statistics by quadrant
    print("\n" + "=" * 80)
    print("QUADRANT SUMMARY STATISTICS")
    print("=" * 80)
    
    summary_stats = []
    for quadrant in ['High Damage / High Capacity', 'High Damage / Low Capacity',
                     'Low Damage / High Capacity', 'Low Damage / Low Capacity']:
        subset = df[df['quadrant'] == quadrant]
        
        stats = {
            'Quadrant': quadrant,
            'Count': len(subset),
            'Percentage': f"{len(subset) / len(df) * 100:.1f}%",
            'Median % Housing Damaged': f"{subset['pct_housing_damaged'].median():.4f}%",
            'Median Weighted Units': f"{subset['weighted_damage_units'].median():.1f}",
            'Median Capacity': f"{subset['construction_capacity'].median():.2f}",
            'Median Recovery (months)': f"{subset['recovery_months'].median():.1f}",
            'Median Recovery (years)': f"{subset['recovery_months'].median() / 12:.1f}"
        }
        summary_stats.append(stats)
        
        print(f"\n{quadrant}:")
        print(f"  Count: {stats['Count']:,} ({stats['Percentage']})")
        print(f"  Median % housing damaged: {stats['Median % Housing Damaged']}")
        print(f"  Median weighted units: {stats['Median Weighted Units']}")
        print(f"  Median capacity: {stats['Median Capacity']} permits/month")
        print(f"  Median recovery: {stats['Median Recovery (years)']} years")
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(OUTPUT_DIR / 'quadrant_summary_statistics_normalized.csv', index=False)
    
    # Save full dataset
    output_file = OUTPUT_DIR / 'event_county_quadrants_normalized.csv'
    df.to_csv(output_file, index=False)
    print(f"\n{len(df):,} event-county pairs saved to: {output_file}")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    create_quadrant_scatter(
        df, damage_threshold, capacity_threshold,
        OUTPUT_DIR / 'damage_recovery_quadrants_scatter_normalized.png'
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nOutput files:")
    print("  - event_county_quadrants_normalized.csv")
    print("  - quadrant_summary_statistics_normalized.csv")
    print("  - damage_recovery_quadrants_scatter_normalized.png")


if __name__ == '__main__':
    main()
