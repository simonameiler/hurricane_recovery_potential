"""
Damage-Recovery Quadrant Analysis (DAMAGE FRACTION VERSION)

Uses damage fraction (repair_cost / replacement_cost) and repair-time weighted units.
This normalizes by county size and damage severity.
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


def load_county_replacement_costs():
    """Load total replacement cost per county from exposure data"""
    print("Loading county replacement costs from exposure data...")
    
    # Get list of states with exposure data in states/ subdirectory
    state_files = list((EXPOSURE_DIR / "states").glob("*.hdf5"))
    
    replacement_costs = []
    
    for i, filepath in enumerate(state_files):
        if i % 5 == 0:
            print(f"  Processing file {i+1}/{len(state_files)}")
        
        try:
            df = pd.read_hdf(filepath, 'exposures')
            # Create FIPS from state and county codes
            if 'stcode' in df.columns and 'ccode' in df.columns:
                df['fips'] = (df['stcode'].astype(str).str.zfill(2) + 
                             df['ccode'].astype(str).str.zfill(3))
            elif 'State' in df.columns and 'County' in df.columns:
                df['fips'] = (df['State'].astype(str).str.zfill(2) + 
                             df['County'].astype(str).str.zfill(3))
            else:
                print(f"  Warning: No FIPS columns in {filepath.name}")
                continue
            
            # Use 'value' column as replacement cost (this is the total exposure value)
            county_agg = df.groupby('fips')['value'].sum().reset_index()
            county_agg.columns = ['fips', 'ReplacementCost']
            
            replacement_costs.append(county_agg)
        except Exception as e:
            print(f"  Warning: Could not load {filepath.name}: {e}")
            continue
    
    # Combine all states
    all_costs = pd.concat(replacement_costs, ignore_index=True)
    all_costs = all_costs.groupby('fips')['ReplacementCost'].sum().reset_index()
    
    print(f"Loaded replacement costs for {len(all_costs)} counties")
    
    return all_costs


def load_all_event_county_pairs():
    """
    Load all event-county pairs with:
    - Damage fraction (repair_cost / replacement_cost)  
    - Weighted damage units (units weighted by repair time)
    """
    print("Loading all event-county pairs...")
    
    # Load county replacement costs
    replacement_df = load_county_replacement_costs()
    replacement_df['fips'] = replacement_df['fips'].astype(str).str.zfill(5)
    
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
    
    # Calculate total units (unweighted)
    damage_df['total_damage_units'] = sum(
        damage_df[col].fillna(0)
        for col in DS_REPAIR_TIMES.keys()
    )
    
    damage_df['repair_cost_total'] = damage_df['repair_cost_sum_scaled'].fillna(0)
    
    print(f"Loaded {len(damage_df):,} damage records")
    
    # Merge with replacement costs to get damage fraction
    damage_df = damage_df.merge(replacement_df[['fips', 'ReplacementCost']], 
                                on='fips', how='left')
    
    # Calculate damage fraction (repair cost / replacement cost)
    damage_df['damage_fraction'] = (damage_df['repair_cost_total'] / 
                                    damage_df['ReplacementCost'].fillna(1))
    
    # Clip to [0, 1] range (some might exceed due to uncertainties)
    damage_df['damage_fraction'] = damage_df['damage_fraction'].clip(0, 1)
    
    # Merge damage and recovery
    merged_df = recovery_df.merge(
        damage_df[['event', 'fips', 'damage_fraction', 'weighted_damage_units', 
                  'total_damage_units', 'repair_cost_total', 'ReplacementCost']],
        on=['event', 'fips'],
        how='inner'
    )
    
    print(f"Merged dataset: {len(merged_df):,} event-county pairs")
    
    # Filter to valid cases
    valid_df = merged_df[
        (merged_df['damage_fraction'] > 0) &
        (merged_df['recovery_potential [months]'] > 0) &
        (merged_df['reconstruction_capacity'] > 0)
    ].copy()
    
    print(f"Valid pairs (damage_fraction > 0, recovery > 0): {len(valid_df):,}")
    
    # Rename columns for clarity
    valid_df = valid_df.rename(columns={
        'reconstruction_capacity': 'construction_capacity',
        'recovery_potential [months]': 'recovery_months'
    })
    
    return valid_df[['event', 'fips', 'construction_capacity', 'recovery_months', 
                     'damage_fraction', 'weighted_damage_units', 'total_damage_units',
                     'repair_cost_total', 'ReplacementCost']]


def assign_quadrants(df, damage_threshold=None, capacity_threshold=None):
    """Assign quadrants based on damage fraction and capacity"""
    df = df.copy()
    
    # Use median splits if no thresholds provided
    if damage_threshold is None:
        damage_threshold = df['damage_fraction'].median()
    if capacity_threshold is None:
        capacity_threshold = df['construction_capacity'].median()
    
    print(f"\nQuadrant thresholds:")
    print(f"  Damage fraction: {damage_threshold:.4f} ({damage_threshold*100:.2f}%)")
    print(f"  Capacity: {capacity_threshold:.2f} permits/month")
    
    # Assign quadrants
    df['quadrant'] = 'Unknown'
    
    high_damage = df['damage_fraction'] >= damage_threshold
    high_capacity = df['construction_capacity'] >= capacity_threshold
    
    df.loc[high_damage & high_capacity, 'quadrant'] = 'High Damage / High Capacity'
    df.loc[high_damage & ~high_capacity, 'quadrant'] = 'High Damage / Low Capacity'
    df.loc[~high_damage & high_capacity, 'quadrant'] = 'Low Damage / High Capacity'
    df.loc[~high_damage & ~high_capacity, 'quadrant'] = 'Low Damage / Low Capacity'
    
    return df, damage_threshold, capacity_threshold


def create_quadrant_scatter(df, damage_threshold, capacity_threshold, output_file):
    """Create scatter plot with damage fraction on y-axis"""
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.05, wspace=0.05, 
                          height_ratios=[1, 3, 0.3], width_ratios=[0.3, 3, 1])
    
    # Main scatter plot
    ax_main = fig.add_subplot(gs[1, 1])
    
    for quadrant, color in QUADRANT_COLORS.items():
        subset = df[df['quadrant'] == quadrant]
        ax_main.scatter(subset['construction_capacity'], subset['damage_fraction'],
                       c=color, label=quadrant, alpha=0.3, s=10, edgecolors='none')
    
    # Quadrant lines
    ax_main.axhline(damage_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_main.axvline(capacity_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax_main.set_xlabel('Construction Capacity (permits/month)', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Damage Fraction (repair cost / replacement cost)', fontsize=12, fontweight='bold')
    ax_main.set_xscale('log')
    ax_main.set_ylim(0, 1)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='upper right', frameon=True, fontsize=9)
    
    # Top histogram (capacity)
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    for quadrant, color in QUADRANT_COLORS.items():
        subset = df[df['quadrant'] == quadrant]
        ax_top.hist(subset['construction_capacity'], bins=50, color=color, 
                   alpha=0.5, edgecolor='none')
    ax_top.axvline(capacity_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_top.set_ylabel('Count', fontsize=10)
    ax_top.set_xscale('log')
    ax_top.tick_params(labelbottom=False)
    
    # Right histogram (damage fraction)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    for quadrant, color in QUADRANT_COLORS.items():
        subset = df[df['quadrant'] == quadrant]
        ax_right.hist(subset['damage_fraction'], bins=50, color=color,
                     alpha=0.5, orientation='horizontal', edgecolor='none')
    ax_right.axhline(damage_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_right.set_xlabel('Count', fontsize=10)
    ax_right.tick_params(labelleft=False)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def main():
    print("=" * 80)
    print("DAMAGE-RECOVERY QUADRANT ANALYSIS (DAMAGE FRACTION VERSION)")
    print("=" * 80)
    
    # Load data
    df = load_all_event_county_pairs()
    
    # Assign quadrants
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
            'Median Damage Fraction': f"{subset['damage_fraction'].median():.3f}",
            'Median Weighted Units': f"{subset['weighted_damage_units'].median():.1f}",
            'Median Capacity': f"{subset['construction_capacity'].median():.2f}",
            'Median Recovery (months)': f"{subset['recovery_months'].median():.1f}",
            'Median Recovery (years)': f"{subset['recovery_months'].median() / 12:.1f}"
        }
        summary_stats.append(stats)
        
        print(f"\n{quadrant}:")
        print(f"  Count: {stats['Count']:,} ({stats['Percentage']})")
        print(f"  Median damage fraction: {stats['Median Damage Fraction']}")
        print(f"  Median weighted units: {stats['Median Weighted Units']}")
        print(f"  Median capacity: {stats['Median Capacity']} permits/month")
        print(f"  Median recovery: {stats['Median Recovery (years)']} years")
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(OUTPUT_DIR / 'quadrant_summary_statistics_damfraction.csv', index=False)
    
    # Save full dataset
    output_file = OUTPUT_DIR / 'event_county_quadrants_damfraction.csv'
    df.to_csv(output_file, index=False)
    print(f"\n{len(df):,} event-county pairs saved to: {output_file}")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    create_quadrant_scatter(
        df, damage_threshold, capacity_threshold,
        OUTPUT_DIR / 'damage_recovery_quadrants_scatter_damfraction.png'
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nOutput files:")
    print("  - event_county_quadrants_damfraction.csv")
    print("  - quadrant_summary_statistics_damfraction.csv")
    print("  - damage_recovery_quadrants_scatter_damfraction.png")


if __name__ == '__main__':
    main()
