"""
Damage-Recovery Quadrant Analysis (REPAIR COST VERSION)

Alternative version using repair cost (damage-state weighted) instead of unit count.
Divides the space into 4 quadrants based on median splits:
- High Damage / High Capacity → "Resilient but Exposed"
- High Damage / Low Capacity → "Critical Vulnerability"
- Low Damage / High Capacity → "Well-Prepared"
- Low Damage / Low Capacity → "Latent Risk"
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

# Color palette for quadrants
QUADRANT_COLORS = {
    'High Damage / High Capacity': '#2c7bb6',      # Blue - Resilient but Exposed
    'High Damage / Low Capacity': '#d7191c',       # Red - Critical Vulnerability
    'Low Damage / High Capacity': '#1a9850',       # Green - Well-Prepared
    'Low Damage / Low Capacity': '#fdae61'         # Orange - Latent Risk
}

QUADRANT_LABELS = {
    'High Damage / High Capacity': 'Resilient\nbut Exposed',
    'High Damage / Low Capacity': 'Critical\nVulnerability',
    'Low Damage / High Capacity': 'Well-\nPrepared',
    'Low Damage / Low Capacity': 'Latent\nRisk'
}


def load_all_event_county_pairs():
    """
    Load all event-county pairs with damage (repair cost) and recovery potential.
    
    Returns:
        DataFrame with columns: event, fips, repair_cost, recovery_months, construction_capacity
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
    
    # Calculate total damage units (for reference)
    damage_df['total_damage_units'] = (
        damage_df['units_DS1_scaled'].fillna(0) +
        damage_df['units_DS2_scaled'].fillna(0) +
        damage_df['units_DS3_scaled'].fillna(0) +
        damage_df['units_DS4_scaled'].fillna(0)
    )
    
    damage_df['repair_cost_total'] = damage_df['repair_cost_sum_scaled'].fillna(0)
    
    print(f"Loaded {len(damage_df):,} damage records")
    
    # Merge damage and recovery
    merged_df = recovery_df.merge(
        damage_df[['event', 'fips', 'total_damage_units', 'repair_cost_total']],
        on=['event', 'fips'],
        how='inner'
    )
    
    print(f"Merged dataset: {len(merged_df):,} event-county pairs")
    
    # Filter to valid cases (positive damage and recovery)
    valid_df = merged_df[
        (merged_df['repair_cost_total'] > 0) &
        (merged_df['recovery_potential [months]'] > 0) &
        (merged_df['reconstruction_capacity'] > 0)
    ].copy()
    
    print(f"Valid pairs (repair cost > 0, recovery > 0): {len(valid_df):,}")
    
    # Rename columns for clarity
    valid_df = valid_df.rename(columns={
        'reconstruction_capacity': 'construction_capacity',
        'recovery_potential [months]': 'recovery_months',
        'repair_cost_total': 'damage_repair_cost',
        'total_damage_units': 'damage_units'  # Keep for reference
    })
    
    return valid_df[['event', 'fips', 'construction_capacity', 'recovery_months', 
                     'damage_repair_cost', 'damage_units']]


def assign_quadrants(df, damage_threshold=None, capacity_threshold=None):
    """
    Assign quadrants based on damage (repair cost) and capacity using median splits.
    
    Args:
        df: DataFrame with damage_repair_cost and construction_capacity columns
        damage_threshold: Custom threshold for damage (default: median)
        capacity_threshold: Custom threshold for capacity (default: median)
    
    Returns:
        DataFrame with quadrant assignments
    """
    df = df.copy()
    
    # Use median splits if no thresholds provided
    if damage_threshold is None:
        damage_threshold = df['damage_repair_cost'].median()
    if capacity_threshold is None:
        capacity_threshold = df['construction_capacity'].median()
    
    print(f"\nQuadrant thresholds:")
    print(f"  Damage (repair cost): ${damage_threshold:,.2f}")
    print(f"  Capacity: {capacity_threshold:.2f} permits/month")
    
    # Assign quadrants
    df['quadrant'] = 'Unknown'
    
    high_damage = df['damage_repair_cost'] >= damage_threshold
    high_capacity = df['construction_capacity'] >= capacity_threshold
    
    df.loc[high_damage & high_capacity, 'quadrant'] = 'High Damage / High Capacity'
    df.loc[high_damage & ~high_capacity, 'quadrant'] = 'High Damage / Low Capacity'
    df.loc[~high_damage & high_capacity, 'quadrant'] = 'Low Damage / High Capacity'
    df.loc[~high_damage & ~high_capacity, 'quadrant'] = 'Low Damage / Low Capacity'
    
    return df, damage_threshold, capacity_threshold


def create_quadrant_scatter(df, damage_threshold, capacity_threshold, output_file):
    """Create scatter plot with quadrants and marginal distributions"""
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.05, wspace=0.05, 
                          height_ratios=[1, 3, 0.3], width_ratios=[0.3, 3, 1])
    
    # Main scatter plot
    ax_main = fig.add_subplot(gs[1, 1])
    
    for quadrant, color in QUADRANT_COLORS.items():
        subset = df[df['quadrant'] == quadrant]
        ax_main.scatter(subset['damage_repair_cost'], subset['construction_capacity'],
                       c=color, label=quadrant, alpha=0.3, s=10, edgecolors='none')
    
    # Quadrant lines
    ax_main.axhline(capacity_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_main.axvline(damage_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax_main.set_xlabel('Repair Cost ($)', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Construction Capacity (permits/month)', fontsize=12, fontweight='bold')
    ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='upper right', frameon=True, fontsize=9)
    
    # Top histogram (damage)
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    for quadrant, color in QUADRANT_COLORS.items():
        subset = df[df['quadrant'] == quadrant]
        ax_top.hist(subset['damage_repair_cost'], bins=50, color=color, 
                   alpha=0.5, edgecolor='none')
    ax_top.axvline(damage_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_top.set_ylabel('Count', fontsize=10)
    ax_top.set_xscale('log')
    ax_top.tick_params(labelbottom=False)
    
    # Right histogram (capacity)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    for quadrant, color in QUADRANT_COLORS.items():
        subset = df[df['quadrant'] == quadrant]
        ax_right.hist(subset['construction_capacity'], bins=50, color=color,
                     alpha=0.5, orientation='horizontal', edgecolor='none')
    ax_right.axhline(capacity_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_right.set_xlabel('Count', fontsize=10)
    ax_right.set_yscale('log')
    ax_right.tick_params(labelleft=False)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def main():
    print("=" * 80)
    print("DAMAGE-RECOVERY QUADRANT ANALYSIS (REPAIR COST VERSION)")
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
            'Median Repair Cost': f"${subset['damage_repair_cost'].median():,.0f}",
            'Median Capacity': f"{subset['construction_capacity'].median():.2f}",
            'Median Recovery (months)': f"{subset['recovery_months'].median():.1f}",
            'Median Recovery (years)': f"{subset['recovery_months'].median() / 12:.1f}"
        }
        summary_stats.append(stats)
        
        print(f"\n{quadrant}:")
        print(f"  Count: {stats['Count']:,} ({stats['Percentage']})")
        print(f"  Median repair cost: {stats['Median Repair Cost']}")
        print(f"  Median capacity: {stats['Median Capacity']} permits/month")
        print(f"  Median recovery: {stats['Median Recovery (years)']} years")
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(OUTPUT_DIR / 'quadrant_summary_statistics_repaircost.csv', index=False)
    
    # Save full dataset
    output_file = OUTPUT_DIR / 'event_county_quadrants_repaircost.csv'
    df.to_csv(output_file, index=False)
    print(f"\n{len(df):,} event-county pairs saved to: {output_file}")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    create_quadrant_scatter(
        df, damage_threshold, capacity_threshold,
        OUTPUT_DIR / 'damage_recovery_quadrants_scatter_repaircost.png'
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nOutput files:")
    print("  - event_county_quadrants_repaircost.csv")
    print("  - quadrant_summary_statistics_repaircost.csv")
    print("  - damage_recovery_quadrants_scatter_repaircost.png")


if __name__ == '__main__':
    main()
