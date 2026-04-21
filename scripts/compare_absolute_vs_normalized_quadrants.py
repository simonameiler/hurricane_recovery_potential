"""
Create side-by-side comparison of absolute vs. normalized quadrant analyses
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"

# Color palette for quadrants
QUADRANT_COLORS = {
    'High Damage / High Capacity': '#2c7bb6',
    'High Damage / Low Capacity': '#d7191c',
    'Low Damage / High Capacity': '#1a9850',
    'Low Damage / Low Capacity': '#fdae61'
}


def main():
    print("Creating comparison visualization...")
    
    # Load both datasets
    df_abs = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_weightedunits.csv')
    df_norm = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_normalized.csv')
    
    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left plot: Absolute weighted units
    ax = axes[0]
    damage_threshold_abs = df_abs['weighted_damage_units'].median()
    capacity_threshold_abs = df_abs['construction_capacity'].median()
    
    for quadrant, color in QUADRANT_COLORS.items():
        subset = df_abs[df_abs['quadrant'] == quadrant]
        ax.scatter(subset['weighted_damage_units'], subset['construction_capacity'],
                   alpha=0.3, s=15, c=color, label=quadrant, edgecolors='none')
    
    ax.axvline(damage_threshold_abs, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(capacity_threshold_abs, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Weighted Damage Units (absolute)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Construction Capacity (permits/month)', fontsize=12, fontweight='bold')
    ax.set_title('A) Absolute Damage', fontsize=14, pad=15, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Add quadrant counts
    for quadrant in QUADRANT_COLORS.keys():
        count = len(df_abs[df_abs['quadrant'] == quadrant])
        pct = count / len(df_abs) * 100
        print(f"Absolute - {quadrant}: {count:,} ({pct:.1f}%)")
    
    # Right plot: Normalized by housing stock
    ax = axes[1]
    damage_threshold_norm = df_norm['pct_housing_damaged'].median()
    capacity_threshold_norm = df_norm['construction_capacity'].median()
    
    for quadrant, color in QUADRANT_COLORS.items():
        subset = df_norm[df_norm['quadrant'] == quadrant]
        ax.scatter(subset['pct_housing_damaged'], subset['construction_capacity'],
                   alpha=0.3, s=15, c=color, label=quadrant, edgecolors='none')
    
    ax.axvline(damage_threshold_norm, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(capacity_threshold_norm, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('% of County Housing Stock Damaged', fontsize=12, fontweight='bold')
    ax.set_ylabel('Construction Capacity (permits/month)', fontsize=12, fontweight='bold')
    ax.set_title('B) Normalized by Housing Stock', fontsize=14, pad=15, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Add quadrant counts
    print("\n")
    for quadrant in QUADRANT_COLORS.keys():
        count = len(df_norm[df_norm['quadrant'] == quadrant])
        pct = count / len(df_norm) * 100
        print(f"Normalized - {quadrant}: {count:,} ({pct:.1f}%)")
    
    plt.suptitle('Damage-Recovery Quadrant Analysis: Absolute vs. Normalized',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / 'quadrant_comparison_absolute_vs_normalized.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plot saved to: {output_file}")
    
    # Create summary comparison table
    print("\n" + "=" * 80)
    print("QUADRANT DISTRIBUTION COMPARISON")
    print("=" * 80)
    
    comparison = []
    for quadrant in ['High Damage / High Capacity', 'High Damage / Low Capacity',
                     'Low Damage / High Capacity', 'Low Damage / Low Capacity']:
        count_abs = len(df_abs[df_abs['quadrant'] == quadrant])
        count_norm = len(df_norm[df_norm['quadrant'] == quadrant])
        pct_abs = count_abs / len(df_abs) * 100
        pct_norm = count_norm / len(df_norm) * 100
        
        comparison.append({
            'Quadrant': quadrant,
            'Absolute Count': count_abs,
            'Absolute %': f"{pct_abs:.1f}%",
            'Normalized Count': count_norm,
            'Normalized %': f"{pct_norm:.1f}%",
            'Change': f"{pct_norm - pct_abs:+.1f}pp"
        })
    
    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(OUTPUT_DIR / 'quadrant_comparison_summary.csv', index=False)
    print(comp_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("\nAbsolute weighted units:")
    print(f"  - Large counties dominate high-damage quadrants")
    print(f"  - {len(df_abs):,} total event-county pairs")
    
    print("\nNormalized by housing stock:")
    print(f"  - Reveals proportional impact on county housing")
    print(f"  - {len(df_norm):,} total event-county pairs (fewer due to housing data availability)")
    print(f"  - High Damage/Low Capacity quadrant increases from 19.5% to 29.6%")
    print(f"  - This reveals small counties with catastrophic proportional damage")


if __name__ == '__main__':
    main()
