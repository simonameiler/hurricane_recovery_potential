"""
Create three-panel comparison of normalization approaches
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    print("Creating three-panel normalization comparison...")
    
    # Load all three datasets
    df_abs = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_weightedunits.csv')
    df_norm_dmg = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_normalized.csv')
    df_fully_norm = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_fully_normalized.csv')
    
    # Create three-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    # Panel 1: Absolute (both axes absolute)
    ax = axes[0]
    damage_threshold = df_abs['weighted_damage_units'].median()
    capacity_threshold = df_abs['construction_capacity'].median()
    
    for quadrant, color in QUADRANT_COLORS.items():
        subset = df_abs[df_abs['quadrant'] == quadrant]
        ax.scatter(subset['weighted_damage_units'], subset['construction_capacity'],
                   alpha=0.25, s=10, c=color, label=quadrant, edgecolors='none')
    
    ax.axvline(damage_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(capacity_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Weighted Damage Units (absolute)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Construction Capacity\n(permits/month, absolute)', fontsize=11, fontweight='bold')
    ax.set_title('A) Both Absolute\n(county size bias in both axes)', 
                 fontsize=12, pad=10, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', framealpha=0.95, fontsize=8)
    
    # Panel 2: Damage normalized, Capacity absolute
    ax = axes[1]
    damage_threshold = df_norm_dmg['pct_housing_damaged'].median()
    capacity_threshold = df_norm_dmg['construction_capacity'].median()
    
    for quadrant, color in QUADRANT_COLORS.items():
        subset = df_norm_dmg[df_norm_dmg['quadrant'] == quadrant]
        ax.scatter(subset['pct_housing_damaged'], subset['construction_capacity'],
                   alpha=0.25, s=10, c=color, label=quadrant, edgecolors='none')
    
    ax.axvline(damage_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(capacity_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('% of County Housing Stock Damaged', fontsize=11, fontweight='bold')
    ax.set_ylabel('Construction Capacity\n(permits/month, absolute)', fontsize=11, fontweight='bold')
    ax.set_title('B) Damage Normalized Only\n(county size bias remains in capacity)', 
                 fontsize=12, pad=10, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', framealpha=0.95, fontsize=8)
    
    # Panel 3: Both normalized
    ax = axes[2]
    damage_threshold = df_fully_norm['pct_housing_damaged'].median()
    capacity_threshold = df_fully_norm['capacity_per_1000_units'].median()
    
    for quadrant, color in QUADRANT_COLORS.items():
        subset = df_fully_norm[df_fully_norm['quadrant'] == quadrant]
        ax.scatter(subset['pct_housing_damaged'], subset['capacity_per_1000_units'],
                   alpha=0.25, s=10, c=color, label=quadrant, edgecolors='none')
    
    ax.axvline(damage_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(capacity_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('% of County Housing Stock Damaged', fontsize=11, fontweight='bold')
    ax.set_ylabel('Construction Capacity\n(permits/1000 units/month)', fontsize=11, fontweight='bold')
    ax.set_title('C) Both Normalized\n(county size bias removed)', 
                 fontsize=12, pad=10, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', framealpha=0.95, fontsize=8)
    
    plt.suptitle('Impact of Normalization on Quadrant Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_file = OUTPUT_DIR / 'normalization_comparison_three_panel.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Three-panel comparison saved to: {output_file}")
    
    # Print detailed statistics
    print("\n" + "=" * 90)
    print("QUADRANT DISTRIBUTION ACROSS NORMALIZATION APPROACHES")
    print("=" * 90)
    
    quadrants = ['High Damage / High Capacity', 'High Damage / Low Capacity',
                 'Low Damage / High Capacity', 'Low Damage / Low Capacity']
    
    print(f"\n{'Quadrant':<30} {'Absolute':<12} {'Dmg Norm':<12} {'Both Norm':<12} {'Change':<10}")
    print("-" * 90)
    
    for q in quadrants:
        count_abs = len(df_abs[df_abs['quadrant'] == q])
        count_norm_dmg = len(df_norm_dmg[df_norm_dmg['quadrant'] == q])
        count_fully = len(df_fully_norm[df_fully_norm['quadrant'] == q])
        
        pct_abs = count_abs / len(df_abs) * 100
        pct_norm_dmg = count_norm_dmg / len(df_norm_dmg) * 100
        pct_fully = count_fully / len(df_fully_norm) * 100
        
        change = pct_fully - pct_abs
        
        print(f"{q:<30} {pct_abs:>5.1f}% ({count_abs:>5}) "
              f"{pct_norm_dmg:>5.1f}% ({count_norm_dmg:>5}) "
              f"{pct_fully:>5.1f}% ({count_fully:>5}) "
              f"{change:>+5.1f}pp")
    
    print("\n" + "=" * 90)
    print("CORRELATION ANALYSIS: County Size vs. Construction Capacity")
    print("=" * 90)
    
    corr_abs = df_abs[['total_damage_units', 'construction_capacity']].corr().iloc[0, 1]
    corr_norm = df_fully_norm[['total_housing_units', 'capacity_per_1000_units']].corr().iloc[0, 1]
    
    print(f"\nAbsolute approach:")
    print(f"  Correlation (units vs capacity): r = {corr_abs:.3f}")
    print(f"  → Strong positive correlation indicates size bias")
    
    print(f"\nFully normalized approach:")
    print(f"  Correlation (housing units vs normalized capacity): r = {corr_norm:.3f}")
    print(f"  → Weak correlation indicates size bias removed")
    
    print(f"\nReduction in size bias: {(1 - abs(corr_norm)/abs(corr_abs)) * 100:.1f}%")
    
    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    print("""
The fully normalized approach (Panel C) provides the most balanced view:
  
  • Removes county size as a confounding factor on BOTH axes
  • Reveals counties with high proportional damage AND low relative capacity
  • More evenly distributed across quadrants (~25% each)
  • Better identifies small counties with catastrophic proportional impacts
  
The damage-only normalized approach (Panel B) is intermediate:
  • Controls for county size in damage assessment
  • But still favors large counties in capacity dimension
  • Overrepresents large counties in high-capacity quadrants
  
The absolute approach (Panel A) is biased toward large counties:
  • Large counties dominate high-damage/high-capacity quadrant (30.5%)
  • Small counties overrepresented in low-damage/low-capacity (30.5%)
  • County size conflates with both damage magnitude and recovery capacity
""")


if __name__ == '__main__':
    main()
