"""
Create side-by-side scatter plots comparing absolute and normalized quadrant analyses
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"

# Load data
print("Loading data...")
df_abs = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_weightedunits.csv')
df_norm = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_fully_normalized.csv')

print(f"Absolute data: {len(df_abs):,} event-county pairs")
print(f"Normalized data: {len(df_norm):,} event-county pairs")

# Color scheme
quadrant_colors = {
    'High Damage / High Capacity': '#2c7bb6',
    'High Damage / Low Capacity': '#d7191c',
    'Low Damage / High Capacity': '#1a9850',
    'Low Damage / Low Capacity': '#fdae61'
}

# Create figure
print("\nCreating side-by-side scatter plots...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Panel A: Absolute analysis
ax = axes[0]
for quadrant, color in quadrant_colors.items():
    subset = df_abs[df_abs['quadrant'] == quadrant]
    ax.scatter(subset['weighted_damage_units'], subset['construction_capacity'],
               alpha=0.5, s=30, c=color, label=f"{quadrant} ({len(subset):,})",
               edgecolors='none')

# Add median threshold lines
abs_median_damage = df_abs['weighted_damage_units'].median()
abs_median_capacity = df_abs['construction_capacity'].median()
ax.axvline(abs_median_damage, color='black', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Damage median ({abs_median_damage:,.0f} units)')
ax.axhline(abs_median_capacity, color='black', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Capacity median ({abs_median_capacity:.0f} permits/month)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Weighted Damage Units (average per event)', fontsize=13, fontweight='bold')
ax.set_ylabel('Construction Capacity (permits/month)', fontsize=13, fontweight='bold')
ax.set_title('(a) Absolute Quadrant Analysis\nRaw Values without Normalization', 
             fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper left', fontsize=9, framealpha=0.95, ncol=1)
ax.grid(True, alpha=0.3, which='both')

# Add correlation annotation
corr_abs = df_abs[['weighted_damage_units', 'construction_capacity']].corr().iloc[0, 1]
ax.text(0.98, 0.02, f'Correlation: r = {corr_abs:.3f}',
        transform=ax.transAxes, fontsize=11, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel B: Normalized analysis
ax = axes[1]
for quadrant, color in quadrant_colors.items():
    subset = df_norm[df_norm['quadrant'] == quadrant]
    ax.scatter(subset['pct_housing_damaged'], subset['capacity_per_1000_units'],
               alpha=0.5, s=30, c=color, label=f"{quadrant} ({len(subset):,})",
               edgecolors='none')

# Add median threshold lines
norm_median_damage = df_norm['pct_housing_damaged'].median()
norm_median_capacity = df_norm['capacity_per_1000_units'].median()
ax.axvline(norm_median_damage, color='black', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Damage median ({norm_median_damage:.1f}%)')
ax.axhline(norm_median_capacity, color='black', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Capacity median ({norm_median_capacity:.3f} per 1000)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('% of Housing Damaged (average per event)', fontsize=13, fontweight='bold')
ax.set_ylabel('Construction Capacity (permits/1000 units/month)', fontsize=13, fontweight='bold')
ax.set_title('(b) Normalized Quadrant Analysis\nScaled by Total Exposed Housing Units', 
             fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper left', fontsize=9, framealpha=0.95, ncol=1)
ax.grid(True, alpha=0.3, which='both')

# Add correlation annotation
corr_norm = df_norm[['pct_housing_damaged', 'capacity_per_1000_units']].corr().iloc[0, 1]
ax.text(0.98, 0.02, f'Correlation: r = {corr_norm:.3f}',
        transform=ax.transAxes, fontsize=11, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Overall title
fig.suptitle('Hurricane Recovery Quadrant Analysis: Absolute vs. Normalized Perspectives', 
             fontsize=16, fontweight='bold', y=0.98)

# Add explanatory text
fig.text(0.5, 0.01,
         '(a) Absolute values favor large counties with high total capacity | ' +
         '(b) Normalization reveals per-capita vulnerability across all county sizes\n' +
         f'Normalization reduces correlation from r={corr_abs:.3f} to r={corr_norm:.3f}, ' +
         f'removing county size bias',
         ha='center', fontsize=10, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

# Save
output_file = OUTPUT_DIR / 'scatter_comparison_absolute_vs_normalized.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved to: {output_file}")

# Print summary statistics
print("\n" + "=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)

print("\nABSOLUTE ANALYSIS:")
print(f"  Weighted damage - Range: {df_abs['weighted_damage_units'].min():.1f} to {df_abs['weighted_damage_units'].max():,.0f} units")
print(f"  Weighted damage - Median: {abs_median_damage:,.0f} units")
print(f"  Capacity - Range: {df_abs['construction_capacity'].min():.1f} to {df_abs['construction_capacity'].max():.0f} permits/month")
print(f"  Capacity - Median: {abs_median_capacity:.0f} permits/month")
print(f"  Correlation (damage vs capacity): r = {corr_abs:.3f}")

print("\nNORMALIZED ANALYSIS:")
print(f"  % Housing damaged - Range: {df_norm['pct_housing_damaged'].min():.4f}% to {df_norm['pct_housing_damaged'].max():.1f}%")
print(f"  % Housing damaged - Median: {norm_median_damage:.2f}%")
print(f"  Capacity/1000 - Range: {df_norm['capacity_per_1000_units'].min():.4f} to {df_norm['capacity_per_1000_units'].max():.2f} permits/1000/month")
print(f"  Capacity/1000 - Median: {norm_median_capacity:.3f} permits/1000/month")
print(f"  Correlation (% damaged vs capacity/1000): r = {corr_norm:.3f}")

print("\nQUADRANT DISTRIBUTION COMPARISON:")
print(f"{'Quadrant':<35} {'Absolute':<15} {'Normalized':<15} {'Change':<15}")
print("-" * 80)
for quadrant in quadrant_colors.keys():
    abs_count = (df_abs['quadrant'] == quadrant).sum()
    abs_pct = 100 * abs_count / len(df_abs)
    norm_count = (df_norm['quadrant'] == quadrant).sum()
    norm_pct = 100 * norm_count / len(df_norm)
    change = norm_pct - abs_pct
    print(f"{quadrant:<35} {abs_count:>6} ({abs_pct:>4.1f}%)   {norm_count:>6} ({norm_pct:>4.1f}%)   {change:>+5.1f}pp")

print("\n" + "=" * 100)
print("COMPLETE")
print("=" * 100)
