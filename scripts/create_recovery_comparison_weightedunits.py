"""
Create recovery comparison plots for weighted units quadrant analysis.
Shows recovery time distribution and damage-to-capacity ratio across quadrants.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = BASE_DIR / 'analysis_output'

# Load quadrant data
quadrant_file = OUTPUT_DIR / 'event_county_quadrants_weightedunits.csv'
df = pd.read_csv(quadrant_file)

print(f"Loaded {len(df)} event-county pairs")

# Color palette (matching scatter plot)
QUADRANT_COLORS = {
    'High Damage / High Capacity': '#2c7bb6',      # Blue
    'High Damage / Low Capacity': '#d7191c',       # Red - Critical Vulnerability
    'Low Damage / High Capacity': '#1a9850',       # Green
    'Low Damage / Low Capacity': '#fdae61'         # Orange
}

QUADRANT_LABELS = {
    'High Damage / High Capacity': 'Resilient\nbut Exposed',
    'High Damage / Low Capacity': 'Critical\nVulnerability',
    'Low Damage / High Capacity': 'Well-\nPrepared',
    'Low Damage / Low Capacity': 'Latent\nRisk'
}

# Create comparison plot
print("\nCreating recovery comparison plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Define quadrant order
quadrant_order = [
    'Low Damage / High Capacity',
    'Low Damage / Low Capacity',
    'High Damage / High Capacity',
    'High Damage / Low Capacity'
]

colors_ordered = [QUADRANT_COLORS[q] for q in quadrant_order]
labels_ordered = [QUADRANT_LABELS[q] for q in quadrant_order]

# Panel 1: Recovery time distribution
ax1 = axes[0]
recovery_data = [df[df['quadrant'] == q]['recovery_months'].values for q in quadrant_order]
bp1 = ax1.boxplot(recovery_data, labels=labels_ordered, patch_artist=True,
                  showfliers=False, widths=0.6)

for patch, color in zip(bp1['boxes'], colors_ordered):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_ylabel('Recovery Time (months)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Quadrant', fontsize=12, fontweight='bold')
ax1.set_title('Recovery Time Distribution by Quadrant', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_yscale('log')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')

# Add median values as text
for i, q in enumerate(quadrant_order):
    median_val = df[df['quadrant'] == q]['recovery_months'].median()
    ax1.text(i+1, median_val, f'{median_val:.0f}', 
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel 2: Weighted damage-to-capacity ratio
ax2 = axes[1]
df['damage_capacity_ratio'] = df['weighted_damage_units'] / df['construction_capacity']
ratio_data = [df[df['quadrant'] == q]['damage_capacity_ratio'].values for q in quadrant_order]
bp2 = ax2.boxplot(ratio_data, labels=labels_ordered, patch_artist=True,
                  showfliers=False, widths=0.6)

for patch, color in zip(bp2['boxes'], colors_ordered):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_ylabel('Weighted Damage-to-Capacity Ratio', fontsize=12, fontweight='bold')
ax2.set_xlabel('Quadrant', fontsize=12, fontweight='bold')
ax2.set_title('Weighted Damage-to-Capacity Ratio by Quadrant', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_yscale('log')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')

# Add median values as text
for i, q in enumerate(quadrant_order):
    median_val = df[df['quadrant'] == q]['damage_capacity_ratio'].median()
    ax2.text(i+1, median_val, f'{median_val:.0f}', 
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()

# Save
output_file = OUTPUT_DIR / 'quadrant_recovery_comparison_weightedunits.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {output_file}")

# Print statistics
print("\n=== RECOVERY TIME STATISTICS (months) ===")
for q in quadrant_order:
    subset = df[df['quadrant'] == q]['recovery_months']
    print(f"\n{q}:")
    print(f"  Median: {subset.median():.1f} ({subset.median()/12:.1f} years)")
    print(f"  Mean: {subset.mean():.1f}")
    print(f"  Q1-Q3: {subset.quantile(0.25):.1f} - {subset.quantile(0.75):.1f}")
    print(f"  Min-Max: {subset.min():.1f} - {subset.max():.1f}")

print("\n=== DAMAGE-TO-CAPACITY RATIO STATISTICS ===")
for q in quadrant_order:
    subset = df[df['quadrant'] == q]['damage_capacity_ratio']
    print(f"\n{q}:")
    print(f"  Median: {subset.median():.1f}")
    print(f"  Mean: {subset.mean():.1f}")
    print(f"  Q1-Q3: {subset.quantile(0.25):.1f} - {subset.quantile(0.75):.1f}")
