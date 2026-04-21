"""
Create side-by-side comparison of dominant quadrant maps:
- Absolute (original weighted units)
- Fully normalized (both damage and capacity normalized)
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'analysis_output'

# Define colors
colors = {
    'High Damage / High Capacity': '#2c7bb6',  # blue
    'High Damage / Low Capacity': '#d7191c',  # red - Critical
    'Low Damage / High Capacity': '#1a9850',  # green
    'Low Damage / Low Capacity': '#fdae61'  # orange
}

# Load both quadrant datasets
print("Loading quadrant data...")
df_abs = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_weightedunits.csv')
df_norm = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_fully_normalized.csv')

# Calculate dominant quadrants for both approaches
def calculate_dominant_quadrants(df):
    county_quadrants = []
    for fips in df['fips'].unique():
        county_data = df[df['fips'] == fips]
        quadrant_counts = Counter(county_data['quadrant'])
        dominant_quadrant = quadrant_counts.most_common(1)[0][0]
        frequency = quadrant_counts.most_common(1)[0][1]
        total_events = len(county_data)
        
        county_quadrants.append({
            'fips': fips,
            'dominant_quadrant': dominant_quadrant,
            'frequency': frequency,
            'total_events': total_events,
            'dominance_pct': 100 * frequency / total_events
        })
    return pd.DataFrame(county_quadrants)

print("Calculating dominant quadrants for absolute approach...")
county_df_abs = calculate_dominant_quadrants(df_abs)

print("Calculating dominant quadrants for fully normalized approach...")
county_df_norm = calculate_dominant_quadrants(df_norm)

# Load county shapefile
print("Loading county shapefile...")
counties = gpd.read_file(DATA_DIR / 'US_counties.shp')

# Filter to coastal states
coastal_state_fips = ['01', '09', '10', '12', '13', '22', '23', '24', '25',
                      '28', '33', '34', '36', '37', '44', '45', '48', '51', '42']
counties = counties[counties['STATEFP'].isin(coastal_state_fips)].copy()
counties['FIPS'] = counties['STATEFP'] + counties['COUNTYFP']

# Prepare data
county_df_abs['fips'] = county_df_abs['fips'].astype(str).str.zfill(5)
county_df_norm['fips'] = county_df_norm['fips'].astype(str).str.zfill(5)

# Merge
counties_abs = counties.merge(county_df_abs, left_on='FIPS', right_on='fips', how='left')
counties_norm = counties.merge(county_df_norm, left_on='FIPS', right_on='fips', how='left')

# Create side-by-side maps
print("\nCreating comparison map...")
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# Left panel: Absolute
ax = axes[0]
from matplotlib.patches import Patch
legend_elements = []

for quadrant, color in colors.items():
    subset = counties_abs[counties_abs['dominant_quadrant'] == quadrant]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.3, alpha=0.8)

no_data = counties_abs[counties_abs['dominant_quadrant'].isna()]
if len(no_data) > 0:
    no_data.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.3)

ax.set_xlim(-100, -65)
ax.set_ylim(24, 48)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('A) Absolute Weighted Units\n(county size bias in both axes)', 
             fontsize=13, pad=15, fontweight='bold')

# Right panel: Fully normalized
ax = axes[1]

for quadrant, color in colors.items():
    subset = counties_norm[counties_norm['dominant_quadrant'] == quadrant]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.3, alpha=0.8)
        count = len(subset)
        legend_elements.append(Patch(facecolor=color, edgecolor='white', 
                                     label=f'{quadrant} ({count})'))

no_data = counties_norm[counties_norm['dominant_quadrant'].isna()]
if len(no_data) > 0:
    no_data.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.3)
    legend_elements.append(Patch(facecolor='lightgray', edgecolor='white',
                                 label=f'No data ({len(no_data)})'))

ax.set_xlim(-100, -65)
ax.set_ylim(24, 48)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('B) Fully Normalized\n(damage & capacity per housing stock)', 
             fontsize=13, pad=15, fontweight='bold')

# Add legend to right panel
ax.legend(handles=legend_elements, loc='lower left', fontsize=10, framealpha=0.95)

# Main title
plt.suptitle('Dominant Recovery Vulnerability Quadrant by County\nComparison: Absolute vs. Fully Normalized', 
             fontsize=16, fontweight='bold', y=0.95)

# Save
output_file = OUTPUT_DIR / 'dominant_quadrant_comparison_map.png'
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Comparison map saved to {output_file}")

# Analyze changes
print("\n" + "=" * 80)
print("COUNTY ASSIGNMENT CHANGES")
print("=" * 80)

# Merge to compare
comparison = county_df_abs[['fips', 'dominant_quadrant']].merge(
    county_df_norm[['fips', 'dominant_quadrant']],
    on='fips', suffixes=('_abs', '_norm'), how='outer'
)

# Counties that changed quadrant
changed = comparison[comparison['dominant_quadrant_abs'] != comparison['dominant_quadrant_norm']]
print(f"\nCounties that changed dominant quadrant: {len(changed)} of {len(comparison)} ({100*len(changed)/len(comparison):.1f}%)")

# Most common transitions
if len(changed) > 0:
    print("\nMost common quadrant transitions:")
    transitions = changed.groupby(['dominant_quadrant_abs', 'dominant_quadrant_norm']).size().sort_values(ascending=False)
    for (from_q, to_q), count in transitions.head(10).items():
        print(f"  {from_q}")
        print(f"    → {to_q}: {count} counties")

# Summary by quadrant
print("\n" + "=" * 80)
print("COUNTY COUNTS BY QUADRANT")
print("=" * 80)
print(f"\n{'Quadrant':<32} {'Absolute':<12} {'Fully Norm':<12} {'Change':<10}")
print("-" * 80)

for quadrant in colors.keys():
    count_abs = (county_df_abs['dominant_quadrant'] == quadrant).sum()
    count_norm = (county_df_norm['dominant_quadrant'] == quadrant).sum()
    change = count_norm - count_abs
    
    print(f"{quadrant:<32} {count_abs:>5} ({count_abs/len(county_df_abs)*100:>4.1f}%)  "
          f"{count_norm:>5} ({count_norm/len(county_df_norm)*100:>4.1f}%)  "
          f"{change:>+4}")

print("\nAnalysis complete!")
