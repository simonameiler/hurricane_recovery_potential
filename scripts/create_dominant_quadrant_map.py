"""
Create spatial map showing dominant quadrant assignment for each county
based on most frequent quadrant across all events.
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

# Load quadrant data
quadrant_file = OUTPUT_DIR / 'event_county_quadrants_weightedunits.csv'
df = pd.read_csv(quadrant_file)

# Calculate dominant quadrant for each county
print("Calculating dominant quadrant per county...")
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

county_df = pd.DataFrame(county_quadrants)

# Save summary
summary_file = OUTPUT_DIR / 'dominant_quadrant_by_county.csv'
county_df.to_csv(summary_file, index=False)
print(f"Saved county summary to {summary_file}")

# Print statistics
print("\n=== DOMINANT QUADRANT DISTRIBUTION ===")
print(county_df['dominant_quadrant'].value_counts())
print(f"\nMean dominance: {county_df['dominance_pct'].mean():.1f}%")
print(f"Median dominance: {county_df['dominance_pct'].median():.1f}%")

# Load county shapefile
print("\nLoading county shapefile...")
counties = gpd.read_file(DATA_DIR / 'US_counties.shp')

# Filter to 19 coastal states (using FIPS state codes)
coastal_state_fips = ['01', '09', '10', '12', '13', '22', '23', '24', '25',
                      '28', '33', '34', '36', '37', '44', '45', '48', '51', '42']
counties = counties[counties['STATEFP'].isin(coastal_state_fips)].copy()

# Ensure FIPS is string with zero-padding (combine STATEFP + COUNTYFP)
counties['FIPS'] = counties['STATEFP'] + counties['COUNTYFP']
county_df['fips'] = county_df['fips'].astype(str).str.zfill(5)

# Merge with quadrant data
counties_merged = counties.merge(county_df, left_on='FIPS', right_on='fips', how='left')

# Create map
print("\nCreating map...")
fig, ax = plt.subplots(figsize=(16, 10))

# Define colors matching analysis (same as scatter plot)
colors = {
    'High Damage / High Capacity': '#2c7bb6',  # blue
    'High Damage / Low Capacity': '#d7191c',  # red - Critical Vulnerability
    'Low Damage / High Capacity': '#1a9850',  # green
    'Low Damage / Low Capacity': '#fdae61'  # orange
}

# Plot each quadrant
from matplotlib.patches import Patch
legend_elements = []

for quadrant, color in colors.items():
    subset = counties_merged[counties_merged['dominant_quadrant'] == quadrant]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.3)
        count = len(subset)
        legend_elements.append(Patch(facecolor=color, edgecolor='white', 
                                     label=f'{quadrant} ({count} counties)'))

# Plot counties with no data
no_data = counties_merged[counties_merged['dominant_quadrant'].isna()]
if len(no_data) > 0:
    no_data.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.3)
    legend_elements.append(Patch(facecolor='lightgray', edgecolor='white',
                                 label=f'No data ({len(no_data)} counties)'))

# Formatting
ax.set_xlim(-100, -65)
ax.set_ylim(24, 48)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Dominant Recovery Vulnerability Quadrant by County\n(Most Frequent Assignment Across All Hurricane Events)', 
             fontsize=14, pad=20)
ax.legend(handles=legend_elements, loc='lower left', fontsize=11, framealpha=0.95)

# Save
output_file = OUTPUT_DIR / 'dominant_quadrant_map.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Map saved to {output_file}")

print("\n=== COUNTIES BY DOMINANT QUADRANT ===")
for quadrant in colors.keys():
    count = (county_df['dominant_quadrant'] == quadrant).sum()
    pct = 100 * count / len(county_df)
    print(f"{quadrant}: {count} counties ({pct:.1f}%)")
