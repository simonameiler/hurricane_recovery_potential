"""
Create 3-panel spatial comparison map:
a) Absolute quadrant analysis
b) Normalized quadrant analysis  
c) Critical county classification
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"
DATA_DIR = BASE_DIR / "data"

# Load data
print("Loading data...")

# 1. Absolute quadrants
df_abs = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_weightedunits.csv')
abs_dominant = df_abs.groupby('fips')['quadrant'].agg(lambda x: x.mode()[0]).reset_index()
abs_dominant.columns = ['fips', 'absolute_quadrant']
abs_dominant['fips'] = abs_dominant['fips'].astype(str).str.zfill(5)

# 2. Normalized quadrants
df_norm = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_fully_normalized.csv')
norm_dominant = df_norm.groupby('fips')['quadrant'].agg(lambda x: x.mode()[0]).reset_index()
norm_dominant.columns = ['fips', 'normalized_quadrant']
norm_dominant['fips'] = norm_dominant['fips'].astype(str).str.zfill(5)

# 3. Critical county classification
critical_df = pd.read_csv(OUTPUT_DIR / 'critical_counties_absolute_vs_normalized.csv')
critical_df['fips'] = critical_df['fips'].astype(str).str.zfill(5)

# Load county boundaries
print("Loading county boundaries...")
counties_shp = gpd.read_file(DATA_DIR / 'US_counties.shp')
coastal_state_fips = ['01', '09', '10', '12', '13', '22', '23', '24', '25',
                      '28', '33', '34', '36', '37', '44', '45', '48', '51', '42']
counties_shp = counties_shp[counties_shp['STATEFP'].isin(coastal_state_fips)].copy()
counties_shp['FIPS'] = counties_shp['STATEFP'] + counties_shp['COUNTYFP']

# Merge data with geometries
print("Merging data with geometries...")
counties_abs = counties_shp.merge(abs_dominant, left_on='FIPS', right_on='fips', how='left')
counties_norm = counties_shp.merge(norm_dominant, left_on='FIPS', right_on='fips', how='left')
counties_crit = counties_shp.merge(critical_df[['fips', 'critical_status']], 
                                   left_on='FIPS', right_on='fips', how='left')

# Define color schemes
quadrant_colors = {
    'High Damage / High Capacity': '#2c7bb6',   # Blue
    'High Damage / Low Capacity': '#d7191c',     # Red
    'Low Damage / High Capacity': '#1a9850',     # Green
    'Low Damage / Low Capacity': '#fdae61'       # Orange
}

critical_colors = {
    'Critical in BOTH': '#8B0000',                      # Dark red (highest priority)
    'Critical in Absolute Only': '#FF6347',             # Tomato red (resource needs)
    'Critical in Normalized Only': '#FFA500',           # Orange (equity focus)
    'Not Critical': '#87CEEB'                           # Sky blue (manageable)
}

# Create figure
print("Creating 3-panel visualization...")
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Panel A: Absolute quadrants
ax = axes[0]
legend_elements_abs = []
for quadrant, color in quadrant_colors.items():
    subset = counties_abs[counties_abs['absolute_quadrant'] == quadrant]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.3, alpha=0.85)
        count = len(subset)
        legend_elements_abs.append(Patch(facecolor=color, edgecolor='white',
                                         label=f'{quadrant} ({count})'))

# No data
no_data = counties_abs[counties_abs['absolute_quadrant'].isna()]
if len(no_data) > 0:
    no_data.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.3)
    legend_elements_abs.append(Patch(facecolor='lightgray', edgecolor='white',
                                     label=f'No data ({len(no_data)})'))

ax.set_xlim(-100, -65)
ax.set_ylim(24, 48)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('(a) Absolute Quadrant Analysis\n(Weighted Damage Units vs. Permits/Month)', 
             fontsize=14, fontweight='bold', pad=10)
ax.legend(handles=legend_elements_abs, loc='lower left', fontsize=10, framealpha=0.95)

# Panel B: Normalized quadrants
ax = axes[1]
legend_elements_norm = []
for quadrant, color in quadrant_colors.items():
    subset = counties_norm[counties_norm['normalized_quadrant'] == quadrant]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.3, alpha=0.85)
        count = len(subset)
        legend_elements_norm.append(Patch(facecolor=color, edgecolor='white',
                                          label=f'{quadrant} ({count})'))

# No data
no_data = counties_norm[counties_norm['normalized_quadrant'].isna()]
if len(no_data) > 0:
    no_data.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.3)
    legend_elements_norm.append(Patch(facecolor='lightgray', edgecolor='white',
                                      label=f'No data ({len(no_data)})'))

ax.set_xlim(-100, -65)
ax.set_ylim(24, 48)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('(b) Normalized Quadrant Analysis\n(% Housing Damaged vs. Permits/1000 Units/Month)', 
             fontsize=14, fontweight='bold', pad=10)
ax.legend(handles=legend_elements_norm, loc='lower left', fontsize=10, framealpha=0.95)

# Panel C: Critical county classification
ax = axes[2]
legend_elements_crit = []
for status, color in critical_colors.items():
    subset = counties_crit[counties_crit['critical_status'] == status]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.3, alpha=0.85)
        count = len(subset)
        legend_elements_crit.append(Patch(facecolor=color, edgecolor='white',
                                          label=f'{status} ({count})'))

# No data
no_data = counties_crit[counties_crit['critical_status'].isna()]
if len(no_data) > 0:
    no_data.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.3)
    legend_elements_crit.append(Patch(facecolor='lightgray', edgecolor='white',
                                      label=f'No data ({len(no_data)})'))

ax.set_xlim(-100, -65)
ax.set_ylim(24, 48)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('(c) Critical County Classification\n(High Damage/Low Capacity Status)', 
             fontsize=14, fontweight='bold', pad=10)
ax.legend(handles=legend_elements_crit, loc='lower left', fontsize=10, framealpha=0.95)

# Overall title
fig.suptitle('Hurricane Recovery Potential: Three-Perspective Spatial Analysis', 
             fontsize=16, fontweight='bold', y=0.98)

# Add explanatory text
fig.text(0.5, 0.02,
         '(a) Absolute: Based on total weighted damage units and permits/month | ' +
         '(b) Normalized: Both metrics scaled by total exposed housing units\n' +
         '(c) Critical: Counties in High Damage/Low Capacity quadrant in absolute (a), normalized (b), or both',
         ha='center', fontsize=9, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Save
output_file = OUTPUT_DIR / 'three_panel_spatial_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved to: {output_file}")

# Print summary statistics
print("\n" + "=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)

print("\nAbsolute Quadrant Distribution:")
for quadrant in sorted(abs_dominant['absolute_quadrant'].value_counts().index):
    count = (abs_dominant['absolute_quadrant'] == quadrant).sum()
    pct = 100 * count / len(abs_dominant)
    print(f"  {quadrant}: {count} ({pct:.1f}%)")

print("\nNormalized Quadrant Distribution:")
for quadrant in sorted(norm_dominant['normalized_quadrant'].value_counts().index):
    count = (norm_dominant['normalized_quadrant'] == quadrant).sum()
    pct = 100 * count / len(norm_dominant)
    print(f"  {quadrant}: {count} ({pct:.1f}%)")

print("\nCritical County Distribution:")
for status in ['Critical in BOTH', 'Critical in Absolute Only', 
               'Critical in Normalized Only', 'Not Critical']:
    count = (critical_df['critical_status'] == status).sum()
    pct = 100 * count / len(critical_df)
    print(f"  {status}: {count} ({pct:.1f}%)")

print("\n" + "=" * 100)
print("COMPLETE")
print("=" * 100)
