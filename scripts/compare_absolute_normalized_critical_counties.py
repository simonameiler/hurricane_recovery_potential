"""
Compare High Damage/Low Capacity counties across absolute and normalized perspectives

Identifies three intervention categories:
a) Critical in BOTH absolute and normalized → highest priority
b) Critical in absolute only → resource-intensive but proportionally manageable
c) Critical in normalized only → equity/community survival issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import geopandas as gpd
from matplotlib.patches import Patch

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"
DATA_DIR = BASE_DIR / "data"

# Load both quadrant analyses
df_absolute = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_weightedunits.csv')
df_normalized = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_fully_normalized.csv')

print("=" * 100)
print("COMPARING HIGH DAMAGE/LOW CAPACITY COUNTIES: ABSOLUTE vs. NORMALIZED")
print("=" * 100)

# Check if each county is EVER in High Damage/Low Capacity in either framework
# This is more policy-relevant than modal quadrant
absolute_critical = df_absolute[df_absolute['quadrant'] == 'High Damage / Low Capacity'].groupby('fips').size().reset_index()
absolute_critical.columns = ['fips', 'abs_critical_count']
absolute_critical['is_absolute_critical'] = True

normalized_critical = df_normalized[df_normalized['quadrant'] == 'High Damage / Low Capacity'].groupby('fips').size().reset_index()
normalized_critical.columns = ['fips', 'norm_critical_count']
normalized_critical['is_normalized_critical'] = True

# Get all unique counties
all_counties = pd.DataFrame({'fips': pd.concat([df_absolute['fips'], df_normalized['fips']]).unique()})

# Merge to identify critical status
comparison = all_counties.copy()
comparison = comparison.merge(absolute_critical[['fips', 'is_absolute_critical']], on='fips', how='left')
comparison = comparison.merge(normalized_critical[['fips', 'is_normalized_critical']], on='fips', how='left')
comparison['is_absolute_critical'] = comparison['is_absolute_critical'].fillna(False)
comparison['is_normalized_critical'] = comparison['is_normalized_critical'].fillna(False)

# Add county-level statistics from normalized data (has both absolute and normalized metrics)
county_stats = df_normalized.groupby('fips').agg({
    'weighted_damage_units': 'mean',
    'pct_housing_damaged': 'mean',
    'total_housing_units': 'first',
    'construction_capacity': 'mean',
    'capacity_per_1000_units': 'mean',
    'recovery_months': 'mean'
}).reset_index()

comparison = comparison.merge(county_stats, on='fips', how='left')

# Add state/county names
comparison['fips'] = comparison['fips'].astype(str).str.zfill(5)
counties_ref = pd.read_csv(DATA_DIR / 'selected_states_counties.csv')
counties_ref['FIPS'] = counties_ref['FIPS'].astype(str).str.zfill(5)
comparison = comparison.merge(
    counties_ref[['FIPS', 'STATE_NAME', 'NAME']], 
    left_on='fips', right_on='FIPS', how='left'
)

# Classify into three categories
def classify_critical_status(row):
    abs_critical = row['is_absolute_critical']
    norm_critical = row['is_normalized_critical']
    
    if abs_critical and norm_critical:
        return 'Critical in BOTH'
    elif abs_critical and not norm_critical:
        return 'Critical in Absolute Only'
    elif not abs_critical and norm_critical:
        return 'Critical in Normalized Only'
    else:
        return 'Not Critical'

comparison['critical_status'] = comparison.apply(classify_critical_status, axis=1)

# Summary statistics
print("\n" + "=" * 100)
print("CRITICAL COUNTY CLASSIFICATION")
print("=" * 100)

for status in ['Critical in BOTH', 'Critical in Absolute Only', 
               'Critical in Normalized Only', 'Not Critical']:
    subset = comparison[comparison['critical_status'] == status]
    count = len(subset)
    pct = 100 * count / len(comparison)
    
    print(f"\n{status}: {count} counties ({pct:.1f}%)")
    
    if count > 0:
        print(f"  Mean total housing: {subset['total_housing_units'].mean():,.0f} units")
        print(f"  Mean absolute damage: {subset['weighted_damage_units'].mean():,.0f} units")
        print(f"  Mean % damaged: {subset['pct_housing_damaged'].mean():.2f}%")
        print(f"  Mean capacity: {subset['construction_capacity'].mean():,.0f} permits/month")
        print(f"  Mean capacity/1000: {subset['capacity_per_1000_units'].mean():.3f} permits/1000 units/month")
        print(f"  Mean recovery: {subset['recovery_months'].mean() / 12:.1f} years")
        
        # Top states
        if 'STATE_NAME' in subset.columns:
            top_states = subset['STATE_NAME'].value_counts().head(3)
            print(f"  Top states: {', '.join([f'{s} ({c})' for s, c in top_states.items()])}")

# Save full comparison
comparison.to_csv(OUTPUT_DIR / 'critical_counties_absolute_vs_normalized.csv', index=False)
print(f"\n\nFull comparison saved to: critical_counties_absolute_vs_normalized.csv")

# Detailed analysis of each critical category
print("\n" + "=" * 100)
print("DETAILED ANALYSIS BY CRITICAL STATUS")
print("=" * 100)

print("\n1. CRITICAL IN BOTH (Absolute AND Normalized)")
print("-" * 100)
both_critical = comparison[comparison['critical_status'] == 'Critical in BOTH'].sort_values('recovery_months', ascending=False)
print(f"   Counties: {len(both_critical)}")
print(f"   Interpretation: High total damage AND high proportional impact")
print(f"   Policy Priority: HIGHEST - need massive resources AND community rebuilding")
print(f"\n   Top 15 counties by recovery time:")
for i, (_, row) in enumerate(both_critical.head(15).iterrows(), 1):
    print(f"   {i:2d}. {row['NAME']}, {row['STATE_NAME']}: {row['recovery_months']/12:.0f} yrs, " +
          f"{row['weighted_damage_units']:,.0f} units ({row['pct_housing_damaged']:.1f}%), " +
          f"capacity: {row['capacity_per_1000_units']:.3f}/1000")

print("\n2. CRITICAL IN ABSOLUTE ONLY")
print("-" * 100)
abs_only = comparison[comparison['critical_status'] == 'Critical in Absolute Only'].sort_values('weighted_damage_units', ascending=False)
print(f"   Counties: {len(abs_only)}")
print(f"   Interpretation: High total damage but LOW proportional impact")
print(f"   Policy Priority: HIGH - need federal resources but community can absorb")
print(f"\n   Top 15 counties by absolute damage:")
for i, (_, row) in enumerate(abs_only.head(15).iterrows(), 1):
    print(f"   {i:2d}. {row['NAME']}, {row['STATE_NAME']}: {row['weighted_damage_units']:,.0f} units ({row['pct_housing_damaged']:.1f}%), " +
          f"{row['total_housing_units']:,.0f} total housing, " +
          f"recovery: {row['recovery_months']/12:.1f} yrs")

print("\n3. CRITICAL IN NORMALIZED ONLY")
print("-" * 100)
norm_only = comparison[comparison['critical_status'] == 'Critical in Normalized Only'].sort_values('pct_housing_damaged', ascending=False)
print(f"   Counties: {len(norm_only)}")
print(f"   Interpretation: Low total damage but HIGH proportional impact")
print(f"   Policy Priority: HIGH (EQUITY FOCUS) - existential threat to small communities")
print(f"\n   Top 15 counties by proportional damage:")
for i, (_, row) in enumerate(norm_only.head(15).iterrows(), 1):
    print(f"   {i:2d}. {row['NAME']}, {row['STATE_NAME']}: {row['pct_housing_damaged']:.1f}% damaged, " +
          f"{row['weighted_damage_units']:,.0f} units, {row['total_housing_units']:,.0f} total, " +
          f"recovery: {row['recovery_months']/12:.1f} yrs")

# Create visualization: scatter plot
print("\n" + "=" * 100)
print("CREATING VISUALIZATIONS")
print("=" * 100)

fig, ax = plt.subplots(figsize=(14, 10))

colors = {
    'Critical in BOTH': '#8B0000',  # Dark red - highest priority
    'Critical in Absolute Only': '#FF8C00',  # Orange - resource intensive
    'Critical in Normalized Only': '#DAA520',  # Gold - equity focus
    'Not Critical': '#90EE90'  # Light green - lower priority
}

for status, color in colors.items():
    subset = comparison[comparison['critical_status'] == status]
    ax.scatter(subset['weighted_damage_units'], subset['pct_housing_damaged'],
               alpha=0.6, s=60, c=color, label=f"{status} ({len(subset)})",
               edgecolors='black', linewidth=0.5)

# Add median threshold lines
abs_median = comparison['weighted_damage_units'].median()
norm_median = comparison['pct_housing_damaged'].median()
ax.axvline(abs_median, color='black', linestyle='--', linewidth=2, alpha=0.7, 
           label=f'Absolute median ({abs_median:,.0f} units)')
ax.axhline(norm_median, color='black', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Normalized median ({norm_median:.1f}%)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Absolute Damage (weighted units, average per event)', fontsize=13, fontweight='bold')
ax.set_ylabel('Proportional Damage (% of housing stock, average per event)', fontsize=13, fontweight='bold')
ax.set_title('Critical Counties: Absolute vs. Normalized High Damage/Low Capacity\n' +
             'Three distinct intervention priorities', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
output_file = OUTPUT_DIR / 'critical_counties_comparison_scatter.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Scatter plot saved to: {output_file}")

# Create spatial map
print("\nCreating spatial map...")

counties_shp = gpd.read_file(DATA_DIR / 'US_counties.shp')
coastal_state_fips = ['01', '09', '10', '12', '13', '22', '23', '24', '25',
                      '28', '33', '34', '36', '37', '44', '45', '48', '51', '42']
counties_shp = counties_shp[counties_shp['STATEFP'].isin(coastal_state_fips)].copy()
counties_shp['FIPS'] = counties_shp['STATEFP'] + counties_shp['COUNTYFP']

counties_merged = counties_shp.merge(comparison, left_on='FIPS', right_on='fips', how='left')

fig, ax = plt.subplots(figsize=(18, 12))

legend_elements = []
for status, color in colors.items():
    subset = counties_merged[counties_merged['critical_status'] == status]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.3, alpha=0.85)
        count = len(subset)
        legend_elements.append(Patch(facecolor=color, edgecolor='white',
                                     label=f'{status} ({count})'))

# No data
no_data = counties_merged[counties_merged['critical_status'].isna()]
if len(no_data) > 0:
    no_data.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.3)
    legend_elements.append(Patch(facecolor='lightgray', edgecolor='white',
                                 label=f'No data ({len(no_data)})'))

ax.set_xlim(-100, -65)
ax.set_ylim(24, 48)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Critical Counties by Intervention Priority:\n' +
             'Comparing Absolute and Normalized High Damage/Low Capacity Status',
             fontsize=16, pad=20, fontweight='bold')
ax.legend(handles=legend_elements, loc='lower left', fontsize=12, framealpha=0.95)

fig.text(0.5, 0.02,
         'Critical in BOTH: Need massive resources + community rebuilding | ' +
         'Absolute Only: Large counties needing federal aid\n' +
         'Normalized Only: Small counties facing existential threat | ' +
         'Not Critical: Lower priority or manageable with local resources',
         ha='center', fontsize=9, style='italic', color='gray')

plt.tight_layout()
output_file = OUTPUT_DIR / 'critical_counties_comparison_map.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Map saved to: {output_file}")

# Create cross-tabulation
print("\n" + "=" * 100)
print("CROSS-TABULATION: CRITICAL STATUS")
print("=" * 100)

crosstab = pd.crosstab(
    comparison['is_absolute_critical'], 
    comparison['is_normalized_critical'],
    margins=True,
    rownames=['Absolute Critical'],
    colnames=['Normalized Critical']
)
print("\n", crosstab)

# Save crosstab
crosstab.to_csv(OUTPUT_DIR / 'quadrant_crosstab_absolute_vs_normalized.csv')

print("\n" + "=" * 100)
print("POLICY IMPLICATIONS")
print("=" * 100)

implications = {
    'Critical in BOTH': {
        'Count': len(both_critical),
        'Resource Needs': 'Massive federal assistance + long-term presence',
        'Pre-disaster': 'Aggressive mitigation, buyouts, relocation programs',
        'Recovery': 'Comprehensive community rebuilding with federal oversight',
        'Equity Concerns': 'Both scale and proportional impact require attention',
        'Timeline': f"{both_critical['recovery_months'].mean() / 12:.0f} years average"
    },
    'Critical in Absolute Only': {
        'Count': len(abs_only),
        'Resource Needs': 'Major federal resources but shorter-term',
        'Pre-disaster': 'Enhanced building codes, insurance requirements',
        'Recovery': 'Efficient resource deployment leveraging local capacity',
        'Equity Concerns': 'Moderate - large counties can mobilize resources',
        'Timeline': f"{abs_only['recovery_months'].mean() / 12:.0f} years average"
    },
    'Critical in Normalized Only': {
        'Count': len(norm_only),
        'Resource Needs': 'Targeted per-capita assistance',
        'Pre-disaster': 'Capacity building, regional cooperation',
        'Recovery': 'Technical assistance + equitable resource distribution',
        'Equity Concerns': 'HIGHEST - small communities face existential crisis',
        'Timeline': f"{norm_only['recovery_months'].mean() / 12:.0f} years average"
    }
}

for category, details in implications.items():
    print(f"\n{category}:")
    for key, value in details.items():
        print(f"  {key}: {value}")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
print("\nOutput files:")
print("  - critical_counties_absolute_vs_normalized.csv")
print("  - critical_counties_comparison_scatter.png")
print("  - critical_counties_comparison_map.png")
print("  - quadrant_crosstab_absolute_vs_normalized.csv")
