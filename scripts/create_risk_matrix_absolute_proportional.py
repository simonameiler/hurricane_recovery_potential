"""
Create 2×2 Risk Matrix combining Absolute Exposure and Proportional Vulnerability

This addresses the dual-perspective problem:
- X-axis: Absolute damage (total units at risk)
- Y-axis: Proportional damage (% of housing stock)

Reveals 4 distinct risk profiles requiring different intervention strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import geopandas as gpd
from matplotlib.patches import Patch, Rectangle

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"
DATA_DIR = BASE_DIR / "data"

# Load normalized data (has both absolute and normalized metrics)
df = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_fully_normalized.csv')

# Calculate county-level averages
print("=" * 100)
print("CREATING 2×2 ABSOLUTE-PROPORTIONAL RISK MATRIX")
print("=" * 100)

county_stats = df.groupby('fips').agg({
    'weighted_damage_units': 'mean',  # Absolute exposure
    'pct_housing_damaged': 'mean',     # Proportional vulnerability
    'total_housing_units': 'first',
    'construction_capacity': 'mean',
    'capacity_per_1000_units': 'mean',
    'recovery_months': 'mean'
}).reset_index()

# Define thresholds
# Use median for balanced distribution
abs_threshold = county_stats['weighted_damage_units'].median()
prop_threshold = county_stats['pct_housing_damaged'].median()

print(f"\nRisk Matrix Thresholds:")
print(f"  Absolute damage threshold: {abs_threshold:,.0f} weighted units")
print(f"  Proportional damage threshold: {prop_threshold:.2f}% of housing")

# Assign to 2×2 matrix
def assign_risk_category(row):
    high_abs = row['weighted_damage_units'] >= abs_threshold
    high_prop = row['pct_housing_damaged'] >= prop_threshold
    
    if high_abs and high_prop:
        return 'Catastrophic Risk'
    elif high_abs and not high_prop:
        return 'High Exposure / Low Vulnerability'
    elif not high_abs and high_prop:
        return 'Low Exposure / High Vulnerability'
    else:
        return 'Manageable Risk'

county_stats['risk_category'] = county_stats.apply(assign_risk_category, axis=1)

# Add state info
county_stats['fips'] = county_stats['fips'].astype(str).str.zfill(5)
county_stats['state_fips'] = county_stats['fips'].str[:2]
counties_ref = pd.read_csv(DATA_DIR / 'selected_states_counties.csv')
counties_ref['FIPS'] = counties_ref['FIPS'].astype(str).str.zfill(5)
county_stats = county_stats.merge(
    counties_ref[['FIPS', 'STATE_NAME', 'NAME']], 
    left_on='fips', right_on='FIPS', how='left'
)

# Print summary
print("\n" + "=" * 100)
print("RISK CATEGORY DISTRIBUTION")
print("=" * 100)

for category in ['Catastrophic Risk', 'High Exposure / Low Vulnerability', 
                 'Low Exposure / High Vulnerability', 'Manageable Risk']:
    subset = county_stats[county_stats['risk_category'] == category]
    count = len(subset)
    pct = 100 * count / len(county_stats)
    
    print(f"\n{category}: {count} counties ({pct:.1f}%)")
    print(f"  Mean total housing: {subset['total_housing_units'].mean():,.0f} units")
    print(f"  Mean absolute damage: {subset['weighted_damage_units'].mean():,.0f} units")
    print(f"  Mean % damaged: {subset['pct_housing_damaged'].mean():.2f}%")
    print(f"  Mean recovery: {subset['recovery_months'].mean() / 12:.1f} years")
    
    # Top states
    top_states = subset['STATE_NAME'].value_counts().head(3)
    print(f"  Top states: {', '.join([f'{s} ({c})' for s, c in top_states.items()])}")

# Save results
county_stats.to_csv(OUTPUT_DIR / 'risk_matrix_absolute_proportional.csv', index=False)
print(f"\n\nRisk matrix data saved to: risk_matrix_absolute_proportional.csv")

# Create scatter plot visualization
print("\n" + "=" * 100)
print("CREATING RISK MATRIX VISUALIZATION")
print("=" * 100)

fig, ax = plt.subplots(figsize=(14, 10))

# Define colors for the 2×2 matrix
colors = {
    'Catastrophic Risk': '#8B0000',  # Dark red - most severe
    'High Exposure / Low Vulnerability': '#FF8C00',  # Orange - needs resources
    'Low Exposure / High Vulnerability': '#DAA520',  # Gold - needs support
    'Manageable Risk': '#228B22'  # Green - least concern
}

# Scatter plot
for category, color in colors.items():
    subset = county_stats[county_stats['risk_category'] == category]
    ax.scatter(subset['weighted_damage_units'], subset['pct_housing_damaged'],
               alpha=0.6, s=60, c=color, label=category, edgecolors='black', linewidth=0.5)

# Add threshold lines
ax.axvline(abs_threshold, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(prop_threshold, color='black', linestyle='--', linewidth=2, alpha=0.7)

# Add quadrant labels
ax.text(abs_threshold * 3, prop_threshold * 3, 'CATASTROPHIC\nRISK',
        ha='center', va='center', fontsize=14, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='#8B0000', alpha=0.3))

ax.text(abs_threshold * 3, prop_threshold / 3, 'HIGH EXPOSURE\nLOW VULNERABILITY',
        ha='center', va='center', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FF8C00', alpha=0.3))

ax.text(abs_threshold / 3, prop_threshold * 3, 'LOW EXPOSURE\nHIGH VULNERABILITY',
        ha='center', va='center', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#DAA520', alpha=0.3))

ax.text(abs_threshold / 3, prop_threshold / 3, 'MANAGEABLE\nRISK',
        ha='center', va='center', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#228B22', alpha=0.3))

# Formatting
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Absolute Damage (weighted units, average per event)', fontsize=13, fontweight='bold')
ax.set_ylabel('Proportional Damage (% of housing stock, average per event)', fontsize=13, fontweight='bold')
ax.set_title('Dual-Perspective Risk Matrix:\nAbsolute Exposure vs. Proportional Vulnerability', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, which='both')

# Add annotation explaining axes
fig.text(0.5, 0.02, 
         'X-axis: Total units at risk (matters for resource allocation) | ' +
         'Y-axis: % of community impacted (matters for community survival)',
         ha='center', fontsize=10, style='italic', color='gray')

plt.tight_layout()
output_file = OUTPUT_DIR / 'risk_matrix_scatter.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Scatter plot saved to: {output_file}")

# Create spatial map
print("\n" + "=" * 100)
print("CREATING SPATIAL MAP")
print("=" * 100)

# Load shapefile
counties_shp = gpd.read_file(DATA_DIR / 'US_counties.shp')
coastal_state_fips = ['01', '09', '10', '12', '13', '22', '23', '24', '25',
                      '28', '33', '34', '36', '37', '44', '45', '48', '51', '42']
counties_shp = counties_shp[counties_shp['STATEFP'].isin(coastal_state_fips)].copy()
counties_shp['FIPS'] = counties_shp['STATEFP'] + counties_shp['COUNTYFP']

# Merge with risk data
counties_merged = counties_shp.merge(county_stats, left_on='FIPS', right_on='fips', how='left')

# Create map
fig, ax = plt.subplots(figsize=(18, 12))

legend_elements = []
for category, color in colors.items():
    subset = counties_merged[counties_merged['risk_category'] == category]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.3, alpha=0.85)
        count = len(subset)
        legend_elements.append(Patch(facecolor=color, edgecolor='white',
                                     label=f'{category} ({count} counties)'))

# Plot counties with no data
no_data = counties_merged[counties_merged['risk_category'].isna()]
if len(no_data) > 0:
    no_data.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.3)
    legend_elements.append(Patch(facecolor='lightgray', edgecolor='white',
                                 label=f'No data ({len(no_data)} counties)'))

# Formatting
ax.set_xlim(-100, -65)
ax.set_ylim(24, 48)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Hurricane Risk Matrix: Absolute Exposure × Proportional Vulnerability\n' +
             'Combining Total Damage Potential with Community Impact',
             fontsize=16, pad=20, fontweight='bold')
ax.legend(handles=legend_elements, loc='lower left', fontsize=12, framealpha=0.95)

# Add explanatory text
fig.text(0.5, 0.02,
         'Catastrophic Risk: High total damage AND high % of community impacted | ' +
         'High Exposure/Low Vulnerability: Large counties with resources to absorb impact\n' +
         'Low Exposure/High Vulnerability: Small counties facing proportionally catastrophic damage | ' +
         'Manageable Risk: Low total damage and low proportional impact',
         ha='center', fontsize=9, style='italic', color='gray', wrap=True)

plt.tight_layout()
output_file = OUTPUT_DIR / 'risk_matrix_map.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Map saved to: {output_file}")

# Detailed analysis by category
print("\n" + "=" * 100)
print("DETAILED CHARACTERISTICS BY RISK CATEGORY")
print("=" * 100)

print("\n1. CATASTROPHIC RISK (High Absolute + High Proportional)")
print("-" * 100)
cat_risk = county_stats[county_stats['risk_category'] == 'Catastrophic Risk'].sort_values('recovery_months', ascending=False)
print(f"   Counties: {len(cat_risk)}")
print(f"   Interpretation: These counties face BOTH high total losses AND catastrophic proportional impact")
print(f"   Policy priority: HIGHEST - need immediate pre-disaster mitigation and robust recovery plans")
print(f"   Average recovery: {cat_risk['recovery_months'].mean() / 12:.0f} years")
print(f"\n   Top 10 by recovery time:")
for _, row in cat_risk.head(10).iterrows():
    print(f"     {row['NAME']}, {row['STATE_NAME']}: {row['recovery_months']/12:.0f} yrs recovery, " +
          f"{row['weighted_damage_units']:,.0f} units ({row['pct_housing_damaged']:.1f}%)")

print("\n2. HIGH EXPOSURE / LOW VULNERABILITY (High Absolute + Low Proportional)")
print("-" * 100)
high_exp = county_stats[county_stats['risk_category'] == 'High Exposure / Low Vulnerability'].sort_values('weighted_damage_units', ascending=False)
print(f"   Counties: {len(high_exp)}")
print(f"   Interpretation: Large counties with major total losses but can absorb proportionally")
print(f"   Policy priority: HIGH - need federal resources but less existential crisis")
print(f"   Average recovery: {high_exp['recovery_months'].mean() / 12:.0f} years")
print(f"\n   Top 10 by absolute damage:")
for _, row in high_exp.head(10).iterrows():
    print(f"     {row['NAME']}, {row['STATE_NAME']}: {row['weighted_damage_units']:,.0f} units ({row['pct_housing_damaged']:.1f}%), " +
          f"{row['total_housing_units']:,.0f} total")

print("\n3. LOW EXPOSURE / HIGH VULNERABILITY (Low Absolute + High Proportional)")
print("-" * 100)
low_exp = county_stats[county_stats['risk_category'] == 'Low Exposure / High Vulnerability'].sort_values('pct_housing_damaged', ascending=False)
print(f"   Counties: {len(low_exp)}")
print(f"   Interpretation: Small counties facing existential threat from proportional damage")
print(f"   Policy priority: HIGH - need targeted equity-focused interventions")
print(f"   Average recovery: {low_exp['recovery_months'].mean() / 12:.0f} years")
print(f"\n   Top 10 by proportional damage:")
for _, row in low_exp.head(10).iterrows():
    print(f"     {row['NAME']}, {row['STATE_NAME']}: {row['pct_housing_damaged']:.1f}% damaged, " +
          f"{row['weighted_damage_units']:,.0f} units, {row['total_housing_units']:,.0f} total")

print("\n4. MANAGEABLE RISK (Low Absolute + Low Proportional)")
print("-" * 100)
manageable = county_stats[county_stats['risk_category'] == 'Manageable Risk']
print(f"   Counties: {len(manageable)}")
print(f"   Interpretation: Limited exposure and proportional impact")
print(f"   Policy priority: LOWER - but still need basic preparedness")
print(f"   Average recovery: {manageable['recovery_months'].mean() / 12:.0f} years")

# Create summary comparison table
print("\n" + "=" * 100)
print("POLICY INTERVENTION STRATEGIES BY RISK CATEGORY")
print("=" * 100)

strategies = {
    'Catastrophic Risk': {
        'Priority': 'CRITICAL',
        'Federal Aid': 'Major assistance needed',
        'Pre-disaster': 'Aggressive mitigation, building codes, buyouts',
        'Recovery': 'Long-term federal presence, community rebuilding',
        'Example': 'Small coastal counties with frequent severe impacts'
    },
    'High Exposure / Low Vulnerability': {
        'Priority': 'HIGH',
        'Federal Aid': 'Substantial resources for total losses',
        'Pre-disaster': 'Enhanced building standards, insurance',
        'Recovery': 'Efficient resource deployment, local capacity',
        'Example': 'Major metro counties that can leverage resources'
    },
    'Low Exposure / High Vulnerability': {
        'Priority': 'HIGH (Equity Focus)',
        'Federal Aid': 'Targeted per-capita assistance',
        'Pre-disaster': 'Capacity building, regional mutual aid',
        'Recovery': 'Technical assistance, equitable recovery',
        'Example': 'Rural communities with limited local capacity'
    },
    'Manageable Risk': {
        'Priority': 'MODERATE',
        'Federal Aid': 'Standard disaster programs',
        'Pre-disaster': 'Basic preparedness, insurance awareness',
        'Recovery': 'Streamlined assistance processes',
        'Example': 'Inland counties with limited exposure'
    }
}

for category, strategy in strategies.items():
    print(f"\n{category}:")
    for key, value in strategy.items():
        print(f"  {key}: {value}")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
print("\nOutput files:")
print("  - risk_matrix_absolute_proportional.csv")
print("  - risk_matrix_scatter.png")
print("  - risk_matrix_map.png")
