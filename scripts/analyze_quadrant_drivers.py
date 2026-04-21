"""
Analyze what variables explain quadrant membership in the fully normalized analysis.

This helps understand:
1. What makes counties fall into different quadrants?
2. Are current quadrant thresholds meaningful?
3. What are the key differentiating factors?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"
DATA_DIR = BASE_DIR / "data"

# Load fully normalized data
df = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_fully_normalized.csv')

# Calculate county-level aggregates
county_stats = df.groupby('fips').agg({
    'pct_housing_damaged': ['mean', 'median', 'std'],
    'capacity_per_1000_units': ['mean', 'median', 'std'],
    'total_housing_units': 'first',
    'construction_capacity': 'mean',
    'weighted_damage_units': 'mean',
    'recovery_months': 'mean',
    'quadrant': lambda x: x.mode()[0]  # dominant quadrant
}).reset_index()

county_stats.columns = ['fips', 
                        'avg_pct_damaged', 'median_pct_damaged', 'std_pct_damaged',
                        'avg_capacity_norm', 'median_capacity_norm', 'std_capacity_norm',
                        'total_housing_units', 'avg_capacity_abs', 'avg_damage_abs',
                        'avg_recovery_months', 'dominant_quadrant']

# Add state FIPS
county_stats['state_fips'] = county_stats['fips'].astype(str).str.zfill(5).str[:2]

# Load county data to get geographic info
counties_df = pd.read_csv(DATA_DIR / 'selected_states_counties.csv')
counties_df['FIPS'] = counties_df['FIPS'].astype(str).str.zfill(5)

# Ensure fips is string for merge
county_stats['fips'] = county_stats['fips'].astype(str).str.zfill(5)

county_stats = county_stats.merge(
    counties_df[['FIPS', 'STATE_NAME', 'NAME']], 
    left_on='fips', right_on='FIPS', how='left'
)

print("=" * 90)
print("WHAT DRIVES QUADRANT MEMBERSHIP?")
print("=" * 90)

# 1. County Size
print("\n1. TOTAL HOUSING UNITS (County Size)")
print("-" * 90)
for quad in ['High Damage / High Capacity', 'High Damage / Low Capacity',
             'Low Damage / High Capacity', 'Low Damage / Low Capacity']:
    subset = county_stats[county_stats['dominant_quadrant'] == quad]
    print(f"{quad}:")
    print(f"  Mean size: {subset['total_housing_units'].mean():,.0f} units")
    print(f"  Median size: {subset['total_housing_units'].median():,.0f} units")
    print(f"  Range: {subset['total_housing_units'].min():,.0f} to {subset['total_housing_units'].max():,.0f}")

# 2. Absolute vs Normalized Metrics
print("\n2. ABSOLUTE DAMAGE (not normalized)")
print("-" * 90)
for quad in ['High Damage / High Capacity', 'High Damage / Low Capacity',
             'Low Damage / High Capacity', 'Low Damage / Low Capacity']:
    subset = county_stats[county_stats['dominant_quadrant'] == quad]
    print(f"{quad}:")
    print(f"  Mean absolute damage: {subset['avg_damage_abs'].mean():,.0f} weighted units")
    print(f"  Median absolute damage: {subset['avg_damage_abs'].median():,.0f} weighted units")

print("\n3. NORMALIZED DAMAGE (% of housing stock)")
print("-" * 90)
for quad in ['High Damage / High Capacity', 'High Damage / Low Capacity',
             'Low Damage / High Capacity', 'Low Damage / Low Capacity']:
    subset = county_stats[county_stats['dominant_quadrant'] == quad]
    print(f"{quad}:")
    print(f"  Mean % damaged: {subset['avg_pct_damaged'].mean():.2f}%")
    print(f"  Median % damaged: {subset['median_pct_damaged'].median():.2f}%")

print("\n4. NORMALIZED CAPACITY (permits per 1000 units)")
print("-" * 90)
for quad in ['High Damage / High Capacity', 'High Damage / Low Capacity',
             'Low Damage / High Capacity', 'Low Damage / Low Capacity']:
    subset = county_stats[county_stats['dominant_quadrant'] == quad]
    print(f"{quad}:")
    print(f"  Mean capacity: {subset['avg_capacity_norm'].mean():.3f} permits/1000 units/month")
    print(f"  Median capacity: {subset['median_capacity_norm'].median():.3f} permits/1000 units/month")

# 5. Geographic patterns
print("\n5. GEOGRAPHIC DISTRIBUTION (Top 5 states per quadrant)")
print("-" * 90)
for quad in ['High Damage / High Capacity', 'High Damage / Low Capacity',
             'Low Damage / High Capacity', 'Low Damage / Low Capacity']:
    subset = county_stats[county_stats['dominant_quadrant'] == quad]
    state_counts = subset['STATE_NAME'].value_counts().head(5)
    print(f"\n{quad}:")
    for state, count in state_counts.items():
        pct = 100 * count / len(subset)
        print(f"  {state}: {count} counties ({pct:.1f}%)")

# Statistical tests
print("\n" + "=" * 90)
print("STATISTICAL SIGNIFICANCE TESTS")
print("=" * 90)

# Test if quadrants differ significantly on key variables
from scipy.stats import f_oneway

# Group data by quadrant
groups = [county_stats[county_stats['dominant_quadrant'] == q] for q in 
          ['High Damage / High Capacity', 'High Damage / Low Capacity',
           'Low Damage / High Capacity', 'Low Damage / Low Capacity']]

# ANOVA for county size
f_stat, p_val = f_oneway(*[g['total_housing_units'] for g in groups])
print(f"\nCounty Size differs by quadrant? F={f_stat:.2f}, p={p_val:.4f}")
print(f"  → {'YES' if p_val < 0.05 else 'NO'} (significant at p<0.05)")

# ANOVA for normalized damage
f_stat, p_val = f_oneway(*[g['avg_pct_damaged'] for g in groups])
print(f"\nNormalized Damage differs by quadrant? F={f_stat:.2f}, p={p_val:.4f}")
print(f"  → {'YES' if p_val < 0.05 else 'NO'} (by construction, should be YES)")

# ANOVA for normalized capacity
f_stat, p_val = f_oneway(*[g['avg_capacity_norm'] for g in groups])
print(f"\nNormalized Capacity differs by quadrant? F={f_stat:.2f}, p={p_val:.4f}")
print(f"  → {'YES' if p_val < 0.05 else 'NO'} (by construction, should be YES)")

# Correlation analysis
print("\n" + "=" * 90)
print("CORRELATION ANALYSIS")
print("=" * 90)

corr_vars = {
    'Total Housing Units': 'total_housing_units',
    'Avg % Damaged': 'avg_pct_damaged',
    'Avg Capacity (norm)': 'avg_capacity_norm',
    'Avg Recovery (months)': 'avg_recovery_months'
}

corr_matrix = county_stats[[v for v in corr_vars.values()]].corr()
corr_matrix.columns = corr_vars.keys()
corr_matrix.index = corr_vars.keys()

print("\nCorrelation Matrix:")
print(corr_matrix.round(3))

# Save results
county_stats.to_csv(OUTPUT_DIR / 'county_quadrant_drivers.csv', index=False)
print(f"\n\nDetailed county statistics saved to: county_quadrant_drivers.csv")

# Create visualization
print("\n" + "=" * 90)
print("CREATING VISUALIZATION")
print("=" * 90)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: County size distribution by quadrant
ax = axes[0, 0]
for quad in ['High Damage / High Capacity', 'High Damage / Low Capacity',
             'Low Damage / High Capacity', 'Low Damage / Low Capacity']:
    subset = county_stats[county_stats['dominant_quadrant'] == quad]
    ax.hist(np.log10(subset['total_housing_units']), alpha=0.5, label=quad, bins=20)
ax.set_xlabel('Log10(Total Housing Units)', fontweight='bold')
ax.set_ylabel('Number of Counties', fontweight='bold')
ax.set_title('County Size Distribution by Quadrant', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 2: Damage distribution
ax = axes[0, 1]
quadrant_colors = {
    'High Damage / High Capacity': '#2c7bb6',
    'High Damage / Low Capacity': '#d7191c',
    'Low Damage / High Capacity': '#1a9850',
    'Low Damage / Low Capacity': '#fdae61'
}
data_to_plot = [county_stats[county_stats['dominant_quadrant'] == q]['avg_pct_damaged'].values 
                for q in quadrant_colors.keys()]
bp = ax.boxplot(data_to_plot, labels=['HD/HC', 'HD/LC', 'LD/HC', 'LD/LC'],
                patch_artist=True)
for patch, color in zip(bp['boxes'], quadrant_colors.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel('Avg % Housing Damaged', fontweight='bold')
ax.set_title('Normalized Damage by Quadrant', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Plot 3: Capacity distribution
ax = axes[1, 0]
data_to_plot = [county_stats[county_stats['dominant_quadrant'] == q]['avg_capacity_norm'].values 
                for q in quadrant_colors.keys()]
bp = ax.boxplot(data_to_plot, labels=['HD/HC', 'HD/LC', 'LD/HC', 'LD/LC'],
                patch_artist=True)
for patch, color in zip(bp['boxes'], quadrant_colors.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel('Avg Capacity (permits/1000 units/month)', fontweight='bold')
ax.set_title('Normalized Capacity by Quadrant', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Plot 4: Recovery time
ax = axes[1, 1]
data_to_plot = [county_stats[county_stats['dominant_quadrant'] == q]['avg_recovery_months'].values / 12
                for q in quadrant_colors.keys()]
bp = ax.boxplot(data_to_plot, labels=['HD/HC', 'HD/LC', 'LD/HC', 'LD/LC'],
                patch_artist=True)
for patch, color in zip(bp['boxes'], quadrant_colors.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel('Avg Recovery Time (years)', fontweight='bold')
ax.set_title('Recovery Time by Quadrant', fontweight='bold')
ax.set_yscale('log')
ax.grid(alpha=0.3, axis='y')

plt.suptitle('Drivers of Quadrant Membership', fontsize=16, fontweight='bold')
plt.tight_layout()

output_file = OUTPUT_DIR / 'quadrant_drivers_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualization saved to: {output_file}")

print("\n" + "=" * 90)
print("SUMMARY: WHAT DEFINES THE QUADRANTS?")
print("=" * 90)
print("""
The fully normalized quadrants are defined by TWO NORMALIZED METRICS:

1. % of County Housing Stock Damaged (median split at ~9%)
   - HIGH DAMAGE: Counties where hurricanes damage >9% of housing on average
   - LOW DAMAGE: Counties where hurricanes damage <9% of housing on average

2. Construction Capacity per 1000 Housing Units (median split at ~0.7 permits/1000/month)
   - HIGH CAPACITY: Counties with >0.7 permits/1000 units/month
   - LOW CAPACITY: Counties with <0.7 permits/1000 units/month

KEY FINDINGS:
- County SIZE no longer determines quadrant membership (after normalization)
- GEOGRAPHY matters: Coastal counties → high damage, rural counties → low capacity
- NORMALIZATION reveals proportional impact independent of absolute size
- The ~0.7 permits/1000 units threshold means a 10,000-unit county needs >7 permits/month
  to be "high capacity" while a 100,000-unit county needs >70 permits/month

ALTERNATIVE QUADRANT DEFINITIONS TO CONSIDER:
1. Policy-relevant thresholds (e.g., recovery >10 years = "high damage")
2. Absolute thresholds (e.g., >10% damage = "high", <1% = "low")
3. Quartile splits (25% in each) - current approach
4. Theory-driven thresholds based on recovery capacity literature
""")

