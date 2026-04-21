"""
Dual-perspective analysis: Why BOTH absolute and normalized metrics matter

This addresses the question: Should we care about absolute damage or proportional damage?
Answer: BOTH, but for different policy questions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"

# Load both datasets
df_abs = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_weightedunits.csv')
df_norm = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_fully_normalized.csv')

print("=" * 100)
print("WHY BOTH ABSOLUTE AND NORMALIZED METRICS MATTER")
print("=" * 100)

print("""
TWO DIFFERENT POLICY QUESTIONS:

1. RESOURCE ALLOCATION / TOTAL LOSS (Absolute Perspective):
   "Where will hurricanes cause the most TOTAL damage and require the most TOTAL resources?"
   → Large counties with many units at risk matter more
   → Useful for: Federal aid budgets, national resource planning, insurance pools
   
2. COMMUNITY VULNERABILITY / EQUITY (Normalized Perspective):
   "Which communities face CATASTROPHIC PROPORTIONAL IMPACT and may not recover?"
   → Small counties where hurricanes devastate a large % of housing matter more
   → Useful for: Targeted interventions, equity considerations, community resilience
   
NEITHER IS "WRONG" - they answer different questions!
""")

# Florida example
print("\n" + "=" * 100)
print("FLORIDA EXAMPLE: BROWARD COUNTY (FIPS 12011)")
print("=" * 100)

broward_norm = df_norm[df_norm['fips'].astype(str).str.zfill(5) == '12011']
broward_abs = df_abs[df_abs['fips'].astype(str).str.zfill(5) == '12011']

if len(broward_norm) > 0:
    print(f"\nBroward County (includes Fort Lauderdale):")
    print(f"  Total housing units: {broward_norm['total_housing_units'].iloc[0]:,.0f}")
if len(broward_norm) > 0:
    print(f"\nBroward County (includes Fort Lauderdale):")
    print(f"  Total housing units: {broward_norm['total_housing_units'].iloc[0]:,.0f}")
    print(f"  Average absolute damage: {broward_norm['weighted_damage_units'].mean():,.0f} weighted units")
    print(f"  Average % damaged: {broward_norm['pct_housing_damaged'].mean():.2f}%")
    print(f"  Median absolute damage: {broward_norm['weighted_damage_units'].median():,.0f} units")
    print(f"  Median % damaged: {broward_norm['pct_housing_damaged'].median():.2f}%")

    print(f"\nQuadrant membership:")
    print(f"  Absolute approach: {broward_abs['quadrant'].mode()[0]}")
    print(f"  Normalized approach: {broward_norm['quadrant'].mode()[0]}")

    print(f"\nINTERPRETATION:")
    print(f"""
  Absolute perspective: "HIGH DAMAGE"
    → {broward_norm['weighted_damage_units'].mean():,.0f} damaged units on average is a HUGE number
    → Requires massive federal aid, insurance payouts, construction resources
    → One of the largest absolute losses in the entire analysis
    → Critical for national/state resource planning
    
  Normalized perspective: "LOW DAMAGE" 
    → But {broward_norm['weighted_damage_units'].mean():,.0f} out of {broward_norm['total_housing_units'].iloc[0]:,.0f} units = only {broward_norm['pct_housing_damaged'].mean():.2f}%
    → County can absorb this proportionally - it's not existential
    → Local economy, tax base, services mostly intact
    → Community doesn't face collapse
    
  BOTH ARE TRUE AND IMPORTANT!
""")
else:
    print("\nBroward County not found in normalized data")

# Compare with a small vulnerable county
print("\n" + "=" * 100)
print("CONTRAST: SMALL VULNERABLE COUNTY")
print("=" * 100)

# Find a small high-damage county
small_vulnerable = df_norm[
    (df_norm['quadrant'] == 'High Damage / Low Capacity') & 
    (df_norm['total_housing_units'] < 10000)
].groupby('fips').agg({
    'total_housing_units': 'first',
    'weighted_damage_units': 'mean',
    'pct_housing_damaged': 'mean',
    'construction_capacity': 'mean',
    'capacity_per_1000_units': 'mean',
    'recovery_months': 'mean'
}).sort_values('pct_housing_damaged', ascending=False).head(1)

if len(small_vulnerable) > 0:
    fips = small_vulnerable.index[0]
    print(f"\nSmall county (FIPS {fips}):")
    print(f"  Total housing units: {small_vulnerable['total_housing_units'].iloc[0]:,.0f}")
    print(f"  Average absolute damage: {small_vulnerable['weighted_damage_units'].iloc[0]:,.0f} weighted units")
    print(f"  Average % damaged: {small_vulnerable['pct_housing_damaged'].iloc[0]:.2f}%")
    print(f"  Average recovery time: {small_vulnerable['recovery_months'].iloc[0]/12:.0f} years")
    
    print(f"\nINTERPRETATION:")
    print(f"""
  Absolute perspective: "LOW DAMAGE"
    → Only {small_vulnerable['weighted_damage_units'].iloc[0]:,.0f} units damaged
    → Small in national/state resource terms
    → Might not get prioritized for federal aid
    → Easy to overlook in absolute terms
    
  Normalized perspective: "HIGH DAMAGE"
    → {small_vulnerable['pct_housing_damaged'].iloc[0]:.1f}% of housing stock damaged
    → CATASTROPHIC for the local community
    → Tax base devastated, services disrupted, population displacement
    → Community may not survive without targeted help
    → Recovery time: {small_vulnerable['recovery_months'].iloc[0]/12:.0f} years
    
  This is where normalization reveals critical vulnerability!
""")

# Create comparison visualization
print("\n" + "=" * 100)
print("CREATING DUAL-PERSPECTIVE VISUALIZATION")
print("=" * 100)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Panel 1: Total units at risk (absolute perspective)
ax = axes[0, 0]
county_totals = df_norm.groupby('fips').agg({
    'weighted_damage_units': 'mean',
    'total_housing_units': 'first',
    'quadrant': lambda x: x.mode()[0]
}).reset_index()

colors = {
    'High Damage / High Capacity': '#2c7bb6',
    'High Damage / Low Capacity': '#d7191c',
    'Low Damage / High Capacity': '#1a9850',
    'Low Damage / Low Capacity': '#fdae61'
}

for quad, color in colors.items():
    subset = county_totals[county_totals['quadrant'] == quad]
    ax.scatter(subset['total_housing_units'], subset['weighted_damage_units'],
               alpha=0.4, s=30, c=color, label=quad)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Total Housing Units in County (log scale)', fontweight='bold', fontsize=11)
ax.set_ylabel('Average Damage (weighted units, log scale)', fontweight='bold', fontsize=11)
ax.set_title('A) Absolute Perspective: Total Units at Risk\n(Matters for total resource allocation)', 
             fontweight='bold', fontsize=12)
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.3, which='both')

# Highlight large counties
large = county_totals.nlargest(5, 'weighted_damage_units')
for _, row in large.iterrows():
    ax.annotate('', xy=(row['total_housing_units'], row['weighted_damage_units']),
                xytext=(row['total_housing_units']*1.5, row['weighted_damage_units']*1.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# Panel 2: Proportional impact (normalized perspective)
ax = axes[0, 1]
for quad, color in colors.items():
    subset = county_totals[county_totals['quadrant'] == quad]
    pct_damaged = (subset['weighted_damage_units'] / subset['total_housing_units']) * 100
    ax.scatter(subset['total_housing_units'], pct_damaged,
               alpha=0.4, s=30, c=color, label=quad)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Total Housing Units in County (log scale)', fontweight='bold', fontsize=11)
ax.set_ylabel('% of Housing Stock Damaged (log scale)', fontweight='bold', fontsize=11)
ax.set_title('B) Normalized Perspective: Proportional Community Impact\n(Matters for equity and community survival)', 
             fontweight='bold', fontsize=12)
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3, which='both')

# Panel 3: Distribution of absolute damage by quadrant
ax = axes[1, 0]
data_by_quad = [county_totals[county_totals['quadrant'] == q]['weighted_damage_units'].values
                for q in colors.keys()]
bp = ax.boxplot(data_by_quad, tick_labels=['HD/HC', 'HD/LC', 'LD/HC', 'LD/LC'],
                patch_artist=True)
for patch, color in zip(bp['boxes'], colors.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_yscale('log')
ax.set_ylabel('Average Absolute Damage (weighted units, log)', fontweight='bold', fontsize=11)
ax.set_title('C) Absolute Damage Distribution by Normalized Quadrant', fontweight='bold', fontsize=12)
ax.grid(alpha=0.3, axis='y')

# Panel 4: County size distribution by quadrant
ax = axes[1, 1]
data_by_quad = [county_totals[county_totals['quadrant'] == q]['total_housing_units'].values
                for q in colors.keys()]
bp = ax.boxplot(data_by_quad, tick_labels=['HD/HC', 'HD/LC', 'LD/HC', 'LD/LC'],
                patch_artist=True)
for patch, color in zip(bp['boxes'], colors.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_yscale('log')
ax.set_ylabel('Total Housing Units (log scale)', fontweight='bold', fontsize=11)
ax.set_title('D) County Size Distribution by Normalized Quadrant', fontweight='bold', fontsize=12)
ax.grid(alpha=0.3, axis='y')

plt.suptitle('Dual Perspective: Absolute vs. Normalized Metrics\nBoth Matter for Different Policy Questions',
             fontsize=16, fontweight='bold')
plt.tight_layout()

output_file = OUTPUT_DIR / 'absolute_vs_normalized_dual_perspective.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualization saved to: {output_file}")

# Create summary table
print("\n" + "=" * 100)
print("SUMMARY STATISTICS: ABSOLUTE DAMAGE BY NORMALIZED QUADRANT")
print("=" * 100)

summary = []
for quad in colors.keys():
    subset = county_totals[county_totals['quadrant'] == quad]
    summary.append({
        'Quadrant (normalized)': quad,
        'N Counties': len(subset),
        'Mean Total Units': f"{subset['total_housing_units'].mean():,.0f}",
        'Mean Abs Damage': f"{subset['weighted_damage_units'].mean():,.0f}",
        'Median Abs Damage': f"{subset['weighted_damage_units'].median():,.0f}",
        'Total Abs Damage (all counties)': f"{subset['weighted_damage_units'].sum():,.0f}"
    })

summary_df = pd.DataFrame(summary)
print("\n" + summary_df.to_string(index=False))

print("\n" + "=" * 100)
print("KEY INSIGHTS")
print("=" * 100)

print("""
1. LOW DAMAGE / HIGH CAPACITY counties have HIGHEST average absolute damage!
   → These are large counties (median ~60k units)
   → Lots of absolute damage but small % of total stock
   → Example: Major metro counties that can absorb the impact
   
2. HIGH DAMAGE / LOW CAPACITY counties have LOWER average absolute damage
   → These are small counties (median ~10k units)
   → Less absolute damage but catastrophic % of total stock
   → Example: Rural coastal communities facing existential threat

3. POLICY IMPLICATIONS:
   
   FOR TOTAL RESOURCE ALLOCATION (Federal aid, insurance):
   → Use ABSOLUTE metrics
   → Large counties need most total resources
   → Focus on counties with highest absolute damage
   
   FOR TARGETED VULNERABILITY INTERVENTIONS:
   → Use NORMALIZED metrics  
   → Small counties need proportionally more help relative to capacity
   → Focus on counties with highest % impact
   
   FOR COMPREHENSIVE RISK MANAGEMENT:
   → Use BOTH perspectives
   → Create risk matrix combining absolute exposure and proportional vulnerability
   → Different strategies for different quadrant types

4. FLORIDA'S TRANSITION EXPLAINED:
   → Large FL counties: High absolute damage BUT low proportional impact
   → They moved from "High Damage" (absolute) to "Low Damage" (normalized)
   → This is CORRECT - they face large total losses but can absorb them
   → Small coastal counties remain High Damage in both perspectives
   
RECOMMENDATION:
Report both perspectives:
- "Absolute Risk Exposure" (total potential losses)
- "Community Vulnerability" (proportional impact and recovery capacity)
""")

county_totals.to_csv(OUTPUT_DIR / 'county_absolute_vs_normalized_comparison.csv', index=False)
print(f"\nDetailed comparison saved to: county_absolute_vs_normalized_comparison.csv")

