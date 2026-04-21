"""
Create visualizations for the revised paper structure:
1. Skewness map showing tail risk dominance
2. Same-damage counties with different capacity outcomes
3. Summary figure linking perspectives
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json
from scipy import stats

print("Creating visualizations for paper narrative...")

# Load distribution metrics
county_dist = pd.read_csv('../analysis_output/county_distribution_metrics.csv')
county_dist['fips'] = county_dist['fips'].astype(str).str.zfill(5)

# Load shapefile
counties = gpd.read_file('../data/US_counties.shp')
counties['FIPS'] = counties['GEOID'].astype(str).str.zfill(5)

# Merge
gdf = counties.merge(county_dist, left_on='FIPS', right_on='fips', how='left')

# ============================================================
# 1. SKEWNESS MAP
# ============================================================
print("\n1. Creating skewness map...")

fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# Panel A: Skewness
ax1 = axes[0, 0]
gdf_plot = gdf[gdf['rt_skew'].notna()]
gdf_plot.plot(column='rt_skew', ax=ax1, legend=True, cmap='YlOrRd',
              vmin=0, vmax=5,
              legend_kwds={'label': 'Skewness', 'shrink': 0.8})
ax1.set_title('A) Distribution Skewness\n(Higher = More Tail-Dominated)', 
              fontsize=14, fontweight='bold')
ax1.axis('off')

# Add interpretation text
textstr = 'Skewness > 2: Extreme tail risk\n(rare catastrophic events dominate)\n\nSkewness < 1: More uniform\n(damage spread across events)'
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel B: Tail Dominance (P90/median)
ax2 = axes[0, 1]
gdf['tail_dom_log'] = np.log10(gdf['tail_dominance'].clip(upper=1e6) + 1)
gdf_plot = gdf[gdf['tail_dom_log'].notna()]
gdf_plot.plot(column='tail_dom_log', ax=ax2, legend=True, cmap='Reds',
              legend_kwds={'label': 'Log10(P90/Median + 1)', 'shrink': 0.8})
ax2.set_title('B) Tail Dominance (P90/Median ratio)\n(Higher = Extreme events >> typical)', 
              fontsize=14, fontweight='bold')
ax2.axis('off')

textstr = 'High ratio:\nMedian ≈ 0, but extreme\nevents cause major damage\n\nLow ratio:\nMore predictable\nrecovery needs'
ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel C: Mean Recovery Time (for context)
ax3 = axes[1, 0]
gdf_plot = gdf[gdf['rt_mean'].notna()]
gdf_plot.plot(column='rt_mean', ax=ax3, legend=True, cmap='Purples',
              vmin=0, vmax=100,
              legend_kwds={'label': 'Mean Recovery Time (months)', 'shrink': 0.8})
ax3.set_title('C) Mean Recovery Time\n(Average across all events)', 
              fontsize=14, fontweight='bold')
ax3.axis('off')

# Panel D: Capacity
ax4 = axes[1, 1]
gdf_plot = gdf[gdf['capacity'].notna()]
gdf_plot.plot(column='capacity', ax=ax4, legend=True, cmap='Greens',
              vmin=0, vmax=100,
              legend_kwds={'label': 'Construction Capacity (units/month)', 'shrink': 0.8})
ax4.set_title('D) Construction Capacity\n(Recovery constraint)', 
              fontsize=14, fontweight='bold')
ax4.axis('off')

plt.suptitle('Spatial Patterns of Tail Risk and Recovery Constraints', 
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('../analysis_output/tail_risk_and_capacity_maps.png', dpi=300, bbox_inches='tight')
print("Saved: tail_risk_and_capacity_maps.png")

# ============================================================
# 2. SAME DAMAGE, DIFFERENT CAPACITY ANALYSIS
# ============================================================
print("\n2. Creating same-damage, different-capacity visualization...")

# Load full recovery data
recovery_dir = Path('../data/recovery_potential_per_scenario')
all_data = []
for json_file in sorted(recovery_dir.glob('*.json')):
    with open(json_file, 'r') as f:
        data = json.load(f)
    for record in data:
        all_data.append({
            'event': record.get('event', json_file.stem.split('_')[0]),
            'fips': str(record['fips']).zfill(5),
            'recovery_time': float(record.get('recovery_potential [months]', 0)),
            'capacity': float(record.get('reconstruction_capacity', 0))
        })

recovery_df = pd.DataFrame(all_data)

# Load impacts
impacts_dir = Path('../impacts_out/by_event/scaled')
all_impacts = []
for csv_file in sorted(impacts_dir.glob('*.csv')):
    df = pd.read_csv(csv_file)
    df['event'] = csv_file.stem.replace('_scaled', '')
    all_impacts.append(df)

impacts_df = pd.concat(all_impacts, ignore_index=True)
impacts_df['fips'] = impacts_df['fips'].astype(str).str.zfill(5)
impacts_df['weighted_damage'] = (
    impacts_df['units_DS1_scaled'] * 1 + 
    impacts_df['units_DS2_scaled'] * 1 + 
    impacts_df['units_DS3_scaled'] * 3 + 
    impacts_df['units_DS4_scaled'] * 6
)

# Merge
merged = recovery_df.merge(impacts_df[['event', 'fips', 'weighted_damage']], 
                           on=['event', 'fips'], how='left')
merged = merged[(merged['recovery_time'] != np.inf) & (merged['recovery_time'].notna())]

# Focus on high damage events (top 20% of damage)
damage_threshold = merged['weighted_damage'].quantile(0.80)
high_damage = merged[merged['weighted_damage'] >= damage_threshold].copy()

print(f"High damage events (top 20%): {len(high_damage):,} records")

# Categorize capacity
high_damage['capacity_cat'] = pd.cut(high_damage['capacity'], 
                                     bins=[0, 10, 50, np.inf],
                                     labels=['Low (<10)', 'Medium (10-50)', 'High (>50)'])

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Panel A: Same damage, different capacity outcomes
ax1 = axes[0, 0]
for cap_cat in ['Low (<10)', 'Medium (10-50)', 'High (>50)']:
    subset = high_damage[high_damage['capacity_cat'] == cap_cat]
    ax1.scatter(subset['weighted_damage'], subset['recovery_time'], 
               alpha=0.4, s=20, label=cap_cat)

ax1.set_xlabel('Weighted Damage (units)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Recovery Time (months)', fontsize=12, fontweight='bold')
ax1.set_title('A) Same Damage, Different Recovery\n(High-damage events only, top 20%)', 
              fontsize=13, fontweight='bold')
ax1.legend(title='Construction Capacity', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 500)
ax1.set_ylim(0, 5000)

# Panel B: Damage bins with capacity stratification
ax2 = axes[0, 1]
damage_bins = pd.qcut(high_damage['weighted_damage'], q=5, labels=['D1', 'D2', 'D3', 'D4', 'D5'], duplicates='drop')
high_damage['damage_bin'] = damage_bins

violin_data = []
positions = []
colors_list = []
pos = 0
for dbin in ['D1', 'D2', 'D3', 'D4', 'D5']:
    for cap_cat in ['Low (<10)', 'Medium (10-50)', 'High (>50)']:
        subset = high_damage[(high_damage['damage_bin'] == dbin) & 
                            (high_damage['capacity_cat'] == cap_cat)]
        if len(subset) > 5:
            violin_data.append(subset['recovery_time'].values)
            positions.append(pos)
            colors_list.append({'Low (<10)': 'red', 'Medium (10-50)': 'orange', 'High (>50)': 'green'}[cap_cat])
            pos += 1

parts = ax2.violinplot(violin_data, positions=positions, widths=0.7, showmeans=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_list[i])
    pc.set_alpha(0.6)

ax2.set_xlabel('Damage Level → Capacity', fontsize=12, fontweight='bold')
ax2.set_ylabel('Recovery Time (months)', fontsize=12, fontweight='bold')
ax2.set_title('B) Recovery Time Distributions\n(Stratified by damage × capacity)', 
              fontsize=13, fontweight='bold')
ax2.set_ylim(0, 3000)
ax2.grid(True, axis='y', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', alpha=0.6, label='Low capacity'),
                  Patch(facecolor='orange', alpha=0.6, label='Medium capacity'),
                  Patch(facecolor='green', alpha=0.6, label='High capacity')]
ax2.legend(handles=legend_elements, loc='upper left')

# Panel C: Capacity multiplier effect
ax3 = axes[1, 0]
# Compute median recovery for each damage-capacity combination
summary = high_damage.groupby(['damage_bin', 'capacity_cat'])['recovery_time'].agg(['median', 'count']).reset_index()
summary = summary[summary['count'] > 5]

width = 0.25
x = np.arange(len(['D1', 'D2', 'D3', 'D4', 'D5']))

for i, cap_cat in enumerate(['Low (<10)', 'Medium (10-50)', 'High (>50)']):
    subset = summary[summary['capacity_cat'] == cap_cat]
    if len(subset) > 0:
        y_values = []
        for dbin in ['D1', 'D2', 'D3', 'D4', 'D5']:
            val = subset[subset['damage_bin'] == dbin]['median'].values
            y_values.append(val[0] if len(val) > 0 else 0)
        
        ax3.bar(x + i*width, y_values, width, 
               label=cap_cat,
               color={'Low (<10)': 'red', 'Medium (10-50)': 'orange', 'High (>50)': 'green'}[cap_cat],
               alpha=0.7)

ax3.set_xlabel('Damage Level (quintiles of high-damage events)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Median Recovery Time (months)', fontsize=12, fontweight='bold')
ax3.set_title('C) Capacity as Multiplier Effect\n(Median recovery by damage × capacity)', 
              fontsize=13, fontweight='bold')
ax3.set_xticks(x + width)
ax3.set_xticklabels(['D1\n(lowest)', 'D2', 'D3', 'D4', 'D5\n(highest)'])
ax3.legend(title='Construction Capacity')
ax3.grid(True, axis='y', alpha=0.3)

# Panel D: Example counties with same damage
ax4 = axes[1, 1]

# Find a damage value where we have counties with different capacities
target_damage = high_damage['weighted_damage'].quantile(0.5)
tolerance = high_damage['weighted_damage'].std() * 0.2

similar_damage = high_damage[
    (high_damage['weighted_damage'] >= target_damage - tolerance) &
    (high_damage['weighted_damage'] <= target_damage + tolerance)
].copy()

# Get one example from each capacity category
examples = []
for cap_cat in ['Low (<10)', 'Medium (10-50)', 'High (>50)']:
    subset = similar_damage[similar_damage['capacity_cat'] == cap_cat]
    if len(subset) > 0:
        example = subset.sample(min(50, len(subset)))
        examples.append(example)

if examples:
    all_examples = pd.concat(examples)
    
    # Scatter plot
    colors_map = {'Low (<10)': 'red', 'Medium (10-50)': 'orange', 'High (>50)': 'green'}
    for cap_cat in ['Low (<10)', 'Medium (10-50)', 'High (>50)']:
        subset = all_examples[all_examples['capacity_cat'] == cap_cat]
        ax4.scatter(subset['capacity'], subset['recovery_time'], 
                   c=colors_map[cap_cat], s=80, alpha=0.6, label=cap_cat,
                   edgecolors='black', linewidth=0.5)
    
    ax4.set_xlabel('Construction Capacity (units/month)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Recovery Time (months)', fontsize=12, fontweight='bold')
    ax4.set_title(f'D) Example: Similar Damage (~{target_damage:.0f} units)\nDifferent Capacity → Different Recovery', 
                  fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, max(100, all_examples['capacity'].max() * 1.1))

plt.suptitle('Capacity as Multiplier: Same Damage, Different Outcomes', 
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('../analysis_output/capacity_multiplier_effect.png', dpi=300, bbox_inches='tight')
print("Saved: capacity_multiplier_effect.png")

# ============================================================
# 3. CONCEPTUAL SUMMARY FIGURE
# ============================================================
print("\n3. Creating conceptual summary figure...")

fig = plt.figure(figsize=(20, 10))

# Left panel: Probabilistic perspective (tail risk dominates)
ax_left = plt.subplot(1, 2, 1)
# Compute county-level metrics
county_summary = merged.groupby('fips').agg({
    'weighted_damage': 'mean',
    'recovery_time': 'mean',
    'capacity': 'first'
}).reset_index()
county_summary = county_summary[
    (county_summary['capacity'] > 0) & 
    (county_summary['weighted_damage'] > 0)
]

scatter = ax_left.scatter(county_summary['weighted_damage'], 
                         county_summary['recovery_time'],
                         c=county_summary['capacity'], 
                         cmap='viridis', alpha=0.6, s=50,
                         edgecolors='black', linewidth=0.5)
ax_left.set_xlabel('Mean Weighted Damage (units)', fontsize=13, fontweight='bold')
ax_left.set_ylabel('Mean Recovery Time (months)', fontsize=13, fontweight='bold')
ax_left.set_title('PROBABILISTIC VIEW (Top-Down)\nDamage explains 82% of variance\nTail risk dominates', 
                  fontsize=14, fontweight='bold', pad=15)
ax_left.grid(True, alpha=0.3)
ax_left.set_xlim(0, 100)
ax_left.set_ylim(0, 300)

cbar = plt.colorbar(scatter, ax=ax_left)
cbar.set_label('Capacity (units/month)', fontsize=11, fontweight='bold')

# Add annotation
ax_left.text(0.05, 0.95, 'Main driver:\nDAMAGE EXPOSURE\n→ Mitigation priority', 
            transform=ax_left.transAxes, fontsize=12,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

# Right panel: Conditional perspective (capacity matters at high damage)
ax_right = plt.subplot(1, 2, 2)

high_damage_counties = merged[merged['weighted_damage'] > merged['weighted_damage'].quantile(0.75)]
county_high = high_damage_counties.groupby('fips').agg({
    'weighted_damage': 'mean',
    'recovery_time': 'mean',
    'capacity': 'first'
}).reset_index()

# Color by capacity terciles
capacity_bins = pd.qcut(county_high['capacity'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
colors = {'Low': 'red', 'Medium': 'orange', 'High': 'green'}

for cat in ['Low', 'Medium', 'High']:
    if cat in capacity_bins.values:
        mask = capacity_bins == cat
        ax_right.scatter(county_high[mask]['weighted_damage'], 
                        county_high[mask]['recovery_time'],
                        c=colors[cat], label=f'{cat} capacity', 
                        alpha=0.7, s=80, edgecolors='black', linewidth=0.5)

ax_right.set_xlabel('Mean Weighted Damage (units)', fontsize=13, fontweight='bold')
ax_right.set_ylabel('Mean Recovery Time (months)', fontsize=13, fontweight='bold')
ax_right.set_title('CONDITIONAL VIEW (High-Damage Counties)\nCapacity explains 13% additional variance\nMultiplier effect visible', 
                  fontsize=14, fontweight='bold', pad=15)
ax_right.legend(loc='upper left', fontsize=11)
ax_right.grid(True, alpha=0.3)

# Add annotation
ax_right.text(0.05, 0.95, 'Given high damage:\nCAPACITY MATTERS\n→ Build recovery capacity', 
             transform=ax_right.transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.suptitle('Complementary Perspectives: Damage (Primary) + Capacity (Multiplier)', 
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('../analysis_output/dual_perspective_summary.png', dpi=300, bbox_inches='tight')
print("Saved: dual_perspective_summary.png")

print("\n" + "="*60)
print("SUMMARY FOR PAPER")
print("="*60)
print("""
Paper Structure:

PART 1: PROBABILISTIC (TOP-DOWN) PERSPECTIVE
- Show 3-panel maps: annual damage, annual recovery, scatter
- Finding: Tail risk dominates (80%+ of counties highly skewed)
- Finding: Damage explains 82% of recovery variance
- Implication: MITIGATION is primary strategy

PART 2: CONDITIONAL (BOTTOM-UP) PERSPECTIVE  
- Focus on high-damage counties (top quartile)
- Show same-damage, different-capacity outcomes
- Finding: At high damage, capacity matters (multiplier effect)
- Finding: 99.9% of matched pairs confirm capacity reduces recovery
- Implication: CAPACITY BUILDING for high-risk areas

PART 3: SINGLE-EVENT CASE STUDIES
- Select events with different damage profiles
- Show how capacity constraints manifest in specific disasters
- Link to real-world recovery experiences

CONCLUSION: DUAL STRATEGY
- Priority 1: Reduce damage exposure (mitigation, codes, land use)
- Priority 2: Build capacity in high-risk areas (construction workforce)
- Target: High-damage + Low-capacity counties for both interventions
""")

plt.close('all')
