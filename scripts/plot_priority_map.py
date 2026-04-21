"""
Create standalone priority category map
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load the results
metrics = pd.read_csv('../analysis_output/recovery_based_top_priority_counties.csv')

# Load all counties to get complete metrics
import json
from pathlib import Path
import numpy as np

# Reload full metrics
recovery_dir = Path('../data/recovery_potential_per_scenario')

all_recovery_data = []
for json_file in sorted(recovery_dir.glob('*.json')):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for record in data:
        all_recovery_data.append({
            'event': record.get('event', json_file.stem.split('_')[0]),
            'fips': str(record['fips']).zfill(5),
            'recovery_time': float(record.get('recovery_potential [months]', 0)),
            'capacity': float(record.get('reconstruction_capacity', 0))
        })

recovery_df = pd.DataFrame(all_recovery_data)

# Compute metrics
n_events = recovery_df['event'].nunique()
event_prob = 0.00067334 / n_events

county_metrics = recovery_df.groupby('fips').agg({
    'recovery_time': [
        ('eart', lambda x: (x * event_prob).sum()),
        ('max_recovery', 'max'),
    ],
    'event': 'count'
}).reset_index()

county_metrics.columns = ['fips', 'eart', 'max_recovery', 'n_events']

# Filter inf
county_metrics = county_metrics[
    (county_metrics['eart'] != np.inf) & 
    (county_metrics['max_recovery'] != np.inf) &
    (~county_metrics['eart'].isna()) &
    (~county_metrics['max_recovery'].isna())
].copy()

# Categorize
def categorize_median_split(values):
    median_val = values.median()
    categories = (values > median_val).astype(int)
    return categories, median_val

county_metrics['eart_cat'], eart_median = categorize_median_split(county_metrics['eart'])
county_metrics['max_recovery_cat'], max_median = categorize_median_split(county_metrics['max_recovery'])

PRIORITY_LABELS = {
    (1, 0): 'P1: Chronic Burden',
    (1, 1): 'P2: Critical',
    (0, 0): 'P3: Resilient',
    (0, 1): 'P4: Tail Risk'
}

PRIORITY_COLORS = {
    'P1: Chronic Burden': '#FFA500',
    'P2: Critical': '#8B0000',
    'P3: Resilient': '#90EE90',
    'P4: Tail Risk': '#FFD700'
}

county_metrics['priority_code'] = list(zip(county_metrics['eart_cat'], 
                                            county_metrics['max_recovery_cat']))
county_metrics['priority_category'] = county_metrics['priority_code'].map(PRIORITY_LABELS)

# Load shapefile
counties = gpd.read_file('../data/US_counties.shp')
counties['FIPS'] = counties['GEOID'].astype(str).str.zfill(5)

# Merge
gdf = counties.merge(county_metrics, left_on='FIPS', right_on='fips', how='left')

# Create map
fig, ax = plt.subplots(figsize=(20, 12))

# Plot by category
for cat in ['P3: Resilient', 'P4: Tail Risk', 'P1: Chronic Burden', 'P2: Critical']:
    subset = gdf[gdf['priority_category'] == cat]
    if len(subset) > 0:
        subset.plot(color=PRIORITY_COLORS[cat], ax=ax, edgecolor='white', linewidth=0.5)

# Plot counties with no data in gray
gdf[gdf['priority_category'].isna()].plot(color='lightgray', ax=ax, edgecolor='white', linewidth=0.3)

ax.set_title('Recovery Priority Categories\n(Based on Expected Annual Recovery Time × Maximum Recovery Time)',
             fontsize=18, fontweight='bold', pad=20)
ax.axis('off')

# Create legend
legend_elements = [
    mpatches.Patch(facecolor=PRIORITY_COLORS['P2: Critical'], 
                   edgecolor='black', label=f'P2: Critical ({(gdf["priority_category"]=="P2: Critical").sum()} counties)'),
    mpatches.Patch(facecolor=PRIORITY_COLORS['P1: Chronic Burden'], 
                   edgecolor='black', label=f'P1: Chronic Burden ({(gdf["priority_category"]=="P1: Chronic Burden").sum()} counties)'),
    mpatches.Patch(facecolor=PRIORITY_COLORS['P4: Tail Risk'], 
                   edgecolor='black', label=f'P4: Tail Risk ({(gdf["priority_category"]=="P4: Tail Risk").sum()} counties)'),
    mpatches.Patch(facecolor=PRIORITY_COLORS['P3: Resilient'], 
                   edgecolor='black', label=f'P3: Resilient ({(gdf["priority_category"]=="P3: Resilient").sum()} counties)'),
    mpatches.Patch(facecolor='lightgray', 
                   edgecolor='black', label=f'No data ({gdf["priority_category"].isna().sum()} counties)')
]

ax.legend(handles=legend_elements, loc='lower left', fontsize=14, 
          frameon=True, fancybox=True, shadow=True)

# Add text box with category definitions
textstr = """
P2: Critical - High EART + High Max (both chronic burden AND tail risk)
P1: Chronic Burden - High EART + Low Max (frequent events, manageable extremes)
P4: Tail Risk - Low EART + High Max (rare but catastrophic events)
P3: Resilient - Low EART + Low Max (limited recovery challenges)
"""

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('../analysis_output/priority_categories_map.png', dpi=300, bbox_inches='tight')
print("Saved: priority_categories_map.png")

# Print distribution and correlation analysis
print("\n" + "="*60)
print("PRIORITY CATEGORY DISTRIBUTION")
print("="*60)
for cat in ['P2: Critical', 'P1: Chronic Burden', 'P4: Tail Risk', 'P3: Resilient']:
    count = (county_metrics['priority_category'] == cat).sum()
    pct = count / len(county_metrics) * 100
    print(f"{cat:25s}: {count:3d} counties ({pct:5.1f}%)")

print(f"\nMedian thresholds:")
print(f"  EART: {eart_median:.4f} months")
print(f"  Max Recovery: {max_median:.1f} months")

# Correlation analysis
print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)
correlation = county_metrics['eart'].corr(county_metrics['max_recovery'])
print(f"Pearson correlation (EART vs Max Recovery): {correlation:.3f}")

# Spearman rank correlation (robust to outliers)
from scipy.stats import spearmanr
spearman_corr, p_value = spearmanr(county_metrics['eart'], county_metrics['max_recovery'])
print(f"Spearman rank correlation: {spearman_corr:.3f} (p-value: {p_value:.2e})")

print("\nInterpretation:")
if abs(correlation) > 0.7:
    print("Strong correlation indicates EART and Max Recovery are driven by similar factors.")
    print("Counties with high expected annual burden also face high tail risk.")
    print("This explains why P1 (high EART, low max) and P4 (low EART, high max) are rare.")
elif abs(correlation) > 0.4:
    print("Moderate correlation suggests some overlap but distinct vulnerability profiles exist.")
else:
    print("Weak correlation suggests EART and Max Recovery capture different aspects of risk.")

plt.close()
