"""
Analyze distributions of recovery times, impacts, and capacity

Rather than just summary statistics, examine the full distribution
to understand the nature of recovery challenges in each county.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats

BASE_DIR = Path(__file__).parent.parent

# Load recovery data
print("Loading recovery data...")
recovery_dir = BASE_DIR / 'data' / 'recovery_potential_per_scenario'

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

# Load impact data
print("Loading impact data...")
impacts_dir = BASE_DIR / 'impacts_out' / 'by_event' / 'scaled'

all_impacts = []
for csv_file in sorted(impacts_dir.glob('*.csv')):
    df = pd.read_csv(csv_file)
    df['event'] = csv_file.stem.replace('_scaled', '')
    all_impacts.append(df)

impacts_df = pd.concat(all_impacts, ignore_index=True)
impacts_df['fips'] = impacts_df['fips'].astype(str).str.zfill(5)

# Compute weighted damage
impacts_df['weighted_damage'] = (
    impacts_df['units_DS1_scaled'] * 1 + 
    impacts_df['units_DS2_scaled'] * 1 + 
    impacts_df['units_DS3_scaled'] * 3 +
    impacts_df['units_DS4_scaled'] * 6
)

# Merge recovery and impacts
merged = recovery_df.merge(impacts_df[['event', 'fips', 'weighted_damage']], 
                           on=['event', 'fips'], how='left')

# Filter out infinite recovery times
merged = merged[(merged['recovery_time'] != np.inf) & (merged['recovery_time'].notna())]

print(f"Loaded {len(merged):,} event-county pairs")
print(f"Counties: {merged['fips'].nunique()}")
print(f"Events: {merged['event'].nunique()}")

# Compute distribution metrics for each county
print("\nComputing distribution metrics...")

def compute_distribution_metrics(group):
    """Compute comprehensive distribution statistics."""
    rt = group['recovery_time'].values
    wd = group['weighted_damage'].values
    cap = group['capacity'].values[0] if len(group['capacity'].values) > 0 else 0
    
    # Filter out zeros for some metrics
    rt_nonzero = rt[rt > 0]
    wd_nonzero = wd[wd > 0]
    
    return pd.Series({
        # Recovery time distribution
        'rt_mean': rt.mean(),
        'rt_median': np.median(rt),
        'rt_std': rt.std(),
        'rt_cv': rt.std() / rt.mean() if rt.mean() > 0 else 0,  # Coefficient of variation
        'rt_skew': stats.skew(rt),
        'rt_p10': np.percentile(rt, 10),
        'rt_p50': np.percentile(rt, 50),
        'rt_p90': np.percentile(rt, 90),
        'rt_p99': np.percentile(rt, 99),
        'rt_max': rt.max(),
        'rt_iqr': np.percentile(rt, 75) - np.percentile(rt, 25),
        
        # Fraction of events causing different recovery burdens
        'frac_rt_zero': (rt == 0).sum() / len(rt),
        'frac_rt_low': ((rt > 0) & (rt < 1)).sum() / len(rt),  # < 1 month
        'frac_rt_medium': ((rt >= 1) & (rt < 12)).sum() / len(rt),  # 1-12 months
        'frac_rt_high': ((rt >= 12) & (rt < 120)).sum() / len(rt),  # 1-10 years
        'frac_rt_extreme': (rt >= 120).sum() / len(rt),  # > 10 years
        
        # Weighted damage distribution
        'wd_mean': wd.mean(),
        'wd_median': np.median(wd),
        'wd_std': wd.std(),
        'wd_cv': wd.std() / wd.mean() if wd.mean() > 0 else 0,
        'wd_skew': stats.skew(wd),
        'wd_p90': np.percentile(wd, 90),
        'wd_p99': np.percentile(wd, 99),
        'wd_max': wd.max(),
        
        # Capacity
        'capacity': cap,
        
        # Relationship between damage and recovery
        'damage_recovery_corr': np.corrcoef(wd, rt)[0, 1] if len(rt) > 1 else 0,
        
        # Event count
        'n_events': len(rt),
        'n_damaging_events': (wd > 0).sum(),
    })

county_dist = merged.groupby('fips').apply(compute_distribution_metrics).reset_index()

print(f"Computed distribution metrics for {len(county_dist)} counties")

# Identify distribution patterns
print("\n" + "="*60)
print("DISTRIBUTION PATTERN ANALYSIS")
print("="*60)

# 1. Coefficient of Variation (variability relative to mean)
print("\n1. RECOVERY TIME VARIABILITY (CV = std/mean)")
print(f"   Low variability (CV < 1): {(county_dist['rt_cv'] < 1).sum()} counties")
print(f"   Moderate variability (1 ≤ CV < 2): {((county_dist['rt_cv'] >= 1) & (county_dist['rt_cv'] < 2)).sum()} counties")
print(f"   High variability (CV ≥ 2): {(county_dist['rt_cv'] >= 2).sum()} counties")
print(f"   Mean CV: {county_dist['rt_cv'].mean():.2f}")

# 2. Skewness (tail behavior)
print("\n2. DISTRIBUTION SHAPE (Skewness)")
print(f"   Left-skewed (<0): {(county_dist['rt_skew'] < 0).sum()} counties")
print(f"   Symmetric (≈0): {((county_dist['rt_skew'] >= -0.5) & (county_dist['rt_skew'] <= 0.5)).sum()} counties")
print(f"   Right-skewed (>0): {(county_dist['rt_skew'] > 0).sum()} counties")
print(f"   Highly right-skewed (>2): {(county_dist['rt_skew'] > 2).sum()} counties - dominated by extreme events")
print(f"   Mean skewness: {county_dist['rt_skew'].mean():.2f}")

# 3. Event frequency patterns
print("\n3. EVENT FREQUENCY PATTERNS")
print(f"   Mostly no damage (>50% zero recovery): {(county_dist['frac_rt_zero'] > 0.5).sum()} counties")
print(f"   Frequent minor events (>30% low recovery): {(county_dist['frac_rt_low'] > 0.3).sum()} counties")
print(f"   Some extreme events (>5% extreme): {(county_dist['frac_rt_extreme'] > 0.05).sum()} counties")

# 4. Tail dominance: compare P90/median ratio
county_dist['tail_dominance'] = county_dist['rt_p90'] / (county_dist['rt_median'] + 0.001)
print("\n4. TAIL DOMINANCE (P90/Median ratio)")
print(f"   Low (<10x): {(county_dist['tail_dominance'] < 10).sum()} counties - relatively uniform")
print(f"   Moderate (10-100x): {((county_dist['tail_dominance'] >= 10) & (county_dist['tail_dominance'] < 100)).sum()} counties")
print(f"   High (≥100x): {(county_dist['tail_dominance'] >= 100).sum()} counties - extreme tail risk")

# 5. Capacity constraint indicator
county_dist['capacity_limited'] = county_dist['capacity'] < 10
print("\n5. CAPACITY CONSTRAINTS")
print(f"   Very low capacity (<10 units/month): {county_dist['capacity_limited'].sum()} counties")
print(f"   Adequate capacity (≥10 units/month): {(~county_dist['capacity_limited']).sum()} counties")

# Create visualization comparing distribution patterns
print("\nCreating distribution comparison plots...")

fig = plt.figure(figsize=(20, 12))

# Plot 1: CV vs Skewness
ax1 = plt.subplot(2, 3, 1)
scatter = ax1.scatter(county_dist['rt_skew'], county_dist['rt_cv'], 
                     c=county_dist['capacity'], cmap='viridis', alpha=0.6, s=30)
ax1.set_xlabel('Skewness (shape)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Coefficient of Variation (variability)', fontsize=11, fontweight='bold')
ax1.set_title('A) Distribution Shape vs Variability', fontsize=12, fontweight='bold')
ax1.axhline(1, color='red', linestyle='--', alpha=0.3, label='CV=1')
ax1.axvline(0, color='red', linestyle='--', alpha=0.3, label='Symmetric')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Capacity')
ax1.set_xlim(-2, 10)
ax1.set_ylim(0, 5)

# Plot 2: Tail dominance
ax2 = plt.subplot(2, 3, 2)
ax2.hist(np.log10(county_dist['tail_dominance'] + 1), bins=50, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Log10(P90/Median)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Number of Counties', fontsize=11, fontweight='bold')
ax2.set_title('B) Tail Dominance Distribution', fontsize=12, fontweight='bold')
ax2.axvline(np.log10(10), color='red', linestyle='--', label='10x threshold')
ax2.axvline(np.log10(100), color='orange', linestyle='--', label='100x threshold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Event fraction breakdown
ax3 = plt.subplot(2, 3, 3)
fractions = county_dist[['frac_rt_zero', 'frac_rt_low', 'frac_rt_medium', 
                         'frac_rt_high', 'frac_rt_extreme']].mean()
ax3.bar(range(5), fractions, color=['gray', 'green', 'yellow', 'orange', 'red'])
ax3.set_xticks(range(5))
ax3.set_xticklabels(['Zero', 'Low\n(<1mo)', 'Medium\n(1-12mo)', 
                     'High\n(1-10yr)', 'Extreme\n(>10yr)'], fontsize=9)
ax3.set_ylabel('Mean Fraction of Events', fontsize=11, fontweight='bold')
ax3.set_title('C) Average Event Severity Distribution', fontsize=12, fontweight='bold')
ax3.grid(True, axis='y', alpha=0.3)

# Plot 4: Damage vs Recovery correlation
ax4 = plt.subplot(2, 3, 4)
scatter = ax4.scatter(county_dist['damage_recovery_corr'], county_dist['capacity'],
                     c=county_dist['rt_mean'], cmap='Reds', alpha=0.6, s=30)
ax4.set_xlabel('Damage-Recovery Correlation', fontsize=11, fontweight='bold')
ax4.set_ylabel('Capacity (units/month)', fontsize=11, fontweight='bold')
ax4.set_title('D) Damage-Recovery Relationship vs Capacity', fontsize=12, fontweight='bold')
ax4.axvline(0.7, color='blue', linestyle='--', alpha=0.3, label='Strong correlation')
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Mean Recovery Time')
ax4.legend()

# Plot 5: Capacity effect on variability
ax5 = plt.subplot(2, 3, 5)
low_cap = county_dist[county_dist['capacity'] < 10]
high_cap = county_dist[county_dist['capacity'] >= 10]
ax5.scatter(low_cap['rt_median'], low_cap['rt_p90'], alpha=0.5, label='Low capacity', s=30)
ax5.scatter(high_cap['rt_median'], high_cap['rt_p90'], alpha=0.5, label='High capacity', s=30)
ax5.plot([0, 1000], [0, 1000], 'k--', alpha=0.3, label='1:1 line')
ax5.set_xlabel('Median Recovery Time (months)', fontsize=11, fontweight='bold')
ax5.set_ylabel('P90 Recovery Time (months)', fontsize=11, fontweight='bold')
ax5.set_title('E) Median vs P90 by Capacity', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, 200)
ax5.set_ylim(0, 5000)

# Plot 6: Percentile ranges
ax6 = plt.subplot(2, 3, 6)
# Select sample counties with different patterns
sample_counties = county_dist.nlargest(10, 'rt_mean')[['fips', 'rt_p10', 'rt_median', 'rt_p90', 'rt_max']].head(10)
y_pos = range(len(sample_counties))
for i, (idx, row) in enumerate(sample_counties.iterrows()):
    ax6.plot([row['rt_p10'], row['rt_p90']], [i, i], 'b-', linewidth=2, alpha=0.6)
    ax6.plot(row['rt_median'], i, 'ro', markersize=6)
    ax6.plot(row['rt_max'], i, 'r^', markersize=6, alpha=0.5)

ax6.set_yticks(y_pos)
ax6.set_yticklabels(sample_counties['fips'].values, fontsize=8)
ax6.set_xlabel('Recovery Time (months)', fontsize=11, fontweight='bold')
ax6.set_title('F) Recovery Time Ranges (Top 10 Mean Recovery)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')
ax6.legend(['P10-P90 range', 'Median', 'Maximum'], loc='best', fontsize=8)

plt.tight_layout()
plt.savefig(BASE_DIR / 'analysis_output' / 'recovery_distribution_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: recovery_distribution_analysis.png")

# Save detailed metrics
county_dist.to_csv(BASE_DIR / 'analysis_output' / 'county_distribution_metrics.csv', index=False)
print("Saved: county_distribution_metrics.csv")

# Identify interesting county patterns
print("\n" + "="*60)
print("NOTABLE COUNTY PATTERNS")
print("="*60)

print("\nMost variable counties (high CV):")
print(county_dist.nlargest(5, 'rt_cv')[['fips', 'rt_cv', 'rt_mean', 'rt_std', 'capacity']])

print("\nMost skewed counties (extreme tail risk):")
print(county_dist.nlargest(5, 'rt_skew')[['fips', 'rt_skew', 'rt_median', 'rt_p99', 'frac_rt_extreme']])

print("\nHighest tail dominance (P90/median):")
print(county_dist.nlargest(5, 'tail_dominance')[['fips', 'tail_dominance', 'rt_median', 'rt_p90', 'capacity']])

print("\nMost predictable (low CV, low skew):")
predictable = county_dist[(county_dist['rt_cv'] < 1) & (county_dist['rt_skew'] < 1) & (county_dist['rt_mean'] > 0)]
print(predictable.nsmallest(5, 'rt_cv')[['fips', 'rt_cv', 'rt_skew', 'rt_mean', 'capacity']])

plt.close()
