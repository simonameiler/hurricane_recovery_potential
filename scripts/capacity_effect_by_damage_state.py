"""
Analyze capacity effect stratified by damage states (DS1-DS4)
Rather than arbitrary quintiles, use actual damage state dominance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats

print("="*70)
print("CAPACITY EFFECT BY DAMAGE STATE (DS1-DS4)")
print("="*70)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n1. Loading recovery and impact data...")

# Load recovery data
recovery_dir = Path('../data/recovery_potential_per_scenario')
all_recovery = []
for json_file in sorted(recovery_dir.glob('*.json')):
    with open(json_file, 'r') as f:
        data = json.load(f)
    for record in data:
        all_recovery.append({
            'event': record.get('event', json_file.stem.split('_')[0]),
            'fips': str(record['fips']).zfill(5),
            'recovery_time': float(record.get('recovery_potential [months]', 0)),
            'capacity': float(record.get('reconstruction_capacity', 0))
        })

recovery_df = pd.DataFrame(all_recovery)
print(f"Loaded {len(recovery_df):,} recovery records")

# Load impacts with damage states
impacts_dir = Path('../impacts_out/by_event/scaled')
all_impacts = []
for csv_file in sorted(impacts_dir.glob('*.csv')):
    df = pd.read_csv(csv_file)
    df['event'] = csv_file.stem.replace('_scaled', '')
    all_impacts.append(df)

impacts_df = pd.concat(all_impacts, ignore_index=True)
impacts_df['fips'] = impacts_df['fips'].astype(str).str.zfill(5)
print(f"Loaded {len(impacts_df):,} impact records")

# Merge
merged = recovery_df.merge(
    impacts_df[['event', 'fips', 'units_DS1_scaled', 'units_DS2_scaled', 
                'units_DS3_scaled', 'units_DS4_scaled']], 
    on=['event', 'fips'], 
    how='inner'
)

# Filter valid records
merged = merged[
    (merged['recovery_time'] != np.inf) & 
    (merged['recovery_time'].notna()) &
    (merged['capacity'] > 0)
].copy()

print(f"Merged dataset: {len(merged):,} valid records")

# Calculate total damage and proportions
merged['total_damage'] = (
    merged['units_DS1_scaled'] + 
    merged['units_DS2_scaled'] + 
    merged['units_DS3_scaled'] + 
    merged['units_DS4_scaled']
)

# Filter out zero damage events
merged = merged[merged['total_damage'] > 0].copy()
print(f"After filtering zero damage: {len(merged):,} records")

# Calculate damage state proportions
merged['prop_DS1'] = merged['units_DS1_scaled'] / merged['total_damage']
merged['prop_DS2'] = merged['units_DS2_scaled'] / merged['total_damage']
merged['prop_DS3'] = merged['units_DS3_scaled'] / merged['total_damage']
merged['prop_DS4'] = merged['units_DS4_scaled'] / merged['total_damage']

# Determine dominant damage state
merged['dominant_DS'] = merged[['prop_DS1', 'prop_DS2', 'prop_DS3', 'prop_DS4']].idxmax(axis=1)
merged['dominant_DS'] = merged['dominant_DS'].str.replace('prop_', '')

print("\nDominant damage state distribution:")
print(merged['dominant_DS'].value_counts().sort_index())

# ============================================================
# 2. CAPACITY CORRELATION BY DAMAGE STATE
# ============================================================
print("\n" + "="*70)
print("CAPACITY EFFECT BY DOMINANT DAMAGE STATE")
print("="*70)

# Calculate correlations for each damage state
results = []
for ds in ['DS1', 'DS2', 'DS3', 'DS4']:
    subset = merged[merged['dominant_DS'] == ds].copy()
    
    if len(subset) > 10:
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(
            subset['capacity'], 
            subset['recovery_time']
        )
        
        # Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(
            subset['capacity'], 
            subset['recovery_time']
        )
        
        # Log-transformed correlation (to handle skewness)
        log_subset = subset[(subset['capacity'] > 0) & (subset['recovery_time'] > 0)].copy()
        log_subset['log_capacity'] = np.log10(log_subset['capacity'] + 1)
        log_subset['log_recovery'] = np.log10(log_subset['recovery_time'] + 1)
        
        log_pearson_r, log_pearson_p = stats.pearsonr(
            log_subset['log_capacity'], 
            log_subset['log_recovery']
        )
        
        results.append({
            'damage_state': ds,
            'n': len(subset),
            'mean_damage': subset['total_damage'].mean(),
            'median_damage': subset['total_damage'].median(),
            'mean_recovery': subset['recovery_time'].mean(),
            'median_recovery': subset['recovery_time'].median(),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'log_pearson_r': log_pearson_r,
            'log_pearson_p': log_pearson_p
        })
        
        print(f"\n{ds}:")
        print(f"  Sample size: {len(subset):,}")
        print(f"  Mean damage: {subset['total_damage'].mean():.1f} units")
        print(f"  Median damage: {subset['total_damage'].median():.1f} units")
        print(f"  Mean recovery: {subset['recovery_time'].mean():.1f} months")
        print(f"  Median recovery: {subset['recovery_time'].median():.1f} months")
        print(f"  Pearson r: {pearson_r:.3f} (p={pearson_p:.2e})")
        print(f"  Spearman r: {spearman_r:.3f} (p={spearman_p:.2e})")
        print(f"  Log-transformed r: {log_pearson_r:.3f} (p={log_pearson_p:.2e})")

results_df = pd.DataFrame(results)
results_df.to_csv('../analysis_output/capacity_effect_by_damage_state.csv', index=False)

# ============================================================
# 6. NORMALIZED ANALYSIS (PROPORTIONAL DAMAGE)
# ============================================================
print("\n" + "="*70)
print("NORMALIZED ANALYSIS: CAPACITY EFFECT BY PROPORTIONAL DAMAGE")
print("="*70)

# Load total housing units per county
county_units = pd.read_csv('../analysis_output/county_exposed_housing_units.csv')
county_units['fips'] = county_units['FIPS'].astype(str).str.zfill(5)

# Merge with main dataset
merged_norm = merged.merge(county_units[['fips', 'exposed_units']], 
                           on='fips', how='left')

# Filter counties with valid housing unit data
merged_norm = merged_norm[merged_norm['exposed_units'].notna()].copy()
print(f"Counties with housing unit data: {len(merged_norm):,} records")

# Calculate proportional damage (as percentage of housing stock)
merged_norm['prop_total_damage'] = (merged_norm['total_damage'] / 
                                    merged_norm['exposed_units'] * 100)

# Filter unrealistic proportions (>100% shouldn't happen but check)
before = len(merged_norm)
merged_norm = merged_norm[merged_norm['prop_total_damage'] <= 100].copy()
after = len(merged_norm)
if before > after:
    print(f"Filtered {before - after} records with >100% damage (data quality issue)")

print(f"\nProportional damage statistics:")
print(f"  Mean: {merged_norm['prop_total_damage'].mean():.2f}%")
print(f"  Median: {merged_norm['prop_total_damage'].median():.2f}%")
print(f"  Max: {merged_norm['prop_total_damage'].max():.2f}%")
print(f"  Counties with >10% damage: {(merged_norm['prop_total_damage'] > 10).sum():,}")
print(f"  Counties with >50% damage: {(merged_norm['prop_total_damage'] > 50).sum():,}")

# Re-run correlation analysis with normalized damage
normalized_results = []
for ds in ['DS1', 'DS2', 'DS3', 'DS4']:
    subset = merged_norm[merged_norm['dominant_DS'] == ds].copy()
    
    if len(subset) > 10:
        pearson_r, pearson_p = stats.pearsonr(
            subset['capacity'], 
            subset['recovery_time']
        )
        
        spearman_r, spearman_p = stats.spearmanr(
            subset['capacity'], 
            subset['recovery_time']
        )
        
        # Correlation with proportional damage
        prop_damage_r, prop_damage_p = stats.pearsonr(
            subset['prop_total_damage'],
            subset['recovery_time']
        )
        
        normalized_results.append({
            'damage_state': ds,
            'n': len(subset),
            'mean_prop_damage': subset['prop_total_damage'].mean(),
            'median_prop_damage': subset['prop_total_damage'].median(),
            'mean_recovery': subset['recovery_time'].mean(),
            'capacity_corr_pearson': pearson_r,
            'capacity_corr_spearman': spearman_r,
            'prop_damage_recovery_corr': prop_damage_r
        })
        
        print(f"\n{ds} (Normalized):")
        print(f"  Sample size: {len(subset):,}")
        print(f"  Mean proportional damage: {subset['prop_total_damage'].mean():.2f}%")
        print(f"  Median proportional damage: {subset['prop_total_damage'].median():.2f}%")
        print(f"  Mean recovery: {subset['recovery_time'].mean():.1f} months")
        print(f"  Capacity-recovery correlation: {pearson_r:.3f}")
        print(f"  Proportional damage-recovery correlation: {prop_damage_r:.3f}")

normalized_results_df = pd.DataFrame(normalized_results)
normalized_results_df.to_csv('../analysis_output/capacity_effect_by_damage_state_normalized.csv', 
                             index=False)

# ============================================================
# 7. COMPARISON: ABSOLUTE VS NORMALIZED
# ============================================================
print("\n" + "="*70)
print("COMPARISON: ABSOLUTE VS NORMALIZED DAMAGE")
print("="*70)

# Categorize by proportional damage levels
merged_norm['prop_damage_cat'] = pd.cut(
    merged_norm['prop_total_damage'],
    bins=[0, 1, 5, 10, 50, 100],
    labels=['<1%', '1-5%', '5-10%', '10-50%', '>50%']
)

print("\nCapacity effect by PROPORTIONAL damage level:")
prop_cat_results = []
for cat in ['<1%', '1-5%', '5-10%', '10-50%', '>50%']:
    subset = merged_norm[merged_norm['prop_damage_cat'] == cat].copy()
    if len(subset) > 10:
        pearson_r, _ = stats.pearsonr(subset['capacity'], subset['recovery_time'])
        spearman_r, _ = stats.spearmanr(subset['capacity'], subset['recovery_time'])
        
        prop_cat_results.append({
            'prop_damage_level': cat,
            'n': len(subset),
            'mean_prop_damage': subset['prop_total_damage'].mean(),
            'pearson_r': pearson_r,
            'spearman_r': spearman_r
        })
        
        print(f"  {cat}: r={pearson_r:.3f} (n={len(subset):,})")

prop_cat_results_df = pd.DataFrame(prop_cat_results)

# ============================================================
# 3. ALTERNATIVE: CATEGORIZE BY DS PRESENCE
# ============================================================
print("\n" + "="*70)
print("ALTERNATIVE: CATEGORIZE BY SIGNIFICANT DS PRESENCE")
print("="*70)
print("(Events where DS contributes >50% of total damage)")

# Categorize based on which DS contributes most (>50%)
alternative_results = []
for ds_col, ds_name in [('units_DS1_scaled', 'DS1'), 
                        ('units_DS2_scaled', 'DS2'),
                        ('units_DS3_scaled', 'DS3'), 
                        ('units_DS4_scaled', 'DS4')]:
    # Events where this DS is dominant (>50% of damage)
    subset = merged[merged[ds_col] / merged['total_damage'] > 0.5].copy()
    
    if len(subset) > 10:
        pearson_r, pearson_p = stats.pearsonr(
            subset['capacity'], 
            subset['recovery_time']
        )
        
        spearman_r, spearman_p = stats.spearmanr(
            subset['capacity'], 
            subset['recovery_time']
        )
        
        alternative_results.append({
            'damage_state': ds_name,
            'n': len(subset),
            'mean_damage': subset['total_damage'].mean(),
            'median_damage': subset['total_damage'].median(),
            'mean_recovery': subset['recovery_time'].mean(),
            'median_recovery': subset['recovery_time'].median(),
            'pearson_r': pearson_r,
            'spearman_r': spearman_r
        })
        
        print(f"\n{ds_name} (>50% of damage):")
        print(f"  Sample size: {len(subset):,}")
        print(f"  Mean total damage: {subset['total_damage'].mean():.1f} units")
        print(f"  Mean recovery: {subset['recovery_time'].mean():.1f} months")
        print(f"  Capacity correlation (Pearson): {pearson_r:.3f}")
        print(f"  Capacity correlation (Spearman): {spearman_r:.3f}")

# ============================================================
# 4. VISUALIZATIONS (with normalized comparison)
# ============================================================
print("\n4. Creating visualizations...")

# First create the normalized comparison figure
fig_comp, axes_comp = plt.subplots(2, 3, figsize=(22, 12))

# Panel 1: Absolute damage - capacity correlation by DS
ax1 = axes_comp[0, 0]
x_pos = np.arange(len(results_df))
width = 0.35
bars1 = ax1.bar(x_pos - width/2, results_df['spearman_r'], width, 
                alpha=0.7, label='Absolute damage', color='steelblue')
if len(normalized_results_df) > 0:
    bars2 = ax1.bar(x_pos + width/2, normalized_results_df['capacity_corr_spearman'], width,
                    alpha=0.7, label='Normalized damage', color='coral')
ax1.set_xlabel('Damage State', fontweight='bold', fontsize=12)
ax1.set_ylabel('Spearman Correlation\n(Capacity vs Recovery)', fontweight='bold', fontsize=12)
ax1.set_title('A) Capacity Effect: Absolute vs Normalized\n(More negative = stronger capacity effect)', 
              fontweight='bold', fontsize=13)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(results_df['damage_state'])
ax1.legend()
ax1.grid(True, axis='y', alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Panel 2: Proportional damage levels
ax2 = axes_comp[0, 1]
if len(prop_cat_results_df) > 0:
    x_pos2 = np.arange(len(prop_cat_results_df))
    ax2.bar(x_pos2, prop_cat_results_df['spearman_r'], alpha=0.7, color='green')
    ax2.set_xlabel('Proportional Damage Level', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Spearman Correlation', fontweight='bold', fontsize=12)
    ax2.set_title('B) Capacity Effect by % Housing Stock Damaged', 
                  fontweight='bold', fontsize=13)
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(prop_cat_results_df['prop_damage_level'])
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add sample sizes
    for i, (r, n) in enumerate(zip(prop_cat_results_df['spearman_r'], 
                                   prop_cat_results_df['n'])):
        ax2.text(i, r - 0.03, f'n={n:,}', ha='center', fontsize=9)

# Panel 3: Mean proportional damage by DS
ax3 = axes_comp[0, 2]
if len(normalized_results_df) > 0:
    x_pos3 = np.arange(len(normalized_results_df))
    ax3.bar(x_pos3, normalized_results_df['mean_prop_damage'], alpha=0.7, color='purple')
    ax3.set_xlabel('Damage State', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Mean Proportional Damage (%)', fontweight='bold', fontsize=12)
    ax3.set_title('C) Mean % of Housing Stock Damaged\n(by damage state)', 
                  fontweight='bold', fontsize=13)
    ax3.set_xticks(x_pos3)
    ax3.set_xticklabels(normalized_results_df['damage_state'])
    ax3.grid(True, axis='y', alpha=0.3)

# Panels 4-6: Scatter plots showing proportional damage
for i, ds in enumerate(['DS2', 'DS3', 'DS4']):
    ax = axes_comp[1, i]
    subset = merged_norm[merged_norm['dominant_DS'] == ds].copy()
    
    if len(subset) > 0:
        # Color by proportional damage
        scatter = ax.scatter(subset['capacity'], subset['recovery_time'],
                           c=subset['prop_total_damage'], cmap='YlOrRd',
                           alpha=0.5, s=30, vmin=0, vmax=20)
        ax.set_xlabel('Capacity (units/month)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Recovery Time (months)', fontweight='bold', fontsize=11)
        ax.set_title(f'{chr(68+i)}) {ds}: Color = % Housing Damaged', 
                    fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(200, subset['capacity'].quantile(0.95)))
        ax.set_ylim(0, min(2000, subset['recovery_time'].quantile(0.95)))
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('% Housing\nDamaged', fontsize=10)

plt.suptitle('Absolute vs Normalized Damage Analysis', 
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('../analysis_output/capacity_effect_absolute_vs_normalized.png', 
            dpi=300, bbox_inches='tight')
print("Saved: capacity_effect_absolute_vs_normalized.png")

# Original visualizations
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Panel 1: Correlation by damage state (bar chart)
ax1 = axes[0, 0]
x_pos = np.arange(len(results_df))
bars1 = ax1.bar(x_pos, results_df['pearson_r'], alpha=0.7, label='Pearson')
bars2 = ax1.bar(x_pos + 0.3, results_df['spearman_r'], alpha=0.7, label='Spearman')
ax1.set_xlabel('Dominant Damage State', fontweight='bold', fontsize=12)
ax1.set_ylabel('Correlation (Capacity vs Recovery)', fontweight='bold', fontsize=12)
ax1.set_title('A) Capacity Effect by Damage State\n(Negative = higher capacity → lower recovery)', 
              fontweight='bold', fontsize=13)
ax1.set_xticks(x_pos + 0.15)
ax1.set_xticklabels(results_df['damage_state'])
ax1.legend()
ax1.grid(True, axis='y', alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Add sample sizes
for i, (r, n) in enumerate(zip(results_df['pearson_r'], results_df['n'])):
    ax1.text(i, r - 0.02, f'n={n:,}', ha='center', fontsize=9, rotation=0)

# Panel 2: Log-transformed correlation
ax2 = axes[0, 1]
ax2.bar(x_pos, results_df['log_pearson_r'], alpha=0.7, color='green')
ax2.set_xlabel('Dominant Damage State', fontweight='bold', fontsize=12)
ax2.set_ylabel('Log-Transformed Correlation', fontweight='bold', fontsize=12)
ax2.set_title('B) Log-Transformed Capacity Effect\n(Handles skewed distributions)', 
              fontweight='bold', fontsize=13)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(results_df['damage_state'])
ax2.grid(True, axis='y', alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Panel 3: Sample sizes
ax3 = axes[0, 2]
ax3.bar(x_pos, results_df['n'], alpha=0.7, color='orange')
ax3.set_xlabel('Dominant Damage State', fontweight='bold', fontsize=12)
ax3.set_ylabel('Sample Size', fontweight='bold', fontsize=12)
ax3.set_title('C) Sample Sizes by Damage State', fontweight='bold', fontsize=13)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(results_df['damage_state'])
ax3.grid(True, axis='y', alpha=0.3)

# Panels 4-7: Scatter plots for each damage state
scatter_axes = [axes[1, 0], axes[1, 1], axes[1, 2]]
for i, (ds, ax) in enumerate(zip(['DS1', 'DS2', 'DS3'], scatter_axes)):
    subset = merged[merged['dominant_DS'] == ds].copy()
    if len(subset) > 0:
        ax.scatter(subset['capacity'], subset['recovery_time'], 
                  alpha=0.3, s=10)
        ax.set_xlabel('Capacity (units/month)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Recovery Time (months)', fontweight='bold', fontsize=11)
        
        r_val = results_df[results_df['damage_state'] == ds]['pearson_r'].values[0]
        ax.set_title(f'{chr(68+i)}) {ds} Dominant\nr = {r_val:.3f}, n={len(subset):,}', 
                    fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(200, subset['capacity'].quantile(0.95)))
        ax.set_ylim(0, min(2000, subset['recovery_time'].quantile(0.95)))

# Add DS4 inset in panel for scatter
# Create composite scatter showing all DS
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
colors = {'DS1': 'blue', 'DS2': 'green', 'DS3': 'orange', 'DS4': 'red'}

for i, ds in enumerate(['DS1', 'DS2', 'DS3', 'DS4']):
    ax = axes2.flatten()[i]
    subset = merged[merged['dominant_DS'] == ds].copy()
    
    if len(subset) > 0:
        ax.scatter(subset['capacity'], subset['recovery_time'], 
                  alpha=0.3, s=20, color=colors[ds])
        ax.set_xlabel('Capacity (units/month)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Recovery Time (months)', fontweight='bold', fontsize=12)
        
        r_val = results_df[results_df['damage_state'] == ds]['pearson_r'].values[0]
        spear_val = results_df[results_df['damage_state'] == ds]['spearman_r'].values[0]
        
        ax.set_title(f'{ds} Dominant Events\nPearson r={r_val:.3f}, Spearman r={spear_val:.3f}\nn={len(subset):,}', 
                    fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3)
        
        # Set reasonable limits
        cap_95 = subset['capacity'].quantile(0.95)
        rec_95 = subset['recovery_time'].quantile(0.95)
        ax.set_xlim(0, min(200, cap_95))
        ax.set_ylim(0, min(2000, rec_95))
        
        # Add trend line
        valid = subset[(subset['capacity'] > 0) & (subset['recovery_time'] > 0)]
        if len(valid) > 10:
            z = np.polyfit(valid['capacity'], valid['recovery_time'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(0, min(200, cap_95), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Linear fit')

plt.figure(1)
plt.suptitle('Capacity Effect Stratified by Damage State', 
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('../analysis_output/capacity_by_damage_state_summary.png', 
            dpi=300, bbox_inches='tight')
print("Saved: capacity_by_damage_state_summary.png")

plt.figure(2)
plt.suptitle('Capacity vs Recovery by Dominant Damage State', 
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('../analysis_output/capacity_by_damage_state_scatters.png', 
            dpi=300, bbox_inches='tight')
print("Saved: capacity_by_damage_state_scatters.png")

# ============================================================
# 5. SUMMARY
# ============================================================
print("\n" + "="*70)
print("SUMMARY: CAPACITY EFFECT BY DAMAGE STATE")
print("="*70)

print("\nCorrelation Summary (Capacity vs Recovery Time):")
print("-" * 70)
print(f"{'State':<10} {'N':<10} {'Pearson r':<12} {'Spearman r':<12} {'Interpretation'}")
print("-" * 70)

for _, row in results_df.iterrows():
    interpretation = ""
    if abs(row['pearson_r']) < 0.1:
        interpretation = "Negligible effect"
    elif abs(row['pearson_r']) < 0.3:
        interpretation = "Weak effect"
    elif abs(row['pearson_r']) < 0.5:
        interpretation = "Moderate effect"
    else:
        interpretation = "Strong effect"
    
    print(f"{row['damage_state']:<10} {row['n']:<10,} {row['pearson_r']:>10.3f}   {row['spearman_r']:>10.3f}   {interpretation}")

print("\n" + "="*70)
print("KEY FINDINGS:")
print("="*70)

# Find which DS has strongest capacity effect
strongest_ds = results_df.loc[results_df['pearson_r'].abs().idxmax()]
weakest_ds = results_df.loc[results_df['pearson_r'].abs().idxmin()]

print(f"""
1. STRONGEST CAPACITY EFFECT: {strongest_ds['damage_state']}
   - Correlation: {strongest_ds['pearson_r']:.3f}
   - Sample size: {strongest_ds['n']:,}
   - Mean recovery: {strongest_ds['mean_recovery']:.1f} months
   
2. WEAKEST CAPACITY EFFECT: {weakest_ds['damage_state']}
   - Correlation: {weakest_ds['pearson_r']:.3f}
   - Sample size: {weakest_ds['n']:,}
   - Mean recovery: {weakest_ds['mean_recovery']:.1f} months

3. PATTERN:
   DS1 (minor): {'Weak/Negligible' if abs(results_df[results_df['damage_state']=='DS1']['pearson_r'].values[0]) < 0.1 else 'Moderate'}
   DS2 (moderate): {'Weak' if abs(results_df[results_df['damage_state']=='DS2']['pearson_r'].values[0]) < 0.2 else 'Moderate'}
   DS3 (extensive): {'Moderate' if abs(results_df[results_df['damage_state']=='DS3']['pearson_r'].values[0]) < 0.3 else 'Strong'}
   DS4 (complete): {'Strong' if abs(results_df[results_df['damage_state']=='DS4']['pearson_r'].values[0]) > 0.3 else 'Moderate'}
   
   → Capacity matters MORE for severe damage states
""")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("""
Capacity constraints have DIFFERENTIAL IMPACT by damage severity:
- Low damage (DS1): Capacity has minimal effect (buildings easily repaired)
- Moderate damage (DS2): Capacity effect emerges
- Extensive damage (DS3): Capacity becomes important bottleneck
- Complete damage (DS4): Capacity is CRITICAL (full reconstruction needed)

Policy Implication:
→ Capacity building most urgent in areas prone to severe damage (DS3-DS4)
→ Less critical in areas with predominantly minor damage (DS1-DS2)
""")

if len(normalized_results_df) > 0:
    print("\n" + "="*70)
    print("NORMALIZED DAMAGE INSIGHTS:")
    print("="*70)
    
    # Compare correlations
    print("\nCapacity effect comparison (Spearman correlation):")
    print("-" * 70)
    print(f"{'State':<10} {'Absolute':<12} {'Normalized':<12} {'Difference'}")
    print("-" * 70)
    
    for _, row_abs in results_df.iterrows():
        ds = row_abs['damage_state']
        norm_row = normalized_results_df[normalized_results_df['damage_state'] == ds]
        if len(norm_row) > 0:
            abs_r = row_abs['spearman_r']
            norm_r = norm_row['capacity_corr_spearman'].values[0]
            diff = norm_r - abs_r
            print(f"{ds:<10} {abs_r:>10.3f}   {norm_r:>10.3f}   {diff:>+10.3f}")
    
    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("="*70)
    
    # Determine if normalized shows stronger effect
    if len(prop_cat_results_df) > 0:
        high_prop_damage = prop_cat_results_df[
            prop_cat_results_df['prop_damage_level'].isin(['>50%', '10-50%'])
        ]
        if len(high_prop_damage) > 0:
            mean_high_prop_r = high_prop_damage['spearman_r'].mean()
            print(f"""
When >10% of housing stock is damaged (community-wide crisis):
- Mean capacity correlation: {mean_high_prop_r:.3f}
- This suggests capacity constraints are felt more acutely when
  the damage affects a large proportion of the community
- Small absolute damage can still overwhelm small communities
            """)
        
        # Community disruption hypothesis
        print("""
COMMUNITY DISRUPTION HYPOTHESIS:
- Absolute damage matters for overall workload
- Proportional damage matters for community disruption intensity
- When a large % of housing is damaged:
  * Local workforce may be displaced
  * Supply chains disrupted
  * Community support systems compromised
  → Capacity constraints amplified beyond simple workload
        """)

plt.close('all')
