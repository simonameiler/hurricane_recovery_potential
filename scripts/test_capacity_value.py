"""
Test the fundamental hypothesis: Does capacity matter beyond damage?

Decompose recovery time variance to understand:
1. How much is explained by damage alone?
2. How much additional variance does capacity explain?
3. When does capacity matter most?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats

print("="*60)
print("TESTING: DOES CAPACITY MATTER BEYOND DAMAGE?")
print("="*60)

# Load data
print("\nLoading data...")
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

# Load impact data for damage
impacts_dir = Path('../impacts_out/by_event/scaled')
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

# Merge
merged = recovery_df.merge(impacts_df[['event', 'fips', 'weighted_damage']], 
                           on=['event', 'fips'], how='left')

# Filter valid data
merged = merged[
    (merged['recovery_time'] != np.inf) & 
    (merged['recovery_time'].notna()) &
    (merged['capacity'] > 0) &
    (merged['weighted_damage'].notna())
].copy()

# Add log transforms for better analysis
merged['log_recovery_time'] = np.log10(merged['recovery_time'] + 0.01)
merged['log_damage'] = np.log10(merged['weighted_damage'] + 0.01)
merged['log_capacity'] = np.log10(merged['capacity'])

print(f"Valid records: {len(merged):,}")

# ============================================================
# 1. VARIANCE DECOMPOSITION
# ============================================================
print("\n" + "="*60)
print("1. VARIANCE DECOMPOSITION")
print("="*60)

# Model 1: Recovery ~ Damage only
X1 = merged[['log_damage']].values
y = merged['log_recovery_time'].values

model1 = LinearRegression()
model1.fit(X1, y)
y_pred1 = model1.predict(X1)
r2_damage_only = r2_score(y, y_pred1)

print(f"\nModel 1: Recovery ~ Damage")
print(f"  R² = {r2_damage_only:.3f}")
print(f"  Variance explained by DAMAGE ALONE: {r2_damage_only*100:.1f}%")

# Model 2: Recovery ~ Capacity only
X2 = merged[['log_capacity']].values
model2 = LinearRegression()
model2.fit(X2, y)
y_pred2 = model2.predict(X2)
r2_capacity_only = r2_score(y, y_pred2)

print(f"\nModel 2: Recovery ~ Capacity")
print(f"  R² = {r2_capacity_only:.3f}")
print(f"  Variance explained by CAPACITY ALONE: {r2_capacity_only*100:.1f}%")

# Model 3: Recovery ~ Damage + Capacity
X3 = merged[['log_damage', 'log_capacity']].values
model3 = LinearRegression()
model3.fit(X3, y)
y_pred3 = model3.predict(X3)
r2_both = r2_score(y, y_pred3)

print(f"\nModel 3: Recovery ~ Damage + Capacity")
print(f"  R² = {r2_both:.3f}")
print(f"  Variance explained by BOTH: {r2_both*100:.1f}%")

# Incremental variance explained
incremental_r2 = r2_both - r2_damage_only
print(f"\n** INCREMENTAL VARIANCE from adding Capacity: {incremental_r2*100:.1f}%")
print(f"** Relative importance: Damage adds {r2_damage_only*100:.1f}%, Capacity adds {incremental_r2*100:.1f}%")

if incremental_r2 < 0.05:
    print("\n⚠️  FINDING: Capacity adds < 5% explanatory power!")
    print("    Recovery is mostly determined by DAMAGE, not capacity constraints.")
elif incremental_r2 < 0.15:
    print("\n→  FINDING: Capacity adds modest (~5-15%) explanatory power")
    print("    Both damage and capacity matter, but damage is primary driver.")
else:
    print("\n✓  FINDING: Capacity adds substantial (>15%) explanatory power")
    print("    Both dimensions are important for understanding recovery.")

# ============================================================
# 2. CONDITIONAL ANALYSIS: Does capacity matter for similar damage levels?
# ============================================================
print("\n" + "="*60)
print("2. CAPACITY EFFECT AT DIFFERENT DAMAGE LEVELS")
print("="*60)

# Bin by damage level
merged['damage_bin'] = pd.qcut(merged['weighted_damage'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')

# For each damage bin, test capacity effect
print("\nCorrelation between Capacity and Recovery Time (by damage level):")
print("(Negative correlation = higher capacity → lower recovery time)\n")

for bin_name in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
    subset = merged[merged['damage_bin'] == bin_name]
    if len(subset) > 10:
        corr = subset['capacity'].corr(subset['recovery_time'])
        print(f"{bin_name:12s} damage: r = {corr:+.3f} (n={len(subset):,})")

# ============================================================
# 3. MATCHED PAIRS: Same damage, different capacity
# ============================================================
print("\n" + "="*60)
print("3. MATCHED COUNTY COMPARISON")
print("="*60)
print("Find counties with similar damage exposure but different capacity\n")

# Compute county-level damage exposure (mean weighted damage)
county_damage = merged.groupby('fips').agg({
    'weighted_damage': 'mean',
    'capacity': 'first',
    'recovery_time': 'mean'
}).reset_index()

county_damage.columns = ['fips', 'mean_damage', 'capacity', 'mean_recovery']

# Find matched pairs: similar damage (within 20%), very different capacity (>3x difference)
matched_pairs = []
counties = county_damage.values

for i in range(len(counties)):
    for j in range(i+1, len(counties)):
        fips1, dam1, cap1, rec1 = counties[i]
        fips2, dam2, cap2, rec2 = counties[j]
        
        if dam1 == 0 or dam2 == 0:
            continue
            
        damage_ratio = max(dam1, dam2) / min(dam1, dam2)
        capacity_ratio = max(cap1, cap2) / min(cap1, cap2)
        
        # Similar damage but very different capacity
        if damage_ratio < 1.2 and capacity_ratio > 3:
            matched_pairs.append({
                'fips1': fips1, 'fips2': fips2,
                'damage1': dam1, 'damage2': dam2,
                'capacity1': cap1, 'capacity2': cap2,
                'recovery1': rec1, 'recovery2': rec2,
                'damage_ratio': damage_ratio,
                'capacity_ratio': capacity_ratio,
                'recovery_ratio': max(rec1, rec2) / min(rec1, rec2) if min(rec1, rec2) > 0 else np.nan
            })

matched_df = pd.DataFrame(matched_pairs)

if len(matched_df) > 0:
    print(f"Found {len(matched_df)} county pairs with similar damage but different capacity\n")
    print("Top 5 pairs (sorted by capacity difference):")
    print(matched_df.nlargest(5, 'capacity_ratio')[['fips1', 'fips2', 'capacity_ratio', 'recovery_ratio']])
    
    # Test: Does higher capacity → lower recovery?
    matched_df['high_cap_has_lower_recovery'] = (
        ((matched_df['capacity1'] > matched_df['capacity2']) & (matched_df['recovery1'] < matched_df['recovery2'])) |
        ((matched_df['capacity2'] > matched_df['capacity1']) & (matched_df['recovery2'] < matched_df['recovery1']))
    )
    
    pct_consistent = matched_df['high_cap_has_lower_recovery'].sum() / len(matched_df) * 100
    print(f"\n** {pct_consistent:.1f}% of pairs: higher capacity → lower recovery time")
    
    if pct_consistent < 60:
        print("   ⚠️  WEAK PATTERN: Capacity doesn't reliably reduce recovery for similar damage")
    elif pct_consistent < 80:
        print("   →  MODERATE PATTERN: Capacity helps but other factors matter too")
    else:
        print("   ✓  STRONG PATTERN: Capacity consistently reduces recovery for similar damage")
else:
    print("No matched pairs found (damage levels too heterogeneous)")

# ============================================================
# 4. VISUALIZATIONS
# ============================================================
print("\n" + "="*60)
print("Creating visualizations...")
print("="*60)

fig = plt.figure(figsize=(18, 12))

# Plot 1: Damage vs Recovery (main relationship)
ax1 = plt.subplot(2, 3, 1)
scatter = ax1.scatter(merged['log_damage'], merged['log_recovery_time'], 
                     c=merged['log_capacity'], cmap='viridis', alpha=0.3, s=10)
ax1.plot(merged['log_damage'], y_pred1, 'r-', linewidth=2, label=f'Damage only (R²={r2_damage_only:.2f})')
ax1.set_xlabel('Log10(Weighted Damage)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Log10(Recovery Time, months)', fontsize=11, fontweight='bold')
ax1.set_title(f'A) Damage Explains {r2_damage_only*100:.0f}% of Recovery Variance', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Log10(Capacity)')

# Plot 2: Capacity vs Recovery (secondary relationship)
ax2 = plt.subplot(2, 3, 2)
scatter = ax2.scatter(merged['log_capacity'], merged['log_recovery_time'], 
                     c=merged['log_damage'], cmap='Reds', alpha=0.3, s=10)
ax2.plot(merged['log_capacity'], y_pred2, 'b-', linewidth=2, label=f'Capacity only (R²={r2_capacity_only:.2f})')
ax2.set_xlabel('Log10(Capacity, units/month)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Log10(Recovery Time, months)', fontsize=11, fontweight='bold')
ax2.set_title(f'B) Capacity Explains {r2_capacity_only*100:.0f}% of Recovery Variance', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax2, label='Log10(Damage)')

# Plot 3: Residuals from damage-only model vs capacity
ax3 = plt.subplot(2, 3, 3)
residuals = y - y_pred1
ax3.scatter(merged['log_capacity'], residuals, alpha=0.3, s=10, c='green')
ax3.axhline(0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Log10(Capacity, units/month)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Residuals from Damage-only Model', fontsize=11, fontweight='bold')
ax3.set_title(f'C) Does Capacity Explain Remaining Variance?\n(After accounting for damage)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Correlation of residuals with capacity
resid_capacity_corr = np.corrcoef(merged['log_capacity'], residuals)[0, 1]
ax3.text(0.05, 0.95, f'Correlation: {resid_capacity_corr:.3f}', 
         transform=ax3.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 4: Variance explained breakdown
ax4 = plt.subplot(2, 3, 4)
components = ['Damage\nalone', 'Capacity\nalone', 'Both\ntogether', 'Unexplained']
values = [r2_damage_only*100, r2_capacity_only*100, r2_both*100, (1-r2_both)*100]
colors = ['red', 'blue', 'purple', 'gray']
bars = ax4.bar(components, values, color=colors, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Variance Explained (%)', fontsize=11, fontweight='bold')
ax4.set_title('D) Variance Decomposition', fontsize=12, fontweight='bold')
ax4.grid(True, axis='y', alpha=0.3)
ax4.set_ylim(0, 100)

# Add value labels
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

# Plot 5: Capacity effect by damage quintile
ax5 = plt.subplot(2, 3, 5)
for i, bin_name in enumerate(['Very Low', 'Low', 'Medium', 'High', 'Very High']):
    subset = merged[merged['damage_bin'] == bin_name]
    if len(subset) > 10:
        ax5.scatter(subset['capacity'], subset['recovery_time'], 
                   alpha=0.4, s=20, label=f'{bin_name} damage')

ax5.set_xlabel('Capacity (units/month)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Recovery Time (months)', fontsize=11, fontweight='bold')
ax5.set_title('E) Capacity Effect Across Damage Levels', fontsize=12, fontweight='bold')
ax5.set_xlim(0, 200)
ax5.set_ylim(0, 1000)
ax5.legend(fontsize=8, loc='upper right')
ax5.grid(True, alpha=0.3)

# Plot 6: Distribution of prediction errors
ax6 = plt.subplot(2, 3, 6)
errors_damage_only = y - y_pred1
errors_both = y - y_pred3
ax6.hist(errors_damage_only, bins=50, alpha=0.5, label='Damage only', color='red', edgecolor='black')
ax6.hist(errors_both, bins=50, alpha=0.5, label='Damage + Capacity', color='purple', edgecolor='black')
ax6.set_xlabel('Prediction Error (log months)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax6.set_title('F) Prediction Error Distribution', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.suptitle('Decomposing Recovery Time: Damage vs Capacity', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('../analysis_output/damage_vs_capacity_decomposition.png', dpi=300, bbox_inches='tight')
print("Saved: damage_vs_capacity_decomposition.png")

# ============================================================
# FINAL VERDICT
# ============================================================
print("\n" + "="*60)
print("FINAL VERDICT")
print("="*60)

if r2_damage_only > 0.80:
    print("\n🔴 DAMAGE DOMINATES (R² > 80%)")
    print("   Recovery is primarily determined by damage exposure.")
    print("   Capacity plays a minor role.")
    print("\n   IMPLICATION: Focus on DAMAGE REDUCTION (mitigation, exposure management)")
    print("                Capacity building is secondary priority.")
    
elif r2_damage_only > 0.60 and incremental_r2 < 0.10:
    print("\n🟠 DAMAGE IS PRIMARY, CAPACITY IS SECONDARY")
    print(f"   Damage explains {r2_damage_only*100:.0f}%, capacity adds {incremental_r2*100:.0f}%")
    print("\n   IMPLICATION: Prioritize damage reduction, but capacity matters at the margins")
    print("                Build capacity in high-damage areas to reduce worst outcomes.")
    
elif r2_damage_only > 0.40 and incremental_r2 > 0.10:
    print("\n🟡 BOTH DAMAGE AND CAPACITY MATTER")
    print(f"   Damage explains {r2_damage_only*100:.0f}%, capacity adds {incremental_r2*100:.0f}%")
    print("\n   IMPLICATION: Need DUAL STRATEGY:")
    print("                1) Reduce damage exposure (mitigation)")
    print("                2) Build recovery capacity (construction workforce)")
    
else:
    print("\n🟢 COMPLEX RELATIONSHIP")
    print(f"   Neither damage alone ({r2_damage_only*100:.0f}%) nor both together ({r2_both*100:.0f}%) explain most variance")
    print("\n   IMPLICATION: Other factors beyond damage and capacity are crucial")
    print("                Consider social vulnerability, pre-event planning, etc.")

print("\n" + "="*60)
