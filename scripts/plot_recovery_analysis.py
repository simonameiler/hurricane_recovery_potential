"""
Interactive plotting script for hurricane recovery analysis.
Run cells individually to create specific plots.
"""

# %% Imports and run main analysis
import matplotlib.pyplot as plt
import numpy as np
from TC_NA_recovery_analysis_reorganized import main

print("Running main analysis...")
results = main()

# Extract results
per_event_analysis_median = results['per_event_analysis_median']
bin_centers = results['bin_centers']
pct_damage_list = results['pct_damage_list']

print("\nAnalysis complete! Run the cells below to create plots.")

# %% Plot 1: Threshold Analysis (Goldilocks Zone)
plt.figure(figsize=(10, 6))
plt.semilogx(bin_centers, pct_damage_list, 'o-', linewidth=2, markersize=10, color='navy')
plt.axhline(50, color='red', linestyle='--', alpha=0.7, linewidth=2, label='50% threshold')
plt.axvspan(10, 50, alpha=0.2, color='green', label='Goldilocks Zone (10-50 permits/month)')
plt.xlabel('Construction Capacity (permits/month)', fontsize=13)
plt.ylabel('% Damage-Driven Counties', fontsize=13)
plt.title('How Damage Dominance Changes with Construction Capacity', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3, which='both')
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# %% Plot 2: Recovery vs Damage (colored by driver)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

damage_driven = per_event_analysis_median[per_event_analysis_median['dominant_driver'] == 'Damage']
capacity_driven = per_event_analysis_median[per_event_analysis_median['dominant_driver'] == 'Capacity']

ax.scatter(damage_driven['log_damage'], damage_driven['log_recovery'], 
           c='crimson', alpha=0.6, s=60, edgecolors='darkred', linewidth=0.5,
           label=f'Damage-driven (n={len(damage_driven)})')
ax.scatter(capacity_driven['log_damage'], capacity_driven['log_recovery'], 
           c='dodgerblue', alpha=0.6, s=60, edgecolors='navy', linewidth=0.5,
           label=f'Capacity-driven (n={len(capacity_driven)})')

ax.set_xlabel('Log(Expected Annual Damage) [USD]', fontsize=13)
ax.set_ylabel('Log(Expected Annual Recovery Time) [months]', fontsize=13)
ax.set_title('Recovery Time vs Damage by Dominant Driver', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% Plot 3: Recovery vs Capacity (colored by driver)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.scatter(damage_driven['log_capacity'], damage_driven['log_recovery'], 
           c='crimson', alpha=0.6, s=60, edgecolors='darkred', linewidth=0.5,
           label=f'Damage-driven (n={len(damage_driven)})')
ax.scatter(capacity_driven['log_capacity'], capacity_driven['log_recovery'], 
           c='dodgerblue', alpha=0.6, s=60, edgecolors='navy', linewidth=0.5,
           label=f'Capacity-driven (n={len(capacity_driven)})')

ax.set_xlabel('Log(Construction Capacity) [permits/month]', fontsize=13)
ax.set_ylabel('Log(Expected Annual Recovery Time) [months]', fontsize=13)
ax.set_title('Recovery Time vs Capacity by Dominant Driver', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% Plot 4: Damage vs Capacity (colored by driver)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.scatter(damage_driven['log_capacity'], damage_driven['log_damage'], 
           c='crimson', alpha=0.6, s=60, edgecolors='darkred', linewidth=0.5,
           label=f'Damage-driven (n={len(damage_driven)})')
ax.scatter(capacity_driven['log_capacity'], capacity_driven['log_damage'], 
           c='dodgerblue', alpha=0.6, s=60, edgecolors='navy', linewidth=0.5,
           label=f'Capacity-driven (n={len(capacity_driven)})')

ax.set_xlabel('Log(Construction Capacity) [permits/month]', fontsize=13)
ax.set_ylabel('Log(Expected Annual Damage) [USD]', fontsize=13)
ax.set_title('Damage vs Capacity by Dominant Driver', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% Summary Statistics
from scipy.stats import pearsonr

print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)

print(f"\nTotal counties analyzed: {len(per_event_analysis_median)}")
print(f"\nDriver Classification:")
driver_counts = per_event_analysis_median['dominant_driver'].value_counts()
for driver, count in driver_counts.items():
    pct = 100 * count / len(per_event_analysis_median)
    print(f"  {driver}-driven: {count:3d} counties ({pct:5.1f}%)")

# Overall correlations
corr_damage, p_damage = pearsonr(per_event_analysis_median['log_damage'], 
                                  per_event_analysis_median['log_recovery'])
corr_capacity, p_capacity = pearsonr(per_event_analysis_median['log_capacity'], 
                                      per_event_analysis_median['log_recovery'])

print(f"\nOverall Correlations with Recovery Time:")
print(f"  Damage:   r = {corr_damage:6.3f} (p = {p_damage:.2e})")
print(f"  Capacity: r = {corr_capacity:6.3f} (p = {p_capacity:.2e})")

# Stratified correlations
print(f"\nDamage-Driven Counties (n={len(damage_driven)}):")
corr_dd_damage, _ = pearsonr(damage_driven['log_damage'], damage_driven['log_recovery'])
corr_dd_capacity, _ = pearsonr(damage_driven['log_capacity'], damage_driven['log_recovery'])
print(f"  Damage correlation:   r = {corr_dd_damage:6.3f}")
print(f"  Capacity correlation: r = {corr_dd_capacity:6.3f}")

print(f"\nCapacity-Driven Counties (n={len(capacity_driven)}):")
corr_cd_damage, _ = pearsonr(capacity_driven['log_damage'], capacity_driven['log_recovery'])
corr_cd_capacity, _ = pearsonr(capacity_driven['log_capacity'], capacity_driven['log_recovery'])
print(f"  Damage correlation:   r = {corr_cd_damage:6.3f}")
print(f"  Capacity correlation: r = {corr_cd_capacity:6.3f}")

print("\n" + "="*80)

# %% Capacity Distribution by Driver
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(damage_driven['construction_capacity'], bins=30, color='crimson', 
             alpha=0.7, edgecolor='darkred')
axes[0].set_xlabel('Construction Capacity (permits/month)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title(f'Damage-Driven Counties (n={len(damage_driven)})', fontsize=13)
axes[0].axvline(damage_driven['construction_capacity'].median(), 
                color='black', linestyle='--', linewidth=2, 
                label=f'Median = {damage_driven["construction_capacity"].median():.1f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(capacity_driven['construction_capacity'], bins=30, color='dodgerblue', 
             alpha=0.7, edgecolor='navy')
axes[1].set_xlabel('Construction Capacity (permits/month)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title(f'Capacity-Driven Counties (n={len(capacity_driven)})', fontsize=13)
axes[1].axvline(capacity_driven['construction_capacity'].median(), 
                color='black', linestyle='--', linewidth=2,
                label=f'Median = {capacity_driven["construction_capacity"].median():.1f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
