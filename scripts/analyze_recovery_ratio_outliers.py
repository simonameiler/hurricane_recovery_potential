"""
Investigate the 11.53% of cases where recovery ratio deviates from 0.5.

This script analyzes what makes these outlier cases different from the expected
perfect halving relationship.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"
DATA_DIR = BASE_DIR / "data"

# Load the comparison data
print("Loading comparison data...")
comparison_df = pd.read_csv(OUTPUT_DIR / 'construction_capacity_comparison_full.csv')

# Filter to valid cases
valid_data = comparison_df[
    comparison_df['recovery_potential [months]_baseline'].notna() &
    comparison_df['recovery_potential [months]_doubled'].notna() &
    (comparison_df['recovery_potential [months]_baseline'] > 0) &
    (comparison_df['recovery_potential [months]_doubled'] > 0)
].copy()

print(f"Total valid cases: {len(valid_data):,}")

# Calculate theoretical ratio first
valid_data['theoretical_ratio'] = (
    valid_data['reconstruction_capacity_baseline'] / 
    valid_data['reconstruction_capacity_doubled']
)

# Check if capacity was actually doubled
valid_data['capacity_actually_doubled'] = np.isclose(
    valid_data['reconstruction_capacity_doubled'],
    valid_data['reconstruction_capacity_baseline'] * 2,
    rtol=1e-5
)

# Define outliers (ratio != 0.5 with ±0.05 tolerance)
tolerance = 0.05
valid_data['is_outlier'] = (
    (valid_data['recovery_ratio'] < 0.5 - tolerance) |
    (valid_data['recovery_ratio'] > 0.5 + tolerance)
)

outliers = valid_data[valid_data['is_outlier']].copy()
normal_cases = valid_data[~valid_data['is_outlier']].copy()

print(f"\nOutliers: {len(outliers):,} ({len(outliers)/len(valid_data)*100:.2f}%)")
print(f"Normal cases: {len(normal_cases):,} ({len(normal_cases)/len(valid_data)*100:.2f}%)")

print("\n" + "="*80)
print("OUTLIER ANALYSIS")
print("="*80)

# 1. Examine the distribution of outlier ratios
print("\nOutlier Recovery Ratio Statistics:")
print(outliers['recovery_ratio'].describe())

print("\nBreakdown of outlier types:")
higher_than_half = outliers[outliers['recovery_ratio'] > 0.5 + tolerance]
lower_than_half = outliers[outliers['recovery_ratio'] < 0.5 - tolerance]
print(f"  Ratio > 0.55 (less than 50% reduction): {len(higher_than_half):,} ({len(higher_than_half)/len(outliers)*100:.1f}%)")
print(f"  Ratio < 0.45 (more than 50% reduction): {len(lower_than_half):,} ({len(lower_than_half)/len(outliers)*100:.1f}%)")

# 2. Compare characteristics of outliers vs normal cases
print("\n" + "="*80)
print("COMPARING OUTLIERS VS NORMAL CASES")
print("="*80)

characteristics = {
    'Baseline Recovery Time (months)': 'recovery_potential [months]_baseline',
    'Doubled Recovery Time (months)': 'recovery_potential [months]_doubled',
    'Baseline Capacity': 'reconstruction_capacity_baseline',
    'Doubled Capacity': 'reconstruction_capacity_doubled',
}

comparison_stats = []
for label, col in characteristics.items():
    outlier_stats = outliers[col].describe()
    normal_stats = normal_cases[col].describe()
    
    print(f"\n{label}:")
    print(f"  Outliers - Mean: {outlier_stats['mean']:.2f}, Median: {outlier_stats['50%']:.2f}, Std: {outlier_stats['std']:.2f}")
    print(f"  Normal   - Mean: {normal_stats['mean']:.2f}, Median: {normal_stats['50%']:.2f}, Std: {normal_stats['std']:.2f}")
    
    comparison_stats.append({
        'Characteristic': label,
        'Outliers_Mean': outlier_stats['mean'],
        'Outliers_Median': outlier_stats['50%'],
        'Normal_Mean': normal_stats['mean'],
        'Normal_Median': normal_stats['50%']
    })

# 3. Check for specific patterns
print("\n" + "="*80)
print("PATTERN ANALYSIS")
print("="*80)

# Check for very small/large baseline capacities
print("\nBaseline capacity patterns:")
print(f"  Outliers with capacity < 1: {len(outliers[outliers['reconstruction_capacity_baseline'] < 1]):,} ({len(outliers[outliers['reconstruction_capacity_baseline'] < 1])/len(outliers)*100:.1f}%)")
print(f"  Normal with capacity < 1: {len(normal_cases[normal_cases['reconstruction_capacity_baseline'] < 1]):,} ({len(normal_cases[normal_cases['reconstruction_capacity_baseline'] < 1])/len(normal_cases)*100:.1f}%)")

print(f"\n  Outliers with capacity < 0.1: {len(outliers[outliers['reconstruction_capacity_baseline'] < 0.1]):,}")
print(f"  Normal with capacity < 0.1: {len(normal_cases[normal_cases['reconstruction_capacity_baseline'] < 0.1]):,}")

# Check for very long recovery times (might be near-infinity)
print("\nRecovery time patterns:")
very_long = 1000  # months threshold
print(f"  Outliers with baseline recovery > {very_long} months: {len(outliers[outliers['recovery_potential [months]_baseline'] > very_long]):,} ({len(outliers[outliers['recovery_potential [months]_baseline'] > very_long])/len(outliers)*100:.1f}%)")
print(f"  Normal with baseline recovery > {very_long} months: {len(normal_cases[normal_cases['recovery_potential [months]_baseline'] > very_long]):,} ({len(normal_cases[normal_cases['recovery_potential [months]_baseline'] > very_long])/len(normal_cases)*100:.1f}%)")

# Check if baseline capacity is exactly zero
print("\nZero capacity cases:")
print(f"  Outliers with baseline capacity = 0: {len(outliers[outliers['reconstruction_capacity_baseline'] == 0]):,}")
print(f"  Normal with baseline capacity = 0: {len(normal_cases[normal_cases['reconstruction_capacity_baseline'] == 0]):,}")

# 4. Examine extreme outliers
print("\n" + "="*80)
print("EXTREME OUTLIER EXAMPLES")
print("="*80)

# Most extreme higher ratios
print("\nTop 10 cases with HIGHEST ratios (least effective doubling):")
extreme_high = outliers.nlargest(10, 'recovery_ratio')[
    ['event', 'fips', 'recovery_ratio', 'percent_reduction',
     'reconstruction_capacity_baseline', 'reconstruction_capacity_doubled',
     'recovery_potential [months]_baseline', 'recovery_potential [months]_doubled']
]
print(extreme_high.to_string(index=False))

# Most extreme lower ratios (if any)
if len(lower_than_half) > 0:
    print("\nTop 10 cases with LOWEST ratios (most effective doubling):")
    extreme_low = outliers.nsmallest(10, 'recovery_ratio')[
        ['event', 'fips', 'recovery_ratio', 'percent_reduction',
         'reconstruction_capacity_baseline', 'reconstruction_capacity_doubled',
         'recovery_potential [months]_baseline', 'recovery_potential [months]_doubled']
    ]
    print(extreme_low.to_string(index=False))

# 5. Check if it's a rounding/precision issue
print("\n" + "="*80)
print("PRECISION ANALYSIS")
print("="*80)

# For perfect doubling, theoretical ratio should be 0.5
print("\nTheoretical capacity ratio (baseline/doubled):")
print(f"  Outliers - Mean: {outliers['theoretical_ratio'].mean():.6f}, Median: {outliers['theoretical_ratio'].median():.6f}")
print(f"  Normal   - Mean: {normal_cases['theoretical_ratio'].mean():.6f}, Median: {normal_cases['theoretical_ratio'].median():.6f}")

print(f"\nCases where capacity was actually doubled:")
print(f"  Outliers: {outliers['capacity_actually_doubled'].sum():,} / {len(outliers):,} ({outliers['capacity_actually_doubled'].sum()/len(outliers)*100:.1f}%)")
print(f"  Normal:   {normal_cases['capacity_actually_doubled'].sum():,} / {len(normal_cases):,} ({normal_cases['capacity_actually_doubled'].sum()/len(normal_cases)*100:.1f}%)")

# 6. Create visualizations
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Recovery ratio distribution comparison
ax = axes[0, 0]
ax.hist(normal_cases['recovery_ratio'], bins=50, alpha=0.7, label='Normal', color='blue', density=True)
ax.hist(outliers['recovery_ratio'], bins=50, alpha=0.7, label='Outliers', color='red', density=True)
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('Recovery Time Ratio')
ax.set_ylabel('Density')
ax.set_title('Recovery Ratio Distribution:\nOutliers vs Normal')
ax.legend()

# Plot 2: Baseline capacity comparison
ax = axes[0, 1]
ax.hist(np.log10(normal_cases['reconstruction_capacity_baseline'].clip(lower=0.001)), 
        bins=50, alpha=0.7, label='Normal', color='blue', density=True)
ax.hist(np.log10(outliers['reconstruction_capacity_baseline'].clip(lower=0.001)), 
        bins=50, alpha=0.7, label='Outliers', color='red', density=True)
ax.set_xlabel('Log10(Baseline Capacity)')
ax.set_ylabel('Density')
ax.set_title('Baseline Capacity Distribution')
ax.legend()

# Plot 3: Baseline recovery time comparison
ax = axes[0, 2]
ax.hist(np.log10(normal_cases['recovery_potential [months]_baseline'].clip(lower=0.1)), 
        bins=50, alpha=0.7, label='Normal', color='blue', density=True)
ax.hist(np.log10(outliers['recovery_potential [months]_baseline'].clip(lower=0.1)), 
        bins=50, alpha=0.7, label='Outliers', color='red', density=True)
ax.set_xlabel('Log10(Baseline Recovery Time [months])')
ax.set_ylabel('Density')
ax.set_title('Baseline Recovery Time Distribution')
ax.legend()

# Plot 4: Scatter - capacity vs ratio
ax = axes[1, 0]
sample_normal = normal_cases.sample(n=min(5000, len(normal_cases)), random_state=42)
sample_outliers = outliers.sample(n=min(len(outliers), 5000), random_state=42)
ax.scatter(sample_normal['reconstruction_capacity_baseline'], 
          sample_normal['recovery_ratio'], 
          alpha=0.3, s=10, c='blue', label='Normal')
ax.scatter(sample_outliers['reconstruction_capacity_baseline'], 
          sample_outliers['recovery_ratio'], 
          alpha=0.6, s=20, c='red', label='Outliers')
ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('Baseline Capacity')
ax.set_ylabel('Recovery Ratio')
ax.set_title('Recovery Ratio vs Baseline Capacity')
ax.set_xscale('log')
ax.set_xlim(0.01, sample_normal['reconstruction_capacity_baseline'].max())
ax.legend()

# Plot 5: Scatter - recovery time vs ratio
ax = axes[1, 1]
ax.scatter(sample_normal['recovery_potential [months]_baseline'], 
          sample_normal['recovery_ratio'], 
          alpha=0.3, s=10, c='blue', label='Normal')
ax.scatter(sample_outliers['recovery_potential [months]_baseline'], 
          sample_outliers['recovery_ratio'], 
          alpha=0.6, s=20, c='red', label='Outliers')
ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('Baseline Recovery Time (months)')
ax.set_ylabel('Recovery Ratio')
ax.set_title('Recovery Ratio vs Baseline Recovery Time')
ax.set_xscale('log')
ax.legend()

# Plot 6: Theoretical vs actual ratio
ax = axes[1, 2]
ax.scatter(sample_normal['theoretical_ratio'], 
          sample_normal['recovery_ratio'], 
          alpha=0.3, s=10, c='blue', label='Normal')
ax.scatter(sample_outliers['theoretical_ratio'], 
          sample_outliers['recovery_ratio'], 
          alpha=0.6, s=20, c='red', label='Outliers')
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect match')
ax.set_xlabel('Theoretical Ratio (Capacity Baseline/Doubled)')
ax.set_ylabel('Actual Recovery Ratio')
ax.set_title('Theoretical vs Actual Recovery Ratio')
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'outlier_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'outlier_analysis.png'}")
plt.close()

# 7. Save detailed outlier data
print("\nSaving detailed outlier data...")
outlier_export = outliers[[
    'event', 'fips', 'recovery_ratio', 'percent_reduction',
    'reconstruction_capacity_baseline', 'reconstruction_capacity_doubled',
    'recovery_potential [months]_baseline', 'recovery_potential [months]_doubled',
    'theoretical_ratio', 'capacity_actually_doubled'
]].sort_values('recovery_ratio', ascending=False)

outlier_export.to_csv(OUTPUT_DIR / 'recovery_ratio_outliers_detailed.csv', index=False)
print(f"Saved: {OUTPUT_DIR / 'recovery_ratio_outliers_detailed.csv'}")

# 8. Summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

summary = {
    'Total Cases': len(valid_data),
    'Outliers': len(outliers),
    'Outlier Percentage': f"{len(outliers)/len(valid_data)*100:.2f}%",
    'Outliers with Ratio > 0.55': len(higher_than_half),
    'Outliers with Ratio < 0.45': len(lower_than_half),
    'Outliers with Very Low Capacity (<0.1)': len(outliers[outliers['reconstruction_capacity_baseline'] < 0.1]),
    'Outliers with Very Long Recovery (>1000 mo)': len(outliers[outliers['recovery_potential [months]_baseline'] > 1000]),
    'Outliers where Capacity Not Actually Doubled': len(outliers[~outliers['capacity_actually_doubled']]),
}

for key, value in summary.items():
    print(f"{key}: {value}")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
