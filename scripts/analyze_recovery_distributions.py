"""
Analyze distributions of recovery times, impacts, and capacity

Computes per-county distribution statistics (mean, percentiles, skewness,
coefficient of variation) of recovery times and weighted damage across the
probabilistic event set, and writes analysis_output/county_distribution_metrics.csv
(consumed by notebooks/probabilistic_analysis.ipynb, skewness map).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

BASE_DIR = Path(__file__).parent.parent

# Load recovery data
print("Loading recovery data...")
recovery_csv = BASE_DIR / 'data' / 'recovery' / 'recovery_potential.csv'

recovery_df_raw = pd.read_csv(recovery_csv, dtype={'fips': str})
recovery_df_raw['fips'] = recovery_df_raw['fips'].astype(str).str.zfill(5)
recovery_df_raw['recovery_potential_months'] = pd.to_numeric(
    recovery_df_raw['recovery_potential_months'], errors='coerce')

recovery_df = pd.DataFrame({
    'event': recovery_df_raw['event_name'].astype(str),
    'fips': recovery_df_raw['fips'],
    # NaN recovery = zero-capacity county (undefined recovery); dropped below
    'recovery_time': recovery_df_raw['recovery_potential_months'],
    'capacity': recovery_df_raw['reconstruction_capacity'].astype(float),
})

# Load impact data
print("Loading impact data...")
impacts_dir = BASE_DIR / 'data' / 'impact' / 'per_event'

all_impacts = []
for csv_file in sorted(impacts_dir.glob('*.csv')):
    df = pd.read_csv(csv_file)
    all_impacts.append(df)

impacts_df = pd.concat(all_impacts, ignore_index=True)
impacts_df = impacts_df.rename(columns={'event_name': 'event'})
impacts_df['fips'] = impacts_df['fips'].astype(str).str.zfill(5)
impacts_df['event'] = impacts_df['event'].astype(str)

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

# Drop undefined recovery (zero-capacity counties) and any infinities
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

# Save detailed metrics
out_csv = BASE_DIR / 'analysis_output' / 'county_distribution_metrics.csv'
out_csv.parent.mkdir(exist_ok=True)
county_dist.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

# Identify interesting county patterns
print("\n" + "="*60)
print("NOTABLE COUNTY PATTERNS")
print("="*60)

print("\nMost variable counties (high CV):")
print(county_dist.nlargest(5, 'rt_cv')[['fips', 'rt_cv', 'rt_mean', 'rt_std', 'capacity']])

print("\nMost skewed counties (extreme tail risk):")
print(county_dist.nlargest(5, 'rt_skew')[['fips', 'rt_skew', 'rt_median', 'rt_p99', 'frac_rt_extreme']])
