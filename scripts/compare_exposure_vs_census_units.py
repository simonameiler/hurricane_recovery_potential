import pandas as pd

# Load exposure units
exp_units = pd.read_csv('analysis_output/county_exposed_housing_units.csv')
exp_units['FIPS'] = exp_units['FIPS'].astype(str).str.zfill(5)

# Load normalized data which has Census units
norm_df = pd.read_csv('analysis_output/event_county_quadrants_fully_normalized.csv')
norm_df['fips'] = norm_df['fips'].astype(str).str.zfill(5)

# Get unique counties with Census totals
census_units = norm_df.groupby('fips')['total_housing_units'].first().reset_index()
census_units.columns = ['FIPS', 'census_units']

# Merge
comparison = exp_units.merge(census_units, on='FIPS', how='inner')
comparison['ratio'] = comparison['exposed_units'] / comparison['census_units']

print(f'Counties in both: {len(comparison)}')
print(f'Mean exposure/census ratio: {comparison["ratio"].mean():.3f}')
print(f'Median ratio: {comparison["ratio"].median():.3f}')
print(f'Min ratio: {comparison["ratio"].min():.3f}')
print(f'Max ratio: {comparison["ratio"].max():.3f}')
print()

print('Counties with VERY different counts (ratio < 0.5):')
low = comparison[comparison['ratio'] < 0.5].sort_values('ratio')
for _, row in low.head(10).iterrows():
    print(f'  {row["county_name"]}: {row["exposed_units"]:,.0f} exposed / {row["census_units"]:,.0f} census = {row["ratio"]:.2f}')
print()

print('Counties with ratio > 1 (more exposed than Census):')
high = comparison[comparison['ratio'] > 1.0].sort_values('ratio', ascending=False)
for _, row in high.head(10).iterrows():
    print(f'  {row["county_name"]}: {row["exposed_units"]:,.0f} exposed / {row["census_units"]:,.0f} census = {row["ratio"]:.2f}')
