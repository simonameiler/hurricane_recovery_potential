"""
Extract total housing units per county from CLIMADA exposure data

This provides internally consistent unit counts that only include 
housing exposed to hurricane hazard (vs Census total housing).
"""

import os
from pathlib import Path
import pandas as pd
from climada.entity.exposures import Exposures

# Define paths
BASE_DIR = Path(__file__).parent.parent
EXP_DIR = BASE_DIR / "data" / "exposure" / "states"
OUTPUT_DIR = BASE_DIR / "analysis_output"

# Get all HDF5 exposure files
exposure_files = list(EXP_DIR.glob("*_exposure.hdf5"))

print("=" * 100)
print("EXTRACTING HOUSING UNITS FROM CLIMADA EXPOSURE DATA")
print("=" * 100)
print(f"\nFound {len(exposure_files)} exposure files")

# Load all exposures and combine
all_exposures = []

for exp_file in exposure_files:
    state_name = exp_file.stem.replace('_exposure', '').upper()
    print(f"Loading {state_name}...", end=" ")
    
    try:
        exp = Exposures.from_hdf5(exp_file)
        exp_df = exp.gdf.copy()
        exp_df['State'] = state_name
        all_exposures.append(exp_df)
        print(f"✓ {len(exp_df):,} buildings")
    except Exception as e:
        print(f"✗ Error: {e}")

# Combine all exposures
combined_exp = pd.concat(all_exposures, ignore_index=True)
print(f"\nTotal buildings loaded: {len(combined_exp):,}")

# Create FIPS code (GEOID format: stcode + ccode, both zero-padded)
combined_exp['FIPS'] = (combined_exp['stcode'].astype(str).str.zfill(2) + 
                        combined_exp['ccode'].astype(str).str.zfill(3))

print("\nColumns available:")
print(combined_exp.columns.tolist())

# Aggregate units by county
county_units = combined_exp.groupby('FIPS').agg({
    'NumberOfUnits': 'sum',
    'County': 'first',
    'stcode': 'first'
}).reset_index()

county_units.columns = ['FIPS', 'exposed_units', 'county_name', 'state_fips']

print("\n" + "=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)
print(f"Total counties: {len(county_units)}")
print(f"Total exposed units: {county_units['exposed_units'].sum():,.0f}")
print(f"Mean units per county: {county_units['exposed_units'].mean():,.0f}")
print(f"Median units per county: {county_units['exposed_units'].median():,.0f}")

print("\nTop 10 counties by exposed units:")
top10 = county_units.nlargest(10, 'exposed_units')
for _, row in top10.iterrows():
    print(f"  {row['county_name']}: {row['exposed_units']:,.0f} units")

# Save to CSV
output_file = OUTPUT_DIR / 'county_exposed_housing_units.csv'
county_units.to_csv(output_file, index=False)
print(f"\n✓ Saved to: {output_file}")

# Compare with Census data if available
census_file = OUTPUT_DIR / 'county_total_housing_units.csv'
if census_file.exists():
    print("\n" + "=" * 100)
    print("COMPARISON: EXPOSURE vs CENSUS HOUSING UNITS")
    print("=" * 100)
    
    census = pd.read_csv(census_file)
    census['FIPS'] = census['FIPS'].astype(str).str.zfill(5)
    
    comparison = county_units.merge(census[['FIPS', 'total_housing_units']], 
                                     on='FIPS', how='inner')
    comparison['exposure_ratio'] = comparison['exposed_units'] / comparison['total_housing_units']
    
    print(f"\nCounties in both datasets: {len(comparison)}")
    print(f"Mean exposure ratio: {comparison['exposure_ratio'].mean():.1%}")
    print(f"Median exposure ratio: {comparison['exposure_ratio'].median():.1%}")
    
    print("\nCounties with most difference (lowest exposure ratio):")
    low_ratio = comparison.nsmallest(5, 'exposure_ratio')
    for _, row in low_ratio.iterrows():
        print(f"  {row['county_name']}: {row['exposed_units']:,.0f} exposed / {row['total_housing_units']:,.0f} total = {row['exposure_ratio']:.1%}")
    
    print("\nCounties with highest exposure ratio:")
    high_ratio = comparison.nlargest(5, 'exposure_ratio')
    for _, row in high_ratio.iterrows():
        print(f"  {row['county_name']}: {row['exposed_units']:,.0f} exposed / {row['total_housing_units']:,.0f} total = {row['exposure_ratio']:.1%}")

print("\n" + "=" * 100)
print("COMPLETE")
print("=" * 100)
