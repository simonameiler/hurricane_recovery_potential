#!/usr/bin/env python3
"""
Analyze where wind-only approach misses surge/rain impacts.

Compares Gori et al. hazard components to identify events/counties where:
- Wind damage is minimal (below threshold)
- But surge + rain damage is non-negligible

This helps quantify the limitation of the multiplicative scaling approach.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def load_gori_hazards(data_dir: Path):
    """Load wind, surge, and rain hazard matrices from Gori et al. .mat files."""
    wind_data = loadmat(data_dir / "maxwindmat_ncep_reanal.mat")
    surge_data = loadmat(data_dir / "maxelev_coastcounty_ncep_reanal.mat")
    rain_data = loadmat(data_dir / "ptot_rain_county_ncep_reanal.mat")
    
    # Extract intensity matrices
    # Wind: (5018 events × 3220 counties) - transpose to match others
    # Surge: (3220 counties × 5018 events) - use scounty (not scounty_mhhw)
    # Rain: (3220 counties × 5018 events)
    wind = wind_data['maxwindmat'].T  # Transpose to (counties × events)
    surge = surge_data['scounty']     # Already (counties × events)
    rain = rain_data['ptot_mat']      # Already (counties × events)
    
    print(f"Wind shape: {wind.shape} (counties × events)")
    print(f"Surge shape: {surge.shape} (counties × events)")
    print(f"Rain shape: {rain.shape} (counties × events)")
    
    return wind, surge, rain


def analyze_zero_wind_regions(wind, surge, rain, output_dir: Path, wind_threshold=25.0):
    """
    Identify events/counties where wind is negligible but surge+rain is significant.
    
    Parameters
    ----------
    wind, surge, rain : np.ndarray
        Intensity matrices (counties × events)
        wind: max wind speed (m/s or mph)
        surge: max elevation (m)
        rain: total precipitation (mm or inches)
    wind_threshold : float
        Wind threshold in original units (default 25 m/s ~ 56 mph, minimal damage)
    """
    print(f"\n=== Analyzing Zero-Wind Limitation ===")
    print(f"Using wind threshold: {wind_threshold} (in original units)")
    print(f"Wind range: {wind.min():.2f} - {wind.max():.2f}")
    print(f"Surge range: {surge.min():.2f} - {surge.max():.2f}")
    print(f"Rain range: {rain.min():.2f} - {rain.max():.2f}")
    
    # Don't normalize - use physical thresholds
    # Wind threshold: 25 m/s (~56 mph) - below this, minimal structural damage expected
    # Surge threshold: 0.5 m - noticeable flooding
    # Rain threshold: 50 mm - significant precipitation
    
    surge_threshold = 0.5  # meters
    rain_threshold = 50.0  # mm
    
    # Identify "negligible wind" cells
    zero_wind_mask = wind < wind_threshold
    
    # Identify significant water hazard cells
    significant_surge = surge > surge_threshold
    significant_rain = rain > rain_threshold
    significant_water = significant_surge | significant_rain
    
    # Find cells with negligible wind but significant water hazard
    missed_cells = zero_wind_mask & significant_water
    
    print(f"\nTotal county-event cells: {wind.size:,}")
    print(f"Cells with negligible wind (< {wind_threshold}): {zero_wind_mask.sum():,} ({100*zero_wind_mask.sum()/wind.size:.1f}%)")
    print(f"Cells with significant surge (> {surge_threshold}m): {significant_surge.sum():,} ({100*significant_surge.sum()/wind.size:.1f}%)")
    print(f"Cells with significant rain (> {rain_threshold}mm): {significant_rain.sum():,} ({100*significant_rain.sum()/wind.size:.1f}%)")
    print(f"Cells with significant water (surge OR rain): {significant_water.sum():,} ({100*significant_water.sum()/wind.size:.1f}%)")
    print(f"Cells MISSED (negligible wind + significant water): {missed_cells.sum():,} ({100*missed_cells.sum()/wind.size:.1f}%)")
    
    # Analyze by county (sum across events)
    missed_by_county = missed_cells.sum(axis=1)  # Sum across events for each county (axis=1 since counties × events)
    surge_by_county = (surge * missed_cells).sum(axis=1)  # Total surge missed per county
    rain_by_county = (rain * missed_cells).sum(axis=1)  # Total rain missed per county
    
    # Summary statistics
    counties_affected = (missed_by_county > 0).sum()
    print(f"\nCounties with at least one missed event: {counties_affected:,}")
    print(f"Max missed events for a single county: {missed_by_county.max()}")
    print(f"Mean missed events per affected county: {missed_by_county[missed_by_county > 0].mean():.1f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Histogram of missed events per county
    ax = axes[0, 0]
    counties_with_missed = missed_by_county[missed_by_county > 0]
    if len(counties_with_missed) > 0:
        ax.hist(counties_with_missed, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Number of Missed Events per County')
        ax.set_ylabel('Number of Counties')
        ax.set_title('Distribution of Missed Events\n(Zero wind but significant water hazard)')
        ax.grid(alpha=0.3)
    
    # 2. Scatter: wind vs water hazard (sample of cells)
    ax = axes[0, 1]
    sample_size = min(10000, wind.size)
    sample_idx = np.random.choice(wind.size, sample_size, replace=False)
    wind_flat = wind.flatten()[sample_idx]
    surge_flat = surge.flatten()[sample_idx]
    rain_flat = rain.flatten()[sample_idx]
    missed_flat = missed_cells.flatten()[sample_idx]
    
    # Use max of surge or rain as "water hazard" for visualization
    water_flat = np.maximum(surge_flat, rain_flat)
    
    ax.scatter(wind_flat[~missed_flat], water_flat[~missed_flat], 
              alpha=0.3, s=10, label='Captured by wind model', color='blue')
    ax.scatter(wind_flat[missed_flat], water_flat[missed_flat], 
              alpha=0.5, s=10, label='Missed (negligible wind)', color='red')
    ax.axvline(wind_threshold, color='red', linestyle='--', linewidth=2, label=f'Wind threshold ({wind_threshold})')
    ax.axhline(max(surge_threshold, rain_threshold), color='orange', linestyle='--', linewidth=2, 
              label=f'Water threshold')
    ax.set_xlabel('Wind Speed')
    ax.set_ylabel('Max Water Hazard (Surge or Rain)')
    ax.set_title('Wind vs Water Hazard Intensity\n(Random sample of county-event cells)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # 3. Contribution of wind vs water to total hazard (normalized)
    ax = axes[1, 0]
    # Normalize for contribution analysis
    wind_norm = wind / wind.max() if wind.max() > 0 else wind
    surge_norm = surge / surge.max() if surge.max() > 0 else surge
    rain_norm = rain / rain.max() if rain.max() > 0 else rain
    water_norm = np.maximum(surge_norm, rain_norm)
    
    total_hazard = wind_norm + water_norm
    wind_contribution = wind_norm / total_hazard
    wind_contribution_flat = wind_contribution.flatten()
    wind_contribution_valid = wind_contribution_flat[~np.isnan(wind_contribution_flat) & (wind_contribution_flat > 0)]
    
    ax.hist(wind_contribution_valid, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Wind Contribution to Total Hazard')
    ax.set_ylabel('Frequency (county-event cells)')
    ax.set_title('Distribution of Wind Contribution\n(Normalized Wind / [Wind + max(Surge, Rain)])')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='50% contribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Missed hazard magnitude
    ax = axes[1, 1]
    missed_surge_values = surge[missed_cells]
    missed_rain_values = rain[missed_cells]
    if len(missed_surge_values) > 0:
        ax.hist(missed_surge_values, bins=50, edgecolor='black', alpha=0.7, color='blue', label='Surge')
        ax.hist(missed_rain_values, bins=50, edgecolor='black', alpha=0.7, color='green', label='Rain')
        ax.set_xlabel('Hazard Intensity (original units)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Missed Water Hazards\n(In negligible-wind cells)')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "zero_wind_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {plot_path}")
    plt.close()
    
    # Save county-level summary
    county_summary = pd.DataFrame({
        'county_index': range(len(missed_by_county)),
        'missed_events_count': missed_by_county,
        'total_missed_surge': surge_by_county,
        'total_missed_rain': rain_by_county
    })
    
    # Add fips from county_region.csv
    county_map = pd.read_csv('data/county_region.csv')
    county_summary = county_summary.merge(
        county_map[['county_index', 'fips', 'stcode', 'ccode']], 
        on='county_index', 
        how='left'
    )
    
    county_summary = county_summary.sort_values('missed_events_count', ascending=False)
    summary_path = output_dir / "missed_water_hazard_by_county.csv"
    county_summary.to_csv(summary_path, index=False)
    print(f"Saved county summary to: {summary_path}")
    
    # Print top counties most affected
    print("\nTop 20 counties with most missed events:")
    print(county_summary.head(20)[['fips', 'stcode', 'ccode', 'missed_events_count', 'total_missed_surge', 'total_missed_rain']])


def main():
    data_dir = Path("data/hazard")
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    print("Loading Gori et al. hazard data...")
    # Note: You may need to adjust the key names in load_gori_hazards
    # based on the actual structure of your .mat files
    try:
        wind, surge, rain = load_gori_hazards(data_dir)
        analyze_zero_wind_regions(wind, surge, rain, output_dir)
    except Exception as e:
        print(f"Error loading hazard data: {e}")
        print("\nPlease check:")
        print("1. The .mat file paths are correct")
        print("2. The matrix key names match your files (may not be 'hazard')")
        print("\nYou can inspect .mat file contents with:")
        print("  python -c \"from scipy.io import loadmat; print(loadmat('path/to/file.mat').keys())\"")


if __name__ == "__main__":
    main()
