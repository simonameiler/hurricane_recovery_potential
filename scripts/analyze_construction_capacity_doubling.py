"""
Analyze the impact of doubling construction capacity on recovery potential.

This script:
1. Loads recovery potential data for baseline and doubled construction capacity scenarios
2. Compares recovery times per event and county
3. Tests if doubling capacity leads to halving of recovery time
4. Creates visualizations including maps showing spatial differences
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import geopandas as gpd
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
BASELINE_DIR = DATA_DIR / "recovery_potential_per_scenario"
DOUBLED_DIR = DATA_DIR / "recovery_potential_double_construction_capacity"
OUTPUT_DIR = BASE_DIR / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load county shapefile
COUNTIES_SHP = DATA_DIR / "US_counties.shp"


def load_recovery_data(directory):
    """Load all recovery potential JSON files from a directory."""
    all_data = []
    json_files = sorted(directory.glob("*_scaled_recovery_potential.json"))
    
    print(f"Loading {len(json_files)} files from {directory.name}...")
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_data.extend(data)
    
    df = pd.DataFrame(all_data)
    
    # Handle infinity values
    df['recovery_potential [months]'] = df['recovery_potential [months]'].replace([np.inf, -np.inf], np.nan)
    
    return df


def compare_scenarios(baseline_df, doubled_df):
    """Compare baseline and doubled construction capacity scenarios."""
    
    # Merge datasets
    merged = baseline_df.merge(
        doubled_df,
        on=['event', 'fips'],
        suffixes=('_baseline', '_doubled')
    )
    
    # Calculate the ratio of recovery times
    merged['recovery_ratio'] = (
        merged['recovery_potential [months]_doubled'] / 
        merged['recovery_potential [months]_baseline']
    )
    
    # Calculate absolute difference
    merged['recovery_diff_months'] = (
        merged['recovery_potential [months]_baseline'] - 
        merged['recovery_potential [months]_doubled']
    )
    
    # Calculate percent reduction
    merged['percent_reduction'] = (
        (merged['recovery_potential [months]_baseline'] - 
         merged['recovery_potential [months]_doubled']) / 
        merged['recovery_potential [months]_baseline'] * 100
    )
    
    return merged


def analyze_capacity_relationship(merged_df):
    """Analyze if doubling capacity leads to halving of recovery time."""
    
    # Filter out nan and infinite values
    valid_data = merged_df[
        merged_df['recovery_potential [months]_baseline'].notna() &
        merged_df['recovery_potential [months]_doubled'].notna() &
        (merged_df['recovery_potential [months]_baseline'] > 0) &
        (merged_df['recovery_potential [months]_doubled'] > 0)
    ].copy()
    
    print("\n" + "="*80)
    print("ANALYSIS: Does doubling construction capacity halve recovery time?")
    print("="*80)
    
    # Calculate statistics on the ratio
    ratio_stats = valid_data['recovery_ratio'].describe()
    print("\nRecovery Time Ratio (Doubled / Baseline) Statistics:")
    print(ratio_stats)
    
    # Test if median ratio is close to 0.5
    median_ratio = valid_data['recovery_ratio'].median()
    mean_ratio = valid_data['recovery_ratio'].mean()
    
    print(f"\nMedian ratio: {median_ratio:.4f}")
    print(f"Mean ratio: {mean_ratio:.4f}")
    print(f"Expected ratio if perfect halving: 0.5000")
    print(f"Deviation from 0.5: {abs(median_ratio - 0.5):.4f}")
    
    # Count how many are close to 0.5 (within 5% tolerance)
    tolerance = 0.05
    close_to_half = valid_data[
        (valid_data['recovery_ratio'] >= 0.5 - tolerance) &
        (valid_data['recovery_ratio'] <= 0.5 + tolerance)
    ]
    
    pct_close = (len(close_to_half) / len(valid_data)) * 100
    print(f"\nPercentage of cases within ±{tolerance} of 0.5: {pct_close:.2f}%")
    print(f"Total valid cases: {len(valid_data):,}")
    
    # Analyze percent reduction
    print("\nPercent Reduction in Recovery Time:")
    print(valid_data['percent_reduction'].describe())
    
    return valid_data


def create_per_event_summary(merged_df):
    """Create summary statistics per event."""
    
    event_summary = merged_df.groupby('event').agg({
        'recovery_potential [months]_baseline': ['mean', 'median', 'std'],
        'recovery_potential [months]_doubled': ['mean', 'median', 'std'],
        'recovery_ratio': ['mean', 'median', 'std'],
        'recovery_diff_months': ['mean', 'median', 'sum'],
        'percent_reduction': ['mean', 'median'],
        'fips': 'count'
    }).round(4)
    
    event_summary.columns = ['_'.join(col).strip() for col in event_summary.columns.values]
    event_summary = event_summary.rename(columns={'fips_count': 'num_counties'})
    event_summary = event_summary.reset_index()
    
    return event_summary


def create_per_county_summary(merged_df):
    """Create summary statistics per county across all events."""
    
    county_summary = merged_df.groupby('fips').agg({
        'recovery_potential [months]_baseline': ['mean', 'median', 'std', 'count'],
        'recovery_potential [months]_doubled': ['mean', 'median', 'std'],
        'recovery_ratio': ['mean', 'median', 'std'],
        'recovery_diff_months': ['mean', 'median', 'sum'],
        'percent_reduction': ['mean', 'median'],
        'event': 'count'
    }).round(4)
    
    county_summary.columns = ['_'.join(col).strip() for col in county_summary.columns.values]
    county_summary = county_summary.rename(columns={'event_count': 'num_events'})
    county_summary = county_summary.reset_index()
    
    return county_summary


def create_visualizations(valid_data, event_summary, county_summary):
    """Create visualizations of the analysis."""
    
    print("\nCreating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Scatter plot: Baseline vs Doubled recovery time
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Subplot 1: Recovery time comparison
    ax = axes[0, 0]
    sample_size = min(10000, len(valid_data))
    sample = valid_data.sample(n=sample_size, random_state=42)
    
    ax.scatter(sample['recovery_potential [months]_baseline'], 
               sample['recovery_potential [months]_doubled'],
               alpha=0.3, s=10, c='blue')
    
    # Add perfect halving line
    max_val = max(sample['recovery_potential [months]_baseline'].max(),
                  sample['recovery_potential [months]_doubled'].max())
    ax.plot([0, max_val], [0, max_val/2], 'r--', label='Perfect Halving (y=x/2)', linewidth=2)
    ax.plot([0, max_val], [0, max_val], 'k--', label='No Change (y=x)', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Baseline Recovery Time (months)', fontsize=12)
    ax.set_ylabel('Doubled Capacity Recovery Time (months)', fontsize=12)
    ax.set_title('Recovery Time: Baseline vs Doubled Construction Capacity', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, np.percentile(sample['recovery_potential [months]_baseline'], 95))
    ax.set_ylim(0, np.percentile(sample['recovery_potential [months]_doubled'], 95))
    
    # Subplot 2: Distribution of recovery ratios
    ax = axes[0, 1]
    ratio_data = valid_data['recovery_ratio']
    ratio_data_filtered = ratio_data[(ratio_data >= 0) & (ratio_data <= 1.5)]
    
    ax.hist(ratio_data_filtered, bins=100, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Expected Ratio (0.5)')
    ax.axvline(x=ratio_data_filtered.median(), color='orange', linestyle='--', 
               linewidth=2, label=f'Median ({ratio_data_filtered.median():.3f})')
    ax.set_xlabel('Recovery Time Ratio (Doubled/Baseline)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Recovery Time Ratios', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Subplot 3: Percent reduction distribution
    ax = axes[1, 0]
    pct_data = valid_data['percent_reduction']
    pct_data_filtered = pct_data[(pct_data >= -10) & (pct_data <= 100)]
    
    ax.hist(pct_data_filtered, bins=100, edgecolor='black', alpha=0.7, color='lightgreen')
    ax.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Expected (50%)')
    ax.axvline(x=pct_data_filtered.median(), color='orange', linestyle='--', 
               linewidth=2, label=f'Median ({pct_data_filtered.median():.1f}%)')
    ax.set_xlabel('Percent Reduction in Recovery Time (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Percent Reduction in Recovery Time', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Subplot 4: Recovery difference by baseline recovery time
    ax = axes[1, 1]
    sample = valid_data.sample(n=sample_size, random_state=42)
    scatter = ax.scatter(sample['recovery_potential [months]_baseline'], 
                        sample['recovery_diff_months'],
                        c=sample['percent_reduction'], 
                        alpha=0.5, s=10, cmap='RdYlGn', vmin=0, vmax=100)
    
    ax.set_xlabel('Baseline Recovery Time (months)', fontsize=12)
    ax.set_ylabel('Reduction in Recovery Time (months)', fontsize=12)
    ax.set_title('Recovery Time Reduction vs Baseline', fontsize=14, fontweight='bold')
    ax.set_xlim(0, np.percentile(sample['recovery_potential [months]_baseline'], 95))
    plt.colorbar(scatter, ax=ax, label='Percent Reduction (%)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'construction_capacity_doubling_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'construction_capacity_doubling_analysis.png'}")
    plt.close()
    
    # 2. Event-level summary plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top events by total recovery time saved
    ax = axes[0]
    top_events = event_summary.nlargest(20, 'recovery_diff_months_sum')
    ax.barh(range(len(top_events)), top_events['recovery_diff_months_sum'], color='steelblue')
    ax.set_yticks(range(len(top_events)))
    ax.set_yticklabels(top_events['event'])
    ax.set_xlabel('Total Recovery Time Saved (months)', fontsize=12)
    ax.set_ylabel('Event ID', fontsize=12)
    ax.set_title('Top 20 Events by Total Recovery Time Saved', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Event-level ratio distribution
    ax = axes[1]
    ax.scatter(event_summary['num_counties'], event_summary['recovery_ratio_mean'], 
               alpha=0.6, s=50, c='coral')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Perfect Halving')
    ax.set_xlabel('Number of Affected Counties', fontsize=12)
    ax.set_ylabel('Mean Recovery Time Ratio', fontsize=12)
    ax.set_title('Event-Level Recovery Time Ratio vs Coverage', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'event_level_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'event_level_analysis.png'}")
    plt.close()


def create_maps(county_summary):
    """Create maps showing spatial differences in recovery time reduction."""
    
    print("\nCreating spatial maps...")
    
    try:
        # Load county shapefile
        counties_gdf = gpd.read_file(COUNTIES_SHP)
        
        # Ensure FIPS codes are strings with proper formatting
        # Use GEOID column which contains the 5-digit FIPS code
        counties_gdf['GEOID'] = counties_gdf['GEOID'].astype(str).str.zfill(5)
        county_summary['fips'] = county_summary['fips'].astype(str).str.zfill(5)
        
        # Merge with county summary
        map_data = counties_gdf.merge(county_summary, left_on='GEOID', right_on='fips', how='left')
        
        # Create figure with multiple maps
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Map 1: Mean recovery time reduction (months)
        ax = axes[0, 0]
        map_data.plot(column='recovery_diff_months_mean', cmap='YlOrRd', 
                     legend=True, ax=ax, edgecolor='black', linewidth=0.1,
                     legend_kwds={'label': 'Mean Recovery Time Saved (months)', 'shrink': 0.8})
        ax.set_title('Mean Recovery Time Saved per County\n(Baseline - Doubled Capacity)', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Map 2: Mean percent reduction
        ax = axes[0, 1]
        map_data.plot(column='percent_reduction_mean', cmap='RdYlGn', 
                     legend=True, ax=ax, edgecolor='black', linewidth=0.1,
                     vmin=0, vmax=100,
                     legend_kwds={'label': 'Mean Percent Reduction (%)', 'shrink': 0.8})
        ax.set_title('Mean Percent Reduction in Recovery Time per County', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Map 3: Mean recovery time ratio
        ax = axes[1, 0]
        # Use a diverging colormap centered at 0.5
        norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1.0)
        map_data.plot(column='recovery_ratio_mean', cmap='RdYlGn_r', 
                     legend=True, ax=ax, edgecolor='black', linewidth=0.1, norm=norm,
                     legend_kwds={'label': 'Mean Recovery Time Ratio\n(Doubled/Baseline)', 'shrink': 0.8})
        ax.set_title('Mean Recovery Time Ratio per County\n(Lower = More Effective)', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Map 4: Number of events affecting each county
        ax = axes[1, 1]
        map_data.plot(column='num_events', cmap='Blues', 
                     legend=True, ax=ax, edgecolor='black', linewidth=0.1,
                     legend_kwds={'label': 'Number of Events', 'shrink': 0.8})
        ax.set_title('Number of Hurricane Events Affecting Each County', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'construction_capacity_doubling_maps.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {OUTPUT_DIR / 'construction_capacity_doubling_maps.png'}")
        plt.close()
        
        # Create focused map on coastal regions
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        
        # Filter to counties with data
        coastal_data = map_data[map_data['num_events'].notna()]
        
        coastal_data.plot(column='recovery_diff_months_mean', cmap='YlOrRd', 
                         legend=True, ax=ax, edgecolor='black', linewidth=0.3,
                         legend_kwds={'label': 'Mean Recovery Time Saved (months)', 'shrink': 0.8})
        
        ax.set_title('Recovery Time Saved by Doubling Construction Capacity\n(Coastal Counties Affected by Hurricanes)', 
                    fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Set appropriate bounds for coastal regions
        ax.set_xlim(-100, -65)
        ax.set_ylim(25, 47)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'coastal_recovery_time_savings_map.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {OUTPUT_DIR / 'coastal_recovery_time_savings_map.png'}")
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not create maps: {e}")


def main():
    """Main analysis function."""
    
    print("\n" + "="*80)
    print("CONSTRUCTION CAPACITY DOUBLING ANALYSIS")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    baseline_df = load_recovery_data(BASELINE_DIR)
    doubled_df = load_recovery_data(DOUBLED_DIR)
    
    print(f"\nBaseline data: {len(baseline_df):,} rows")
    print(f"Doubled capacity data: {len(doubled_df):,} rows")
    
    # Compare scenarios
    print("\n2. Comparing scenarios...")
    merged_df = compare_scenarios(baseline_df, doubled_df)
    print(f"Merged data: {len(merged_df):,} rows")
    
    # Analyze capacity relationship
    print("\n3. Analyzing capacity-recovery relationship...")
    valid_data = analyze_capacity_relationship(merged_df)
    
    # Create per-event summary
    print("\n4. Creating per-event summary...")
    event_summary = create_per_event_summary(valid_data)
    print(f"Event summary: {len(event_summary)} events")
    
    # Create per-county summary
    print("\n5. Creating per-county summary...")
    county_summary = create_per_county_summary(valid_data)
    print(f"County summary: {len(county_summary)} counties")
    
    # Save summary tables
    print("\n6. Saving summary tables...")
    merged_df.to_csv(OUTPUT_DIR / 'construction_capacity_comparison_full.csv', index=False)
    print(f"Saved: {OUTPUT_DIR / 'construction_capacity_comparison_full.csv'}")
    
    event_summary.to_csv(OUTPUT_DIR / 'construction_capacity_event_summary.csv', index=False)
    print(f"Saved: {OUTPUT_DIR / 'construction_capacity_event_summary.csv'}")
    
    county_summary.to_csv(OUTPUT_DIR / 'construction_capacity_county_summary.csv', index=False)
    print(f"Saved: {OUTPUT_DIR / 'construction_capacity_county_summary.csv'}")
    
    # Create visualizations
    print("\n7. Creating visualizations...")
    create_visualizations(valid_data, event_summary, county_summary)
    
    # Create maps
    print("\n8. Creating spatial maps...")
    create_maps(county_summary)
    
    # Print final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal events analyzed: {len(event_summary)}")
    print(f"Total counties analyzed: {len(county_summary)}")
    print(f"Total event-county combinations: {len(valid_data):,}")
    
    median_ratio = valid_data['recovery_ratio'].median()
    median_pct_reduction = valid_data['percent_reduction'].median()
    
    print(f"\nMedian recovery time ratio: {median_ratio:.4f}")
    print(f"Median percent reduction: {median_pct_reduction:.2f}%")
    
    if abs(median_ratio - 0.5) < 0.01:
        print("\n✓ Doubling construction capacity DOES lead to approximately halving recovery time!")
    else:
        print(f"\n⚠ Doubling construction capacity leads to a {median_pct_reduction:.1f}% reduction,")
        print(f"  which deviates from the expected 50% by {abs(median_pct_reduction - 50):.1f} percentage points.")
    
    print("\n" + "="*80)
    print("Analysis complete! Check the analysis_output directory for results.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
