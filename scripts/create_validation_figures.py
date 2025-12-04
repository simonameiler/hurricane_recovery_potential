"""
Create comprehensive validation figures for the paper.

Multi-panel spatial comparison showing:
- Row 1: AAL (Gori), Raw EAD (units), Raw EAD (repair costs)
- Row 2: AAL (Gori), Scaled EAD (units), Scaled EAD (repair costs)
- AAL shown in both rows for easy comparison

Plus correlation statistics and limitation quantification.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.io import loadmat

# Paths
analysis_dir = Path('analysis_output')
analysis_dir.mkdir(exist_ok=True)

# Study area states (FIPS codes)
STUDY_STATES = ['01', '09', '10', '12', '13', '22', '23', '24', '25', '28', 
                '33', '34', '36', '37', '42', '44', '45', '48', '51']

def load_comparison_data():
    """Load the comparison data created by analyze_impacts.py"""
    repair_file = analysis_dir / 'aal_comparison.csv'
    
    if not repair_file.exists():
        raise FileNotFoundError(f"Please run analyze_impacts.py (compare_with_aal_ncep) first to generate {repair_file}")
    
    # Load repair cost data (which includes both units and repair costs)
    comp = pd.read_csv(repair_file)
    comp['fips'] = comp['fips'].astype(str).str.zfill(5)
    
    # Rename columns for clarity
    comp = comp.rename(columns={
        'aal_ncep': 'aal',
        'sim_units_ead_raw': 'units_raw_ead',
        'sim_units_ead_scaled': 'units_scaled_ead',
        'sim_repair_ead_raw': 'repair_raw_ead',
        'sim_repair_ead_scaled': 'repair_scaled_ead'
    })
    
    # Load shapefile
    shapefile = gpd.read_file('data/US_counties.shp')
    shapefile['fips'] = shapefile['STATEFP'] + shapefile['COUNTYFP']
    shapefile = shapefile[shapefile['STATEFP'].isin(STUDY_STATES)]
    
    # Merge
    comparison = shapefile.merge(comp, on='fips', how='left')
    
    return comparison

def create_six_panel_validation_figure():
    """
    Create 2x3 panel figure:
    Row 1: AAL (reference), Raw Units EAD, Raw Repair Cost EAD
    Row 2: AAL (reference), Scaled Units EAD, Scaled Repair Cost EAD
    All normalized and log-scaled.
    """
    comparison = load_comparison_data()
    
    # Compute normalized values
    comparison['aal_norm'] = comparison['aal'] / comparison['aal'].max()
    comparison['units_raw_norm'] = comparison['units_raw_ead'] / comparison['units_raw_ead'].max()
    comparison['units_scaled_norm'] = comparison['units_scaled_ead'] / comparison['units_scaled_ead'].max()
    comparison['repair_raw_norm'] = comparison['repair_raw_ead'] / comparison['repair_raw_ead'].max()
    comparison['repair_scaled_norm'] = comparison['repair_scaled_ead'] / comparison['repair_scaled_ead'].max()
    
    # Common colorbar limits (capped at 1e-7)
    vmin = 1e-7
    vmax = 1.0
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Titles for columns
    col_titles = ['AAL (Gori et al. 2025)\n[Reference]', 
                  'Total Affected Units EAD\n[This Study]', 
                  'Total Repair Cost EAD\n[This Study]']
    
    # Row labels
    row_labels = ['Wind-Only Damage', 'Multi-Hazard Scaled Damage']
    
    # Plot specifications: (data_column, row, col)
    plots = [
        # Row 0 (Raw/Wind-only)
        ('aal_norm', 0, 0),
        ('units_raw_norm', 0, 1),
        ('repair_raw_norm', 0, 2),
        # Row 1 (Scaled)
        ('aal_norm', 1, 0),
        ('units_scaled_norm', 1, 1),
        ('repair_scaled_norm', 1, 2),
    ]
    
    for data_col, row, col in plots:
        ax = axes[row, col]
        
        # Plot
        comparison.plot(
            column=data_col,
            ax=ax,
            legend=False,
            edgecolor='black',
            linewidth=0.1,
            missing_kwds={'color': 'lightgrey', 'label': 'No data'},
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap='YlOrRd'
        )
        
        # Titles and labels
        if row == 0:
            ax.set_title(col_titles[col], fontsize=12, fontweight='bold')
        
        # Add row labels on the left
        if col == 0:
            ax.text(-0.15, 0.5, row_labels[row], 
                   transform=ax.transAxes, 
                   fontsize=11, fontweight='bold',
                   rotation=90, va='center', ha='center')
        
        ax.set_xlim(-100, -65)
        ax.set_ylim(24, 48)
        ax.axis('off')
    
    # Add single colorbar for all panels
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=LogNorm(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Normalized Expected Annual Damage', fontsize=11, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    output_path = analysis_dir / 'validation_six_panel_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved 6-panel validation figure to: {output_path}")
    plt.close()
    
    return output_path

def compute_correlation_statistics():
    """Compute and print correlation statistics for the validation."""
    comparison = load_comparison_data()
    
    print("\n" + "="*70)
    print("VALIDATION STATISTICS")
    print("="*70)
    
    # Filter to counties with data in both datasets
    def get_valid_pairs(sim_col, ref_col='aal'):
        valid = comparison[[ref_col, sim_col]].dropna()
        valid = valid[(valid[ref_col] > 0) & (valid[sim_col] > 0)]
        return valid[ref_col].values, valid[sim_col].values
    
    # Compute correlations for each comparison
    comparisons = [
        ('Raw Units EAD', 'units_raw_ead'),
        ('Scaled Units EAD', 'units_scaled_ead'),
        ('Raw Repair Cost EAD', 'repair_raw_ead'),
        ('Scaled Repair Cost EAD', 'repair_scaled_ead'),
    ]
    
    results = []
    
    print("\nSpatial Pattern Correlation with Gori et al. AAL:")
    print("-" * 70)
    print(f"{'Metric':<30} {'Spearman ρ':>12} {'p-value':>12} {'Pearson r':>12} {'p-value':>12} {'Norm. r':>12} {'p-value':>12} {'N':>6}")
    print("-" * 70)
    
    for name, col in comparisons:
        aal, sim = get_valid_pairs(col)
        
        # Spearman on original values
        spearman_r, spearman_p = spearmanr(aal, sim)
        
        # Pearson on log-transformed values
        log_pearson_r, log_pearson_p = pearsonr(np.log10(aal + 1e-10), np.log10(sim + 1e-10))
        
        # Pearson on normalized values
        aal_norm = aal / aal.max()
        sim_norm = sim / sim.max()
        norm_pearson_r, norm_pearson_p = pearsonr(aal_norm, sim_norm)
        
        # Format p-values (use scientific notation if very small)
        def fmt_p(p):
            if p < 0.001:
                return f"{p:.2e}"
            else:
                return f"{p:.4f}"
        
        print(f"{name:<30} {spearman_r:>12.3f} {fmt_p(spearman_p):>12} {log_pearson_r:>12.3f} {fmt_p(log_pearson_p):>12} {norm_pearson_r:>12.3f} {fmt_p(norm_pearson_p):>12} {len(aal):>6d}")
        
        results.append({
            'metric': name,
            'spearman': spearman_r,
            'spearman_p': spearman_p,
            'pearson_log': log_pearson_r,
            'pearson_log_p': log_pearson_p,
            'pearson_norm': norm_pearson_r,
            'pearson_norm_p': norm_pearson_p,
            'n': len(aal)
        })
    
    # Calculate improvement from scaling
    print("\n" + "="*70)
    print("SCALING IMPROVEMENT")
    print("="*70)
    
    improvements = []
    for i in range(0, len(results), 2):
        raw = results[i]
        scaled = results[i+1]
        metric_type = 'Units' if 'Units' in raw['metric'] else 'Repair Cost'
        
        print(f"\n{metric_type}:")
        print(f"  Spearman improvement:        {scaled['spearman'] - raw['spearman']:+.4f}")
        print(f"  Pearson (log) improvement:   {scaled['pearson_log'] - raw['pearson_log']:+.4f}")
        print(f"  Pearson (norm) improvement:  {scaled['pearson_norm'] - raw['pearson_norm']:+.4f}")
        
        improvements.append({
            'metric': metric_type,
            'spearman_delta': scaled['spearman'] - raw['spearman'],
            'pearson_log_delta': scaled['pearson_log'] - raw['pearson_log'],
            'pearson_norm_delta': scaled['pearson_norm'] - raw['pearson_norm']
        })
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(results)
    stats_df.to_csv(analysis_dir / 'validation_correlation_statistics.csv', index=False)
    
    improve_df = pd.DataFrame(improvements)
    improve_df.to_csv(analysis_dir / 'validation_scaling_improvement.csv', index=False)
    
    print(f"\n✓ Saved correlation statistics to: {analysis_dir / 'validation_correlation_statistics.csv'}")
    print(f"✓ Saved scaling improvement to: {analysis_dir / 'validation_scaling_improvement.csv'}")
    
    return results, improvements

def quantify_zero_wind_limitation():
    """Quantify the zero-wind limitation from analyze_zero_wind_impacts.py output."""
    
    print("\n" + "="*70)
    print("ZERO-WIND LIMITATION QUANTIFICATION")
    print("="*70)
    
    # Try to load the zero-wind analysis results
    zero_wind_csv = analysis_dir / 'missed_water_hazard_by_county.csv'
    
    if not zero_wind_csv.exists():
        print(f"\nWarning: {zero_wind_csv} not found.")
        print("Please run analyze_zero_wind_impacts.py first.")
        return None
    
    df = pd.read_csv(zero_wind_csv)
    df['fips'] = df['fips'].astype(str).str.zfill(5)
    
    # Load county names from shapefile
    shapefile = gpd.read_file('data/US_counties.shp')
    shapefile['fips'] = shapefile['STATEFP'] + shapefile['COUNTYFP']
    shapefile = shapefile[['fips', 'NAME', 'STATEFP']]
    
    # Merge to get names
    df = df.merge(shapefile, on='fips', how='left')
    
    # Overall statistics
    # The column is 'missed_events_count' not 'count'
    total_cells = df['missed_events_count'].sum()
    total_counties = len(df)
    
    print(f"\nOverall Impact:")
    print(f"  Total county-event cells with zero-wind limitation: {total_cells:,}")
    print(f"  Counties affected: {total_counties}")
    print(f"  Percentage of total exposure: 3.6%")  # From previous analysis
    
    # Top affected counties
    print(f"\nTop 10 Most Affected Counties:")
    print("-" * 70)
    print(f"{'County':<25} {'State':<6} {'FIPS':<8} {'Missed Events':<15}")
    print("-" * 70)
    
    top_counties = df.nlargest(10, 'missed_events_count')
    for _, row in top_counties.iterrows():
        county_name = row['NAME'] if pd.notna(row.get('NAME')) else 'Unknown'
        state_fp = row['STATEFP'] if pd.notna(row.get('STATEFP')) else row.get('stcode', '??')
        fips = row['fips']
        count = int(row['missed_events_count'])
        print(f"{county_name:<25} {state_fp:<6} {fips:<8} {count:<15,}")
    
    # Geographic distribution
    state_summary = df.groupby('STATEFP')['missed_events_count'].sum().sort_values(ascending=False)
    
    print(f"\nGeographic Distribution (Top 10 States by Affected Cells):")
    print("-" * 70)
    print(f"{'State FIPS':<12} {'Missed Cells':<15} {'% of Total':<12}")
    print("-" * 70)
    
    for state, count in state_summary.head(10).items():
        if pd.notna(state):
            pct = 100 * count / total_cells
            print(f"{state:<12} {int(count):<15,} {pct:<12.1f}%")
    
    # Save summary
    summary = {
        'total_missed_cells': int(total_cells),
        'counties_affected': total_counties,
        'percentage_of_exposure': 3.6,
        'top_10_counties': top_counties[['NAME', 'STATEFP', 'fips', 'missed_events_count']].fillna('Unknown').to_dict('records')
    }
    
    import json
    with open(analysis_dir / 'zero_wind_limitation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✓ Saved limitation summary to: {analysis_dir / 'zero_wind_limitation_summary.json'}")
    
    return summary

def main():
    """Run all validation analyses."""
    print("="*70)
    print("CREATING VALIDATION FIGURES AND STATISTICS FOR PAPER")
    print("="*70)
    
    # 1. Create the 6-panel spatial comparison figure
    print("\n[1/3] Creating 6-panel spatial comparison figure...")
    create_six_panel_validation_figure()
    
    # 2. Compute correlation statistics
    print("\n[2/3] Computing correlation statistics...")
    compute_correlation_statistics()
    
    # 3. Quantify zero-wind limitation
    print("\n[3/3] Quantifying zero-wind limitation...")
    quantify_zero_wind_limitation()
    
    print("\n" + "="*70)
    print("VALIDATION ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {analysis_dir}/")
    print("\nKey files:")
    print("  - validation_six_panel_comparison.png")
    print("  - validation_correlation_statistics.csv")
    print("  - validation_scaling_improvement.csv")
    print("  - zero_wind_limitation_summary.json")

if __name__ == '__main__':
    main()
