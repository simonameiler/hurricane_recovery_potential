#!/usr/bin/env python3
"""
Compare AAL with total affected units (summed across all damage states).

This is more relevant for recovery analysis than repair costs, since recovery
modeling focuses on the NUMBER of units affected, not just dollar values.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def compute_total_units_ead(events_dir, freq=0.00067334):
    """Compute EAD for total affected units (sum across all DS)."""
    print("Loading event data and computing total affected units...")
    
    # Load raw and scaled event data
    raw_files = list(events_dir.glob("raw/*.csv"))
    scaled_files = list(events_dir.glob("scaled/*.csv"))
    
    print(f"Found {len(raw_files)} raw files, {len(scaled_files)} scaled files")
    
    # Load and concatenate
    raw_dfs = [pd.read_csv(f) for f in raw_files]
    scaled_dfs = [pd.read_csv(f) for f in scaled_files]
    
    raw_df = pd.concat(raw_dfs, ignore_index=True)
    scaled_df = pd.concat(scaled_dfs, ignore_index=True)
    
    # Compute total units affected per county (sum across all events and DS)
    print("\nComputing total affected units per county...")
    
    # Raw: sum all DS columns
    raw_units = raw_df.groupby('fips').agg({
        'units_DS1_raw': 'sum',
        'units_DS2_raw': 'sum',
        'units_DS3_raw': 'sum',
        'units_DS4_raw': 'sum'
    }).reset_index()
    raw_units['fips'] = raw_units['fips'].astype(str).str.zfill(5)
    raw_units['total_units'] = (
        raw_units['units_DS1_raw'] + 
        raw_units['units_DS2_raw'] + 
        raw_units['units_DS3_raw'] + 
        raw_units['units_DS4_raw']
    )
    raw_units['total_units_ead'] = raw_units['total_units'] * freq
    
    # Scaled: sum all DS columns
    scaled_units = scaled_df.groupby('fips').agg({
        'units_DS1_scaled': 'sum',
        'units_DS2_scaled': 'sum',
        'units_DS3_scaled': 'sum',
        'units_DS4_scaled': 'sum'
    }).reset_index()
    scaled_units['fips'] = scaled_units['fips'].astype(str).str.zfill(5)
    scaled_units['total_units'] = (
        scaled_units['units_DS1_scaled'] + 
        scaled_units['units_DS2_scaled'] + 
        scaled_units['units_DS3_scaled'] + 
        scaled_units['units_DS4_scaled']
    )
    scaled_units['total_units_ead'] = scaled_units['total_units'] * freq
    
    return raw_units, scaled_units


def compare_units_with_aal(raw_units, scaled_units, aal_comparison):
    """Compare total affected units EAD with AAL."""
    print("\n" + "="*70)
    print("ANALYSIS: Normalized AAL vs Normalized Total Affected Units")
    print("="*70)
    
    # Merge units with AAL comparison data
    comparison = aal_comparison[['fips', 'aal_ncep']].copy()
    comparison['fips'] = comparison['fips'].astype(str).str.zfill(5)
    comparison = comparison.merge(
        raw_units[['fips', 'total_units_ead']], 
        on='fips', 
        how='left'
    ).rename(columns={'total_units_ead': 'units_ead_raw'})
    
    comparison = comparison.merge(
        scaled_units[['fips', 'total_units_ead']], 
        on='fips', 
        how='left'
    ).rename(columns={'total_units_ead': 'units_ead_scaled'})
    
    comparison = comparison.fillna(0)
    
    # Normalize all metrics to 0-1
    comparison['aal_norm'] = comparison['aal_ncep'] / comparison['aal_ncep'].max()
    comparison['units_raw_norm'] = comparison['units_ead_raw'] / comparison['units_ead_raw'].max()
    comparison['units_scaled_norm'] = comparison['units_ead_scaled'] / comparison['units_ead_scaled'].max()
    
    # Summary statistics
    print(f"\nTotal affected units EAD (raw):    {comparison['units_ead_raw'].sum():,.0f} units/year")
    print(f"Total affected units EAD (scaled): {comparison['units_ead_scaled'].sum():,.0f} units/year")
    
    # Correlations - units vs AAL
    mask_raw = (comparison['aal_ncep'] > 0) & (comparison['units_ead_raw'] > 0)
    mask_scaled = (comparison['aal_ncep'] > 0) & (comparison['units_ead_scaled'] > 0)
    
    print(f"\n{'Comparison':<35} {'Pearson r':<12} {'Spearman r':<12} {'N':<8}")
    print("-"*70)
    
    if mask_raw.sum() > 0:
        # Raw units vs AAL (log-log)
        pr_raw_log, pp_raw_log = pearsonr(
            np.log10(comparison.loc[mask_raw, 'aal_ncep']), 
            np.log10(comparison.loc[mask_raw, 'units_ead_raw'])
        )
        sr_raw, sp_raw = spearmanr(
            comparison.loc[mask_raw, 'aal_ncep'], 
            comparison.loc[mask_raw, 'units_ead_raw']
        )
        print(f"{'Units (raw) vs AAL (log-log)':<35} {pr_raw_log:<12.3f} {sr_raw:<12.3f} {mask_raw.sum():<8}")
        
        # Normalized
        mask_norm = (comparison['aal_norm'] > 0) & (comparison['units_raw_norm'] > 0)
        pr_raw_norm, pp_raw_norm = pearsonr(
            comparison.loc[mask_norm, 'aal_norm'], 
            comparison.loc[mask_norm, 'units_raw_norm']
        )
        print(f"{'Units (raw) vs AAL (normalized)':<35} {pr_raw_norm:<12.3f} {sr_raw:<12.3f} {mask_norm.sum():<8}")
    
    if mask_scaled.sum() > 0:
        # Scaled units vs AAL (log-log)
        pr_scaled_log, pp_scaled_log = pearsonr(
            np.log10(comparison.loc[mask_scaled, 'aal_ncep']), 
            np.log10(comparison.loc[mask_scaled, 'units_ead_scaled'])
        )
        sr_scaled, sp_scaled = spearmanr(
            comparison.loc[mask_scaled, 'aal_ncep'], 
            comparison.loc[mask_scaled, 'units_ead_scaled']
        )
        print(f"{'Units (scaled) vs AAL (log-log)':<35} {pr_scaled_log:<12.3f} {sr_scaled:<12.3f} {mask_scaled.sum():<8}")
        
        # Normalized
        mask_norm = (comparison['aal_norm'] > 0) & (comparison['units_scaled_norm'] > 0)
        pr_scaled_norm, pp_scaled_norm = pearsonr(
            comparison.loc[mask_norm, 'aal_norm'], 
            comparison.loc[mask_norm, 'units_scaled_norm']
        )
        print(f"{'Units (scaled) vs AAL (normalized)':<35} {pr_scaled_norm:<12.3f} {sr_scaled:<12.3f} {mask_norm.sum():<8}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Top left: Raw units vs AAL (log-log)
    ax = axes[0, 0]
    if mask_raw.sum() > 0:
        ax.scatter(comparison.loc[mask_raw, 'aal_ncep'], 
                  comparison.loc[mask_raw, 'units_ead_raw'],
                  alpha=0.5, s=30, edgecolor='black', linewidth=0.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('AAL ($/year)')
        ax.set_ylabel('Total Units EAD (units/year)')
        ax.set_title(f'Raw Units vs AAL\nPearson r={pr_raw_log:.3f}, Spearman r={sr_raw:.3f}')
        ax.grid(alpha=0.3)
    
    # Top right: Scaled units vs AAL (log-log)
    ax = axes[0, 1]
    if mask_scaled.sum() > 0:
        ax.scatter(comparison.loc[mask_scaled, 'aal_ncep'], 
                  comparison.loc[mask_scaled, 'units_ead_scaled'],
                  alpha=0.5, s=30, edgecolor='black', linewidth=0.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('AAL ($/year)')
        ax.set_ylabel('Total Units EAD (units/year)')
        ax.set_title(f'Scaled Units vs AAL\nPearson r={pr_scaled_log:.3f}, Spearman r={sr_scaled:.3f}')
        ax.grid(alpha=0.3)
    
    # Bottom left: Normalized raw units vs AAL
    ax = axes[1, 0]
    mask_norm = (comparison['aal_norm'] > 0) & (comparison['units_raw_norm'] > 0)
    if mask_norm.sum() > 0:
        ax.scatter(comparison.loc[mask_norm, 'aal_norm'], 
                  comparison.loc[mask_norm, 'units_raw_norm'],
                  alpha=0.5, s=30, edgecolor='black', linewidth=0.5)
        # Add 1:1 line
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label='1:1 line')
        ax.set_xlabel('Normalized AAL')
        ax.set_ylabel('Normalized Units EAD (raw)')
        ax.set_title(f'Normalized: Raw Units vs AAL\nPearson r={pr_raw_norm:.3f}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Bottom right: Normalized scaled units vs AAL
    ax = axes[1, 1]
    mask_norm = (comparison['aal_norm'] > 0) & (comparison['units_scaled_norm'] > 0)
    if mask_norm.sum() > 0:
        ax.scatter(comparison.loc[mask_norm, 'aal_norm'], 
                  comparison.loc[mask_norm, 'units_scaled_norm'],
                  alpha=0.5, s=30, edgecolor='black', linewidth=0.5)
        # Add 1:1 line
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label='1:1 line')
        ax.set_xlabel('Normalized AAL')
        ax.set_ylabel('Normalized Units EAD (scaled)')
        ax.set_title(f'Normalized: Scaled Units vs AAL\nPearson r={pr_scaled_norm:.3f}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = 'analysis_output/units_vs_aal_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {plot_path}")
    plt.close()
    
    # Create additional plot: Normalized AAL vs Normalized Units side-by-side maps would need geopandas
    # Instead, create a focused scatter plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Raw units
    ax = axes[0]
    mask_norm = (comparison['aal_norm'] > 0) & (comparison['units_raw_norm'] > 0)
    if mask_norm.sum() > 0:
        ax.scatter(comparison.loc[mask_norm, 'aal_norm'], 
                  comparison.loc[mask_norm, 'units_raw_norm'],
                  alpha=0.6, s=40, edgecolor='black', linewidth=0.5, c='blue')
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.6, linewidth=2, label='1:1 line (perfect match)')
        ax.set_xlabel('Normalized AAL (Gori et al.)', fontsize=11)
        ax.set_ylabel('Normalized Total Units Affected EAD', fontsize=11)
        ax.set_title(f'Raw Simulation\nPearson r = {pr_raw_norm:.3f}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
    
    # Right: Scaled units
    ax = axes[1]
    mask_norm = (comparison['aal_norm'] > 0) & (comparison['units_scaled_norm'] > 0)
    if mask_norm.sum() > 0:
        ax.scatter(comparison.loc[mask_norm, 'aal_norm'], 
                  comparison.loc[mask_norm, 'units_scaled_norm'],
                  alpha=0.6, s=40, edgecolor='black', linewidth=0.5, c='green')
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.6, linewidth=2, label='1:1 line (perfect match)')
        ax.set_xlabel('Normalized AAL (Gori et al.)', fontsize=11)
        ax.set_ylabel('Normalized Total Units Affected EAD', fontsize=11)
        ax.set_title(f'Scaled Simulation\nPearson r = {pr_scaled_norm:.3f}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
    
    plt.suptitle('Normalized AAL vs Normalized Total Units Affected EAD\n(County-level comparison)', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_path = 'analysis_output/normalized_aal_vs_units.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved normalized comparison plot to: {plot_path}")
    plt.close()
    
    # Create spatial map comparison (like normalized_spatial_comparison.png)
    print("\nCreating spatial maps of normalized AAL vs normalized units...")
    create_normalized_spatial_maps(comparison)
    
    # Save comparison data
    comparison.to_csv('analysis_output/units_aal_comparison.csv', index=False)
    print(f"Saved comparison data to: analysis_output/units_aal_comparison.csv")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("\nComparing UNITS (recovery-relevant metric) vs AAL ($):")
    print("\n1. This is MORE MEANINGFUL for your recovery analysis because:")
    print("   - Recovery focuses on NUMBER of damaged buildings")
    print("   - Not just dollar values")
    print("   - Directly usable for recovery potential modeling")
    
    print("\n2. The normalized correlation tells you:")
    print("   - How well your model captures RELATIVE risk patterns")
    print("   - Independent of magnitude/units differences")
    print("   - High correlation = spatial patterns align well")
    
    print("\n3. Normalized AAL vs Normalized Units comparison is valid because:")
    print("   - Both represent risk metrics ($ vs units)")
    print("   - Normalization removes unit differences")
    print("   - Tests if high-$ counties also have high unit counts")
    print("   - Assumes AAL somewhat correlates with affected units")
    
    if pr_scaled_norm > 0.7:
        print(f"\n✓ Strong correlation (r={pr_scaled_norm:.3f}) confirms your model")
        print("  captures the right spatial distribution of unit-level risk!")
    elif pr_scaled_norm > 0.5:
        print(f"\n✓ Moderate correlation (r={pr_scaled_norm:.3f}) shows reasonable")
        print("  alignment of spatial patterns between units and AAL")
    
    return comparison


def create_normalized_spatial_maps(comparison):
    """Create spatial maps comparing normalized AAL with normalized total units."""
    import geopandas as gpd
    from matplotlib.colors import LogNorm
    from pathlib import Path
    
    # Load shapefile
    shapefile_path = Path("data/US_counties.shp")
    if not shapefile_path.exists():
        print(f"Warning: Shapefile not found. Skipping spatial maps.")
        return
    
    counties = gpd.read_file(shapefile_path)
    if "STATEFP" in counties.columns and "COUNTYFP" in counties.columns:
        counties["fips"] = (counties["STATEFP"].astype(str) + counties["COUNTYFP"].astype(str)).str.zfill(5)
    
    # Filter to study states
    study_states = ['01', '09', '10', '12', '13', '22', '23', '24', '25', '28', 
                    '33', '34', '36', '37', '42', '44', '45', '48', '51']
    counties = counties[counties["STATEFP"].isin(study_states)].copy()
    
    # Compute difference
    comparison['diff_units'] = comparison['units_scaled_norm'] - comparison['aal_norm']
    
    # Determine common color scale for first two panels
    vmin_common = min(
        comparison['aal_norm'][comparison['aal_norm'] > 0].min(),
        comparison['units_scaled_norm'][comparison['units_scaled_norm'] > 0].min()
    )
    vmax_common = 1.0  # Both normalized to max of 1
    
    # Create 3-panel figure: AAL, Units, Difference
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    # Panel 1: AAL normalized
    merged = counties.merge(comparison[['fips', 'aal_norm']], on='fips', how='left').fillna(0)
    merged['aal_norm_nz'] = merged['aal_norm'].replace(0, np.nan)
    if merged['aal_norm_nz'].max() > 0:
        merged.plot(column='aal_norm_nz', ax=axes[0], legend=True, cmap='YlOrRd',
                   norm=LogNorm(vmin=vmin_common, vmax=vmax_common),
                   legend_kwds={'label': 'Normalized value (log scale)', 'shrink': 0.8},
                   edgecolor='black', linewidth=0.1, missing_kwds={'color': 'lightgrey'})
    axes[0].set_title('AAL (Gori et al.) - Normalized', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: Units scaled normalized
    merged = counties.merge(comparison[['fips', 'units_scaled_norm']], on='fips', how='left').fillna(0)
    merged['units_scaled_norm_nz'] = merged['units_scaled_norm'].replace(0, np.nan)
    if merged['units_scaled_norm_nz'].max() > 0:
        merged.plot(column='units_scaled_norm_nz', ax=axes[1], legend=True, cmap='YlOrRd',
                   norm=LogNorm(vmin=vmin_common, vmax=vmax_common),
                   legend_kwds={'label': 'Normalized value (log scale)', 'shrink': 0.8},
                   edgecolor='black', linewidth=0.1, missing_kwds={'color': 'lightgrey'})
    axes[1].set_title('Total Units Affected EAD (Scaled Sim) - Normalized', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    
    # Panel 3: Difference (Units - AAL)
    merged = counties.merge(comparison[['fips', 'diff_units']], on='fips', how='left').fillna(0)
    vmax_diff = max(abs(merged['diff_units'].min()), abs(merged['diff_units'].max()))
    merged.plot(column='diff_units', ax=axes[2], legend=True, cmap='RdBu_r',
               vmin=-vmax_diff, vmax=vmax_diff,
               legend_kwds={'label': 'Difference (Units - AAL)', 'shrink': 0.8},
               edgecolor='black', linewidth=0.1, missing_kwds={'color': 'lightgrey'})
    axes[2].set_title('Difference (Sim Units - AAL)\nRed: Sim > AAL, Blue: AAL > Sim', 
                     fontsize=13, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle('Normalized Spatial Comparison: AAL vs Total Units Affected EAD', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plot_path = 'analysis_output/normalized_spatial_aal_vs_units.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved spatial comparison map to: {plot_path}")
    plt.close()
    
    # Create 3-panel comparison: AAL, Units, Repair Cost (shared colorbar)
    print("Creating 3-panel comparison with shared colorbar...")
    create_three_panel_comparison(comparison)


def create_three_panel_comparison(comparison):
    """Create 3-panel spatial comparison: AAL, Units, Repair Cost with shared colorbar."""
    import geopandas as gpd
    from matplotlib.colors import LogNorm
    from pathlib import Path
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # Load shapefile
    shapefile_path = Path("data/US_counties.shp")
    if not shapefile_path.exists():
        print(f"Warning: Shapefile not found. Skipping 3-panel map.")
        return
    
    counties = gpd.read_file(shapefile_path)
    if "STATEFP" in counties.columns and "COUNTYFP" in counties.columns:
        counties["fips"] = (counties["STATEFP"].astype(str) + counties["COUNTYFP"].astype(str)).str.zfill(5)
    
    # Filter to study states
    study_states = ['01', '09', '10', '12', '13', '22', '23', '24', '25', '28', 
                    '33', '34', '36', '37', '42', '44', '45', '48', '51']
    counties = counties[counties["STATEFP"].isin(study_states)].copy()
    
    # Load repair cost normalized data from the AAL comparison
    aal_comparison = pd.read_csv('analysis_output/aal_comparison.csv')
    aal_comparison['fips'] = aal_comparison['fips'].astype(str).str.zfill(5)
    aal_comparison['repair_scaled_norm'] = aal_comparison['sim_repair_ead_scaled'] / aal_comparison['sim_repair_ead_scaled'].max()
    
    # Merge repair cost into comparison
    comparison = comparison.merge(
        aal_comparison[['fips', 'repair_scaled_norm']], 
        on='fips', 
        how='left'
    ).fillna(0)
    
    # Determine common color scale
    vmin_common = max(
        1e-7,  # Cap lower bound at 10^-7
        min(
            comparison['aal_norm'][comparison['aal_norm'] > 0].min(),
            comparison['units_scaled_norm'][comparison['units_scaled_norm'] > 0].min(),
            comparison['repair_scaled_norm'][comparison['repair_scaled_norm'] > 0].min()
        )
    )
    vmax_common = 1.0
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    
    # Panel 1: AAL
    merged = counties.merge(comparison[['fips', 'aal_norm']], on='fips', how='left').fillna(0)
    merged['aal_norm_nz'] = merged['aal_norm'].replace(0, np.nan)
    if merged['aal_norm_nz'].max() > 0:
        merged.plot(column='aal_norm_nz', ax=axes[0], legend=False, cmap='YlOrRd',
                   norm=LogNorm(vmin=vmin_common, vmax=vmax_common),
                   edgecolor='black', linewidth=0.1, missing_kwds={'color': 'lightgrey'})
    axes[0].set_title('AAL\n(Gori et al.)', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: Total Units
    merged = counties.merge(comparison[['fips', 'units_scaled_norm']], on='fips', how='left').fillna(0)
    merged['units_scaled_norm_nz'] = merged['units_scaled_norm'].replace(0, np.nan)
    if merged['units_scaled_norm_nz'].max() > 0:
        merged.plot(column='units_scaled_norm_nz', ax=axes[1], legend=False, cmap='YlOrRd',
                   norm=LogNorm(vmin=vmin_common, vmax=vmax_common),
                   edgecolor='black', linewidth=0.1, missing_kwds={'color': 'lightgrey'})
    axes[1].set_title('Total Units Affected EAD\n(Scaled Simulation)', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    
    # Panel 3: Repair Cost EAD
    merged = counties.merge(comparison[['fips', 'repair_scaled_norm']], on='fips', how='left').fillna(0)
    merged['repair_scaled_norm_nz'] = merged['repair_scaled_norm'].replace(0, np.nan)
    if merged['repair_scaled_norm_nz'].max() > 0:
        im = merged.plot(column='repair_scaled_norm_nz', ax=axes[2], legend=False, cmap='YlOrRd',
                   norm=LogNorm(vmin=vmin_common, vmax=vmax_common),
                   edgecolor='black', linewidth=0.1, missing_kwds={'color': 'lightgrey'})
    axes[2].set_title('Repair Cost EAD\n(Scaled Simulation)', fontsize=13, fontweight='bold')
    axes[2].axis('off')
    
    # Add single colorbar on the right
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=LogNorm(vmin=vmin_common, vmax=vmax_common))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Normalized Value (log scale)', fontsize=11, fontweight='bold')
    
    plt.suptitle('Normalized Risk Metrics: AAL vs Simulation Outputs', 
                 fontsize=14, fontweight='bold', x=0.5, y=0.98)
    
    plot_path = 'analysis_output/three_panel_normalized_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved 3-panel comparison map to: {plot_path}")
    plt.close()


def main():
    from pathlib import Path
    
    events_dir = Path("impacts_out/by_event")
    
    # Compute total units EAD
    raw_units, scaled_units = compute_total_units_ead(events_dir)
    
    # Load AAL comparison
    aal_comparison = pd.read_csv('analysis_output/aal_comparison.csv')
    
    # Compare units with AAL
    compare_units_with_aal(raw_units, scaled_units, aal_comparison)


if __name__ == "__main__":
    main()
