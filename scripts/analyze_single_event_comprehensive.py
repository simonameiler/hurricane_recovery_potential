"""
Single Event Analysis with Spatial Maps + Damage-Recovery Scatter Plot

Creates 4-panel visualization for a specific event:
- Top row: 3 spatial maps (damage, capacity, recovery)
- Bottom row: Scatter plot (damage vs recovery) to identify patterns/clustering
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm
import numpy as np
from pathlib import Path
import json

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"
DATA_DIR = BASE_DIR / "data"
RECOVERY_DIR = DATA_DIR / "recovery_potential_per_scenario"

def load_event_data(event_id):
    """Load data for a specific event from both absolute and normalized datasets"""
    
    # Load from normalized dataset
    df_norm = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_fully_normalized.csv')
    event_data_norm = df_norm[df_norm['event'] == int(event_id)].copy()
    
    # Load from absolute dataset
    df_abs = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_weightedunits.csv')
    event_data_abs = df_abs[df_abs['event'] == int(event_id)].copy()
    
    if len(event_data_norm) == 0:
        print(f"No data found for event {event_id}")
        return None
    
    # Merge both datasets to get both quadrant classifications
    event_data_norm['fips'] = event_data_norm['fips'].astype(str).str.zfill(5)
    event_data_abs['fips'] = event_data_abs['fips'].astype(str).str.zfill(5)
    
    # Merge absolute quadrant into normalized data
    event_data = event_data_norm.merge(
        event_data_abs[['fips', 'quadrant']],
        on='fips',
        suffixes=('_normalized', '_absolute'),
        how='left'
    )
    
    # Rename for clarity
    event_data.rename(columns={
        'quadrant_normalized': 'quadrant_norm',
        'quadrant_absolute': 'quadrant_abs'
    }, inplace=True)
    
    return event_data

def create_event_analysis(event_id, output_suffix=""):
    """
    Create comprehensive analysis for a single event
    """
    
    print(f"\n{'='*100}")
    print(f"ANALYZING EVENT: {event_id}")
    print(f"{'='*100}")
    
    # Load event data
    event_data = load_event_data(event_id)
    if event_data is None:
        return
    
    print(f"\nAffected counties: {len(event_data)}")
    print(f"Total weighted damage: {event_data['weighted_damage_units'].sum():,.0f} units")
    print(f"Mean recovery time: {event_data['recovery_months'].mean() / 12:.1f} years")
    
    # Load county boundaries
    counties_shp = gpd.read_file(DATA_DIR / 'US_counties.shp')
    coastal_state_fips = ['01', '09', '10', '12', '13', '22', '23', '24', '25',
                          '28', '33', '34', '36', '37', '44', '45', '48', '51', '42']
    counties_shp = counties_shp[counties_shp['STATEFP'].isin(coastal_state_fips)].copy()
    counties_shp['FIPS'] = counties_shp['STATEFP'] + counties_shp['COUNTYFP']
    
    # Merge event data with geometries
    counties_merged = counties_shp.merge(event_data, left_on='FIPS', right_on='fips', how='left')
    
    # ========================================================================
    # FIGURE 1: Spatial Maps (a, b, c)
    # ========================================================================
    fig1 = plt.figure(figsize=(24, 8))
    
    # ========== PANEL 1: Weighted Damage Units ==========
    ax1 = fig1.add_subplot(1, 3, 1)
    
    # Plot affected counties
    affected = counties_merged[counties_merged['weighted_damage_units'].notna()]
    if len(affected) > 0:
        affected.plot(
            column='weighted_damage_units',
            cmap='YlOrRd',
            norm=LogNorm(vmin=affected['weighted_damage_units'].min(), 
                        vmax=affected['weighted_damage_units'].max()),
            edgecolor='black',
            linewidth=0.5,
            ax=ax1,
            legend=True,
            legend_kwds={'label': 'Weighted Damage Units', 'shrink': 0.6}
        )
    
    # Plot unaffected counties in gray
    unaffected = counties_merged[counties_merged['weighted_damage_units'].isna()]
    if len(unaffected) > 0:
        unaffected.plot(ax=ax1, color='lightgray', edgecolor='white', linewidth=0.3)
    
    ax1.set_xlim(-100, -65)
    ax1.set_ylim(24, 48)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('(a) Weighted Damage Units', fontsize=14, fontweight='bold', pad=10)
    
    # ========== PANEL 2: Construction Capacity ==========
    ax2 = fig1.add_subplot(1, 3, 2)
    
    if len(affected) > 0:
        affected.plot(
            column='construction_capacity',
            cmap='Greens',
            norm=LogNorm(vmin=max(affected['construction_capacity'].min(), 0.1),
                        vmax=affected['construction_capacity'].max()),
            edgecolor='black',
            linewidth=0.5,
            ax=ax2,
            legend=True,
            legend_kwds={'label': 'Permits/Month', 'shrink': 0.6}
        )
    
    if len(unaffected) > 0:
        unaffected.plot(ax=ax2, color='lightgray', edgecolor='white', linewidth=0.3)
    
    ax2.set_xlim(-100, -65)
    ax2.set_ylim(24, 48)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('(b) Construction Capacity', fontsize=14, fontweight='bold', pad=10)
    
    # ========== PANEL 3: Recovery Time ==========
    ax3 = fig1.add_subplot(1, 3, 3)
    
    if len(affected) > 0:
        # Convert to years for display
        affected_copy = affected.copy()
        affected_copy['recovery_years'] = affected_copy['recovery_months'] / 12
        
        affected_copy.plot(
            column='recovery_years',
            cmap='RdPu',
            norm=LogNorm(vmin=max(affected_copy['recovery_years'].min(), 0.01),
                        vmax=affected_copy['recovery_years'].max()),
            edgecolor='black',
            linewidth=0.5,
            ax=ax3,
            legend=True,
            legend_kwds={'label': 'Recovery Time (years)', 'shrink': 0.6}
        )
    
    if len(unaffected) > 0:
        unaffected.plot(ax=ax3, color='lightgray', edgecolor='white', linewidth=0.3)
    
    ax3.set_xlim(-100, -65)
    ax3.set_ylim(24, 48)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('(c) Recovery Time', fontsize=14, fontweight='bold', pad=10)
    
    # Overall title for Figure 1
    fig1.suptitle(f'Event {event_id}: Spatial Distribution of Impact and Recovery Potential', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save Figure 1
    output_file1 = OUTPUT_DIR / f'event_{event_id}_spatial_maps{output_suffix}.png'
    plt.savefig(output_file1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Spatial maps saved to: {output_file1}")
    
    # ========================================================================
    # FIGURE 2: Damage-Recovery Scatter Plots (d, e)
    # ========================================================================
    fig2, (ax4, ax5) = plt.subplots(1, 2, figsize=(18, 8))
    
    # ========== PANEL 4: Absolute Damage vs Recovery Scatter Plot ==========
    
    # Color by quadrant
    quadrant_colors = {
        'High Damage / High Capacity': '#2c7bb6',
        'High Damage / Low Capacity': '#d7191c',
        'Low Damage / High Capacity': '#1a9850',
        'Low Damage / Low Capacity': '#fdae61'
    }
    
    for quadrant, color in quadrant_colors.items():
        subset = event_data[event_data['quadrant_abs'] == quadrant]
        if len(subset) > 0:
            ax4.scatter(subset['recovery_months'] / 12, subset['weighted_damage_units'],
                       alpha=0.7, s=100, c=color, label=quadrant,
                       edgecolors='black', linewidth=0.5)
    
    # Add county labels for top 3 by damage (fewer to avoid clutter)
    top3_damage = event_data.nlargest(3, 'weighted_damage_units')
    for _, row in top3_damage.iterrows():
        county_name = counties_merged[counties_merged['fips'] == row['fips']]['NAME'].values
        if len(county_name) > 0:
            ax4.annotate(county_name[0], 
                        xy=(row['recovery_months']/12, row['weighted_damage_units']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.7)
    
    # Use log scale
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Recovery Time (years)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Absolute Damage (Weighted Units)', fontsize=13, fontweight='bold')
    ax4.set_title('(d) Absolute Damage-Recovery Relationship', 
                 fontsize=14, fontweight='bold', pad=15)
    ax4.legend(loc='best', fontsize=9, framealpha=0.95)
    ax4.grid(True, alpha=0.3, which='both')
    
    # Add correlation info
    if len(event_data) > 1:
        corr = event_data[['weighted_damage_units', 'recovery_months']].corr().iloc[0, 1]
        ax4.text(0.02, 0.98, f'Correlation: r = {corr:.3f}\nColors = Absolute Quadrant',
                transform=ax4.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== PANEL 5: Normalized Damage vs Recovery Scatter Plot ==========
    # Color by NORMALIZED quadrants
    
    for quadrant, color in quadrant_colors.items():
        subset = event_data[event_data['quadrant_norm'] == quadrant]
        if len(subset) > 0:
            ax5.scatter(subset['recovery_months'] / 12, subset['pct_housing_damaged'],
                       alpha=0.7, s=100, c=color, label=quadrant,
                       edgecolors='black', linewidth=0.5)
    
    # Add county labels for top 3 by % damaged
    top3_pct = event_data.nlargest(3, 'pct_housing_damaged')
    for _, row in top3_pct.iterrows():
        county_name = counties_merged[counties_merged['fips'] == row['fips']]['NAME'].values
        if len(county_name) > 0:
            ax5.annotate(county_name[0], 
                        xy=(row['recovery_months']/12, row['pct_housing_damaged']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.7)
    
    # Use log scale
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_xlabel('Recovery Time (years)', fontsize=13, fontweight='bold')
    ax5.set_ylabel('Proportional Damage (% of Housing)', fontsize=13, fontweight='bold')
    ax5.set_title('(e) Normalized Damage-Recovery Relationship', 
                 fontsize=14, fontweight='bold', pad=15)
    ax5.legend(loc='best', fontsize=9, framealpha=0.95)
    ax5.grid(True, alpha=0.3, which='both')
    
    # Add correlation info
    if len(event_data) > 1:
        corr_norm = event_data[['pct_housing_damaged', 'recovery_months']].corr().iloc[0, 1]
        ax5.text(0.02, 0.98, f'Correlation: r = {corr_norm:.3f}\nColors = Normalized Quadrant',
                transform=ax5.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Overall title for Figure 2
    fig2.suptitle(f'Event {event_id}: Damage-Recovery Relationships (Absolute vs. Normalized)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save Figure 2
    output_file2 = OUTPUT_DIR / f'event_{event_id}_scatter_plots{output_suffix}.png'
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Scatter plots saved to: {output_file2}")
    
    # Print statistics
    print(f"\n{'='*100}")
    print(f"EVENT {event_id} STATISTICS")
    print(f"{'='*100}")
    
    print(f"\nDamage Distribution:")
    print(f"  Min: {event_data['weighted_damage_units'].min():,.0f} units")
    print(f"  Median: {event_data['weighted_damage_units'].median():,.0f} units")
    print(f"  Max: {event_data['weighted_damage_units'].max():,.0f} units")
    print(f"  Total: {event_data['weighted_damage_units'].sum():,.0f} units")
    
    print(f"\nRecovery Time Distribution:")
    print(f"  Min: {event_data['recovery_months'].min() / 12:.1f} years")
    print(f"  Median: {event_data['recovery_months'].median() / 12:.1f} years")
    print(f"  Max: {event_data['recovery_months'].max() / 12:.1f} years")
    print(f"  Counties >50 years: {(event_data['recovery_months'] > 600).sum()} ({100*(event_data['recovery_months'] > 600).sum()/len(event_data):.1f}%)")
    
    print(f"\nQuadrant Distribution (Normalized):")
    for quadrant in quadrant_colors.keys():
        count = (event_data['quadrant_norm'] == quadrant).sum()
        pct = 100 * count / len(event_data)
        print(f"  {quadrant}: {count} counties ({pct:.1f}%)")
    
    print(f"\n{'='*100}\n")
    
    return event_data


if __name__ == "__main__":
    # Find top events by total damage
    df = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_fully_normalized.csv')
    event_totals = df.groupby('event')['weighted_damage_units'].sum().sort_values(ascending=False)
    
    print("Top 10 events by total weighted damage:")
    for i, (event, damage) in enumerate(event_totals.head(10).items(), 1):
        n_counties = len(df[df['event'] == event])
        print(f"  {i}. Event {event}: {damage:,.0f} units ({n_counties} counties)")
    
    # Events to analyze (user specified)
    events_to_analyze = [765, 3907, 350, 4967, 2082, 3713]
    
    print(f"\n{'='*100}")
    print(f"Analyzing {len(events_to_analyze)} specified events")
    print(f"{'='*100}\n")
    
    # Analyze each specified event
    for event_id in events_to_analyze:
        create_event_analysis(event_id)
