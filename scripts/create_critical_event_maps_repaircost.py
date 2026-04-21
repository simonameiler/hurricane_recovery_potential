#!/usr/bin/env python3
"""
Create 3-panel maps for the 5 critical events (repair cost version).
Shows contrast using repair cost-weighted damage metric.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def create_event_map(event_id, gdf_all, event_data, output_dir='/Users/simonameiler/Documents/work/03_code/repos/hurricane_recovery_potential/analysis_output'):
    """Create 3-panel map for a single event"""
    
    # Get event summary info
    event_summary = event_data.groupby('event').agg({
        'damage_repair_cost': 'sum',
        'recovery_months': 'median',
        'fips': 'count'
    }).loc[event_id]
    
    # Determine state(s)
    event_counties = event_data[event_data['event'] == event_id].copy()
    event_counties['state'] = event_counties['fips'].astype(str).str.zfill(5).str[:2]
    state_map = {
        '01': 'AL', '09': 'CT', '10': 'DE', '12': 'FL', '13': 'GA', 
        '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '28': 'MS',
        '33': 'NH', '34': 'NJ', '36': 'NY', '37': 'NC', '44': 'RI',
        '45': 'SC', '48': 'TX', '51': 'VA'
    }
    event_counties['state_name'] = event_counties['state'].map(state_map)
    states = event_counties['state_name'].value_counts()
    
    if len(states) > 1:
        state_label = f"{states.index[0]}/{states.index[1]}"
    else:
        state_label = states.index[0]
    
    event_name = f"Event {event_id} ({state_label})"
    
    # Merge event data with all counties
    gdf = gdf_all.merge(event_counties, on='fips', how='left')
    
    # Create 3-panel map
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Hurricane Event Analysis: {event_name}', 
                fontsize=14, fontweight='bold')
    
    # Panel 1: Repair Cost (log scale)
    ax1 = axes[0]
    damage_data = gdf['damage_repair_cost'].dropna()
    if len(damage_data) > 0 and damage_data.max() > 0:
        vmin_damage = max(damage_data[damage_data > 0].min(), 1)
        vmax_damage = damage_data.max()
        gdf.plot(
            column='damage_repair_cost',
            cmap='YlOrRd',
            linewidth=0.1,
            edgecolor='0.5',
            legend=True,
            ax=ax1,
            norm=LogNorm(vmin=vmin_damage, vmax=vmax_damage),
            legend_kwds={'label': 'Repair Cost $ (log scale)', 'shrink': 0.7},
            missing_kwds={'color': '#f0f0f0'}
        )
    else:
        gdf.plot(ax=ax1, color='#f0f0f0', linewidth=0.1, edgecolor='0.5')
    
    ax1.set_title('Total Damage (Repair Cost)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Capacity (log scale)
    ax2 = axes[1]
    capacity_data = gdf['construction_capacity'].dropna()
    if len(capacity_data) > 0 and capacity_data.max() > 0:
        vmin_capacity = max(capacity_data[capacity_data > 0].min(), 0.1)
        vmax_capacity = capacity_data.max()
        gdf.plot(
            column='construction_capacity',
            cmap='Greens',
            linewidth=0.1,
            edgecolor='0.5',
            legend=True,
            ax=ax2,
            norm=LogNorm(vmin=vmin_capacity, vmax=vmax_capacity),
            legend_kwds={'label': 'Permits/month (log scale)', 'shrink': 0.7},
            missing_kwds={'color': '#f0f0f0'}
        )
    else:
        gdf.plot(ax=ax2, color='#f0f0f0', linewidth=0.1, edgecolor='0.5')
    
    ax2.set_title('Construction Capacity', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Panel 3: Recovery Potential (log scale)
    ax3 = axes[2]
    recovery_data = gdf['recovery_months'].dropna()
    if len(recovery_data) > 0 and recovery_data.max() > 0:
        vmin_recovery = max(recovery_data[recovery_data > 0].min(), 0.1)
        vmax_recovery = recovery_data.max()
        gdf.plot(
            column='recovery_months',
            cmap='Purples',
            linewidth=0.1,
            edgecolor='0.5',
            legend=True,
            ax=ax3,
            norm=LogNorm(vmin=vmin_recovery, vmax=vmax_recovery),
            legend_kwds={'label': 'Months (log scale)', 'shrink': 0.7},
            missing_kwds={'color': '#f0f0f0'}
        )
    else:
        gdf.plot(ax=ax3, color='#f0f0f0', linewidth=0.1, edgecolor='0.5')
    
    ax3.set_title('Recovery Potential', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_file = f'{output_dir}/critical_event_{event_id}_repaircost_3panel.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file, event_summary

def main():
    print("=" * 80)
    print("CREATING 3-PANEL MAPS FOR CRITICAL EVENTS (REPAIR COST VERSION)")
    print("=" * 80)
    
    # Load county boundaries
    print("\nLoading county boundaries...")
    gdf_counties = gpd.read_file('/Users/simonameiler/Documents/work/03_code/repos/hurricane_recovery_potential/data/US_counties.shp')
    gdf_counties['fips'] = gdf_counties['GEOID'].astype(str).str.zfill(5)
    
    # Filter to only 19 coastal states
    coastal_states = ['01', '09', '10', '12', '13', '22', '23', '24', '25', '28', 
                     '33', '34', '36', '37', '44', '45', '48', '51', '42']
    gdf_counties['state_fips'] = gdf_counties['fips'].str[:2]
    gdf_counties = gdf_counties[gdf_counties['state_fips'].isin(coastal_states)]
    
    # Load event-county data
    print("Loading event-county data...")
    df = pd.read_csv('/Users/simonameiler/Documents/work/03_code/repos/hurricane_recovery_potential/analysis_output/event_county_quadrants_repaircost.csv')
    df['fips'] = df['fips'].astype(str).str.zfill(5)
    
    # Top 5 critical events (by median recovery)
    critical_events = [2082, 1208, 803, 1984, 4967]
    
    for event_id in critical_events:
        print("\n" + "=" * 80)
        event_data = df[df['event'] == event_id]
        
        # Get state label
        event_data_copy = event_data.copy()
        event_data_copy['state'] = event_data_copy['fips'].astype(str).str.zfill(5).str[:2]
        state_map = {
            '01': 'Alabama', '12': 'Florida', '13': 'Georgia', '22': 'Louisiana',
            '28': 'Mississippi', '37': 'North Carolina', '45': 'South Carolina',
            '48': 'Texas'
        }
        event_data_copy['state_name'] = event_data_copy['state'].map(state_map)
        states = event_data_copy['state_name'].value_counts()
        state_label = states.index[0] if len(states) == 1 else f"{states.index[0]}/{states.index[1]}"
        
        print(f"Processing Event {event_id} ({state_label})")
        print("=" * 80)
        
        total_repair_cost = event_data['damage_repair_cost'].sum()
        num_counties = len(event_data)
        median_recovery = event_data['recovery_months'].median()
        
        print(f"Counties affected: {num_counties}")
        print(f"Total repair cost: ${int(total_repair_cost):,}")
        print(f"Median recovery: {int(median_recovery)} months")
        
        output_file, _ = create_event_map(event_id, gdf_counties, df)
        print(f"Saved: {output_file.split('/')[-1]}")
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nAll maps saved to: /Users/simonameiler/Documents/work/03_code/repos/hurricane_recovery_potential/analysis_output")
    print("\nFiles created:")
    for event_id in critical_events:
        print(f"  - critical_event_{event_id}_repaircost_3panel.png")

if __name__ == '__main__':
    main()
