"""
Create 3-panel maps for the 5 critical events that predominantly hit
the Critical Vulnerability quadrant.

Uses the same format as the existing single_event_*_3panel.png figures.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"
DATA_DIR = BASE_DIR / "data"

# The 5 critical events
critical_events = {
    803.0: "Event 803 (Louisiana)",
    1846.0: "Event 1846 (Florida)", 
    4094.0: "Event 4094 (Florida)",
    4442.0: "Event 4442 (Florida/Georgia)",
    4967.0: "Event 4967 (Louisiana)"
}

print("="*80)
print("CREATING 3-PANEL MAPS FOR CRITICAL EVENTS")
print("="*80)

# Load county shapefile
print("\nLoading county boundaries...")
counties = gpd.read_file(DATA_DIR / "US_counties.shp")
counties['fips'] = (counties['STATEFP'].astype(str).str.zfill(2) + 
                   counties['COUNTYFP'].astype(str).str.zfill(3))

# Load event-county quadrant data
print("Loading event-county data...")
df = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants.csv')
df['event'] = pd.to_numeric(df['event'], errors='coerce')
df['fips'] = df['fips'].astype(str).str.zfill(5)

# Filter to coastal states
coastal_states = ['01', '12', '13', '22', '28', '37', '45', '48', '51',  # Gulf/Atlantic
                 '09', '10', '24', '25', '33', '34', '36', '44']  # Northeast
coastal_counties = counties[counties['STATEFP'].isin(coastal_states)].copy()

# Create maps for each critical event
for event_id, event_name in critical_events.items():
    print(f"\n{'='*80}")
    print(f"Processing {event_name}")
    print('='*80)
    
    # Get event data
    event_data = df[df['event'] == event_id].copy()
    
    if len(event_data) == 0:
        print(f"No data found for event {event_id}")
        continue
    
    print(f"Counties affected: {len(event_data)}")
    print(f"Total damage: {event_data['damage_units'].sum():,.0f} units")
    print(f"Median recovery: {event_data['recovery_months'].median():.0f} months")
    
    # Merge with counties
    gdf = coastal_counties.merge(
        event_data[['fips', 'damage_units', 'construction_capacity', 'recovery_months']],
        left_on='fips',
        right_on='fips',
        how='left'
    )
    
    # Create 3-panel map
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Hurricane Event Analysis: {event_name}', 
                fontsize=14, fontweight='bold')
    
    # Panel 1: Damage (log scale)
    ax1 = axes[0]
    damage_data = gdf['damage_units'].dropna()
    if len(damage_data) > 0 and damage_data.max() > 0:
        vmin_damage = max(damage_data[damage_data > 0].min(), 0.1)
        vmax_damage = damage_data.max()
        gdf.plot(
            column='damage_units',
            cmap='YlOrRd',
            linewidth=0.1,
            edgecolor='0.5',
            legend=True,
            ax=ax1,
            norm=LogNorm(vmin=vmin_damage, vmax=vmax_damage),
            legend_kwds={'label': 'Units damaged (log scale)', 'shrink': 0.7},
            missing_kwds={'color': '#f0f0f0'}
        )
    else:
        gdf.plot(ax=ax1, color='#f0f0f0', linewidth=0.1, edgecolor='0.5')
    
    ax1.set_title('Total Damage', fontsize=12, fontweight='bold')
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
    
    # Save with event ID in filename
    filename = f"critical_event_{int(event_id)}_3panel.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print(f"\nAll maps saved to: {OUTPUT_DIR}")
print("\nFiles created:")
for event_id in critical_events.keys():
    print(f"  - critical_event_{int(event_id)}_3panel.png")
