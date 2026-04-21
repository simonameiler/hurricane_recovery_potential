"""
Create 3-panel maps for the 5 critical vulnerability events using weighted damage metric.
Panel 1: Weighted damage units
Panel 2: Construction capacity
Panel 3: Recovery time
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'analysis_output'

# Load critical events summary
critical_events_file = OUTPUT_DIR / 'critical_events_summary_weightedunits.csv'
critical_events = pd.read_csv(critical_events_file)
event_ids = critical_events['event'].tolist()

print(f"Creating maps for {len(event_ids)} critical events: {event_ids}")

# Load quadrant data
quadrant_file = OUTPUT_DIR / 'event_county_quadrants_weightedunits.csv'
df = pd.read_csv(quadrant_file)

# Load county shapefile
counties = gpd.read_file(DATA_DIR / 'US_counties.shp')

# Filter to 19 coastal states (using FIPS state codes)
coastal_state_fips = ['01', '09', '10', '12', '13', '22', '23', '24', '25',
                      '28', '33', '34', '36', '37', '44', '45', '48', '51', '42']
counties = counties[counties['STATEFP'].isin(coastal_state_fips)].copy()

# Ensure FIPS is string with zero-padding (combine STATEFP + COUNTYFP)
counties['FIPS'] = counties['STATEFP'] + counties['COUNTYFP']

# Create maps for each critical event
for event_id in event_ids:
    print(f"\nCreating map for event {event_id}...")
    
    # Get data for this event
    event_data = df[df['event'] == event_id].copy()
    event_data['fips'] = event_data['fips'].astype(str).str.zfill(5)
    
    # Merge with counties
    counties_merged = counties.merge(event_data, left_on='FIPS', right_on='fips', how='left')
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Weighted Damage Units
    ax = axes[0]
    counties_merged.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.3)
    
    # Plot counties with damage (log scale)
    has_damage = counties_merged[counties_merged['weighted_damage_units'].notna() & 
                                  (counties_merged['weighted_damage_units'] > 0)]
    if len(has_damage) > 0:
        vmin = max(has_damage['weighted_damage_units'].min(), 1)
        vmax = has_damage['weighted_damage_units'].max()
        
        has_damage.plot(ax=ax, column='weighted_damage_units',
                       cmap='YlOrRd', edgecolor='white', linewidth=0.3,
                       legend=True, legend_kwds={'label': 'Weighted Damage Units', 'shrink': 0.8},
                       norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
    
    ax.set_xlim(-100, -65)
    ax.set_ylim(24, 48)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Weighted Damage Units', fontsize=12, pad=10)
    
    # Panel 2: Construction Capacity
    ax = axes[1]
    counties_merged.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.3)
    
    has_capacity = counties_merged[counties_merged['construction_capacity'].notna() & 
                                    (counties_merged['construction_capacity'] > 0)]
    if len(has_capacity) > 0:
        vmin = max(has_capacity['construction_capacity'].min(), 0.1)
        vmax = has_capacity['construction_capacity'].max()
        
        has_capacity.plot(ax=ax, column='construction_capacity',
                         cmap='Blues', edgecolor='white', linewidth=0.3,
                         legend=True, legend_kwds={'label': 'Permits/Month', 'shrink': 0.8},
                         norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
    
    ax.set_xlim(-100, -65)
    ax.set_ylim(24, 48)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Construction Capacity', fontsize=12, pad=10)
    
    # Panel 3: Recovery Time
    ax = axes[2]
    counties_merged.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.3)
    
    has_recovery = counties_merged[counties_merged['recovery_months'].notna() & 
                                    (counties_merged['recovery_months'] > 0)]
    if len(has_recovery) > 0:
        vmin = max(has_recovery['recovery_months'].min(), 0.1)
        vmax = has_recovery['recovery_months'].max()
        
        has_recovery.plot(ax=ax, column='recovery_months',
                         cmap='RdPu', edgecolor='white', linewidth=0.3,
                         legend=True, legend_kwds={'label': 'Recovery (months)', 'shrink': 0.8},
                         norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
    
    ax.set_xlim(-100, -65)
    ax.set_ylim(24, 48)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Recovery Time', fontsize=12, pad=10)
    
    # Overall title
    critical_pct = critical_events[critical_events['event'] == event_id]['critical_pct'].values[0]
    median_recovery = event_data['recovery_months'].median()
    plt.suptitle(f'Event {event_id}: Critical Vulnerability = {critical_pct:.1f}%, Median Recovery = {median_recovery:.0f} months',
                 fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    # Save
    output_file = OUTPUT_DIR / f'critical_event_{event_id}_weightedunits_3panel.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")

print("\n=== All critical event maps created ===")
