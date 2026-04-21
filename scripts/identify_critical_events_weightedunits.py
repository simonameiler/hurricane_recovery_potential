"""
Identify the 5 events with highest percentage of counties in Critical Vulnerability quadrant
using the weighted damage units metric.
"""

import pandas as pd
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = BASE_DIR / 'analysis_output'

# Load quadrant assignments
quadrant_file = OUTPUT_DIR / 'event_county_quadrants_weightedunits.csv'
df = pd.read_csv(quadrant_file)

print(f"Loaded {len(df)} event-county pairs")
print(f"Unique events: {df['event'].nunique()}")
print(f"Unique counties: {df['fips'].nunique()}")

# Calculate percentage of counties in High Damage / Low Capacity (Critical Vulnerability) for each event
event_stats = []

for event_id in df['event'].unique():
    event_data = df[df['event'] == event_id]
    
    total_counties = len(event_data)
    critical_counties = len(event_data[event_data['quadrant'] == 'High Damage / Low Capacity'])
    critical_pct = 100 * critical_counties / total_counties
    
    total_damage = event_data['weighted_damage_units'].sum()
    total_capacity = event_data['construction_capacity'].sum()
    median_recovery = event_data['recovery_months'].median()
    
    # Get state distribution
    states = event_data['fips'].astype(str).str[:2].value_counts()
    top_states = ', '.join([f"{s}:{c}" for s, c in states.head(3).items()])
    
    event_stats.append({
        'event': event_id,
        'total_counties': total_counties,
        'critical_counties': critical_counties,
        'critical_pct': critical_pct,
        'total_weighted_damage': total_damage,
        'total_capacity': total_capacity,
        'median_recovery_months': median_recovery,
        'top_states': top_states
    })

event_df = pd.DataFrame(event_stats)

# Sort by critical percentage
event_df = event_df.sort_values('critical_pct', ascending=False)

# Get top 5
top5 = event_df.head(5)

print("\n=== TOP 5 CRITICAL VULNERABILITY EVENTS (WEIGHTED) ===")
print(top5[['event', 'critical_counties', 'total_counties', 'critical_pct', 
            'median_recovery_months', 'top_states']])

# Save results
output_file = OUTPUT_DIR / 'critical_events_summary_weightedunits.csv'
top5.to_csv(output_file, index=False)
print(f"\nSaved top 5 to {output_file}")

# Also save full ranking
full_file = OUTPUT_DIR / 'all_events_critical_ranking_weightedunits.csv'
event_df.to_csv(full_file, index=False)
print(f"Saved full ranking to {full_file}")

# Print detailed breakdown for top 5
print("\n=== DETAILED BREAKDOWN ===")
for idx, row in top5.iterrows():
    event_id = row['event']
    event_data = df[df['event'] == event_id]
    
    print(f"\nEvent {event_id}:")
    print(f"  Critical Vulnerability: {row['critical_pct']:.1f}% ({row['critical_counties']}/{row['total_counties']})")
    print(f"  Median recovery: {row['median_recovery_months']:.0f} months")
    print(f"  Total weighted damage: {row['total_weighted_damage']:,.0f}")
    print(f"  Quadrant distribution:")
    for quadrant in event_data['quadrant'].unique():
        count = len(event_data[event_data['quadrant'] == quadrant])
        pct = 100 * count / row['total_counties']
        print(f"    {quadrant}: {count} ({pct:.1f}%)")
