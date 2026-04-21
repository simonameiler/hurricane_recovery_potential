"""
Identify the 5 events with highest total weighted damage units.
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

# Calculate total damage for each event
event_stats = []

for event_id in df['event'].unique():
    event_data = df[df['event'] == event_id]
    
    total_damage = event_data['weighted_damage_units'].sum()
    total_capacity = event_data['construction_capacity'].sum()
    total_counties = len(event_data)
    median_recovery = event_data['recovery_months'].median()
    
    # Quadrant breakdown
    quadrant_counts = event_data['quadrant'].value_counts()
    critical_pct = 100 * quadrant_counts.get('High Damage / Low Capacity', 0) / total_counties
    
    # State distribution
    states = event_data['fips'].astype(str).str[:2].value_counts()
    top_states = ', '.join([f"{s}:{c}" for s, c in states.head(3).items()])
    
    event_stats.append({
        'event': event_id,
        'total_weighted_damage': total_damage,
        'total_counties': total_counties,
        'total_capacity': total_capacity,
        'median_recovery_months': median_recovery,
        'critical_pct': critical_pct,
        'top_states': top_states
    })

event_df = pd.DataFrame(event_stats)

# Sort by total damage
event_df = event_df.sort_values('total_weighted_damage', ascending=False)

# Get top 5
top5 = event_df.head(5)

print("\n=== TOP 5 HIGHEST WEIGHTED DAMAGE EVENTS ===")
print(top5[['event', 'total_weighted_damage', 'total_counties', 
            'median_recovery_months', 'critical_pct', 'top_states']])

# Save results
output_file = OUTPUT_DIR / 'highest_damage_events_summary_weightedunits.csv'
top5.to_csv(output_file, index=False)
print(f"\nSaved top 5 to {output_file}")

# Print detailed breakdown
print("\n=== DETAILED BREAKDOWN ===")
for idx, row in top5.iterrows():
    event_id = row['event']
    event_data = df[df['event'] == event_id]
    
    print(f"\nEvent {event_id}:")
    print(f"  Total weighted damage: {row['total_weighted_damage']:,.0f}")
    print(f"  Affected counties: {row['total_counties']}")
    print(f"  Total capacity: {row['total_capacity']:.1f} permits/month")
    print(f"  Median recovery: {row['median_recovery_months']:.0f} months ({row['median_recovery_months']/12:.1f} years)")
    print(f"  Critical Vulnerability: {row['critical_pct']:.1f}%")
    print(f"  Quadrant distribution:")
    for quadrant in event_data['quadrant'].unique():
        count = len(event_data[event_data['quadrant'] == quadrant])
        pct = 100 * count / row['total_counties']
        print(f"    {quadrant}: {count} ({pct:.1f}%)")
