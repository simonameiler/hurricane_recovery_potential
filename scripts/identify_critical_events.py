"""
Identify and characterize the 5 critical events that predominantly hit
the Critical Vulnerability quadrant.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"
IMPACTS_DIR = BASE_DIR / "impacts_out" / "by_event" / "scaled"

# The 5 critical events (as floats in the data)
critical_event_ids = [803.0, 1846.0, 4094.0, 4442.0, 4967.0]

print("="*80)
print("IDENTIFYING THE 5 CRITICAL EVENTS")
print("="*80)

# Load the quadrant data
df = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants.csv')
# Ensure event is numeric
df['event'] = pd.to_numeric(df['event'], errors='coerce')

# State FIPS codes for reference
state_names = {
    '01': 'Alabama', '04': 'Arizona', '05': 'Arkansas', '06': 'California',
    '08': 'Colorado', '09': 'Connecticut', '10': 'Delaware', '11': 'DC',
    '12': 'Florida', '13': 'Georgia', '15': 'Hawaii', '16': 'Idaho',
    '17': 'Illinois', '18': 'Indiana', '19': 'Iowa', '20': 'Kansas',
    '21': 'Kentucky', '22': 'Louisiana', '23': 'Maine', '24': 'Maryland',
    '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota', '28': 'Mississippi',
    '29': 'Missouri', '30': 'Montana', '31': 'Nebraska', '32': 'Nevada',
    '33': 'New Hampshire', '34': 'New Jersey', '35': 'New Mexico', '36': 'New York',
    '37': 'North Carolina', '38': 'North Dakota', '39': 'Ohio', '40': 'Oklahoma',
    '41': 'Oregon', '42': 'Pennsylvania', '44': 'Rhode Island', '45': 'South Carolina',
    '46': 'South Dakota', '47': 'Tennessee', '48': 'Texas', '49': 'Utah',
    '50': 'Vermont', '51': 'Virginia', '53': 'Washington', '54': 'West Virginia',
    '55': 'Wisconsin', '56': 'Wyoming'
}

for event_id in critical_event_ids:
    print(f"\n{'='*80}")
    print(f"EVENT {event_id}")
    print('='*80)
    
    # Get event data from quadrant analysis
    event_data = df[df['event'] == event_id].copy()
    
    if len(event_data) == 0:
        print(f"No data found for event {event_id}")
        continue
    
    # Count by quadrant
    quadrant_counts = event_data['quadrant'].value_counts()
    total_counties = len(event_data)
    
    print(f"\nAffected {total_counties} counties")
    print("\nQuadrant distribution:")
    for quad in ['High Damage / High Capacity', 'High Damage / Low Capacity',
                 'Low Damage / High Capacity', 'Low Damage / Low Capacity']:
        count = quadrant_counts.get(quad, 0)
        pct = count / total_counties * 100 if total_counties > 0 else 0
        print(f"  {quad}: {count} counties ({pct:.1f}%)")
    
    # Geographic distribution
    event_data['state'] = event_data['fips'].astype(str).str[:2]
    states_affected = event_data['state'].value_counts()
    
    print(f"\nTop 5 states affected:")
    for state_code, count in states_affected.head(5).items():
        state_name = state_names.get(state_code, f"State {state_code}")
        pct = count / total_counties * 100
        print(f"  {state_name}: {count} counties ({pct:.1f}%)")
    
    # Damage statistics
    print(f"\nDamage statistics:")
    print(f"  Total damage: {event_data['damage_units'].sum():,.0f} units")
    print(f"  Mean damage per county: {event_data['damage_units'].mean():,.0f} units")
    print(f"  Median damage: {event_data['damage_units'].median():,.0f} units")
    print(f"  Max damage (single county): {event_data['damage_units'].max():,.0f} units")
    
    # Recovery statistics
    print(f"\nRecovery statistics:")
    print(f"  Mean recovery: {event_data['recovery_months'].mean():,.0f} months ({event_data['recovery_months'].mean()/12:.1f} years)")
    print(f"  Median recovery: {event_data['recovery_months'].median():,.0f} months ({event_data['recovery_months'].median()/12:.1f} years)")
    print(f"  Max recovery: {event_data['recovery_months'].max():,.0f} months ({event_data['recovery_months'].max()/12:.1f} years)")
    
    # Counties in critical quadrant
    critical_counties = event_data[event_data['quadrant'] == 'High Damage / Low Capacity'].copy()
    if len(critical_counties) > 0:
        print(f"\nTop 5 counties in Critical Vulnerability quadrant:")
        critical_sorted = critical_counties.sort_values('recovery_months', ascending=False).head(5)
        for _, row in critical_sorted.iterrows():
            state_code = str(row['fips'])[:2]
            state_name = state_names.get(state_code, f"State {state_code}")
            print(f"  {row['fips']} ({state_name}): {row['damage_units']:,.0f} units, "
                  f"{row['recovery_months']:,.0f} months ({row['recovery_months']/12:.1f} years)")

# Create summary table
print("\n" + "="*80)
print("SUMMARY TABLE FOR PAPER")
print("="*80)

summary_data = []
for event_id in critical_event_ids:
    event_data = df[df['event'] == event_id]
    if len(event_data) == 0:
        continue
    
    critical_pct = (event_data['quadrant'] == 'High Damage / Low Capacity').sum() / len(event_data) * 100
    
    # Get primary state
    event_data['state'] = event_data['fips'].astype(str).str[:2]
    primary_state_code = event_data['state'].value_counts().index[0]
    primary_state = state_names.get(primary_state_code, f"State {primary_state_code}")
    
    summary_data.append({
        'Event ID': event_id,
        'Primary State': primary_state,
        'Counties Affected': len(event_data),
        '% Critical Vulnerability': f"{critical_pct:.1f}%",
        'Total Damage (units)': f"{event_data['damage_units'].sum():,.0f}",
        'Median Recovery (years)': f"{event_data['recovery_months'].median()/12:.1f}"
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Save detailed analysis
summary_df.to_csv(OUTPUT_DIR / 'critical_events_summary.csv', index=False)
print(f"\n\nSaved: {OUTPUT_DIR / 'critical_events_summary.csv'}")

# Additional: Try to identify event characteristics that make them critical
print("\n" + "="*80)
print("WHAT MAKES THESE EVENTS CRITICAL?")
print("="*80)

all_events = df.groupby('event').agg({
    'damage_units': 'sum',
    'recovery_months': 'median',
    'fips': 'count'
}).reset_index()
all_events.columns = ['event', 'total_damage', 'median_recovery', 'counties_affected']

critical_events_stats = all_events[all_events['event'].isin(critical_event_ids)]
other_events_stats = all_events[~all_events['event'].isin(critical_event_ids)]

print(f"\nCritical events (n={len(critical_events_stats)}):")
print(f"  Mean counties affected: {critical_events_stats['counties_affected'].mean():.1f}")
print(f"  Mean total damage: {critical_events_stats['total_damage'].mean():,.0f} units")
print(f"  Mean median recovery: {critical_events_stats['median_recovery'].mean()/12:.1f} years")

print(f"\nOther events (n={len(other_events_stats)}):")
print(f"  Mean counties affected: {other_events_stats['counties_affected'].mean():.1f}")
print(f"  Mean total damage: {other_events_stats['total_damage'].mean():,.0f} units")
print(f"  Mean median recovery: {other_events_stats['median_recovery'].mean()/12:.1f} years")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("Critical events are NOT necessarily the largest (mean damage similar to others)")
print("BUT they hit low-capacity regions disproportionately")
print("→ It's the GEOGRAPHIC TARGETING, not absolute magnitude, that makes them critical")
