"""
Analyze the events that predominantly hit the Critical Vulnerability quadrant.
This supports the hypothesis about damage thresholds.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"

# Load event patterns
events = pd.read_csv(OUTPUT_DIR / 'event_quadrant_patterns.csv')

# Find events that predominantly hit High Damage / Low Capacity (>50% of counties)
critical_events = events[events['High Damage / Low Capacity'] > 50].sort_values(
    'High Damage / Low Capacity', ascending=False
)

print("Events predominantly hitting Critical Vulnerability quadrant (>50% of counties):")
print("="*80)
for _, row in critical_events.iterrows():
    print(f"\nEvent {row['event']}:")
    print(f"  High Damage / Low Capacity: {row['High Damage / Low Capacity']:.1f}%")
    print(f"  Counties affected: {int(row['counties_affected'])}")
    print(f"  Total damage: {row['total_damage']:,.0f} units")
    print(f"  Mean damage per county: {row['mean_damage']:,.1f} units")
    print(f"  Mean recovery: {row['mean_recovery']:.1f} months")
    print(f"  Median recovery: {row['median_recovery']:.1f} months")
    print(f"  Max recovery: {row['max_recovery']:.1f} months")

# Compare to overall statistics
print("\n" + "="*80)
print("COMPARISON TO ALL EVENTS:")
print("="*80)
print(f"Total events analyzed: {len(events)}")
print(f"\nMean total damage (all events): {events['total_damage'].mean():,.0f} units")
print(f"Mean damage per county (all events): {events['mean_damage'].mean():,.1f} units")
print(f"Mean recovery (all events): {events['mean_recovery'].mean():.1f} months")

print("\n" + "="*80)
print("DAMAGE DISTRIBUTION:")
print("="*80)
percentiles = [50, 75, 90, 95, 99, 99.9]
for p in percentiles:
    print(f"{p}th percentile total damage: {events['total_damage'].quantile(p/100):,.0f} units")

# Check where these 5 events rank
print("\n" + "="*80)
print("RANKING OF CRITICAL EVENTS:")
print("="*80)
events_sorted = events.sort_values('total_damage', ascending=False).reset_index(drop=True)
for event_id in critical_events['event']:
    rank = events_sorted[events_sorted['event'] == event_id].index[0] + 1
    percentile = (1 - rank / len(events)) * 100
    total_dmg = events_sorted[events_sorted['event'] == event_id]['total_damage'].values[0]
    print(f"Event {event_id}: Rank {rank}/{len(events)} (top {percentile:.1f}%), "
          f"Total damage: {total_dmg:,.0f} units")

# Analyze quadrant patterns for high-damage events
print("\n" + "="*80)
print("QUADRANT PATTERNS FOR TOP 100 MOST DAMAGING EVENTS:")
print("="*80)
top_100 = events_sorted.head(100)
for col in ['High Damage / High Capacity', 'High Damage / Low Capacity', 
            'Low Damage / High Capacity', 'Low Damage / Low Capacity']:
    mean_pct = top_100[col].mean()
    print(f"{col}: {mean_pct:.1f}% (mean)")

print("\nVersus all events:")
for col in ['High Damage / High Capacity', 'High Damage / Low Capacity',
            'Low Damage / High Capacity', 'Low Damage / Low Capacity']:
    mean_pct = events[col].mean()
    print(f"{col}: {mean_pct:.1f}% (mean)")

# Test threshold hypothesis
print("\n" + "="*80)
print("THRESHOLD HYPOTHESIS TEST:")
print("="*80)
print("Do very damaging events hit different quadrants than smaller events?")
print()

# Split into damage bins
events['damage_bin'] = pd.qcut(events['total_damage'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

for bin_name in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
    bin_events = events[events['damage_bin'] == bin_name]
    print(f"\n{bin_name} damage events (n={len(bin_events)}):")
    print(f"  Damage range: {bin_events['total_damage'].min():,.0f} - {bin_events['total_damage'].max():,.0f} units")
    print(f"  High Damage / High Capacity: {bin_events['High Damage / High Capacity'].mean():.1f}%")
    print(f"  High Damage / Low Capacity: {bin_events['High Damage / Low Capacity'].mean():.1f}%")
    print(f"  Mean recovery time: {bin_events['mean_recovery'].mean():.1f} months")
