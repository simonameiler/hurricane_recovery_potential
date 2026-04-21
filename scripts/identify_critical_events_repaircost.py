#!/usr/bin/env python3
"""
Identify critical events based on repair cost quadrants.
Events where >50% of counties fall in "Critical Vulnerability" quadrant.
"""

import pandas as pd
import numpy as np

def main():
    print("=" * 80)
    print("IDENTIFYING CRITICAL EVENTS (REPAIR COST VERSION)")
    print("=" * 80)
    
    # Load repair cost quadrant data
    df = pd.read_csv('../analysis_output/event_county_quadrants_repaircost.csv')
    
    # Calculate per-event statistics
    event_stats = []
    
    for event_id in df['event'].unique():
        event_data = df[df['event'] == event_id]
        
        # Count counties in each quadrant
        quadrant_counts = event_data['quadrant'].value_counts()
        critical_count = quadrant_counts.get('High Damage / Low Capacity', 0)
        total_counties = len(event_data)
        critical_pct = (critical_count / total_counties) * 100
        
        # Only consider events with >50% in critical quadrant
        if critical_pct > 50:
            # Get state distribution
            event_data['state'] = event_data['fips'].astype(str).str.zfill(5).str[:2]
            state_map = {
                '01': 'Alabama', '09': 'Connecticut', '10': 'Delaware', '12': 'Florida',
                '13': 'Georgia', '22': 'Louisiana', '23': 'Maine', '24': 'Maryland',
                '25': 'Massachusetts', '28': 'Mississippi', '33': 'New Hampshire',
                '34': 'New Jersey', '36': 'New York', '37': 'North Carolina',
                '44': 'Rhode Island', '45': 'South Carolina', '48': 'Texas',
                '51': 'Virginia'
            }
            event_data['state_name'] = event_data['state'].map(state_map)
            state_counts = event_data['state_name'].value_counts()
            primary_state = state_counts.index[0] if len(state_counts) > 0 else "Unknown"
            
            event_stats.append({
                'Event ID': event_id,
                'Primary State': primary_state,
                'Counties Affected': total_counties,
                '% Critical Vulnerability': f"{critical_pct:.1f}%",
                'Total Repair Cost': f"${int(event_data['damage_repair_cost'].sum()):,}",
                'Total Damage (units)': f"{int(event_data['damage_units'].sum()):,}",
                'Median Recovery (years)': event_data['recovery_months'].median() / 12
            })
    
    # Sort by % critical vulnerability
    results_df = pd.DataFrame(event_stats)
    results_df = results_df.sort_values('Median Recovery (years)', ascending=False)
    
    print(f"\nFound {len(results_df)} events with >50% counties in Critical Vulnerability quadrant")
    print("\n" + "=" * 80)
    print("TOP 10 CRITICAL EVENTS (by median recovery time)")
    print("=" * 80)
    
    for idx, row in results_df.head(10).iterrows():
        print(f"\nEvent {row['Event ID']} ({row['Primary State']}):")
        print(f"  Counties: {row['Counties Affected']}")
        print(f"  Critical vulnerability: {row['% Critical Vulnerability']}")
        print(f"  Total repair cost: {row['Total Repair Cost']}")
        print(f"  Total damage: {row['Total Damage (units)']}")
        print(f"  Median recovery: {row['Median Recovery (years)']:.1f} years")
    
    # Save all critical events
    output_file = '../analysis_output/critical_events_summary_repaircost.csv'
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nSaved {len(results_df)} critical events to: {output_file}")
    
    # Print top 5 for mapping
    print("\nTop 5 critical events for mapping:")
    top5 = results_df.head(5)
    for idx, row in top5.iterrows():
        print(f"  Event {row['Event ID']}: {row['Primary State']}, {row['Median Recovery (years)']:.1f} years recovery")

if __name__ == '__main__':
    main()
