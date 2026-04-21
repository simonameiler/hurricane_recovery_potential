#!/usr/bin/env python3
"""
Identify and characterize the 5 highest total damage events
to contrast with the "critical vulnerability" events.
"""

import pandas as pd
import numpy as np

def main():
    print("=" * 80)
    print("IDENTIFYING HIGHEST DAMAGE EVENTS")
    print("=" * 80)
    
    # Load event-county quadrant data
    df = pd.read_csv('../analysis_output/event_county_quadrants.csv')
    
    # Calculate total damage per event
    event_totals = df.groupby('event').agg({
        'damage_units': 'sum',
        'repair_cost_total': 'sum',
        'recovery_months': 'median',
        'construction_capacity': 'median',
        'fips': 'count'
    }).reset_index()
    
    event_totals.columns = ['event', 'total_damage_units', 'total_repair_cost', 
                           'median_recovery_months', 'median_capacity', 'num_counties']
    
    # Sort by total damage and get top 5
    top_damage_events = event_totals.nlargest(5, 'total_damage_units')
    
    print("\n" + "=" * 80)
    print("TOP 5 HIGHEST DAMAGE EVENTS")
    print("=" * 80)
    
    results = []
    
    for idx, row in top_damage_events.iterrows():
        event_id = int(row['event'])
        
        print(f"\n{'=' * 80}")
        print(f"Event {event_id}")
        print(f"{'=' * 80}")
        
        # Get detailed info for this event
        event_data = df[df['event'] == event_id].copy()
        
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
        
        # Get quadrant distribution
        quadrant_dist = event_data['quadrant'].value_counts()
        critical_pct = (quadrant_dist.get('High Damage / Low Capacity', 0) / len(event_data)) * 100
        
        print(f"Counties affected: {int(row['num_counties'])}")
        print(f"Total damage: {int(row['total_damage_units']):,} units")
        print(f"Total repair cost: ${int(row['total_repair_cost']):,}")
        print(f"Median recovery: {int(row['median_recovery_months'])} months ({row['median_recovery_months']/12:.1f} years)")
        print(f"Median capacity: {row['median_capacity']:.2f} permits/month")
        print(f"\nStates affected:")
        for state, count in state_counts.items():
            print(f"  {state}: {count} counties")
        print(f"\nQuadrant distribution:")
        for quad, count in quadrant_dist.items():
            pct = (count / len(event_data)) * 100
            print(f"  {quad}: {count} counties ({pct:.1f}%)")
        
        # Determine primary state
        primary_state = state_counts.index[0] if len(state_counts) > 0 else "Unknown"
        if len(state_counts) > 1:
            secondary_state = state_counts.index[1]
            state_label = f"{primary_state}/{secondary_state}"
        else:
            state_label = primary_state
        
        results.append({
            'event': event_id,
            'state': state_label,
            'num_counties': int(row['num_counties']),
            'total_damage_units': int(row['total_damage_units']),
            'total_repair_cost': int(row['total_repair_cost']),
            'median_recovery_months': int(row['median_recovery_months']),
            'median_recovery_years': row['median_recovery_months'] / 12,
            'median_capacity': row['median_capacity'],
            'critical_vulnerability_pct': critical_pct
        })
    
    # Save summary
    results_df = pd.DataFrame(results)
    output_file = '../analysis_output/highest_damage_events_summary.csv'
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH CRITICAL VULNERABILITY EVENTS")
    print("=" * 80)
    
    # Load critical events for comparison
    try:
        critical_events = pd.read_csv('../analysis_output/critical_events_summary.csv')
        # Parse the damage column (remove commas)
        critical_events['Total Damage (units)'] = critical_events['Total Damage (units)'].str.replace(',', '').astype(int)
        
        print("\nCritical Vulnerability Events (High Damage + Low Capacity):")
        print(f"  Median recovery: {critical_events['Median Recovery (years)'].median():.1f} years")
        print(f"  Total damage range: {critical_events['Total Damage (units)'].min():,} - {critical_events['Total Damage (units)'].max():,} units")
        
        print("\nHighest Damage Events:")
        print(f"  Median recovery: {results_df['median_recovery_years'].median():.1f} years")
        print(f"  Total damage range: {results_df['total_damage_units'].min():,} - {results_df['total_damage_units'].max():,} units")
        
        print("\nKey insight: Highest damage events have {:.1f}x more total damage but {:.1f}x FASTER recovery".format(
            results_df['total_damage_units'].median() / critical_events['Total Damage (units)'].median(),
            critical_events['Median Recovery (years)'].median() / results_df['median_recovery_years'].median()
        ))
        
    except FileNotFoundError:
        print("\nNote: Could not find critical_events_summary.csv for comparison")
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nSummary saved to: {output_file}")
    print("\nHighest damage events identified:")
    for _, row in results_df.iterrows():
        print(f"  Event {row['event']}: {row['total_damage_units']:,} units, {row['median_recovery_years']:.1f} years recovery ({row['state']})")

if __name__ == '__main__':
    main()
