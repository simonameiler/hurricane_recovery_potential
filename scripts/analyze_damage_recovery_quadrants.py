"""
Damage-Recovery Quadrant Analysis

Systematic analysis of the damage-recovery space using all event-county pairs.
Divides the space into 4 quadrants based on median splits:
- High Damage / High Capacity → "Resilient but Exposed"
- High Damage / Low Capacity → "Critical Vulnerability"
- Low Damage / High Capacity → "Well-Prepared"
- Low Damage / Low Capacity → "Latent Risk"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import geopandas as gpd
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"
DATA_DIR = BASE_DIR / "data"
IMPACTS_DIR = BASE_DIR / "impacts_out" / "by_event" / "scaled"
RECOVERY_DIR = DATA_DIR / "recovery_potential_per_scenario"

# Color palette for quadrants
QUADRANT_COLORS = {
    'High Damage / High Capacity': '#2c7bb6',      # Blue - Resilient but Exposed
    'High Damage / Low Capacity': '#d7191c',       # Red - Critical Vulnerability
    'Low Damage / High Capacity': '#1a9850',       # Green - Well-Prepared
    'Low Damage / Low Capacity': '#fdae61'         # Orange - Latent Risk
}

QUADRANT_LABELS = {
    'High Damage / High Capacity': 'Resilient\nbut Exposed',
    'High Damage / Low Capacity': 'Critical\nVulnerability',
    'Low Damage / High Capacity': 'Well-\nPrepared',
    'Low Damage / Low Capacity': 'Latent\nRisk'
}


def load_all_event_county_pairs():
    """
    Load all event-county pairs with damage and recovery potential.
    
    Returns:
        DataFrame with columns: event, fips, damage_units, recovery_months, construction_capacity
    """
    print("Loading all event-county pairs...")
    
    # Load all recovery potential files
    recovery_data = []
    recovery_files = sorted(RECOVERY_DIR.glob("*_scaled_recovery_potential.json"))
    
    print(f"Found {len(recovery_files)} recovery potential files")
    
    for i, filepath in enumerate(recovery_files):
        if i % 1000 == 0:
            print(f"  Processing file {i}/{len(recovery_files)}")
        
        with open(filepath, 'r') as f:
            event_data = json.load(f)
            recovery_data.extend(event_data)
    
    recovery_df = pd.DataFrame(recovery_data)
    recovery_df['event'] = recovery_df['event'].astype(str)
    recovery_df['fips'] = recovery_df['fips'].astype(str).str.zfill(5)
    recovery_df['reconstruction_capacity'] = recovery_df['reconstruction_capacity'].astype(float)
    recovery_df['recovery_potential [months]'] = recovery_df['recovery_potential [months]'].astype(float)
    
    print(f"Loaded {len(recovery_df):,} recovery potential records")
    
    # Load all damage impact files
    damage_data = []
    impact_files = sorted(IMPACTS_DIR.glob("*_scaled.csv"))
    
    print(f"Found {len(impact_files)} impact files")
    
    for i, filepath in enumerate(impact_files):
        if i % 1000 == 0:
            print(f"  Processing file {i}/{len(impact_files)}")
        
        df = pd.read_csv(filepath)
        df['event'] = filepath.stem.replace('_scaled', '')
        damage_data.append(df)
    
    damage_df = pd.concat(damage_data, ignore_index=True)
    damage_df['fips'] = damage_df['fips'].astype(str).str.zfill(5)
    
    # Calculate total damage units
    damage_df['total_damage_units'] = (
        damage_df['units_DS1_scaled'].fillna(0) +
        damage_df['units_DS2_scaled'].fillna(0) +
        damage_df['units_DS3_scaled'].fillna(0) +
        damage_df['units_DS4_scaled'].fillna(0)
    )
    
    damage_df['repair_cost_total'] = damage_df['repair_cost_sum_scaled'].fillna(0)
    
    print(f"Loaded {len(damage_df):,} damage records")
    
    # Merge damage and recovery
    merged_df = recovery_df.merge(
        damage_df[['event', 'fips', 'total_damage_units', 'repair_cost_total']],
        on=['event', 'fips'],
        how='inner'
    )
    
    print(f"Merged dataset: {len(merged_df):,} event-county pairs")
    
    # Filter to valid cases (positive damage and recovery)
    valid_df = merged_df[
        (merged_df['total_damage_units'] > 0) &
        (merged_df['recovery_potential [months]'] > 0) &
        (merged_df['reconstruction_capacity'] > 0)
    ].copy()
    
    print(f"Valid pairs (damage > 0, recovery > 0): {len(valid_df):,}")
    
    # Rename columns for clarity
    valid_df = valid_df.rename(columns={
        'reconstruction_capacity': 'construction_capacity',
        'recovery_potential [months]': 'recovery_months',
        'total_damage_units': 'damage_units'
    })
    
    return valid_df


def assign_quadrants(df):
    """
    Assign each event-county pair to a quadrant based on median splits.
    
    Args:
        df: DataFrame with damage_units and construction_capacity
        
    Returns:
        DataFrame with quadrant assignments
    """
    print("\nAssigning quadrants based on median splits...")
    
    # Calculate medians
    median_damage = df['damage_units'].median()
    median_capacity = df['construction_capacity'].median()
    
    print(f"Median damage: {median_damage:.1f} units")
    print(f"Median capacity: {median_capacity:.2f} permits/month")
    
    # Assign quadrants
    conditions = [
        (df['damage_units'] >= median_damage) & (df['construction_capacity'] >= median_capacity),
        (df['damage_units'] >= median_damage) & (df['construction_capacity'] < median_capacity),
        (df['damage_units'] < median_damage) & (df['construction_capacity'] >= median_capacity),
        (df['damage_units'] < median_damage) & (df['construction_capacity'] < median_capacity)
    ]
    
    quadrant_labels = [
        'High Damage / High Capacity',
        'High Damage / Low Capacity',
        'Low Damage / High Capacity',
        'Low Damage / Low Capacity'
    ]
    
    df['quadrant'] = np.select(conditions, quadrant_labels, default='Unknown')
    
    # Print quadrant statistics
    print("\nQuadrant distribution:")
    quadrant_counts = df['quadrant'].value_counts()
    for quad in quadrant_labels:
        count = quadrant_counts.get(quad, 0)
        pct = count / len(df) * 100
        print(f"  {quad}: {count:,} ({pct:.1f}%)")
    
    # Calculate statistics by quadrant
    print("\nQuadrant statistics:")
    for quad in quadrant_labels:
        quad_data = df[df['quadrant'] == quad]
        print(f"\n{quad}:")
        print(f"  Mean damage: {quad_data['damage_units'].mean():.1f} units")
        print(f"  Mean capacity: {quad_data['construction_capacity'].mean():.2f} permits/month")
        print(f"  Mean recovery: {quad_data['recovery_months'].mean():.1f} months")
        print(f"  Median recovery: {quad_data['recovery_months'].median():.1f} months")
    
    return df, median_damage, median_capacity


def create_quadrant_scatter(df, median_damage, median_capacity, output_dir):
    """
    Create scatter plot with quadrants and marginal histograms.
    """
    print("\nCreating quadrant scatter plot...")
    
    # Set up the figure with GridSpec for better control
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05,
                         left=0.1, right=0.95, top=0.95, bottom=0.08)
    
    # Main scatter plot
    ax_main = fig.add_subplot(gs[1:, :-1])
    
    # Marginal histograms
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    
    # Plot data by quadrant
    for quad, color in QUADRANT_COLORS.items():
        quad_data = df[df['quadrant'] == quad]
        ax_main.scatter(
            quad_data['damage_units'],
            quad_data['construction_capacity'],
            c=color,
            alpha=0.4,
            s=20,
            label=QUADRANT_LABELS[quad],
            edgecolors='none'
        )
    
    # Add median lines
    ax_main.axvline(median_damage, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Median')
    ax_main.axhline(median_capacity, color='black', linestyle='--', linewidth=2, alpha=0.7)
    
    # Set log scales
    ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    
    # Labels
    ax_main.set_xlabel('Damage (housing units per event)', fontsize=13, fontweight='bold')
    ax_main.set_ylabel('Construction Capacity (permits/month)', fontsize=13, fontweight='bold')
    ax_main.grid(True, alpha=0.3, which='both')
    ax_main.legend(loc='upper left', framealpha=0.9, fontsize=10)
    
    # Marginal histogram for damage (top)
    for quad, color in QUADRANT_COLORS.items():
        quad_data = df[df['quadrant'] == quad]
        ax_top.hist(
            quad_data['damage_units'],
            bins=np.logspace(np.log10(df['damage_units'].min()),
                           np.log10(df['damage_units'].max()), 50),
            color=color,
            alpha=0.5,
            edgecolor='none'
        )
    ax_top.axvline(median_damage, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax_top.set_xscale('log')
    ax_top.set_ylabel('Count', fontsize=10)
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, alpha=0.3, axis='y')
    
    # Marginal histogram for capacity (right)
    for quad, color in QUADRANT_COLORS.items():
        quad_data = df[df['quadrant'] == quad]
        ax_right.hist(
            quad_data['construction_capacity'],
            bins=np.logspace(np.log10(df['construction_capacity'].min()),
                           np.log10(df['construction_capacity'].max()), 50),
            color=color,
            alpha=0.5,
            orientation='horizontal',
            edgecolor='none'
        )
    ax_right.axhline(median_capacity, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax_right.set_yscale('log')
    ax_right.set_xlabel('Count', fontsize=10)
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Damage-Recovery Quadrant Analysis\nAll Event-County Pairs',
                fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(output_dir / 'damage_recovery_quadrants_scatter.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'damage_recovery_quadrants_scatter.png'}")
    plt.close()


def create_recovery_comparison_plot(df, output_dir):
    """
    Create box plots comparing recovery times across quadrants.
    """
    print("\nCreating recovery comparison plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot of recovery times
    quadrant_order = [
        'Low Damage / High Capacity',
        'Low Damage / Low Capacity',
        'High Damage / High Capacity',
        'High Damage / Low Capacity'
    ]
    
    colors_ordered = [QUADRANT_COLORS[q] for q in quadrant_order]
    labels_ordered = [QUADRANT_LABELS[q] for q in quadrant_order]
    
    # Recovery time distribution
    ax1 = axes[0]
    recovery_data = [df[df['quadrant'] == q]['recovery_months'].values for q in quadrant_order]
    bp1 = ax1.boxplot(recovery_data, labels=labels_ordered, patch_artist=True,
                      showfliers=False, widths=0.6)
    
    for patch, color in zip(bp1['boxes'], colors_ordered):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Recovery Time (months)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Quadrant', fontsize=12, fontweight='bold')
    ax1.set_title('Recovery Time Distribution by Quadrant', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Damage-to-capacity ratio
    ax2 = axes[1]
    df['damage_capacity_ratio'] = df['damage_units'] / df['construction_capacity']
    ratio_data = [df[df['quadrant'] == q]['damage_capacity_ratio'].values for q in quadrant_order]
    bp2 = ax2.boxplot(ratio_data, labels=labels_ordered, patch_artist=True,
                      showfliers=False, widths=0.6)
    
    for patch, color in zip(bp2['boxes'], colors_ordered):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Damage-to-Capacity Ratio', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Quadrant', fontsize=12, fontweight='bold')
    ax2.set_title('Damage-to-Capacity Ratio by Quadrant', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quadrant_recovery_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'quadrant_recovery_comparison.png'}")
    plt.close()


def analyze_county_quadrant_membership(df, output_dir):
    """
    Analyze which counties appear in which quadrants across all events.
    """
    print("\nAnalyzing county quadrant membership...")
    
    # Count how many times each county appears in each quadrant
    county_quadrants = df.groupby(['fips', 'quadrant']).size().reset_index(name='count')
    county_totals = df.groupby('fips').size().reset_index(name='total_events')
    
    # Calculate dominant quadrant for each county
    county_dominant = county_quadrants.loc[
        county_quadrants.groupby('fips')['count'].idxmax()
    ][['fips', 'quadrant', 'count']]
    
    county_dominant = county_dominant.merge(county_totals, on='fips')
    county_dominant['pct_dominant'] = county_dominant['count'] / county_dominant['total_events'] * 100
    
    # Calculate average metrics by county
    county_metrics = df.groupby('fips').agg({
        'damage_units': 'mean',
        'construction_capacity': 'first',  # Capacity is constant per county
        'recovery_months': 'mean'
    }).reset_index()
    
    county_dominant = county_dominant.merge(county_metrics, on='fips')
    
    print(f"\nTotal unique counties: {len(county_dominant)}")
    print("\nDominant quadrant distribution:")
    for quad in QUADRANT_COLORS.keys():
        count = (county_dominant['quadrant'] == quad).sum()
        pct = count / len(county_dominant) * 100
        print(f"  {quad}: {count} counties ({pct:.1f}%)")
    
    # Save summary
    county_dominant.to_csv(output_dir / 'county_quadrant_membership.csv', index=False)
    print(f"\nSaved: {output_dir / 'county_quadrant_membership.csv'}")
    
    return county_dominant


def create_geographic_map(county_dominant, output_dir):
    """
    Create map showing dominant quadrant for each county.
    """
    print("\nCreating geographic map...")
    
    # Load county shapefile
    counties = gpd.read_file(DATA_DIR / "US_counties.shp")
    counties['fips'] = (counties['STATEFP'].astype(str).str.zfill(2) + 
                       counties['COUNTYFP'].astype(str).str.zfill(3))
    
    # Merge with quadrant data
    counties = counties.merge(
        county_dominant[['fips', 'quadrant', 'pct_dominant', 'recovery_months']],
        on='fips',
        how='left'
    )
    
    # Filter to coastal states (approximate)
    coastal_states = ['01', '12', '13', '22', '28', '37', '45', '48', '51',  # Gulf/Atlantic
                     '09', '10', '24', '25', '33', '34', '36', '44']  # Northeast
    counties = counties[counties['STATEFP'].isin(coastal_states)]
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Plot counties colored by dominant quadrant
    for quad, color in QUADRANT_COLORS.items():
        quad_counties = counties[counties['quadrant'] == quad]
        quad_counties.plot(ax=ax, color=color, edgecolor='white', linewidth=0.3,
                          label=QUADRANT_LABELS[quad], alpha=0.8)
    
    # Plot counties with no data
    counties[counties['quadrant'].isna()].plot(ax=ax, color='lightgray', 
                                               edgecolor='white', linewidth=0.3)
    
    ax.set_xlim(-100, -65)
    ax.set_ylim(24, 48)
    ax.axis('off')
    ax.set_title('Dominant Quadrant Membership by County\n(Based on Event-County Pairs)',
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='lower left', framealpha=0.95, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quadrant_geographic_map.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'quadrant_geographic_map.png'}")
    plt.close()


def analyze_event_patterns(df, output_dir):
    """
    Analyze which events predominantly affect which quadrants.
    """
    print("\nAnalyzing event patterns...")
    
    # Count how many counties in each quadrant for each event
    event_quadrants = df.groupby(['event', 'quadrant']).size().reset_index(name='counties')
    event_totals = df.groupby('event').size().reset_index(name='total_counties')
    
    event_quadrants = event_quadrants.merge(event_totals, on='event')
    event_quadrants['pct'] = event_quadrants['counties'] / event_quadrants['total_counties'] * 100
    
    # Find events that predominantly hit one quadrant (>50% of affected counties)
    dominant_events = event_quadrants[event_quadrants['pct'] > 50].copy()
    
    print(f"\nEvents predominantly affecting one quadrant (>50% of counties):")
    for quad in QUADRANT_COLORS.keys():
        quad_events = dominant_events[dominant_events['quadrant'] == quad]
        print(f"  {quad}: {len(quad_events)} events")
    
    # Calculate average damage and recovery per event
    event_metrics = df.groupby('event').agg({
        'damage_units': ['mean', 'sum'],
        'recovery_months': ['mean', 'median', 'max'],
        'fips': 'count'
    }).reset_index()
    
    event_metrics.columns = ['event', 'mean_damage', 'total_damage', 
                            'mean_recovery', 'median_recovery', 'max_recovery', 'counties_affected']
    
    # Save event summary
    event_summary = event_quadrants.pivot(index='event', columns='quadrant', 
                                         values='pct').reset_index()
    event_summary = event_summary.merge(event_metrics, on='event')
    event_summary.to_csv(output_dir / 'event_quadrant_patterns.csv', index=False)
    print(f"\nSaved: {output_dir / 'event_quadrant_patterns.csv'}")
    
    return event_summary


def create_summary_statistics(df, county_dominant, output_dir):
    """
    Create comprehensive summary statistics table.
    """
    print("\nCreating summary statistics...")
    
    summary_stats = []
    
    for quad in QUADRANT_COLORS.keys():
        quad_data = df[df['quadrant'] == quad]
        
        stats = {
            'Quadrant': quad,
            'Event-County Pairs': len(quad_data),
            'Pct of Total Pairs': len(quad_data) / len(df) * 100,
            'Unique Counties': quad_data['fips'].nunique(),
            'Pct of Counties': quad_data['fips'].nunique() / df['fips'].nunique() * 100,
            'Mean Damage (units)': quad_data['damage_units'].mean(),
            'Median Damage (units)': quad_data['damage_units'].median(),
            'Mean Capacity (permits/mo)': quad_data['construction_capacity'].mean(),
            'Median Capacity (permits/mo)': quad_data['construction_capacity'].median(),
            'Mean Recovery (months)': quad_data['recovery_months'].mean(),
            'Median Recovery (months)': quad_data['recovery_months'].median(),
            'Max Recovery (months)': quad_data['recovery_months'].max(),
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / 'quadrant_summary_statistics.csv', index=False)
    print(f"\nSaved: {output_dir / 'quadrant_summary_statistics.csv'}")
    
    # Print formatted summary
    print("\n" + "="*80)
    print("QUADRANT SUMMARY STATISTICS")
    print("="*80)
    for _, row in summary_df.iterrows():
        print(f"\n{row['Quadrant']}:")
        print(f"  Event-County Pairs: {row['Event-County Pairs']:,} ({row['Pct of Total Pairs']:.1f}%)")
        print(f"  Unique Counties: {int(row['Unique Counties'])} ({row['Pct of Counties']:.1f}%)")
        print(f"  Mean Damage: {row['Mean Damage (units)']:.1f} units")
        print(f"  Mean Capacity: {row['Mean Capacity (permits/mo)']:.2f} permits/month")
        print(f"  Mean Recovery: {row['Mean Recovery (months)']:.1f} months")
        print(f"  Median Recovery: {row['Median Recovery (months)']:.1f} months")
    
    return summary_df


def create_threshold_hypothesis_visualization(df, output_dir):
    """
    Visualize evidence for the damage threshold hypothesis.
    Shows how event damage levels relate to quadrant distribution.
    """
    print("\nCreating threshold hypothesis visualization...")
    
    # Create damage bins
    df_events = df.groupby('event').agg({
        'damage_units': 'sum',
        'recovery_months': 'mean',
        'fips': 'count',
        'quadrant': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
    }).reset_index()
    
    df_events.columns = ['event', 'total_damage', 'mean_recovery', 'counties', 'dominant_quadrant']
    
    # Create damage quintiles
    df_events['damage_quintile'] = pd.qcut(df_events['total_damage'], q=5, 
                                           labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Count quadrant membership by damage quintile
    quadrant_by_damage = df.copy()
    quadrant_by_damage = quadrant_by_damage.merge(
        df_events[['event', 'total_damage', 'damage_quintile']], 
        on='event'
    )
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Stacked bar chart of quadrant distribution by damage level
    ax1 = axes[0, 0]
    pivot_data = pd.crosstab(quadrant_by_damage['damage_quintile'], 
                            quadrant_by_damage['quadrant'], 
                            normalize='index') * 100
    
    pivot_data = pivot_data[[q for q in QUADRANT_COLORS.keys() if q in pivot_data.columns]]
    colors = [QUADRANT_COLORS[q] for q in pivot_data.columns]
    
    pivot_data.plot(kind='bar', stacked=True, ax=ax1, color=colors, width=0.7, alpha=0.8)
    ax1.set_xlabel('Event Damage Level', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Percentage of Event-County Pairs', fontsize=12, fontweight='bold')
    ax1.set_title('Quadrant Distribution by Event Damage Level', fontsize=13, fontweight='bold')
    ax1.legend([QUADRANT_LABELS[q] for q in pivot_data.columns], 
              loc='upper left', fontsize=9)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Mean recovery time by damage level and quadrant
    ax2 = axes[0, 1]
    for quad, color in QUADRANT_COLORS.items():
        quad_data = quadrant_by_damage[quadrant_by_damage['quadrant'] == quad]
        recovery_by_damage = quad_data.groupby('damage_quintile')['recovery_months'].mean()
        ax2.plot(range(len(recovery_by_damage)), recovery_by_damage.values, 
                'o-', color=color, linewidth=2, markersize=8, 
                label=QUADRANT_LABELS[quad], alpha=0.8)
    
    ax2.set_xlabel('Event Damage Level', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Recovery Time (months)', fontsize=12, fontweight='bold')
    ax2.set_title('Recovery Time vs Event Damage Level', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Distribution of total damage for events hitting each quadrant
    ax3 = axes[1, 0]
    quadrant_order = ['Low Damage / High Capacity', 'Low Damage / Low Capacity',
                     'High Damage / High Capacity', 'High Damage / Low Capacity']
    
    # Get events that predominantly hit each quadrant (>40% of counties)
    event_quadrants = df.groupby(['event', 'quadrant']).size().reset_index(name='count')
    event_totals = df.groupby('event').size().reset_index(name='total')
    event_quadrants = event_quadrants.merge(event_totals, on='event')
    event_quadrants['pct'] = event_quadrants['count'] / event_quadrants['total'] * 100
    
    dominant_event_quads = event_quadrants[event_quadrants['pct'] > 40].copy()
    dominant_event_quads = dominant_event_quads.merge(df_events[['event', 'total_damage']], on='event')
    
    damage_by_quad = [dominant_event_quads[dominant_event_quads['quadrant'] == q]['total_damage'].values 
                     for q in quadrant_order]
    
    bp = ax3.boxplot(damage_by_quad, 
                     labels=[QUADRANT_LABELS[q] for q in quadrant_order],
                     patch_artist=True, showfliers=True, widths=0.6)
    
    for patch, quad in zip(bp['boxes'], quadrant_order):
        patch.set_facecolor(QUADRANT_COLORS[quad])
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Total Event Damage (housing units)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Dominant Quadrant', fontsize=12, fontweight='bold')
    ax3.set_title('Event Damage Distribution by Dominant Quadrant\n(Events with >40% counties in quadrant)', 
                 fontsize=13, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # 4. Critical events highlighted
    ax4 = axes[1, 1]
    
    # Scatter: total damage vs % in critical quadrant
    event_critical_pct = event_quadrants[event_quadrants['quadrant'] == 'High Damage / Low Capacity'].copy()
    event_critical_pct = event_critical_pct.merge(df_events[['event', 'total_damage']], on='event')
    
    # Plot all events
    ax4.scatter(df_events['total_damage'], 
               [0] * len(df_events),  # Placeholder, will update
               c='lightgray', s=30, alpha=0.3, label='Other events')
    
    # Get percentage in critical quadrant for all events
    critical_pcts = []
    for event in df_events['event']:
        pct_row = event_quadrants[(event_quadrants['event'] == event) & 
                                 (event_quadrants['quadrant'] == 'High Damage / Low Capacity')]
        if len(pct_row) > 0:
            critical_pcts.append(pct_row['pct'].values[0])
        else:
            critical_pcts.append(0)
    
    df_events['critical_pct'] = critical_pcts
    
    # Plot all events
    scatter = ax4.scatter(df_events['total_damage'], 
                         df_events['critical_pct'],
                         c=df_events['critical_pct'], 
                         cmap='YlOrRd', s=50, alpha=0.6, 
                         vmin=0, vmax=100)
    
    # Highlight the 5 critical events
    critical_event_ids = event_critical_pct[event_critical_pct['pct'] > 50]['event'].values
    critical_events_data = df_events[df_events['event'].isin(critical_event_ids)]
    ax4.scatter(critical_events_data['total_damage'],
               critical_events_data['critical_pct'],
               c='red', s=200, marker='*', edgecolors='black', linewidths=1.5,
               label='Critical events (>50%)', zorder=5)
    
    ax4.set_xlabel('Total Event Damage (housing units)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('% of Counties in Critical Quadrant', fontsize=12, fontweight='bold')
    ax4.set_title('Events Hitting Critical Vulnerability Quadrant', fontsize=13, fontweight='bold')
    ax4.set_xscale('log')
    ax4.axhline(50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='50% threshold')
    ax4.legend(fontsize=10, loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('% in Critical Quadrant', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_hypothesis_evidence.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'threshold_hypothesis_evidence.png'}")
    plt.close()


def main():
    """Main execution function."""
    print("="*80)
    print("DAMAGE-RECOVERY QUADRANT ANALYSIS")
    print("="*80)
    
    # Load data
    df = load_all_event_county_pairs()
    
    # Assign quadrants
    df, median_damage, median_capacity = assign_quadrants(df)
    
    # Save full dataset
    df.to_csv(OUTPUT_DIR / 'event_county_quadrants.csv', index=False)
    print(f"\nSaved full dataset: {OUTPUT_DIR / 'event_county_quadrants.csv'}")
    
    # Create visualizations
    create_quadrant_scatter(df, median_damage, median_capacity, OUTPUT_DIR)
    create_recovery_comparison_plot(df, OUTPUT_DIR)
    create_threshold_hypothesis_visualization(df, OUTPUT_DIR)
    
    # Analyze patterns
    county_dominant = analyze_county_quadrant_membership(df, OUTPUT_DIR)
    create_geographic_map(county_dominant, OUTPUT_DIR)
    
    event_summary = analyze_event_patterns(df, OUTPUT_DIR)
    
    # Create summary statistics
    summary_df = create_summary_statistics(df, county_dominant, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nTotal event-county pairs analyzed: {len(df):,}")
    print(f"Unique events: {df['event'].nunique()}")
    print(f"Unique counties: {df['fips'].nunique()}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
