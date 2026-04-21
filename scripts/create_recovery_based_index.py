"""
Create Recovery-Based Priority Index

Uses actual recovery time outputs to identify priority counties.
This is more direct and interpretable than using damage and capacity separately.

Approach: Compute Expected Annual Recovery Time (EART) and Maximum Recovery Time,
then classify counties based on both typical and extreme recovery challenges.

Priority Matrix (2x2):
                    Max Recovery Time
                    Low         High
Expected   High  |  P1      |   P2   |
Annual     Low   |  P3      |   P4   |

P1 = Chronic burden (high EART, low max) - Frequent moderate events
P2 = Critical (high EART, high max) - Both chronic AND extreme challenges
P3 = Resilient (low EART, low max) - Limited recovery challenges
P4 = Tail risk (low EART, high max) - Rare extreme events
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
from pathlib import Path

# Define priority labels
PRIORITY_LABELS = {
    (1, 0): 'P1: Chronic Burden',        # High EART, Low max
    (1, 1): 'P2: Critical',              # High EART, High max
    (0, 0): 'P3: Resilient',             # Low EART, Low max
    (0, 1): 'P4: Tail Risk'              # Low EART, High max
}

PRIORITY_COLORS = {
    'P1: Chronic Burden': '#FFA500',     # Orange - frequent but manageable
    'P2: Critical': '#8B0000',           # Dark red - worst case
    'P3: Resilient': '#90EE90',          # Light green - best case
    'P4: Tail Risk': '#FFD700'           # Gold - rare but severe
}

def load_data():
    """Load county geometries and recovery data."""
    print("Loading data...")
    
    # Load county shapefile
    counties = gpd.read_file('/Users/simonameiler/Documents/work/03_code/repos/hurricane_recovery_potential/data/US_counties.shp')
    counties['FIPS'] = counties['GEOID'].astype(str).str.zfill(5)
    
    # Load all recovery times from JSON files
    recovery_dir = Path('/Users/simonameiler/Documents/work/03_code/repos/hurricane_recovery_potential/data/recovery_potential_per_scenario')
    
    all_recovery_data = []
    for json_file in sorted(recovery_dir.glob('*.json')):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Data is a list of dictionaries
        for record in data:
            all_recovery_data.append({
                'event': record.get('event', json_file.stem.split('_')[0]),
                'fips': str(record['fips']).zfill(5),
                'recovery_time': float(record.get('recovery_potential [months]', 0)),
                'capacity': float(record.get('reconstruction_capacity', 0))
            })
    
    recovery_df = pd.DataFrame(all_recovery_data)
    
    print(f"Loaded {len(recovery_df):,} event-county recovery records")
    print(f"Unique events: {recovery_df['event'].nunique()}")
    print(f"Unique counties: {recovery_df['fips'].nunique()}")
    
    return counties, recovery_df

def compute_recovery_metrics(recovery_df):
    """Compute expected annual and maximum recovery times."""
    print("Computing recovery metrics...")
    
    # Get event probability (uniform)
    n_events = recovery_df['event'].nunique()
    event_frequency = 0.00067334  # events per year
    event_prob = event_frequency / n_events
    
    # Compute Expected Annual Recovery Time (EART)
    county_metrics = recovery_df.groupby('fips').agg({
        'recovery_time': [
            ('eart', lambda x: (x * event_prob).sum()),  # Expected annual
            ('max_recovery', 'max'),                      # Maximum
            ('median_recovery', 'median'),                # Median
            ('mean_recovery', 'mean')                     # Mean
        ],
        'capacity': [('mean_capacity', 'mean')],
        'event': 'count'
    }).reset_index()
    
    # Flatten column names
    county_metrics.columns = ['fips', 'eart', 'max_recovery', 'median_recovery', 
                              'mean_recovery', 'mean_capacity', 'n_events']
    
    # Filter out counties with infinite recovery times (no capacity data)
    n_before = len(county_metrics)
    county_metrics = county_metrics[
        (county_metrics['eart'] != np.inf) & 
        (county_metrics['max_recovery'] != np.inf) &
        (~county_metrics['eart'].isna()) &
        (~county_metrics['max_recovery'].isna())
    ].copy()
    n_after = len(county_metrics)
    
    print(f"\nComputed metrics for {n_before} counties")
    print(f"Filtered out {n_before - n_after} counties with infinite recovery (no capacity data)")
    print(f"Analyzing {n_after} counties with valid data")
    
    print(f"\nEART Statistics (months):")
    print(county_metrics['eart'].describe())
    print(f"\nMax Recovery Statistics (months):")
    print(county_metrics['max_recovery'].describe())
    
    return county_metrics

def categorize_median_split(values, labels=['Low', 'High']):
    """Categorize values into two groups split at median."""
    median_val = values.median()
    categories = (values > median_val).astype(int)  # 0 for Low, 1 for High
    return categories, median_val

def assign_priority_categories(metrics):
    """Assign priority categories based on 2x2 matrix.
    
    Categories are defined by median splits on two dimensions:
    
    1. Expected Annual Recovery Time (EART):
       - Represents the probabilistically-weighted average recovery burden
       - EART = Σ(recovery_time_i × probability_i) across all events
       - High EART (>median): County faces frequent or severe recovery challenges
       - Low EART (≤median): County has lower typical recovery burden
    
    2. Maximum Recovery Time (tail risk):
       - Represents worst-case scenario across all events
       - High Max (>median): County vulnerable to extreme events requiring long recovery
       - Low Max (≤median): Even worst events are manageable
    
    Priority Matrix:
    - P2 Critical (High EART + High Max): Chronic burden AND extreme tail risk
    - P1 Chronic Burden (High EART + Low Max): Frequent challenges, manageable extremes  
    - P4 Tail Risk (Low EART + High Max): Rare but catastrophic events
    - P3 Resilient (Low EART + Low Max): Limited recovery challenges overall
    """
    print("\nAssigning priority categories...")
    print("\nMethodology:")
    print("- EART (Expected Annual Recovery Time) = probabilistically-weighted recovery burden")
    print("- Max Recovery = worst-case recovery time across all events")
    print("- Both metrics split at median to create 2×2 priority matrix")
    
    # Categorize EART - High EART = High burden (1)
    metrics['eart_cat'], eart_median = categorize_median_split(metrics['eart'])
    
    # Categorize max recovery - High max = High tail risk (1)
    metrics['max_recovery_cat'], max_median = categorize_median_split(metrics['max_recovery'])
    
    # Assign priority based on combination
    metrics['priority_code'] = list(zip(metrics['eart_cat'], 
                                        metrics['max_recovery_cat']))
    metrics['priority_category'] = metrics['priority_code'].map(PRIORITY_LABELS)
    
    # Add numerical priority for sorting
    priority_rank = {
        'P2: Critical': 1,       # Highest priority - both high
        'P1: Chronic Burden': 2,
        'P4: Tail Risk': 3,
        'P3: Resilient': 4       # Lowest priority - both low
    }
    metrics['priority_rank'] = metrics['priority_category'].map(priority_rank)
    
    print(f"\nMedian thresholds:")
    print(f"  EART median: {eart_median:.2f} months")
    print(f"  Max recovery median: {max_median:.1f} months")
    
    return metrics, eart_median, max_median

def create_multipanel_map(gdf, eart_median, max_median):
    """Create multi-panel visualization with EART, max recovery, and priority maps."""
    print("\nCreating multi-panel visualization...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Panel 1: Expected Annual Recovery Time
    ax1 = plt.subplot(2, 3, 1, projection=None)
    gdf_plot = gdf[gdf['eart'].notna()]
    gdf_plot.plot(column='eart', ax=ax1, legend=True, cmap='YlOrRd',
                  legend_kwds={'label': 'Expected Annual Recovery Time (months)', 'shrink': 0.8})
    ax1.set_title('A) Expected Annual Recovery Time (EART)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax1.text(0.02, 0.98, f'Median: {eart_median:.2f} months',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 2: Maximum Recovery Time
    ax2 = plt.subplot(2, 3, 2, projection=None)
    gdf_plot = gdf[gdf['max_recovery'].notna()]
    gdf_plot.plot(column='max_recovery', ax=ax2, legend=True, cmap='Reds',
                  legend_kwds={'label': 'Maximum Recovery Time (months)', 'shrink': 0.8})
    ax2.set_title('B) Maximum Recovery Time (Tail Risk)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    ax2.text(0.02, 0.98, f'Median: {max_median:.1f} months',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 3: Priority Categories
    ax3 = plt.subplot(2, 3, 3, projection=None)
    
    gdf['color'] = gdf['priority_category'].map(PRIORITY_COLORS)
    gdf_valid = gdf[gdf['priority_category'].notna()]
    
    for cat in gdf_valid['priority_category'].unique():
        gdf_valid[gdf_valid['priority_category'] == cat].plot(
            color=PRIORITY_COLORS[cat], ax=ax3, edgecolor='white', linewidth=0.3
        )
    
    ax3.set_title('C) Recovery Priority Categories', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Create custom legend
    legend_elements = [mpatches.Patch(facecolor=PRIORITY_COLORS[cat], 
                                      edgecolor='black', label=cat)
                      for cat in ['P2: Critical', 'P1: Chronic Burden', 
                                 'P4: Tail Risk', 'P3: Resilient']]
    ax3.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
              frameon=True, fontsize=11)
    
    # Panel 4-6: Priority distribution table and statistics
    ax4 = plt.subplot(2, 3, (4, 6))
    ax4.axis('off')
    
    # Count by priority category
    priority_counts = gdf['priority_category'].value_counts()
    priority_pcts = (priority_counts / priority_counts.sum() * 100).round(1)
    
    # Create summary table
    table_data = []
    for cat in ['P2: Critical', 'P1: Chronic Burden', 'P4: Tail Risk', 'P3: Resilient']:
        if cat in priority_counts.index:
            count = priority_counts[cat]
            pct = priority_pcts[cat]
            avg_eart = gdf[gdf['priority_category'] == cat]['eart'].mean()
            avg_max = gdf[gdf['priority_category'] == cat]['max_recovery'].mean()
            table_data.append([cat, count, f'{pct}%', f'{avg_eart:.2f}', f'{avg_max:.1f}'])
    
    # Create table
    table = ax4.table(cellText=table_data,
                     colLabels=['Priority Category', 'Counties', '%', 'Avg EART', 'Avg Max RT'],
                     cellLoc='left',
                     loc='upper left',
                     bbox=[0, 0.5, 1, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#CCCCCC')
        table[(0, i)].set_text_props(weight='bold')
    
    # Color code rows
    for i, cat in enumerate(['P2: Critical', 'P1: Chronic Burden', 'P4: Tail Risk', 'P3: Resilient'], 1):
        if i <= len(table_data):
            table[(i, 0)].set_facecolor(PRIORITY_COLORS[cat])
    
    # Add interpretation text
    interpretation = """
Priority Interpretation:

P2: Critical - High expected annual recovery burden AND high tail risk
    → Urgent comprehensive intervention: reduce exposure + build capacity
    
P1: Chronic Burden - High expected annual recovery burden, manageable tail risk
    → Focus on reducing frequent impacts through mitigation and preparedness
    
P4: Tail Risk - Low annual burden but high extreme event recovery challenges
    → Build capacity for rare catastrophic events; emergency planning focus
    
P3: Resilient - Low annual burden and low tail risk
    → Maintain current preparedness standards
    """
    ax4.text(0, 0.35, interpretation, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.suptitle('Recovery-Based Priority Index: 2×2 Classification\n(Expected Annual Recovery Time × Maximum Recovery Time)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('/Users/simonameiler/Documents/work/03_code/repos/hurricane_recovery_potential/analysis_output/recovery_based_priority_index.png',
                dpi=300, bbox_inches='tight')
    print("Saved: recovery_based_priority_index.png")
    
    plt.close()

def create_priority_matrix_heatmap(metrics):
    """Create a 2x2 matrix heatmap showing county distribution."""
    print("Creating priority matrix heatmap...")
    
    # Create cross-tabulation
    matrix = pd.crosstab(metrics['eart_cat'], 
                         metrics['max_recovery_cat'])
    
    # Ensure we have a 2x2 matrix
    for i in [0, 1]:
        for j in [0, 1]:
            if i not in matrix.index:
                matrix.loc[i] = 0
            if j not in matrix.columns:
                matrix[j] = 0
    
    matrix = matrix.sort_index().sort_index(axis=1)
    matrix = matrix.iloc[::-1]  # Reverse rows so high EART is at top
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(matrix.values, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(['Low', 'High'], fontsize=14)
    ax.set_yticklabels(['High', 'Low'], fontsize=14)
    
    ax.set_xlabel('Maximum Recovery Time (Tail Risk)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Expected Annual Recovery Time', fontsize=16, fontweight='bold')
    ax.set_title('County Distribution Across Recovery Priority Matrix', fontsize=18, fontweight='bold', pad=20)
    
    # Add annotations
    for i in range(2):
        for j in range(2):
            eart_cat = 1 - i
            max_cat = j
            count = int(matrix.iloc[i, j])
            priority_label = PRIORITY_LABELS[(eart_cat, max_cat)]
            
            text = ax.text(j, i, f'{priority_label}\n({count} counties)',
                          ha="center", va="center", color="black", fontsize=12,
                          weight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Counties', rotation=270, labelpad=20, fontsize=14)
    
    plt.tight_layout()
    plt.savefig('/Users/simonameiler/Documents/work/03_code/repos/hurricane_recovery_potential/analysis_output/recovery_priority_matrix_heatmap.png',
                dpi=300, bbox_inches='tight')
    print("Saved: recovery_priority_matrix_heatmap.png")
    
    plt.close()

def create_scatter_plots(metrics):
    """Create scatter plots showing relationship between EART and max recovery."""
    print("Creating scatter plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: EART vs Max Recovery colored by priority
    ax1 = axes[0]
    for cat in ['P3: Resilient', 'P4: Tail Risk', 'P1: Chronic Burden', 'P2: Critical']:
        data = metrics[metrics['priority_category'] == cat]
        ax1.scatter(data['max_recovery'], data['eart'], 
                   c=PRIORITY_COLORS[cat], label=cat, alpha=0.6, s=50)
    
    ax1.set_xlabel('Maximum Recovery Time (months)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Expected Annual Recovery Time (months)', fontsize=12, fontweight='bold')
    ax1.set_title('Recovery Metrics by Priority Category', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Add median lines
    eart_median = metrics['eart'].median()
    max_median = metrics['max_recovery'].median()
    ax1.axhline(eart_median, color='gray', linestyle='--', alpha=0.5, label=f'EART median: {eart_median:.2f}')
    ax1.axvline(max_median, color='gray', linestyle='--', alpha=0.5, label=f'Max median: {max_median:.1f}')
    
    # Plot 2: EART vs Median Recovery
    ax2 = axes[1]
    for cat in ['P3: Resilient', 'P4: Tail Risk', 'P1: Chronic Burden', 'P2: Critical']:
        data = metrics[metrics['priority_category'] == cat]
        ax2.scatter(data['median_recovery'], data['eart'], 
                   c=PRIORITY_COLORS[cat], label=cat, alpha=0.6, s=50)
    
    ax2.set_xlabel('Median Recovery Time (months)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Expected Annual Recovery Time (months)', fontsize=12, fontweight='bold')
    ax2.set_title('Expected vs Median Recovery Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/simonameiler/Documents/work/03_code/repos/hurricane_recovery_potential/analysis_output/recovery_scatter_analysis.png',
                dpi=300, bbox_inches='tight')
    print("Saved: recovery_scatter_analysis.png")
    
    plt.close()

def create_top_counties_table(metrics, n=30):
    """Create detailed table of top priority counties."""
    print(f"\nGenerating top {n} priority counties table...")
    
    top_counties = metrics.sort_values(['priority_rank', 'eart'], 
                                       ascending=[True, False]).head(n)
    
    output = top_counties[['fips', 'priority_category', 'priority_rank', 
                           'eart', 'max_recovery', 'median_recovery',
                           'mean_capacity', 'n_events']].copy()
    output.columns = ['FIPS', 'Priority_Category', 'Priority_Rank', 
                     'EART_months', 'Max_Recovery_months', 'Median_Recovery_months',
                     'Mean_Capacity', 'N_Events']
    
    output.to_csv('/Users/simonameiler/Documents/work/03_code/repos/hurricane_recovery_potential/analysis_output/recovery_based_top_priority_counties.csv',
                  index=False)
    print(f"Saved: recovery_based_top_priority_counties.csv")
    
    return output

def main():
    """Main execution function."""
    print("="*60)
    print("Creating Recovery-Based Priority Index")
    print("="*60)
    
    # Load data
    counties, recovery_df = load_data()
    
    # Compute recovery metrics
    metrics = compute_recovery_metrics(recovery_df)
    
    # Assign priority categories
    metrics, eart_median, max_median = assign_priority_categories(metrics)
    
    # Merge with geometries
    gdf = counties.merge(metrics, left_on='FIPS', right_on='fips', how='left')
    
    # Create visualizations
    create_multipanel_map(gdf, eart_median, max_median)
    create_priority_matrix_heatmap(metrics)
    create_scatter_plots(metrics)
    
    # Create top counties table
    top_counties = create_top_counties_table(metrics, n=30)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PRIORITY DISTRIBUTION SUMMARY")
    print("="*60)
    for cat in ['P2: Critical', 'P1: Chronic Burden', 'P4: Tail Risk', 'P3: Resilient']:
        count = (metrics['priority_category'] == cat).sum()
        pct = count / len(metrics) * 100
        print(f"{cat:25s}: {count:3d} counties ({pct:5.1f}%)")
    
    print(f"\nTotal counties analyzed: {len(metrics)}")
    print(f"\nTop 5 Priority Counties:")
    print(top_counties.head()[['FIPS', 'Priority_Category', 'EART_months', 'Max_Recovery_months']].to_string(index=False))
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == '__main__':
    main()
