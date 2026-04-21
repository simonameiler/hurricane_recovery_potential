"""
Create 2x2 Priority Index for Recovery Potential Assessment

Combines damage burden (weighted EAD) with capacity adequacy to identify
priority counties for different intervention types.

Priority Matrix (2x2):
                    Capacity Adequacy
                    Low         High
Damage    High  |  P1      |   P2   |
Burden    Low   |  P3      |   P4   |

P1 = Critical (high damage, low capacity) - Urgent capacity building
P2 = Exposure management (high damage, high capacity) - Mitigation focus
P3 = Capacity building (low damage, low capacity) - Baseline capacity needed
P4 = Low priority (low damage, high capacity) - Adequate resources
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Define priority labels
PRIORITY_LABELS = {
    (1, 0): 'P1: Critical',              # High damage, Low capacity
    (1, 1): 'P2: Exposure Mgmt',         # High damage, High capacity
    (0, 0): 'P3: Capacity Building',     # Low damage, Low capacity
    (0, 1): 'P4: Low Priority'           # Low damage, High capacity
}

PRIORITY_COLORS = {
    'P1: Critical': '#8B0000',           # Dark red
    'P2: Exposure Mgmt': '#FFA500',      # Orange
    'P3: Capacity Building': '#FFD700',  # Gold
    'P4: Low Priority': '#90EE90'        # Light green
}

def load_data():
    """Load county geometries and annual metrics."""
    print("Loading data...")
    
    # Load county shapefile
    counties = gpd.read_file('/Users/simonameiler/Documents/work/03_code/repos/hurricane_recovery_potential/data/US_counties.shp')
    counties['FIPS'] = counties['GEOID'].astype(str).str.zfill(5)
    
    # Load annual metrics from compare_median_vs_max_events.py output
    # We need to recompute this here since we need the raw values
    
    # Load construction capacity
    permits = pd.read_csv('/Users/simonameiler/Documents/work/03_code/repos/hurricane_recovery_potential/data/selected_states_counties_with_permits.csv')
    permits['fips'] = permits['FIPS'].astype(str).str.zfill(5)
    permits_dict = permits.set_index('fips')['Average_Building_Permits(12 months)'].to_dict()
    
    # Load all event impacts to compute EAWUA
    impacts_dir = Path('/Users/simonameiler/Documents/work/03_code/repos/hurricane_recovery_potential/impacts_out/by_event/scaled')
    
    all_data = []
    for csv_file in sorted(impacts_dir.glob('*.csv')):
        df = pd.read_csv(csv_file)
        event_name = csv_file.stem
        df['event'] = event_name
        all_data.append(df)
    
    all_events = pd.concat(all_data, ignore_index=True)
    all_events['fips'] = all_events['fips'].astype(str).str.zfill(5)
    
    print(f"Loaded {len(all_events):,} county-event impact records")
    
    return counties, all_events, permits_dict

def compute_weighted_damage(row):
    """Compute weighted damage using DS recovery time weights."""
    ds1 = row.get('units_DS1_scaled', 0) if 'units_DS1_scaled' in row else 0
    ds2 = row.get('units_DS2_scaled', 0) if 'units_DS2_scaled' in row else 0
    ds3 = row.get('units_DS3_scaled', 0) if 'units_DS3_scaled' in row else 0
    ds4 = row.get('units_DS4_scaled', 0) if 'units_DS4_scaled' in row else 0
    return ds1 * 1 + ds2 * 1 + ds3 * 3 + ds4 * 6

def compute_annual_metrics(all_events, permits_dict):
    """Compute annual metrics for each county."""
    print("Computing annual metrics...")
    
    # Compute weighted damage for each event-county
    all_events['weighted_damage'] = all_events.apply(compute_weighted_damage, axis=1)
    
    # Get event frequency (assuming uniform probability)
    n_events = all_events['event'].nunique()
    event_frequency = 0.00067334  # events per year
    event_prob = event_frequency / n_events
    
    # Compute EAWUA (Expected Annual Weighted Units Affected)
    county_metrics = all_events.groupby('fips').agg({
        'weighted_damage': lambda x: (x * event_prob).sum(),
        'event': 'count'
    }).reset_index()
    
    county_metrics.columns = ['fips', 'eawua', 'n_events']
    
    # Add construction capacity
    county_metrics['capacity'] = county_metrics['fips'].map(permits_dict)
    
    # Filter out counties with missing data
    county_metrics = county_metrics.dropna()
    
    print(f"Computed metrics for {len(county_metrics)} counties")
    
    return county_metrics

def categorize_median_split(values, labels=['Low', 'High']):
    """Categorize values into two groups split at median."""
    median_val = values.median()
    categories = (values > median_val).astype(int)  # 0 for Low, 1 for High
    return categories, median_val

def assign_priority_categories(metrics):
    """Assign priority categories based on 2x2 matrix."""
    print("Assigning priority categories...")
    
    # Categorize damage burden (EAWUA) - High damage = High burden (1), Low damage = Low burden (0)
    metrics['damage_burden_cat'], damage_median = categorize_median_split(metrics['eawua'])
    
    # Categorize capacity adequacy - High capacity = High adequacy (1), Low capacity = Low adequacy (0)
    metrics['capacity_adequacy_cat'], capacity_median = categorize_median_split(metrics['capacity'])
    
    # Assign priority based on combination
    metrics['priority_code'] = list(zip(metrics['damage_burden_cat'], 
                                        metrics['capacity_adequacy_cat']))
    metrics['priority_category'] = metrics['priority_code'].map(PRIORITY_LABELS)
    
    # Add numerical priority for sorting (1=highest, 4=lowest)
    priority_rank = {
        'P1: Critical': 1,
        'P2: Exposure Mgmt': 2,
        'P3: Capacity Building': 3,
        'P4: Low Priority': 4
    }
    metrics['priority_rank'] = metrics['priority_category'].map(priority_rank)
    
    print(f"\nDamage burden median: {damage_median:.1f}")
    print(f"Capacity adequacy median: {capacity_median:.1f}")
    
    return metrics, damage_median, capacity_median

def create_multipanel_map(gdf, damage_median, capacity_median):
    """Create multi-panel visualization with damage, capacity, and priority maps."""
    print("Creating multi-panel visualization...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Panel 1: Damage Burden (EAWUA)
    ax1 = plt.subplot(2, 3, 1, projection=None)
    gdf_plot = gdf[gdf['eawua'].notna()]
    gdf_plot.plot(column='eawua', ax=ax1, legend=True, cmap='YlOrRd',
                  legend_kwds={'label': 'Expected Annual Weighted Units', 'shrink': 0.8})
    ax1.set_title('A) Damage Burden (EAWUA)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Add median line to legend
    ax1.text(0.02, 0.98, f'Median: {damage_median:.1f}',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 2: Capacity Adequacy
    ax2 = plt.subplot(2, 3, 2, projection=None)
    gdf_plot = gdf[gdf['capacity'].notna()]
    gdf_plot.plot(column='capacity', ax=ax2, legend=True, cmap='YlGn',
                  legend_kwds={'label': 'Construction Capacity (units/month)', 'shrink': 0.8})
    ax2.set_title('B) Capacity Adequacy', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add median line to legend
    ax2.text(0.02, 0.98, f'Median: {capacity_median:.1f}',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 3: Priority Categories
    ax3 = plt.subplot(2, 3, 3, projection=None)
    
    # Create color mapping
    gdf['color'] = gdf['priority_category'].map(PRIORITY_COLORS)
    gdf_valid = gdf[gdf['priority_category'].notna()]
    
    for cat in gdf_valid['priority_category'].unique():
        gdf_valid[gdf_valid['priority_category'] == cat].plot(
            color=PRIORITY_COLORS[cat], ax=ax3, edgecolor='white', linewidth=0.3
        )
    
    ax3.set_title('C) Priority Categories', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Create custom legend
    legend_elements = [mpatches.Patch(facecolor=PRIORITY_COLORS[cat], 
                                      edgecolor='black', label=cat)
                      for cat in sorted(PRIORITY_COLORS.keys(), 
                                       key=lambda x: int(x[1]))]
    ax3.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
              frameon=True, fontsize=11)
    
    # Panel 4-6: Priority distribution table and statistics
    ax4 = plt.subplot(2, 3, (4, 6))
    ax4.axis('off')
    
    # Count by priority category
    priority_counts = gdf['priority_category'].value_counts().sort_index()
    priority_pcts = (priority_counts / priority_counts.sum() * 100).round(1)
    
    # Create summary table
    table_data = []
    for cat in sorted(priority_counts.index, key=lambda x: int(x[1])):
        count = priority_counts[cat]
        pct = priority_pcts[cat]
        avg_damage = gdf[gdf['priority_category'] == cat]['eawua'].mean()
        avg_capacity = gdf[gdf['priority_category'] == cat]['capacity'].mean()
        table_data.append([cat, count, f'{pct}%', f'{avg_damage:.1f}', f'{avg_capacity:.1f}'])
    
    # Create table
    table = ax4.table(cellText=table_data,
                     colLabels=['Priority Category', 'Counties', '%', 'Avg EAWUA', 'Avg Capacity'],
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
    for i, cat in enumerate(sorted(priority_counts.index, key=lambda x: int(x[1])), 1):
        table[(i, 0)].set_facecolor(PRIORITY_COLORS[cat])
    
    # Add interpretation text
    interpretation = """
Priority Interpretation:

P1: Critical - High damage burden with limited recovery capacity
    → Urgent need for capacity building and disaster preparedness programs
    
P2: Exposure Management - High damage burden but adequate capacity
    → Focus on mitigation to reduce exposure (building codes, land use)
    
P3: Capacity Building - Low damage burden but limited capacity
    → Build baseline recovery capacity for future resilience
    
P4: Low Priority - Low damage burden with adequate capacity
    → Maintain current preparedness standards
    """
    ax4.text(0, 0.35, interpretation, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.suptitle('Recovery Priority Index: 2×2 Classification\n(Damage Burden × Capacity Adequacy)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('/Users/simonameiler/Documents/work/03_code/repos/hurricane_recovery_potential/analysis_output/priority_index_2x2_multipanel.png',
                dpi=300, bbox_inches='tight')
    print("Saved: priority_index_2x2_multipanel.png")
    
    plt.close()

def create_top_counties_table(metrics, n=30):
    """Create detailed table of top priority counties."""
    print(f"\nGenerating top {n} priority counties table...")
    
    # Sort by priority rank, then by EAWUA within each priority
    top_counties = metrics.sort_values(['priority_rank', 'eawua'], 
                                       ascending=[True, False]).head(n)
    
    # Format for output
    output = top_counties[['fips', 'priority_category', 'priority_rank', 
                           'eawua', 'capacity', 'n_events']].copy()
    output.columns = ['FIPS', 'Priority_Category', 'Priority_Rank', 
                     'EAWUA', 'Capacity', 'N_Events']
    
    # Save to CSV
    output.to_csv('/Users/simonameiler/Documents/work/03_code/repos/hurricane_recovery_potential/analysis_output/top_priority_counties.csv',
                  index=False)
    print(f"Saved: top_priority_counties.csv")
    
    return output

def create_priority_matrix_heatmap(metrics):
    """Create a 2x2 matrix heatmap showing county distribution."""
    print("Creating priority matrix heatmap...")
    
    # Create cross-tabulation
    matrix = pd.crosstab(metrics['damage_burden_cat'], 
                         metrics['capacity_adequacy_cat'])
    
    # Ensure we have a 2x2 matrix (fill missing cells with 0)
    for i in [0, 1]:
        for j in [0, 1]:
            if i not in matrix.index:
                matrix.loc[i] = 0
            if j not in matrix.columns:
                matrix[j] = 0
    
    # Sort to ensure proper order
    matrix = matrix.sort_index().sort_index(axis=1)
    
    # Reorder to match conventional layout (high damage at top)
    matrix = matrix.iloc[::-1]  # Reverse rows so high damage is at top
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(matrix.values, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(['Low', 'High'], fontsize=14)
    ax.set_yticklabels(['High', 'Low'], fontsize=14)
    
    ax.set_xlabel('Capacity Adequacy', fontsize=16, fontweight='bold')
    ax.set_ylabel('Damage Burden', fontsize=16, fontweight='bold')
    ax.set_title('County Distribution Across Priority Matrix', fontsize=18, fontweight='bold', pad=20)
    
    # Add text annotations with counts and priority labels
    for i in range(2):
        for j in range(2):
            damage_cat = 1 - i  # Reverse mapping due to flipped rows
            capacity_cat = j
            count = int(matrix.iloc[i, j])
            priority_label = PRIORITY_LABELS[(damage_cat, capacity_cat)]
            
            text = ax.text(j, i, f'{priority_label}\n({count} counties)',
                          ha="center", va="center", color="black", fontsize=12,
                          weight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Counties', rotation=270, labelpad=20, fontsize=14)
    
    plt.tight_layout()
    plt.savefig('/Users/simonameiler/Documents/work/03_code/repos/hurricane_recovery_potential/analysis_output/priority_matrix_heatmap.png',
                dpi=300, bbox_inches='tight')
    print("Saved: priority_matrix_heatmap.png")
    
    plt.close()

def main():
    """Main execution function."""
    print("="*60)
    print("Creating 2×2 Priority Index")
    print("="*60)
    
    # Load data
    counties, all_events, permits_dict = load_data()
    
    # Compute annual metrics
    metrics = compute_annual_metrics(all_events, permits_dict)
    
    # Assign priority categories
    metrics, damage_median, capacity_median = assign_priority_categories(metrics)
    
    # Merge with geometries
    gdf = counties.merge(metrics, left_on='FIPS', right_on='fips', how='left')
    
    # Create visualizations
    create_multipanel_map(gdf, damage_median, capacity_median)
    create_priority_matrix_heatmap(metrics)
    
    # Create top counties table
    top_counties = create_top_counties_table(metrics, n=30)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PRIORITY DISTRIBUTION SUMMARY")
    print("="*60)
    for cat in sorted(metrics['priority_category'].unique(), key=lambda x: int(x[1])):
        count = (metrics['priority_category'] == cat).sum()
        pct = count / len(metrics) * 100
        print(f"{cat:25s}: {count:3d} counties ({pct:5.1f}%)")
    
    print(f"\nTotal counties analyzed: {len(metrics)}")
    print(f"\nTop 5 Priority Counties:")
    print(top_counties.head()[['FIPS', 'Priority_Category', 'EAWUA', 'Capacity']].to_string(index=False))
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

if __name__ == '__main__':
    main()
