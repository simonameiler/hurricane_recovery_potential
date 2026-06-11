"""
Create bivariate choropleth maps for US Atlantic coast counties.

Map A: Risk (EAUA) × Recovery Potential  (inverted EARP: high EARP = low recovery potential)
Map B: Risk (EAUA) × Construction Capacity (CC)

Outputs (saved to analysis_output/):
  bivariate_map_A_risk_vs_recovery.png / .pdf
  bivariate_map_B_risk_vs_capacity.png  / .pdf
  bivariate_maps_combined.png           / .pdf

Run with:
  conda activate climada_env && python scripts/create_bivariate_maps.py
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'analysis_output'

DEFAULT_FREQ = 0.00067334  # events per year (Poisson rate used throughout the study)

COASTAL_STATE_FIPS = [
    '01', '09', '10', '12', '13', '22', '23', '24', '25', '28',
    '33', '34', '36', '37', '42', '44', '45', '48', '51',
]

MAP_XLIM = (-107, -65)
MAP_YLIM = (24, 48)

NO_DATA_COLOR = "#bab9b9" #bab9b9, "#e0e0e0"

# ---------------------------------------------------------------------------
# Bivariate color grids
# ---------------------------------------------------------------------------
# Indexing convention for both grids:
#   color = GRID[y_tercile - 1][x_tercile - 1]   (y = row axis, x = column axis)
#
# Map A  — GRID_A[earp_tercile - 1][eaua_tercile - 1]
#   Row 0 = EARP tercile 1  (low EARP  = high recovery potential)
#   Row 2 = EARP tercile 3  (high EARP = low  recovery potential)
#   Col 0 = EAUA tercile 1  (low risk)
#   Col 2 = EAUA tercile 3  (high risk)
#
# Steven's pink-purple-blue bivariate scheme
# Row indexed by earp_tercile: 1=low EARP=high recovery (good), 3=high EARP=low recovery (bad)
GRID_A = [
    ['#e8e8e8', '#ace4e4', '#5ac8c8'],   # EARP tercile 1: low EARP = high recovery potential (light = safe)
    ['#dfb0d6', '#a5add3', '#5698b9'],   # EARP tercile 2: medium
    ['#be64ac', '#8c62aa', '#3b4994'],   # EARP tercile 3: high EARP = low recovery potential (dark = concerning)
]

# Map B — Risk (blue) x Construction Capacity (green/teal)
# Columns: EAUA low → high = light → blue (SAME as Map A)
# Rows indexed by inv_cc_tercile (=4-cc_tercile):
#   inv=1 (High CC, good) → row 0 = light/safe colours
#   inv=3 (Low CC, bad)   → row 2 = dark/concerning colours
GRID_B = [
    ['#e8e8e8', '#ace4e4', '#5ac8c8'],   # inv_cc=1: high capacity (light = safe)
    ['#b8d6be', '#90b2b3', '#567994'],   # inv_cc=2: medium
    ['#73ae80', '#5a9178', '#2a5a5b'],   # inv_cc=3: low capacity  (dark = concerning)
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data():
    """Load EAUA, EARP, CC, and coastal county shapefile."""

    # Expected Annual Units Affected (EAUA)
    print("  Loading damage metrics...")
    damage_df = pd.read_csv(OUTPUT_DIR / 'county_event_frequency_damage_metrics.csv')
    damage_df['fips'] = damage_df['fips'].astype(str).str.zfill(5)
    # total_weighted_damage_units = Σ (weighted_damage) across all events per county
    # EAUA = Σ(weighted_damage) × event_frequency
    damage_df['eaua'] = damage_df['total_weighted_damage_units'] * DEFAULT_FREQ
    eaua_df = damage_df[['fips', 'eaua']].copy()

    # Expected Annual Recovery Potential (EARP)
    print("  Loading EARP metrics...")
    earp_raw = pd.read_csv(OUTPUT_DIR / 'earp_per_county.csv')
    earp_raw['fips'] = earp_raw['fips'].astype(str).str.zfill(5)
    earp_df = earp_raw[['fips', 'earp_months_per_year']].rename(
        columns={'earp_months_per_year': 'earp'}
    )

    # Construction Capacity (CC)
    print("  Loading construction capacity...")
    permits_df = pd.read_csv(DATA_DIR / 'selected_states_counties_with_permits.csv')
    permits_df['fips'] = permits_df['FIPS'].astype(str).str.zfill(5)
    permits_df['cc'] = permits_df['Average_Building_Permits(12 months)'] / 12
    cc_df = permits_df[['fips', 'cc']].copy()

    # County shapefile
    print("  Loading county shapefile...")
    counties = gpd.read_file(DATA_DIR / 'US_counties.shp')
    coastal_counties = counties[counties['STATEFP'].isin(COASTAL_STATE_FIPS)].copy()
    coastal_counties['GEOID'] = (
        coastal_counties['STATEFP'] + coastal_counties['COUNTYFP']
    ).str.zfill(5)

    return eaua_df, earp_df, cc_df, coastal_counties


# ---------------------------------------------------------------------------
# Tercile assignment
# ---------------------------------------------------------------------------

def assign_tercile(series):
    """
    Assign tercile labels 1 / 2 / 3 (as float) to a pandas Series.
    Only counties with strictly positive, finite values are included in the
    tercile computation; all others receive NaN.
    """
    valid_mask = series.notna() & np.isfinite(series) & (series > 0)
    result = pd.Series(np.nan, index=series.index, dtype=float)
    if valid_mask.sum() < 3:
        return result
    try:
        labels = pd.qcut(series[valid_mask], q=3, labels=[1, 2, 3],
                         duplicates='drop')
    except ValueError:
        labels = pd.cut(series[valid_mask], bins=3, labels=[1, 2, 3])
    result[valid_mask] = labels.astype(float)
    return result


# ---------------------------------------------------------------------------
# Metrics preparation & color assignment
# ---------------------------------------------------------------------------

def _get_color(row, row_col, col_col, grid):
    t_row = row[row_col]
    t_col = row[col_col]
    if pd.isna(t_row) or pd.isna(t_col):
        return NO_DATA_COLOR
    return grid[int(t_row) - 1][int(t_col) - 1]


def prepare_geodataframe(coastal_counties, eaua_df, earp_df, cc_df):
    """
    Merge metrics with the county GeoDataFrame and compute bivariate colors.
    Returns the merged GeoDataFrame with columns:
      eaua, earp, cc,
      eaua_tercile, earp_tercile, cc_tercile,
      color_a, color_b
    """
    metrics = (
        eaua_df
        .merge(earp_df, on='fips', how='outer')
        .merge(cc_df,   on='fips', how='outer')
    )

    # Zero / negative values → no meaningful data
    for col in ('eaua', 'earp', 'cc'):
        metrics.loc[~(metrics[col] > 0), col] = np.nan

    metrics['eaua_tercile'] = assign_tercile(metrics['eaua'])
    metrics['earp_tercile'] = assign_tercile(metrics['earp'])
    metrics['cc_tercile']   = assign_tercile(metrics['cc'])

    gdf = coastal_counties.merge(metrics, left_on='GEOID', right_on='fips', how='left')

    gdf['color_a'] = gdf.apply(
        lambda r: _get_color(r, 'earp_tercile', 'eaua_tercile', GRID_A), axis=1
    )
    # Invert CC tercile so high CC (good) → row 0 and low CC (bad) → row 2,
    # matching the EARP inversion logic in Map A.
    gdf['inv_cc_tercile'] = 4 - gdf['cc_tercile']  # NaN stays NaN
    gdf['color_b'] = gdf.apply(
        lambda r: _get_color(r, 'inv_cc_tercile', 'eaua_tercile', GRID_B), axis=1
    )
    return gdf


# ---------------------------------------------------------------------------
# Bivariate legend
# ---------------------------------------------------------------------------

def add_bivariate_legend(ax, colors_display, xlabel, ylabel,
                          x_low='Low', x_high='High',
                          y_low='Low', y_high='High',
                          bbox=(0.72, 0.01, 0.27, 0.34)):
    """
    Draw a 3×3 bivariate colour legend as an inset axes.

    colors_display : 3×3 list of hex strings
        colors_display[0][...] → bottom row (y_low)
        colors_display[2][...] → top row    (y_high)
    bbox : (x0, y0, width, height) in axes-fraction coordinates
    """
    axins = ax.inset_axes(bbox)

    for row_idx in range(3):
        for col_idx in range(3):
            rect = mpatches.Rectangle(
                (col_idx, row_idx), 1, 1,
                facecolor=colors_display[row_idx][col_idx],
                edgecolor='white', linewidth=0.5,
                transform=axins.transData,
            )
            axins.add_patch(rect)

    axins.set_xlim(-1.0, 3.5)
    axins.set_ylim(-1.0, 3.5)
    axins.set_aspect('equal')
    axins.axis('off')

    # Arrows along x and y
    kw_arrow = dict(arrowstyle='->', color='black', lw=0.8)
    axins.annotate('', xy=(3.0, -0.5), xytext=(0.0, -0.5),
                   xycoords='data', textcoords='data',
                   arrowprops=kw_arrow, annotation_clip=False)
    # Y-axis: plain line only (no arrowhead)
    axins.annotate('', xy=(-0.5, 3.0), xytext=(-0.5, 0.0),
                   xycoords='data', textcoords='data',
                   arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                   annotation_clip=False)

    fs_label = 7
    fs_tick  = 6

    # Axis labels (shifted slightly left / down relative to arrows)
    axins.text(1.3, -1.05, xlabel, ha='center', va='top',
               fontsize=fs_label, transform=axins.transData, clip_on=False)
    axins.text(-1.3, 1.3, ylabel, ha='center', va='center',
               fontsize=fs_label, rotation=90,
               transform=axins.transData, clip_on=False)

    # Low / High tick labels
    axins.text(0.0, -0.65, x_low,  ha='left',  va='top',
               fontsize=fs_tick, transform=axins.transData, clip_on=False)
    axins.text(3.0, -0.65, x_high, ha='right', va='top',
               fontsize=fs_tick, transform=axins.transData, clip_on=False)
    axins.text(-0.65, 0.0, y_low,  ha='right', va='bottom',
               fontsize=fs_tick, rotation=90,
               transform=axins.transData, clip_on=False)
    axins.text(-0.65, 3.0, y_high, ha='right', va='top',
               fontsize=fs_tick, rotation=90,
               transform=axins.transData, clip_on=False)

    return axins


# ---------------------------------------------------------------------------
# Map rendering
# ---------------------------------------------------------------------------

def render_map(gdf, state_gdf, color_col, ax,
               colors_display=None, xlabel='', ylabel='',
               y_low='Low', y_high='High',
               panel_label=None):
    """Plot bivariate choropleth with optional legend inset."""
    # County fill
    gdf.plot(ax=ax, color=gdf[color_col],
             edgecolor='#aaaaaa', linewidth=0.15)
    # State borders
    state_gdf.plot(ax=ax, facecolor='none',
                   edgecolor='#444444', linewidth=0.7)

    ax.set_xlim(MAP_XLIM)
    ax.set_ylim(MAP_YLIM)
    ax.set_aspect('equal')
    ax.axis('off')

    if panel_label is not None:
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left')

    if colors_display is not None:
        add_bivariate_legend(ax, colors_display,
                             xlabel=xlabel, ylabel=ylabel,
                             y_low=y_low, y_high=y_high)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(gdf):
    print("\n=== Tercile thresholds ===")
    for col, name in [('eaua', 'EAUA (weighted units/yr)'),
                      ('earp', 'EARP (months/yr)'),
                      ('cc',   'CC (permits/month)')]:
        valid = gdf[col].dropna() if col in gdf.columns else pd.Series(dtype=float)
        valid = valid[valid > 0]
        if len(valid) > 0:
            p33 = valid.quantile(1 / 3)
            p67 = valid.quantile(2 / 3)
            print(f"  {name:35s}  p33={p33:.4f}  p67={p67:.4f}  n={len(valid)}")

    print("\n=== Map A: Risk (EAUA) × Recovery Potential (inverted EARP) ===")
    print(f"  {'EARP tercile':15s}  {'EAUA tercile':15s}  {'n counties':>10s}")
    total_a = 0
    for earp_t in [1, 2, 3]:
        for eaua_t in [1, 2, 3]:
            mask = (gdf['earp_tercile'] == earp_t) & (gdf['eaua_tercile'] == eaua_t)
            n = int(mask.sum())
            total_a += n
            color = GRID_A[earp_t - 1][eaua_t - 1]
            print(f"  earp={earp_t} (row {earp_t})      eaua={eaua_t} (col {eaua_t})      {n:>10d}  {color}")
    n_nd = int(gdf['color_a'].eq(NO_DATA_COLOR).sum())
    print(f"  No data:                                    {n_nd:>10d}")
    print(f"  Total:                                      {total_a + n_nd:>10d}")

    print("\n=== Map B: Risk (EAUA) × Construction Capacity (CC) ===")
    print(f"  {'CC tercile':15s}   {'EAUA tercile':15s}  {'n counties':>10s}")
    total_b = 0
    for cc_t in [1, 2, 3]:
        for eaua_t in [1, 2, 3]:
            mask = (gdf['cc_tercile'] == cc_t) & (gdf['eaua_tercile'] == eaua_t)
            n = int(mask.sum())
            total_b += n
            color = GRID_B[cc_t - 1][eaua_t - 1]
            print(f"  cc={cc_t} (row {cc_t})          eaua={eaua_t} (col {eaua_t})      {n:>10d}  {color}")
    n_nd = int(gdf['color_b'].eq(NO_DATA_COLOR).sum())
    print(f"  No data:                                    {n_nd:>10d}")
    print(f"  Total:                                      {total_b + n_nd:>10d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    eaua_df, earp_df, cc_df, coastal_counties = load_all_data()

    print("\nMerging metrics and computing terciles...")
    gdf = prepare_geodataframe(coastal_counties, eaua_df, earp_df, cc_df)

    # State boundaries for overlay (dissolve counties by state FIPS)
    state_gdf = coastal_counties.dissolve(by='STATEFP').reset_index()

    # Summary statistics
    print_summary(gdf)

    # ------------------------------------------------------------------
    # Legend colour order for display
    #   Display grid: [0] = bottom row (low on y-axis), [2] = top row (high on y-axis)
    #
    # Map A:  GRID_A[0]=light=High recovery (bottom), GRID_A[2]=dark=Low recovery (top)
    #         → use as-is; y-axis labels are inverted (High at bottom, Low at top)
    colors_a_display = GRID_A
    #
    # Map B:  GRID_B[0]=light=High CC (bottom), GRID_B[2]=dark=Low CC (top)
    #         → use as-is; y-axis labels are inverted (High at bottom, Low at top)
    colors_b_display = GRID_B
    # ------------------------------------------------------------------

    # ---- Map A --------------------------------------------------------
    print("\nCreating Map A (Risk × Recovery Potential)...")
    fig_a, ax_a = plt.subplots(1, 1, figsize=(8, 6))
    render_map(gdf, state_gdf, 'color_a', ax_a,
               colors_display=colors_a_display,
               xlabel='Risk (EAUA)',
               ylabel='Recovery potential',
               y_low='High', y_high='Low')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out = OUTPUT_DIR / f'bivariate_map_A_risk_vs_recovery.{ext}'
        plt.savefig(out, dpi=300, bbox_inches='tight')
    print("✓ Saved Map A (PNG + PDF)")
    plt.close()

    # ---- Map B --------------------------------------------------------
    print("\nCreating Map B (Risk × Capacity)...")
    fig_b, ax_b = plt.subplots(1, 1, figsize=(8, 6))
    render_map(gdf, state_gdf, 'color_b', ax_b,
               colors_display=colors_b_display,
               xlabel='Risk (EAUA)',
               ylabel='Construction capacity',
               y_low='High', y_high='Low')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out = OUTPUT_DIR / f'bivariate_map_B_risk_vs_capacity.{ext}'
        plt.savefig(out, dpi=300, bbox_inches='tight')
    print("✓ Saved Map B (PNG + PDF)")
    plt.close()

    # ---- Combined 2-panel figure --------------------------------------
    # Publication figsize: 174 mm wide (2-column Nature/AGU width) × 80 mm tall
    print("\nCreating combined 2-panel figure...")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    # Panel a (left): Map B — Risk × Construction Capacity
    render_map(gdf, state_gdf, 'color_b', axes[0],
               colors_display=colors_b_display,
               xlabel='Risk (EAUA)',
               ylabel='Construction capacity',
               y_low='High', y_high='Low',
               panel_label='a')
    # Panel b (right): Map A — Risk × Recovery Potential
    render_map(gdf, state_gdf, 'color_a', axes[1],
               colors_display=colors_a_display,
               xlabel='Risk (EAUA)',
               ylabel='Recovery potential',
               y_low='High', y_high='Low',
               panel_label='b')
    plt.tight_layout(pad=0.3)
    for ext in ('png', 'pdf'):
        out = OUTPUT_DIR / f'bivariate_maps_combined.{ext}'
        plt.savefig(out, dpi=300, bbox_inches='tight')
    print("✓ Saved combined figure (PNG + PDF)")
    plt.close()

    print("\nDone.")


if __name__ == '__main__':
    main()
