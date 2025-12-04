#!/usr/bin/env python3
#%%
import os
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from shapely.ops import unary_union

# Directories - update as needed
res_dir = Path('/Users/simonameiler/Documents/work/03_code/repos/recovery/data/results/')
exp_dir = Path('/Users/simonameiler/Documents/work/03_code/repos/recovery/data/exposure/')
fig_dir = Path('/Users/simonameiler/Documents/work/03_code/repos/recovery/plots/')

# Event identifier
event = "2022266N12294"

# ------------------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------------------

# Damage dataframe
damage_df = pd.read_csv(res_dir / 'FL_HIIIMMN_capra_cbsa.csv')

# CBSA and county shapefiles
cbsa = gpd.read_file(exp_dir / 'CoreBasedStatisticalAreas_USA.geojson')
counties = gpd.read_file(exp_dir / 'tl_2024_us_county')

# Filter Florida counties and get state outline
fl_counties = counties[counties['STATEFP'] == '12']
fl_state = fl_counties.dissolve()

# Ensure CRS alignment and clip CBSA to Florida
cbsa = cbsa.to_crs(fl_state.crs)
cbsa_fl = gpd.clip(cbsa, fl_state)

# Load recovery results and filter by event
recovery = pd.read_csv(res_dir / f'recovery_potential_{event}.csv')
recovery_event = recovery[recovery['event'] == event].copy()

# Merge recovery metrics into CBSA GeoDataFrame
cbsa_recovery = cbsa_fl.merge(
    recovery_event,
    left_on='NAME',
    right_on='cbsa_name',
    how='left'
)

# Fill missing recovery metrics with zeros
for col in ['reconstruction_capacity', 'recovery_time [months]']:
    if col in cbsa_recovery.columns:
        cbsa_recovery[col] = cbsa_recovery[col].fillna(0)
    else:
        raise KeyError(f"Column {col} missing in recovery data")

# ------------------------------------------------------------------------------
# 2. Metric-adding functions
# ------------------------------------------------------------------------------
def add_total_repair_cost(gdf, df_damage, event):
    repair_col = f'RepairCost_{event}'
    if repair_col not in df_damage.columns:
        raise KeyError(f"{repair_col} not found in damage_df")
    costs = df_damage.groupby('cbsa_name')[repair_col].sum()
    gdf['total_repair_cost'] = gdf['cbsa_name'].map(costs).fillna(0)
    return gdf


def add_units_affected(gdf, df_damage, event):
    repair_col = f'RepairCost_{event}'
    if repair_col not in df_damage.columns:
        raise KeyError(f"{repair_col} not found in damage_df")
    mask = df_damage[repair_col] > 0
    units = df_damage[mask].groupby('cbsa_name')['NumberOfUnits'].sum()
    gdf['units_affected'] = gdf['cbsa_name'].map(units).fillna(0)
    return gdf

# ------------------------------------------------------------------------------
# 3. Plotting helpers
# ------------------------------------------------------------------------------
def safe_plot(gdf, ax, **kwargs):
    if not gdf.empty:
        gdf.plot(ax=ax, **kwargs)


def plot_metrics_panels(gdf, fl_state, fl_counties, metrics, figsize=(15,5), save_path=None):
    """
    Plot side-by-side panels.
    metrics: list of tuples (col_name, subplot_title, cbar_label, cmap).
    If cbar_label is None, no ylabel is drawn.
    Recovery-time panel is inverted with 'high' at top, 'low' at bottom.
    """
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (col, subplot_title, cbar_label, cmap_name) in zip(axes, metrics):
        # prepare normalization
        data = gdf[col].replace(0, np.nan)
        vmin = data.min(skipna=True) or 1
        vmax = data.max(skipna=True)
        norm = LogNorm(vmin=vmin, vmax=vmax)

        # handle colormap inversion for recovery_time
        cmap_use = cmap_name + '_r' if col == 'recovery_time [months]' else cmap_name

        # plot outlines
        fl_state.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
        fl_counties.boundary.plot(ax=ax, edgecolor='black', linewidth=0.1)

        # missing, zero, and valid data
        safe_plot(gdf[gdf[col].isna()], ax, color='lightgrey', edgecolor='white', linewidth=0.8)
        safe_plot(gdf[gdf[col] == 0], ax, color='lightgrey', edgecolor='black', linewidth=0.8)
        safe_plot(gdf[gdf[col] > 0], ax,
                  column=col, cmap=cmap_use, norm=norm,
                  edgecolor='black', linewidth=0.8)

        # subplot title
        ax.set_title(subplot_title, fontsize=16)
        ax.axis('off')
        ax.set_aspect('equal')

        # colorbar
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_use)
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)

        if col == 'recovery_time [months]':
            cbar.set_ticks([norm.vmin, norm.vmax])
            cbar.set_ticklabels(['high', 'low'])
            cbar.ax.invert_yaxis()
            cbar.ax.tick_params(labelsize=16)
        else:
            if cbar_label is not None:
                cbar.ax.set_ylabel(cbar_label, fontsize=16)
            cbar.ax.tick_params(labelsize=16)

    # legend under the middle panel
    if n > 1:
        legend_elems = [
            Patch(facecolor='lightgrey', edgecolor='white', label='No Data'),
            Patch(facecolor='white',    edgecolor='black', label='Outside CBSA')
        ]
        mid = axes[n//2]
        mid.legend(handles=legend_elems,
                   loc='lower center',
                   bbox_to_anchor=(0.5, -0.12),
                   frameon=True,
                   fontsize=14, ncol=2)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
#%%
# ------------------------------------------------------------------------------
# 4. Plotting calls with title and label in metrics tuple
# ------------------------------------------------------------------------------

# 4.1 Two-panel: capacity and recovery potential
metrics_2 = [
    ('reconstruction_capacity', 'Construction Capacity', 'building permits/month', 'Greens'),
    ('recovery_time [months]',  'Recovery Potential',     None,                    'Purples'),
]
plot_metrics_panels(
    cbsa_recovery, fl_state, fl_counties, metrics_2,
    figsize=(12, 6),
    save_path=fig_dir / f'recovery_two_panel_{event}.png'
)
#%%
# 4.2 Three-panel: add total repair cost
cbsa_recovery = add_total_repair_cost(cbsa_recovery, damage_df, event)
metrics_3_cost = [
    ('reconstruction_capacity', 'Construction Capacity',  'building permits/month', 'Greens'),
    ('total_repair_cost',       'Total Repair Cost',      'USD',                     'YlOrBr'),
    ('recovery_time [months]',  'Recovery Potential',     None,                      'Purples'),
]
plot_metrics_panels(
    cbsa_recovery, fl_state, fl_counties, metrics_3_cost,
    figsize=(18, 6),
    save_path=fig_dir / f'recovery_cost_three_panel_{event}.png'
)
#%%
# 4.3 Three-panel: add units affected
cbsa_recovery = add_units_affected(cbsa_recovery, damage_df, event)
metrics_3_units = [
    ('units_affected',          'Total Units Affected',   'number of units',        'Oranges'),
    ('reconstruction_capacity', 'Construction Capacity',  'building permits/month', 'Greens'),
    ('recovery_time [months]',  'Recovery Potential',     None,                     'Purples'),
]
plot_metrics_panels(
    cbsa_recovery, fl_state, fl_counties, metrics_3_units,
    figsize=(18, 6),
    save_path=fig_dir / f'recovery_units_three_panel_{event}.png'
)

# %%
# 4.3 Three-panel: add units affected
cbsa_recovery = add_units_affected(cbsa_recovery, damage_df, event)
cbsa_recovery = add_total_repair_cost(cbsa_recovery, damage_df, event)
metrics_3_units = [
    ('units_affected',          'Total Units Affected',   'number of units',        'Oranges'),
    ('total_repair_cost',       'Total Repair Cost',      'USD',                     'Blues'),
    ('recovery_time [months]',  'Recovery Potential',     None,                     'Purples'),
]
plot_metrics_panels(
    cbsa_recovery, fl_state, fl_counties, metrics_3_units,
    figsize=(18, 6),
    save_path=fig_dir / f'recovery_cost-units_three_panel_{event}.png'
)

# %%
