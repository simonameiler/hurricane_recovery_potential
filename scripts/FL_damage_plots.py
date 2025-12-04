#!/usr/bin/env python3
#%%
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
res_dir = Path('../data/results')     # path to your damage_df CSV
exp_dir = Path('../data/exposure')    # path to CBSA and county shapefiles
fig_dir = Path('../plots')            # where to save figures

event = '2022266N12294'               # storm ID
ds_col = f'DS_{event}'                # damage-state column in damage_df

# ------------------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------------------
#damage_csv = res_dir / 'FL_HIIIMMN_capra_cbsa.csv'
damage_csv = res_dir / 'summary_cbsa_damage_all_2014-2024'
df_damage = pd.read_csv(damage_csv)
df_damage['damage_state'] = (
    df_damage[ds_col]
      .str.extract(r'DS(\d)', expand=False)
      .fillna('0')
      .astype(int)
)
cbsa = gpd.read_file(exp_dir / 'CoreBasedStatisticalAreas_USA.geojson')
counties = gpd.read_file(exp_dir / 'tl_2024_us_county')
fl_counties = counties[counties['STATEFP']=='12']
fl_state = fl_counties.dissolve()
gdf_cbsa = cbsa.to_crs(fl_state.crs)
gdf_cbsa_fl = gpd.clip(gdf_cbsa[['NAME','geometry']], fl_state)

# ------------------------------------------------------------------------------
# 2. SUMMARIZE METRICS BY DAMAGE STATE
# ------------------------------------------------------------------------------
def summarize_metric(df, metric_col):
    df2 = df[df['damage_state'].between(1,4)]
    grp = (
        df2.groupby(['cbsa_name','damage_state'])[metric_col]
           .sum()
           .reset_index(name=metric_col)
    )
    wide = grp.pivot(index='cbsa_name', columns='damage_state', values=metric_col)
    dcols = {ds: f"{metric_col}_DS{ds}" for ds in wide.columns}
    wide = wide.rename(columns=dcols).fillna(0).reset_index()
    return wide

df_cost = summarize_metric(df_damage, f'RepairCost_{event}')
df_units = summarize_metric(df_damage, 'NumberOfUnits')

def merge_summaries(gdf, df_summary):
    merged = gdf.merge(df_summary, left_on='NAME', right_on='cbsa_name', how='left')
    for col in df_summary.columns:
        if col.startswith((f'RepairCost_{event}', 'NumberOfUnits')):
            merged[col] = merged[col].fillna(0)
    return merged

gdf_repair = merge_summaries(gdf_cbsa_fl, df_cost)
gdf_units  = merge_summaries(gdf_cbsa_fl, df_units)

# ------------------------------------------------------------------------------
# 3. PLOTTING FUNCTION WITH SHARED COLORBAR
# ------------------------------------------------------------------------------
def plot_ds_panels(gdf, metric_prefix, title_fmt, cmap, cbar_label, out_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(24,6), constrained_layout=True)
    # compute global vmin/vmax across DS1-4
    cols = [f"{metric_prefix}_DS{ds}" for ds in [1,2,3,4]]
    all_data = gdf[cols].replace(0, np.nan)
    vmin = all_data.min(skipna=True).min() or 1
    vmax = all_data.max(skipna=True).max()
    norm = LogNorm(vmin=vmin, vmax=vmax)

    for ax, ds in zip(axes, [1,2,3,4]):
        col = f"{metric_prefix}_DS{ds}"
        fl_state.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
        fl_counties.boundary.plot(ax=ax, edgecolor='black', linewidth=0.1)
        gdf[gdf[col]==0].plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.8)
        gdf[gdf[col]>0].plot(
            ax=ax, column=col, cmap=cmap, norm=norm,
            edgecolor='black', linewidth=0.8
        )
        ax.set_title(title_fmt.format(ds), fontsize=16)
        ax.axis('off')
        ax.set_aspect('equal')

    # shared colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm._A = []
    cbar = fig.colorbar(
        sm, ax=axes.tolist(), orientation='vertical',
        fraction=0.08, pad=0.02, shrink=0.8
    )
    cbar.set_label(cbar_label, fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    # legend below figure
    legend_elems = [
        Patch(facecolor='lightgrey', edgecolor='white', label='No Data'),
        Patch(facecolor='white',    edgecolor='black', label='Outside CBSA')
    ]
    fig.legend(
        handles=legend_elems,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        fontsize=16,
        frameon=True
    )
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()

# ------------------------------------------------------------------------------
# 4. GENERATE PLOTS
# ------------------------------------------------------------------------------
plot_ds_panels(
    gdf_repair,
    metric_prefix=f'RepairCost_{event}',
    title_fmt='DS{}',
    cmap='Blues',
    cbar_label='Repair Cost (USD)',
    out_path=fig_dir / f'repair_cost_by_DS_{event}.png'
)

plot_ds_panels(
    gdf_units,
    metric_prefix='NumberOfUnits',
    title_fmt='DS{}',
    cmap='Oranges',
    cbar_label='Units Affected',
    out_path=fig_dir / f'units_by_DS_{event}.png'
)
