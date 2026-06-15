"""
Create all manuscript figures (excluding bivariate maps) for the hurricane
recovery potential study.

Figures produced
----------------
Main text
  Figure 2  annual_3panel.png / .pdf           (EAUA, CC, EARP triptych)
  Figure 3  recovery_drivers_annual_vs_median.png / .pdf  (2×2 scatterplot)

Supplementary
  Figure S1 na_coast_hazard_overview.png / .pdf  (wind/surge hazard overview)
  Figure S2 event_350_3panel_map.png / .pdf
            event_4347_3panel_map.png / .pdf     (single-event 3-panel)
  Figure S5 median_event_3panel.png / .pdf
  Figure S6 max_event_3panel.png / .pdf
  Figure S7 recovery_drivers_annual_max.png / .pdf (2×2 scatter annual + max)
  Figure S8 skewness_maps.png / .pdf

All outputs are saved to analysis_output/ as PNG (300 dpi).

Run with:
  conda activate climada_env && python scripts/create_manuscript_figures.py

To generate only specific figures:
  python scripts/create_manuscript_figures.py --figures fig2 fig3 figS2
"""

import argparse
import json
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Print dimensions (IOPP: 8.5 cm single / 15 cm double column) + font settings
# ---------------------------------------------------------------------------
W_SINGLE = 3.35   # inches — 8.5 cm, single column
W_DOUBLE = 5.91   # inches — 15.0 cm, double column

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
})

# ---------------------------------------------------------------------------
# Configuration – visual style matching create_bivariate_maps.py
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "analysis_output"
FIGURES_DIR = OUTPUT_DIR

DEFAULT_FREQ = 0.00067334  # events per year (Poisson rate)
RECOVERY_WEIGHTS = {"DS1": 1.0, "DS2": 1.0, "DS3": 3.0, "DS4": 6.0}

COASTAL_STATE_FIPS = [
    "01", "09", "10", "12", "13", "22", "23", "24", "25", "28",
    "33", "34", "36", "37", "42", "44", "45", "48", "51",
]

MAP_XLIM = (-107, -65)
MAP_YLIM = (24, 48)

NO_DATA_COLOR = "#bab9b9"
COUNTY_EDGECOLOR = "#aaaaaa"
COUNTY_LINEWIDTH = 0.1
STATE_EDGECOLOR = "#444444"
STATE_LINEWIDTH = 0.3

# ---------------------------------------------------------------------------
# Bivariate color grids (Figure 4)
# ---------------------------------------------------------------------------
# Map A (EARP × EAUA) — Steven's pink-purple-blue scheme
#   Row 0 = EARP tercile 1 (low EARP = high recovery, safe)
#   Row 2 = EARP tercile 3 (high EARP = low recovery, concerning)
#   Col 0 = EAUA tercile 1 (low risk), Col 2 = EAUA tercile 3 (high risk)
GRID_A = [
    ["#e8e8e8", "#ace4e4", "#5ac8c8"],
    ["#dfb0d6", "#a5add3", "#5698b9"],
    ["#be64ac", "#8c62aa", "#3b4994"],
]
# Map B (inv-CC × EAUA) — green scheme
#   Row 0 = inv_cc=1 (high capacity, safe), Row 2 = inv_cc=3 (low capacity, concerning)
GRID_B = [
    ["#e8e8e8", "#ace4e4", "#5ac8c8"],
    ["#b8d6be", "#90b2b3", "#567994"],
    ["#73ae80", "#5a9178", "#2a5a5b"],
]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_fig(fig, stem: str):
    """Save figure as PNG (300 dpi) to FIGURES_DIR."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    print(f"  Saved  {stem}.png  →  {FIGURES_DIR}")


# ---------------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------------

def _choropleth_panel(gdf, state_gdf, ax, col, cmap, cbar_label,
                      norm=None, use_log=True, invert_cbar=False):
    """
    Draw a single choropleth panel on *ax* with a horizontal colorbar below.

    Parameters
    ----------
    gdf        : GeoDataFrame containing column *col* and geometry
    state_gdf  : GeoDataFrame used for state-border overlay
    ax         : matplotlib Axes to draw on
    col        : column name in *gdf* to plot
    cmap       : colormap string or object
    cbar_label : label for the colorbar
    norm       : matplotlib Norm; auto-computed from data if None
    use_log    : if True, use LogNorm (ignored when norm is supplied explicitly)
    """
    tmp = gdf[[col, "geometry"]].copy()
    tmp.loc[~(tmp[col] > 0), col] = np.nan

    valid = tmp[col].dropna()

    if norm is None and use_log and len(valid) > 0:
        vmin = valid.min()
        vmax = valid.max()
        if np.isfinite(vmin) and np.isfinite(vmax) and vmin > 0:
            norm = LogNorm(vmin=vmin / 2, vmax=vmax)

    # Colorbar axis slightly wider than the map panel
    cax = ax.inset_axes([0.125, -0.12, 0.75, 0.05])

    tmp.plot(
        column=col, cmap=cmap, norm=norm,
        edgecolor=COUNTY_EDGECOLOR, linewidth=COUNTY_LINEWIDTH,
        legend=True, ax=ax, cax=cax,
        legend_kwds={"orientation": "horizontal"},
        missing_kwds={
            "color": NO_DATA_COLOR,
            "edgecolor": COUNTY_EDGECOLOR,
            "linewidth": COUNTY_LINEWIDTH,
        },
    )

    # State borders overlay
    state_gdf.plot(ax=ax, facecolor="none",
                   edgecolor=STATE_EDGECOLOR, linewidth=STATE_LINEWIDTH)

    ax.set_xlim(MAP_XLIM)
    ax.set_ylim(MAP_YLIM)
    ax.set_aspect("equal")
    ax.axis("off")

    # Colorbar styling
    if invert_cbar:
        cax.invert_xaxis()
        cax.set_xticks([])
        cax.set_xlabel("Low     Recovery potential     High", fontsize=7, labelpad=3)
    else:
        cax.set_xlabel(cbar_label, fontsize=7, labelpad=2)
        cax.tick_params(labelsize=6)
        cax.tick_params(which="minor", length=0)
    for spine in cax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.5)


def _scatter_panel(ax, x, y, xlabel, ylabel, panel_label=None,
                   alpha=0.35, s=12, color="#2166ac"):
    """
    Draw a log-log scatter plot with Spearman ρ annotation.

    Parameters
    ----------
    ax           : matplotlib Axes
    x, y         : pandas Series (may contain NaN; zero values excluded)
    xlabel/ylabel: axis labels
    panel_label  : bold letter label in top-left corner
    """
    valid = (~x.isna()) & (~y.isna()) & (x > 0) & (y > 0)
    xv, yv = x[valid].values, y[valid].values

    ax.scatter(xv, yv, alpha=alpha, s=s, color=color, linewidths=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(axis="x", which="major", labelsize=6)
    ax.tick_params(axis="x", which="minor", length=0, width=0)
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    for spine in ax.spines.values():
        spine.set_edgecolor("0.4")
        spine.set_linewidth(0.8)

    if len(xv) >= 3:
        r, _ = spearmanr(xv, yv)
        n = len(xv)
        ax.text(
            0.04, 0.96,
            f"ρ = {r:+.2f}\nn = {n:,}",
            transform=ax.transAxes, fontsize=6,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="0.7", alpha=0.85),
        )

    if panel_label is not None:
        ax.text(0.02, 0.98, f"{panel_label})", transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")


def _color_scatter_panel(ax, x, y, c, xlabel, ylabel, clabel,
                          cmap, panel_label=None,
                          alpha=0.6, s=20, show_yticks=True):
    """
    Color-encoded log-log scatter with vertical colorbar and inverted y-axis.

    The y-axis is inverted so short recovery time (high RP) appears at the top.
    Correlation is Spearman ρ on log-transformed values.

    Parameters
    ----------
    ax           : matplotlib Axes
    x, y, c      : pandas Series – NaN and non-positive values are excluded
    xlabel/ylabel: axis labels
    clabel       : colorbar label
    cmap         : colormap string for the color variable
    panel_label  : letter rendered as "(a)" etc. in top-left corner
    show_yticks  : False on right panels to reduce clutter
    """
    valid = (
        (~x.isna()) & (~y.isna()) & (~c.isna())
        & (x > 0) & (y > 0) & (c > 0)
    )
    xv, yv, cv = x[valid].values, y[valid].values, c[valid].values

    sc = ax.scatter(
        xv, yv, c=cv, cmap=cmap, alpha=alpha, s=s,
        norm=LogNorm(vmin=cv.min(), vmax=cv.max()),
        linewidths=0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=7)
    if show_yticks:
        ax.set_ylabel(ylabel, fontsize=7)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_edgecolor("0.4")
        spine.set_linewidth(0.8)
    ax.tick_params(color="0.4", labelcolor="0.2")

    ax.tick_params(axis="x", which="major", labelsize=6)
    ax.tick_params(axis="x", which="minor", length=0, width=0)
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    # Side colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(clabel, fontsize=7)
    cbar.ax.tick_params(which="both", labelsize=6)
    cbar.ax.tick_params(which="minor", length=0)
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.5)

    # Spearman ρ annotation
    if len(xv) >= 3:
        r, _ = spearmanr(xv, yv)
        corr_x = 0.05 if show_yticks else 0.35
        ax.text(
            corr_x, 0.02,
            f"\u03c1 = {r:+.2f}\nn = {len(xv):,}",
            transform=ax.transAxes, fontsize=6, va="bottom",
        )

    if panel_label is not None:
        ax.text(0.02, 0.98, f"{panel_label})", transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")

    return cbar


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_spatial_data():
    """Load coastal county shapefile and derive state-boundary overlay."""
    print("  Loading county shapefile …")
    counties = gpd.read_file(DATA_DIR / "US_counties.shp")
    coastal = counties[counties["STATEFP"].isin(COASTAL_STATE_FIPS)].copy()
    coastal["GEOID"] = (coastal["STATEFP"] + coastal["COUNTYFP"]).str.zfill(5)
    state_gdf = coastal.dissolve(by="STATEFP").reset_index()
    print(f"    {len(coastal)} coastal counties, {len(state_gdf)} states")
    return coastal, state_gdf


def load_annual_metrics():
    """
    Load EAUA, EARP, and CC (all as fips-keyed DataFrames).

    Returns
    -------
    eaua_df : DataFrame with columns [fips, eaua]
    earp_df : DataFrame with columns [fips, earp]
    cc_df   : DataFrame with columns [fips, cc]
    """
    print("  Loading annual metrics …")

    # Expected Annual Units Affected (EAUA)
    dmg = pd.read_csv(OUTPUT_DIR / "county_event_frequency_damage_metrics.csv")
    dmg["fips"] = dmg["fips"].astype(str).str.zfill(5)
    dmg["eaua"] = dmg["total_weighted_damage_units"] * DEFAULT_FREQ
    eaua_df = dmg[["fips", "eaua"]].copy()

    # Expected Annual Recovery Potential (EARP; months/yr)
    earp_raw = pd.read_csv(OUTPUT_DIR / "earp_per_county.csv")
    earp_raw["fips"] = earp_raw["fips"].astype(str).str.zfill(5)
    earp_df = earp_raw[["fips", "earp_months_per_year"]].rename(
        columns={"earp_months_per_year": "earp"}
    )

    # Construction Capacity (CC; permits/month)
    permits = pd.read_csv(DATA_DIR / "selected_states_counties_with_permits.csv")
    permits["fips"] = permits["FIPS"].astype(str).str.zfill(5)
    permits["cc"] = permits["Average_Building_Permits(12 months)"] / 12
    cc_df = permits[["fips", "cc"]].copy()

    print(f"    EAUA: {len(eaua_df)} counties")
    print(f"    EARP: {len(earp_df)} counties")
    print(f"    CC:   {len(cc_df)} counties")
    return eaua_df, earp_df, cc_df


def load_event_level_metrics():
    """
    Load per-county median and max event metrics.

    Uses the pre-computed CSV if available, otherwise raises an error
    (the CSV is produced by compare_median_vs_max_events.py).

    Returns
    -------
    DataFrame with columns: fips, median_weighted_damage, median_recovery_months,
                            max_weighted_damage, max_recovery_months
    """
    print("  Loading per-event metrics …")
    csv_path = OUTPUT_DIR / "median_vs_max_event_comparison.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. "
            "Run scripts/compare_median_vs_max_events.py first."
        )
    df = pd.read_csv(csv_path)
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    print(f"    Median/max metrics: {len(df)} counties")
    return df


def load_distribution_metrics():
    """Load county-level distribution metrics (skewness etc.)."""
    print("  Loading distribution metrics …")
    df = pd.read_csv(OUTPUT_DIR / "county_distribution_metrics.csv")
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    print(f"    Distribution metrics: {len(df)} counties")
    return df


def load_hazard_data():
    """
    Load wind and surge hazard matrices and compute per-county statistics.

    The wind matrix is (n_events × n_counties) = (5018 × 3220).
    County ordering follows county_region.csv (county_index column).

    Returns None if scipy is not available (S1 is then skipped).
    """
    print("  Loading hazard matrices …")
    try:
        from scipy.io import loadmat
    except ImportError:
        print("    scipy not found – skipping hazard data (Figure S1 unavailable)")
        return None

    county_map = pd.read_csv(DATA_DIR / "county_region.csv")
    county_map["fips"] = county_map["fips"].astype(str).str.zfill(5)
    n_map = len(county_map)

    wind_mat = loadmat(
        DATA_DIR / "hazard" / "maxwindmat_ncep_reanal.mat"
    )["maxwindmat"]  # (5018, 3220) events × counties

    surge_raw = loadmat(
        DATA_DIR / "hazard" / "maxelev_coastcounty_ncep_reanal.mat"
    )
    surge_key = "scounty_mhhw" if "scounty_mhhw" in surge_raw else "scounty"
    surge_mat = surge_raw[surge_key]  # may be (3220, 5018) or (5018, 3220)
    if surge_mat.shape[0] != wind_mat.shape[0]:
        surge_mat = surge_mat.T  # ensure (events, counties)

    rain_mat = loadmat(
        DATA_DIR / "hazard" / "ptot_rain_county_ncep_reanal.mat"
    )["ptot_mat"]  # (3220, 5018) counties × events
    # Ensure counties are on axis-0
    if rain_mat.shape[0] != wind_mat.shape[1]:
        rain_mat = rain_mat.T
    max_rain = rain_mat.max(axis=1)[:n_map]  # max across events per county

    max_wind = wind_mat.max(axis=0)[:n_map]
    mean_wind = wind_mat.mean(axis=0)[:n_map]
    max_surge = surge_mat.max(axis=0)[:n_map]
    pct_ts_wind = (wind_mat > 17.5).mean(axis=0)[:n_map] * 100

    hazard_df = county_map[["fips"]].copy()
    hazard_df["max_wind_ms"] = max_wind
    hazard_df["mean_wind_ms"] = mean_wind
    hazard_df["max_rain_mm"] = max_rain
    hazard_df["max_surge_m"] = max_surge
    hazard_df["pct_events_ts_wind"] = pct_ts_wind

    print(f"    Hazard data: {len(hazard_df)} counties")
    print(f"    Wind range:  {max_wind.min():.1f}–{max_wind.max():.1f} m/s")
    print(f"    Rain range:  {max_rain.min():.0f}–{max_rain.max():.0f} mm")
    print(f"    Surge range: {max_surge.min():.2f}–{max_surge.max():.2f} m")
    return hazard_df


def load_single_event(event_id):
    """
    Load damage and recovery data for a specific event.

    Returns
    -------
    DataFrame with columns: fips, weighted_damage, cc, recovery_months
    """
    event_id = str(event_id)
    print(f"  Loading event {event_id} …")

    impact_file = (
        BASE_DIR / "impacts_out" / "by_event" / "scaled"
        / f"{event_id}_scaled.csv"
    )
    if not impact_file.exists():
        raise FileNotFoundError(f"Impact file not found: {impact_file}")

    impact_df = pd.read_csv(impact_file)
    impact_df["fips"] = impact_df["fips"].astype(str).str.zfill(5)
    impact_df["weighted_damage"] = (
        impact_df["units_DS1_scaled"] * RECOVERY_WEIGHTS["DS1"]
        + impact_df["units_DS2_scaled"] * RECOVERY_WEIGHTS["DS2"]
        + impact_df["units_DS3_scaled"] * RECOVERY_WEIGHTS["DS3"]
        + impact_df["units_DS4_scaled"] * RECOVERY_WEIGHTS["DS4"]
    )

    rec_file = (
        DATA_DIR / "recovery_potential_per_scenario"
        / f"{event_id}_scaled_recovery_potential.json"
    )
    if not rec_file.exists():
        raise FileNotFoundError(f"Recovery file not found: {rec_file}")

    with open(rec_file) as fh:
        rec_data = json.load(fh)

    rec_rows = []
    for r in rec_data:
        val = r.get("recovery_potential [months]", np.nan)
        rec_rows.append({
            "fips": str(r["fips"]).zfill(5),
            "cc": float(r.get("reconstruction_capacity", np.nan)),
            "recovery_months": np.nan if not np.isfinite(float(val)) else float(val),
        })
    rec_df = pd.DataFrame(rec_rows)

    merged = impact_df[["fips", "weighted_damage"]].merge(
        rec_df, on="fips", how="outer"
    )
    n_aff = int((merged["weighted_damage"] > 0).sum())
    print(f"    {n_aff} counties with positive damage")
    return merged


# ---------------------------------------------------------------------------
# Main GeoDataFrame builder
# ---------------------------------------------------------------------------

def build_main_gdf(coastal_counties, eaua_df, earp_df, cc_df,
                   event_df, dist_df):
    """
    Merge all tabular metrics into the coastal county GeoDataFrame.

    Returns a GeoDataFrame with columns (in addition to shapefile attributes):
    eaua, earp, cc, median_weighted_damage, median_recovery_months,
    max_weighted_damage, max_recovery_months, wd_skew, rt_skew
    """
    # Combine tabular metrics
    metrics = (
        eaua_df
        .merge(earp_df, on="fips", how="outer")
        .merge(cc_df,   on="fips", how="outer")
    )

    event_cols = [
        "fips", "median_weighted_damage", "median_recovery_months",
        "max_weighted_damage", "max_recovery_months",
    ]
    metrics = metrics.merge(
        event_df[[c for c in event_cols if c in event_df.columns]],
        on="fips", how="outer",
    )

    dist_cols = ["fips", "wd_skew", "rt_skew"]
    metrics = metrics.merge(
        dist_df[[c for c in dist_cols if c in dist_df.columns]],
        on="fips", how="outer",
    )

    # Zero / negative → NaN for strictly positive metrics
    for col in ("eaua", "earp", "cc",
                "median_weighted_damage", "median_recovery_months",
                "max_weighted_damage", "max_recovery_months"):
        if col in metrics.columns:
            metrics.loc[~(metrics[col] > 0), col] = np.nan

    gdf = coastal_counties.merge(
        metrics, left_on="GEOID", right_on="fips", how="left"
    )
    return gdf


def _print_metric_summary(gdf, col, label):
    valid = gdf[col].dropna()
    valid = valid[valid > 0] if col not in ("wd_skew", "rt_skew") else valid.dropna()
    if len(valid) > 0:
        print(f"    {label:45s}  n={len(valid):4d}  "
              f"min={valid.min():.4g}  median={valid.median():.4g}  "
              f"max={valid.max():.4g}")


# ---------------------------------------------------------------------------
# Figure 2 – Annual triptych map
# ---------------------------------------------------------------------------

def fig2_annual_triptych(gdf, state_gdf):
    """
    Figure 2: Three-panel choropleth showing EAUA, Construction Capacity,
    and EARP across the US Atlantic coast.
    """
    print("\nFigure 2: Annual triptych …")

    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 3.3))

    panels = [
        ("eaua", "cividis",  "EAUA [weighted units yr⁻¹]",              False),
        ("cc",   "Greens",   "CC [permits month⁻¹]",  False),
        ("earp", "Purples_r",  "Recovery potential",                       True),
    ]
    labels = ["a", "b", "c"]

    for ax, (col, cmap, cbar_label, inv), lbl in zip(axes, panels, labels):
        _choropleth_panel(gdf, state_gdf, ax, col, cmap, cbar_label, invert_cbar=inv)
        ax.set_title(f"{lbl})", loc="left", fontsize=8, fontweight="bold", pad=2)

    print("  Summary:")
    for col, _, label, *__ in panels:
        _print_metric_summary(gdf, col, label)

    plt.tight_layout(pad=0.5)
    _save_fig(fig, "annual_3panel")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3 – 2×2 scatterplot: annual vs. median-event drivers
# ---------------------------------------------------------------------------

def fig3_recovery_drivers_scatter(gdf):
    """
    Figure 3: 2×2 color-encoded log-log scatter.
    Row 0 (Annual):       EARP vs EAUA / CC,   colored by CC / EAUA
    Row 1 (Median event): MRP  vs WUA  / CC,   colored by CC / WUA
    Y-axis inverted: short recovery time (high potential) at top.
    """
    print("\nFigure 3: Recovery drivers scatter (annual + median event) …")

    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, 4.2))

    # Row 0 – annual
    _color_scatter_panel(
        axes[0, 0], gdf["eaua"], gdf["earp"], gdf["cc"],
        xlabel="EAUA",
        ylabel="RP (low\u2013high)",
        clabel="CC",
        cmap="viridis",
        panel_label="a",
        show_yticks=True,
    )
    _color_scatter_panel(
        axes[0, 1], gdf["cc"], gdf["earp"], gdf["eaua"],
        xlabel="CC",
        ylabel="RP (low\u2013high)",
        clabel="EAUA",
        cmap="plasma",
        panel_label="b",
        show_yticks=False,
    )

    # Row 1 – median event
    _color_scatter_panel(
        axes[1, 0], gdf["median_weighted_damage"], gdf["median_recovery_months"],
        gdf["cc"],
        xlabel="WUA",
        ylabel="RP (low\u2013high)",
        clabel="CC",
        cmap="viridis",
        panel_label="c",
        show_yticks=True,
    )
    _color_scatter_panel(
        axes[1, 1], gdf["cc"], gdf["median_recovery_months"],
        gdf["median_weighted_damage"],
        xlabel="CC",
        ylabel="RP (low\u2013high)",
        clabel="WUA",
        cmap="plasma",
        panel_label="d",
        show_yticks=False,
    )

    # Row labels
    fig.text(0.02, 0.78, "annual", rotation=90, va="center", ha="center",
             fontsize=7, fontweight="bold")
    fig.text(0.02, 0.30, "median event", rotation=90, va="center", ha="center",
             fontsize=7, fontweight="bold")

    # Legend (no box)
    fig.text(
        0.9, 0.50,
        "EAUA = expected annual affected units\nWUA = weighted affected units\nCC = construction capacity\nRP = recovery potential",
        va="center", ha="left", fontsize=6,
    )

    plt.tight_layout(rect=[0.05, 0, 0.90, 1])
    _save_fig(fig, "recovery_drivers_annual_vs_median")
    plt.close()


# ---------------------------------------------------------------------------
# Figure S1 – Hazard overview map
# ---------------------------------------------------------------------------

def figS1_hazard_overview(coastal_counties, state_gdf, hazard_df):
    """
    Figure S1: Three-panel choropleth – maximum wind speed, maximum rainfall,
    and maximum surge height per county across all synthetic events.
    """
    print("\nFigure S1: Hazard overview …")

    if hazard_df is None:
        print("  Skipped (hazard data not available – run with climada_env)")
        return

    gdf = coastal_counties.merge(
        hazard_df, left_on="GEOID", right_on="fips", how="left"
    )

    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 3.3))

    panels = [
        ("max_wind_ms",  "YlOrRd",  "Max wind speed [m s⁻¹]",       False),
        ("max_rain_mm",  "Blues",   "Max rainfall [mm]",           False),
        ("max_surge_m",  "viridis", "Max storm surge [m]",   False),
    ]
    labels = ["a", "b", "c"]

    for ax, (col, cmap, cbar_label, inv), lbl in zip(axes, panels, labels):
        _choropleth_panel(gdf, state_gdf, ax, col, cmap, cbar_label,
                          invert_cbar=inv, use_log=False)
        ax.set_title(f"{lbl})", loc="left", fontsize=8, fontweight="bold", pad=2)

    print("  Summary:")
    for col, _, label, *__ in panels:
        _print_metric_summary(gdf, col, label)

    plt.tight_layout(pad=0.5)
    _save_fig(fig, "na_coast_hazard_overview")
    plt.close()


# ---------------------------------------------------------------------------
# Figure S2 – Single-event 3-panel maps
# ---------------------------------------------------------------------------

def figS2_single_event_map(coastal_counties, state_gdf, event_id):
    """
    Figure S2: Three-panel choropleth for a single synthetic event:
    (a) Weighted damage units, (b) Construction capacity, (c) Recovery time.
    """
    print(f"\nFigure S2: Event {event_id} map …")

    event_df = load_single_event(event_id)

    gdf = coastal_counties.merge(
        event_df, left_on="GEOID", right_on="fips", how="left"
    )
    for col in ("weighted_damage", "cc", "recovery_months"):
        gdf.loc[~(gdf[col] > 0), col] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 3.3))

    panels = [
        ("weighted_damage", "cividis", "WUA [units]",                False),
        ("cc",              "Greens",  "CC [permits month⁻¹]", False),
        ("recovery_months", "Purples_r", "Recovery potential",                      True),
    ]
    labels = ["a", "b", "c"]

    n_affected = int((gdf["weighted_damage"] > 0).sum())

    for ax, (col, cmap, cbar_label, inv), lbl in zip(axes, panels, labels):
        _choropleth_panel(gdf, state_gdf, ax, col, cmap, cbar_label, invert_cbar=inv)
        ax.set_title(f"{lbl})", loc="left", fontsize=8, fontweight="bold", pad=2)

    print(f"  Affected counties: {n_affected}")
    print("  Summary:")
    for col, _, label, *__ in panels:
        _print_metric_summary(gdf, col, label)

    plt.tight_layout(pad=0.5)
    _save_fig(fig, f"event_{event_id}_3panel_map")
    plt.close()


# ---------------------------------------------------------------------------
# Figure S5 – Median event triptych
# ---------------------------------------------------------------------------

def figS5_median_event_triptych(gdf, state_gdf):
    """
    Figure S5: Three-panel choropleth – median per-event WUA, CC, MRP.
    """
    print("\nFigure S5: Median event triptych …")

    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 3.3))

    panels = [
        ("median_weighted_damage", "cividis", "WUA [units]",              False),
        ("cc",                     "Greens",  "CC [permits month⁻¹]",  False),
        ("median_recovery_months", "Purples_r", "Recovery potential",                       True),
    ]
    labels = ["a", "b", "c"]

    for ax, (col, cmap, cbar_label, inv), lbl in zip(axes, panels, labels):
        _choropleth_panel(gdf, state_gdf, ax, col, cmap, cbar_label, invert_cbar=inv)
        ax.set_title(f"{lbl})", loc="left", fontsize=8, fontweight="bold", pad=2)

    print("  Summary:")
    for col, _, label, *__ in panels:
        _print_metric_summary(gdf, col, label)

    plt.tight_layout(pad=0.5)
    _save_fig(fig, "median_event_3panel")
    plt.close()


# ---------------------------------------------------------------------------
# Figure S6 – Max event triptych
# ---------------------------------------------------------------------------

def figS6_max_event_triptych(gdf, state_gdf):
    """
    Figure S6: Three-panel choropleth – max per-event WUA, CC, max RP.
    """
    print("\nFigure S6: Max event triptych …")

    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 3.3))

    panels = [
        ("max_weighted_damage",  "cividis", "WUA [units]",                False),
        ("cc",                   "Greens",  "CC [permits month⁻¹]",  False),
        ("max_recovery_months",  "Purples_r", "Recovery potential",                       True),
    ]
    labels = ["a", "b", "c"]

    for ax, (col, cmap, cbar_label, inv), lbl in zip(axes, panels, labels):
        _choropleth_panel(gdf, state_gdf, ax, col, cmap, cbar_label, invert_cbar=inv)
        ax.set_title(f"{lbl})", loc="left", fontsize=8, fontweight="bold", pad=2)

    print("  Summary:")
    for col, _, label, *__ in panels:
        _print_metric_summary(gdf, col, label)

    plt.tight_layout(pad=0.5)
    _save_fig(fig, "max_event_3panel")
    plt.close()


# ---------------------------------------------------------------------------
# Figure S7 – Annual + Max event scatterplot (2×2)
# ---------------------------------------------------------------------------

def figS7_annual_max_scatter(gdf):
    """
    Figure S7: Max-event recovery drivers – 1×2 color-encoded log-log scatter.
    (a) Max RP vs max WUA, colored by CC  (Greens)
    (b) Max RP vs CC,      colored by max WUA (cividis)
    Y-axis inverted: short recovery time (high potential) at top.
    """
    print("\nFigure S7: Max-event recovery drivers scatter …")

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 2.2))

    _color_scatter_panel(
        axes[0], gdf["max_weighted_damage"], gdf["max_recovery_months"], gdf["cc"],
        xlabel="WUA",
        ylabel="RP (low\u2013high)",
        clabel="CC",
        cmap="viridis",
        panel_label="a",
        show_yticks=True,
    )
    _color_scatter_panel(
        axes[1], gdf["cc"], gdf["max_recovery_months"], gdf["max_weighted_damage"],
        xlabel="CC",
        ylabel="RP (low\u2013high)",
        clabel="WUA",
        cmap="plasma",
        panel_label="b",
        show_yticks=False,
    )

    fig.text(0.02, 0.55, "maximum value", rotation=90, va="center", ha="center",
             fontsize=7, fontweight="bold")
    fig.text(
        0.95, 0.50,
        "WUA = weighted affected units\nCC = construction capacity\nRP = recovery potential",
        va="center", ha="left", fontsize=6,
    )

    plt.tight_layout(rect=[0.05, 0, 0.93, 1])
    _save_fig(fig, "recovery_drivers_annual_max")
    plt.close()


# ---------------------------------------------------------------------------
# Figure S8 – Skewness maps
# ---------------------------------------------------------------------------

def _skewness_single_panel(gdf, state_gdf, col, cbar_label, stem):
    """
    Draw and save a single skewness choropleth with a right-side colorbar.
    """
    tmp = gdf[[col, "geometry"]].copy()
    valid = tmp[col].dropna()

    if len(valid) == 0:
        print(f"  No valid data for {col!r} – skipping")
        return

    vmax = float(np.percentile(valid.clip(lower=0), 95))
    norm = Normalize(vmin=0, vmax=max(vmax, 1e-6))

    fig, ax = plt.subplots(1, 1, figsize=(W_DOUBLE * 0.65, 3.3))

    # No-data counties
    tmp[tmp[col].isna()].plot(
        ax=ax, color=NO_DATA_COLOR,
        edgecolor=COUNTY_EDGECOLOR, linewidth=COUNTY_LINEWIDTH,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)

    tmp[tmp[col].notna()].plot(
        column=col, cmap="YlOrRd", norm=norm,
        edgecolor=COUNTY_EDGECOLOR, linewidth=COUNTY_LINEWIDTH,
        legend=True, ax=ax, cax=cax,
        legend_kwds={"orientation": "vertical"},
    )

    state_gdf.plot(ax=ax, facecolor="none",
                   edgecolor=STATE_EDGECOLOR, linewidth=STATE_LINEWIDTH)

    ax.set_xlim(MAP_XLIM)
    ax.set_ylim(MAP_YLIM)
    ax.set_aspect("equal")
    ax.axis("off")

    cax.set_ylabel(cbar_label, fontsize=7, labelpad=3)
    cax.tick_params(labelsize=6)
    cax.tick_params(which="minor", length=0)
    for spine in cax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.5)

    plt.tight_layout(pad=0.3)
    _save_fig(fig, stem)
    plt.close()

    v = gdf[col].dropna()
    print(f"    {cbar_label:55s}  n={len(v):4d}  "
          f"median={v.median():.2f}  p95={np.percentile(v, 95):.2f}  "
          f"max={v.max():.2f}")


def figS8_skewness_maps(gdf, state_gdf):
    """
    Figure S8: Two single-panel choropleths saved separately.
      skewness_wd.png / .pdf  – skewness of per-event damage
      skewness_rt.png / .pdf  – skewness of per-event recovery time
    """
    print("\nFigure S8: Skewness maps …")

    for col in ("wd_skew", "rt_skew"):
        if col not in gdf.columns:
            print(f"  Column {col!r} missing – skipping Figure S8")
            return

    print("  Summary:")
    _skewness_single_panel(
        gdf, state_gdf,
        col="wd_skew",
        cbar_label="Skewness",
        stem="skewness_wd",
    )
    _skewness_single_panel(
        gdf, state_gdf,
        col="rt_skew",
        cbar_label="Skewness",
        stem="skewness_rt",
    )


# ---------------------------------------------------------------------------
# Figure 4 – Bivariate choropleth maps
# ---------------------------------------------------------------------------

def _bv_assign_tercile(series):
    """Assign tercile labels 1/2/3 (float); non-positive / non-finite → NaN."""
    valid = series.notna() & np.isfinite(series) & (series > 0)
    result = pd.Series(np.nan, index=series.index, dtype=float)
    if valid.sum() < 3:
        return result
    try:
        labels = pd.qcut(series[valid], q=3, labels=[1, 2, 3], duplicates="drop")
    except ValueError:
        labels = pd.cut(series[valid], bins=3, labels=[1, 2, 3])
    result[valid] = labels.astype(float)
    return result


def _bv_get_color(row, row_col, col_col, grid):
    t_row, t_col = row[row_col], row[col_col]
    if pd.isna(t_row) or pd.isna(t_col):
        return NO_DATA_COLOR
    return grid[int(t_row) - 1][int(t_col) - 1]


def _bv_prepare_gdf(coastal_counties, eaua_df, earp_df, cc_df):
    """Merge metrics into coastal county GDF and compute bivariate colors."""
    metrics = (
        eaua_df
        .merge(earp_df, on="fips", how="outer")
        .merge(cc_df,   on="fips", how="outer")
    )
    for col in ("eaua", "earp", "cc"):
        metrics.loc[~(metrics[col] > 0), col] = np.nan

    metrics["eaua_tercile"] = _bv_assign_tercile(metrics["eaua"])
    metrics["earp_tercile"] = _bv_assign_tercile(metrics["earp"])
    metrics["cc_tercile"]   = _bv_assign_tercile(metrics["cc"])

    gdf = coastal_counties.merge(metrics, left_on="GEOID", right_on="fips", how="left")

    gdf["color_a"] = gdf.apply(
        lambda r: _bv_get_color(r, "earp_tercile", "eaua_tercile", GRID_A), axis=1
    )
    # Invert CC so high CC (safe) → row 0, low CC (risky) → row 2
    gdf["inv_cc_tercile"] = 4 - gdf["cc_tercile"]
    gdf["color_b"] = gdf.apply(
        lambda r: _bv_get_color(r, "inv_cc_tercile", "eaua_tercile", GRID_B), axis=1
    )
    return gdf


def _bv_legend(ax, colors_display, xlabel, ylabel,
               x_low="Low", x_high="High", y_low="Low", y_high="High",
               bbox=(0.72, 0.01, 0.27, 0.34),
               xlabel_pos=(2.5, -1.8),
               ylabel_pos=(-2.2, 1.3),
               tick_spread=0.4,
               tick_offset=1.1):
    """
    Draw a 3×3 bivariate colour legend as an inset on *ax*.

    Label tuning parameters
    -----------------------
    xlabel_pos   : (x, y) data-coords for the x-axis name (e.g. "Risk")
    ylabel_pos   : (x, y) data-coords for the y-axis name (e.g. "CC")
    tick_spread  : how far Low/High labels extend beyond the arrow endpoints
                   (arrow runs 0→3; spread=0.4 → Low at -0.4, High at 3.4)
    tick_offset  : perpendicular distance of Low/High labels from the arrow line
    """
    axins = ax.inset_axes(bbox)
    for row_idx in range(3):
        for col_idx in range(3):
            rect = mpatches.Rectangle(
                (col_idx, row_idx), 1, 1,
                facecolor=colors_display[row_idx][col_idx],
                edgecolor="white", linewidth=0.5,
                transform=axins.transData,
            )
            axins.add_patch(rect)
    axins.set_xlim(-1.0, 3.5)
    axins.set_ylim(-1.0, 3.5)
    axins.set_aspect("equal")
    axins.axis("off")

    kw = dict(arrowstyle="->", color="black", lw=0.8)
    axins.annotate("", xy=(3.0, -0.5), xytext=(0.0, -0.5),
                   xycoords="data", textcoords="data",
                   arrowprops=kw, annotation_clip=False)
    axins.annotate("", xy=(-0.5, 3.0), xytext=(-0.5, 0.0),
                   xycoords="data", textcoords="data",
                   arrowprops=kw, annotation_clip=False)

    axins.text(*xlabel_pos, xlabel, ha="center", va="top",
               fontsize=7, transform=axins.transData, clip_on=False)
    axins.text(*ylabel_pos, ylabel, ha="center", va="center",
               fontsize=7, rotation=90, transform=axins.transData, clip_on=False)
    axins.text(-tick_spread, -tick_offset, x_low,  ha="left",  va="top",
               fontsize=6, transform=axins.transData, clip_on=False)
    axins.text(3 + tick_spread, -tick_offset, x_high, ha="right", va="top",
               fontsize=6, transform=axins.transData, clip_on=False)
    axins.text(-tick_offset, -tick_spread, y_low,  ha="right", va="bottom",
               fontsize=6, rotation=90, transform=axins.transData, clip_on=False)
    axins.text(-tick_offset, 3 + tick_spread, y_high, ha="right", va="top",
               fontsize=6, rotation=90, transform=axins.transData, clip_on=False)
    return axins


def _bv_render_map(gdf, state_gdf, color_col, ax,
                   colors_display=None, xlabel="", ylabel="",
                   y_low="Low", y_high="High", panel_label=None,
                   legend_bbox=(0.72, 0.01, 0.27, 0.34),
                   **legend_kwargs):
    """Plot bivariate choropleth with optional legend inset on *ax*.

    Extra keyword arguments (xlabel_pos, ylabel_pos, tick_spread, tick_offset)
    are forwarded directly to _bv_legend.
    """
    gdf.plot(ax=ax, color=gdf[color_col],
             edgecolor=COUNTY_EDGECOLOR, linewidth=0.15)
    state_gdf.plot(ax=ax, facecolor="none",
                   edgecolor=STATE_EDGECOLOR, linewidth=0.4)
    ax.set_xlim(MAP_XLIM)
    ax.set_ylim(MAP_YLIM)
    ax.set_aspect("equal")
    ax.axis("off")
    if panel_label is not None:
        ax.set_title(f"{panel_label})", loc="left", fontsize=8, fontweight="bold", pad=2)
    if colors_display is not None:
        _bv_legend(ax, colors_display, xlabel=xlabel, ylabel=ylabel,
                   y_low=y_low, y_high=y_high, bbox=legend_bbox,
                   **legend_kwargs)


def _bv_print_summary(gdf):
    print("  Tercile thresholds:")
    for col, name in [("eaua", "EAUA (weighted units/yr)"),
                      ("earp", "EARP (months/yr)"),
                      ("cc",   "CC (permits/month)")]:
        valid = gdf[col].dropna() if col in gdf.columns else pd.Series(dtype=float)
        valid = valid[valid > 0]
        if len(valid) > 0:
            print(f"    {name:35s}  p33={valid.quantile(1/3):.4f}"
                  f"  p67={valid.quantile(2/3):.4f}  n={len(valid)}")


def _bv_single_panel(gdf, state_gdf, color_col, grid, xlabel, ylabel,
                     y_low, y_high, stem,
                     legend_bbox=(0.667, 0.01, 0.33, 0.36),
                     **legend_kwargs):
    """Save one W_SINGLE bivariate map with its own legend inset.

    Extra keyword arguments (xlabel_pos, ylabel_pos, tick_spread, tick_offset)
    are forwarded to _bv_legend via _bv_render_map.
    """
    fig, ax = plt.subplots(1, 1, figsize=(W_SINGLE, 3))
    _bv_render_map(gdf, state_gdf, color_col, ax,
                   colors_display=grid,
                   xlabel=xlabel, ylabel=ylabel,
                   y_low=y_low, y_high=y_high,
                   legend_bbox=legend_bbox,
                   **legend_kwargs)
    plt.tight_layout()
    _save_fig(fig, stem)
    plt.close()


def fig4_bivariate_maps(coastal_counties, state_gdf, eaua_df, earp_df, cc_df):
    """
    Figure 4: bivariate choropleths.

    Combined (W_DOUBLE, 1×2) with abbreviation legend — main publication figure.
    Individual panels (W_SINGLE) saved separately for supplementary/reuse.
    """
    print("\nFigure 4: Bivariate maps …")
    gdf = _bv_prepare_gdf(coastal_counties, eaua_df, earp_df, cc_df)
    _bv_print_summary(gdf)

    # ── Combined 1×2 figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3))
    _bv_render_map(gdf, state_gdf, "color_b", axes[0],
                   colors_display=GRID_B,
                   xlabel="Risk", ylabel="CC",
                   y_low="High", y_high="Low", panel_label="a",
                   xlabel_pos=(1.5, -1.5), ylabel_pos=(-2.0, 1.5),
                   tick_spread=0.4, tick_offset=0.8)
    _bv_render_map(gdf, state_gdf, "color_a", axes[1],
                   colors_display=GRID_A,
                   xlabel="Risk", ylabel="RP",
                   y_low="High", y_high="Low", panel_label="b",
                   xlabel_pos=(1.5, -1.5), ylabel_pos=(-2.0, 1.5),
                   tick_spread=0.4, tick_offset=0.8)
    plt.tight_layout(pad=0.3)
    _save_fig(fig, "bivariate_maps_combined")
    plt.close()

    # ── Individual W_SINGLE panels ────────────────────────────────────────────
    _bv_single_panel(gdf, state_gdf, "color_b", GRID_B,
                     xlabel="Risk", ylabel="CC",
                     y_low="High", y_high="Low",
                     xlabel_pos=(1.5, -1.5), ylabel_pos=(-1.8, 1.5),
                     tick_spread=0.4, tick_offset=0.8,
                     stem="bivariate_map_B_risk_vs_capacity")
    _bv_single_panel(gdf, state_gdf, "color_a", GRID_A,
                     xlabel="Risk", ylabel="RP",
                     y_low="High", y_high="Low",
                     xlabel_pos=(1.5, -1.5), ylabel_pos=(-1.8, 1.5),
                     tick_spread=0.4, tick_offset=0.8,
                     stem="bivariate_map_A_risk_vs_recovery")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

ALL_FIGS = {
    "fig2":  "Figure 2  – Annual triptych map (EAUA, CC, EARP)",
    "fig3":  "Figure 3  – Annual recovery drivers scatter (1×2, color-coded)",
    "fig4":  "Figure 4  – Bivariate choropleth maps (Risk × CC, Risk × Recovery Potential)",
    "figS1": "Figure S1 – Hazard overview map (max wind, max surge)",
    "figS2": "Figure S2 – Single-event 3-panel maps (events 350 & 4347)",
    "figS5": "Figure S5 – Median event triptych (MUA, CC, MRP)",
    "figS6": "Figure S6 – Max event triptych (Max WUA, CC, Max RP)",
    "figS7": "Figure S7 – Max-event recovery drivers scatter (1×2, color-coded)",
    "figS8": "Figure S8 – Skewness maps",
}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create manuscript figures for the hurricane recovery potential "
            "study.  Run inside climada_env for full functionality."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available figures:\n"
        + "\n".join(f"  {k}: {v}" for k, v in ALL_FIGS.items()),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=list(ALL_FIGS.keys()),
        default=list(ALL_FIGS.keys()),
        metavar="FIG",
        help="Which figures to generate (default: all).",
    )
    args, _ = parser.parse_known_args()
    figs_to_run = set(args.figures)

    print("=" * 70)
    print("Hurricane Recovery Potential – Manuscript Figures")
    print("=" * 70)
    print(f"Output directory : {FIGURES_DIR}")
    print(f"Figures requested: {', '.join(sorted(figs_to_run))}")

    # ── Spatial data (always required) ────────────────────────────────────
    print("\nLoading spatial data …")
    coastal_counties, state_gdf = load_spatial_data()

    # ── Tabular metrics (required for most figures) ────────────────────────
    METRIC_FIGS = {"fig2", "fig3", "figS5", "figS6", "figS7", "figS8"}
    BIVARIATE_FIGS = {"fig4"}
    gdf = None
    eaua_df = earp_df = cc_df = None

    if figs_to_run & (METRIC_FIGS | BIVARIATE_FIGS):
        print("\nLoading annual metrics …")
        eaua_df, earp_df, cc_df = load_annual_metrics()

    if figs_to_run & METRIC_FIGS:
        event_df = load_event_level_metrics()
        dist_df = load_distribution_metrics()
        gdf = build_main_gdf(
            coastal_counties, eaua_df, earp_df, cc_df, event_df, dist_df
        )
        print(f"  Main GeoDataFrame: {len(gdf)} counties")

    # ── Hazard data (Figure S1 only) ───────────────────────────────────────
    hazard_df = None
    if "figS1" in figs_to_run:
        hazard_df = load_hazard_data()

    # ── Generate figures ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Generating figures …")
    print("=" * 70)

    if "fig2" in figs_to_run:
        fig2_annual_triptych(gdf, state_gdf)

    if "fig3" in figs_to_run:
        fig3_recovery_drivers_scatter(gdf)

    if "fig4" in figs_to_run:
        fig4_bivariate_maps(coastal_counties, state_gdf, eaua_df, earp_df, cc_df)

    if "figS1" in figs_to_run:
        figS1_hazard_overview(coastal_counties, state_gdf, hazard_df)

    if "figS2" in figs_to_run:
        for eid in (350, 4347):
            try:
                figS2_single_event_map(coastal_counties, state_gdf, eid)
            except FileNotFoundError as exc:
                print(f"  Warning: {exc}")

    if "figS5" in figs_to_run:
        figS5_median_event_triptych(gdf, state_gdf)

    if "figS6" in figs_to_run:
        figS6_max_event_triptych(gdf, state_gdf)

    if "figS7" in figs_to_run:
        figS7_annual_max_scatter(gdf)

    if "figS8" in figs_to_run:
        figS8_skewness_maps(gdf, state_gdf)

    print("\n" + "=" * 70)
    print(f"Done.  All figures saved to: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
