"""
Create 2-panel plot contrasting Construction Capacity vs. Damage, colored by Recovery Potential

Panel 1: Annual metrics (CC vs EAUA, colored by EARP)
Panel 2: Median per-event metrics (CC vs MUA, colored by MRP)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
from scipy.stats import pearsonr
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"
DATA_DIR = BASE_DIR / "data"
RECOVERY_DIR = DATA_DIR / "recovery_potential_per_scenario"

print("="*80)
print("RECOVERY DRIVERS: CC vs. DAMAGE (COLORED BY RECOVERY POTENTIAL)")
print("="*80)

# Load required data
print("\n### LOADING DATA ###")

# Load event-county quadrant files (which have all the data we need)
df_abs = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_weightedunits.csv')
df_norm = pd.read_csv(OUTPUT_DIR / 'event_county_quadrants_fully_normalized.csv')

# Load EARP data
earp_df = pd.read_csv(OUTPUT_DIR / "earp_per_county.csv")

# Load EAD data (in long format)
ead_long = pd.read_csv(OUTPUT_DIR / "ead_per_county_per_ds_scaled.csv")
# Sum across damage states to get total_ead per county
ead_wide = ead_long.groupby('fips')['ead'].sum().reset_index()
ead_wide.columns = ['fips', 'total_ead']

print("✓ Data loaded")

# ============================================================================
# PREPARE ANNUAL METRICS
# ============================================================================
print("\n### PREPARING ANNUAL METRICS ###")

# Get construction capacity and housing units from normalized quadrant data
capacity_housing_df = df_norm.groupby('fips').agg({
    'construction_capacity': 'first',
    'total_housing_units': 'first'
}).reset_index()

driver_analysis = earp_df[['fips', 'earp_months_per_year']].copy()
driver_analysis = driver_analysis.merge(ead_wide, on='fips', how='inner')
driver_analysis = driver_analysis.merge(capacity_housing_df, on='fips', how='inner')

# Calculate normalized metrics
driver_analysis['pct_housing_ead'] = (driver_analysis['total_ead'] / driver_analysis['total_housing_units']) * 100
driver_analysis['capacity_per_1000'] = (driver_analysis['construction_capacity'] / driver_analysis['total_housing_units']) * 1000

# Remove rows with missing or zero values
driver_analysis = driver_analysis[
    (driver_analysis['earp_months_per_year'] > 0) & 
    (driver_analysis['total_ead'] > 0) & 
    (driver_analysis['construction_capacity'] > 0) &
    (driver_analysis['total_housing_units'] > 0)
]

print(f"Counties with complete annual data: {len(driver_analysis)}")

# ============================================================================
# PREPARE MEDIAN PER-EVENT METRICS
# ============================================================================
print("\n### PREPARING MEDIAN PER-EVENT METRICS ###")

# Get median recovery time and damage per county across all events (from normalized file)
per_event_analysis_median = df_norm.groupby('fips').agg({
    'recovery_months': 'median',
    'weighted_damage_units': 'median',
    'pct_housing_damaged': 'median',
    'capacity_per_1000_units': 'first',  # Same across all events for a county
    'construction_capacity': 'first'
}).reset_index()

per_event_analysis_median.columns = ['fips', 'median_recovery_months', 
                                       'median_damage_units', 'median_pct_damaged',
                                       'capacity_per_1000', 'construction_capacity']

# Remove rows with missing or zero values
per_event_analysis_median = per_event_analysis_median[
    (per_event_analysis_median['median_recovery_months'] > 0) & 
    (per_event_analysis_median['median_damage_units'] > 0) & 
    (per_event_analysis_median['construction_capacity'] > 0)
]

print(f"Counties with complete median per-event data: {len(per_event_analysis_median)}")

# ============================================================================
# CREATE 2-PANEL PLOT (ABSOLUTE VALUES)
# ============================================================================
print("\n### CREATING ABSOLUTE VALUES PLOT ###")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

label_fs = 12
tick_fs = 9
cbar_label_fs = 11

# ---------------- PANEL 1: Annual Metrics (Absolute) ----------------
ax1 = axes[0]

# Calculate correlation between CC and EAUA
corr_cc_eaua, p_cc_eaua = pearsonr(np.log10(driver_analysis['construction_capacity']), 
                                    np.log10(driver_analysis['total_ead']))

scatter1 = ax1.scatter(
    driver_analysis['construction_capacity'], 
    driver_analysis['total_ead'],
    c=driver_analysis['earp_months_per_year'],
    cmap='RdYlGn_r',  # Red (high recovery time) to Green (low recovery time)
    alpha=0.6,
    s=30,
    norm=LogNorm()
)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('CC (permits/month)', fontsize=label_fs)
ax1.set_ylabel('EAUA (# units)', fontsize=label_fs)
ax1.grid(False)

cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('EARP (months/year)', fontsize=cbar_label_fs)
cbar1.ax.tick_params(which='both', labelsize=tick_fs)
cbar1.ax.tick_params(which='minor', length=0)

ax1.text(0.05, 0.98, f'r = {corr_cc_eaua:+.3f}\nn = {len(driver_analysis):,}', 
         transform=ax1.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel label
ax1.text(0.02, 0.02, 'a', transform=ax1.transAxes, fontsize=12, 
         fontweight='bold', va='bottom', ha='left')

# ---------------- PANEL 2: Median Per-Event Metrics (Absolute) ----------------
ax2 = axes[1]

# Calculate correlation between CC and MUA
corr_cc_mua, p_cc_mua = pearsonr(np.log10(per_event_analysis_median['construction_capacity']), 
                                  np.log10(per_event_analysis_median['median_damage_units']))

scatter2 = ax2.scatter(
    per_event_analysis_median['construction_capacity'], 
    per_event_analysis_median['median_damage_units'],
    c=per_event_analysis_median['median_recovery_months'],
    cmap='RdYlGn_r',  # Red (high recovery time) to Green (low recovery time)
    alpha=0.6,
    s=30,
    norm=LogNorm()
)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('CC (permits/month)', fontsize=label_fs)
ax2.set_ylabel('MUA (# units)', fontsize=label_fs)
ax2.grid(False)

cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('MRP (months)', fontsize=cbar_label_fs)
cbar2.ax.tick_params(which='both', labelsize=tick_fs)
cbar2.ax.tick_params(which='minor', length=0)

ax2.text(0.05, 0.98, f'r = {corr_cc_mua:+.3f}\nn = {len(per_event_analysis_median):,}', 
         transform=ax2.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel label
ax2.text(0.02, 0.02, 'b', transform=ax2.transAxes, fontsize=12, 
         fontweight='bold', va='bottom', ha='left')

# ---------------- Styling ----------------
# Colorbar borders
for cbar in [cbar1, cbar2]:
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)

# Panel borders
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_edgecolor('0.4')
        spine.set_linewidth(0.8)
    ax.tick_params(color='0.4', labelcolor='0.2')
    
    # Axis ticks
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.tick_params(axis='x', which='major', bottom=True, top=False,
                   labelbottom=True, labelsize=tick_fs)
    ax.tick_params(axis='y', which='major', left=True, right=False,
                   labelleft=True, labelsize=tick_fs)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "recovery_drivers_cc_vs_damage_by_earp_absolute.png",
            dpi=300, bbox_inches="tight")
plt.show()

print("\n✓ Absolute values plot saved")
print(f"  Annual: CC vs EAUA correlation: r = {corr_cc_eaua:+.3f} (p={p_cc_eaua:.2e})")
print(f"  Median: CC vs MUA correlation: r = {corr_cc_mua:+.3f} (p={p_cc_mua:.2e})")

# ============================================================================
# CREATE 2-PANEL PLOT (NORMALIZED VALUES)
# ============================================================================
print("\n### CREATING NORMALIZED VALUES PLOT ###")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# ---------------- PANEL 1: Annual Metrics (Normalized) ----------------
ax1 = axes[0]

# Calculate correlation between normalized CC and normalized EAUA
corr_cc_eaua_norm, p_cc_eaua_norm = pearsonr(np.log10(driver_analysis['capacity_per_1000']), 
                                               np.log10(driver_analysis['pct_housing_ead']))

scatter1 = ax1.scatter(
    driver_analysis['capacity_per_1000'], 
    driver_analysis['pct_housing_ead'],
    c=driver_analysis['earp_months_per_year'],
    cmap='RdYlGn_r',
    alpha=0.6,
    s=30,
    norm=LogNorm()
)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('CC (permits/1000 units/month)', fontsize=label_fs)
ax1.set_ylabel('EAUA (% of housing)', fontsize=label_fs)
ax1.grid(False)

cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('EARP (months/year)', fontsize=cbar_label_fs)
cbar1.ax.tick_params(which='both', labelsize=tick_fs)
cbar1.ax.tick_params(which='minor', length=0)

ax1.text(0.05, 0.98, f'r = {corr_cc_eaua_norm:+.3f}\nn = {len(driver_analysis):,}', 
         transform=ax1.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel label
ax1.text(0.02, 0.02, 'a', transform=ax1.transAxes, fontsize=12, 
         fontweight='bold', va='bottom', ha='left')

# ---------------- PANEL 2: Median Per-Event Metrics (Normalized) ----------------
ax2 = axes[1]

# Calculate correlation between normalized CC and normalized MUA
corr_cc_mua_norm, p_cc_mua_norm = pearsonr(np.log10(per_event_analysis_median['capacity_per_1000']), 
                                            np.log10(per_event_analysis_median['median_pct_damaged']))

scatter2 = ax2.scatter(
    per_event_analysis_median['capacity_per_1000'], 
    per_event_analysis_median['median_pct_damaged'],
    c=per_event_analysis_median['median_recovery_months'],
    cmap='RdYlGn_r',
    alpha=0.6,
    s=30,
    norm=LogNorm()
)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('CC (permits/1000 units/month)', fontsize=label_fs)
ax2.set_ylabel('MUA (% of housing)', fontsize=label_fs)
ax2.grid(False)

cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('MRP (months)', fontsize=cbar_label_fs)
cbar2.ax.tick_params(which='both', labelsize=tick_fs)
cbar2.ax.tick_params(which='minor', length=0)

ax2.text(0.05, 0.98, f'r = {corr_cc_mua_norm:+.3f}\nn = {len(per_event_analysis_median):,}', 
         transform=ax2.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel label
ax2.text(0.02, 0.02, 'b', transform=ax2.transAxes, fontsize=12, 
         fontweight='bold', va='bottom', ha='left')

# ---------------- Styling ----------------
# Colorbar borders
for cbar in [cbar1, cbar2]:
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)

# Panel borders
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_edgecolor('0.4')
        spine.set_linewidth(0.8)
    ax.tick_params(color='0.4', labelcolor='0.2')
    
    # Axis ticks
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.tick_params(axis='x', which='major', bottom=True, top=False,
                   labelbottom=True, labelsize=tick_fs)
    ax.tick_params(axis='y', which='major', left=True, right=False,
                   labelleft=True, labelsize=tick_fs)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "recovery_drivers_cc_vs_damage_by_earp_normalized.png",
            dpi=300, bbox_inches="tight")
plt.show()

print("\n✓ Normalized values plot saved")
print(f"  Annual: CC/1000 vs % EAUA correlation: r = {corr_cc_eaua_norm:+.3f} (p={p_cc_eaua_norm:.2e})")
print(f"  Median: CC/1000 vs % MUA correlation: r = {corr_cc_mua_norm:+.3f} (p={p_cc_mua_norm:.2e})")

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
print(f"\nAbsolute values (county size effect included):")
print(f"  Both correlations POSITIVE → Capacity scales with county size, not risk")
print(f"\nNormalized values (county size effect removed):")
print(f"  Correlation change reveals TRUE capacity-damage relationship")
if corr_cc_eaua_norm < 0 and corr_cc_mua_norm < 0:
    print(f"  Both NEGATIVE → Higher capacity intensity reduces damage intensity (efficient allocation)")
elif abs(corr_cc_eaua_norm) < 0.1 and abs(corr_cc_mua_norm) < 0.1:
    print(f"  Both NEAR ZERO → Capacity intensity UNRELATED to damage intensity (inefficient allocation)")
else:
    print(f"  Mixed signals → Complex relationship between capacity and damage intensity")
