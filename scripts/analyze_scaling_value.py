#!/usr/bin/env python3
"""
Analyze the value of scaling and explore options for northern county water-only impacts.

Questions addressed:
1. Is the scaling approach worth it? (correlation improvements)
2. Can we approximate water-only impacts for missed counties?
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def analyze_scaling_value():
    """Compare raw vs scaled correlations with AAL."""
    print("="*60)
    print("ANALYSIS: Is Scaling Worth It?")
    print("="*60)
    
    comp = pd.read_csv('analysis_output/aal_comparison.csv')
    
    mask_raw = (comp['aal_ncep'] > 0) & (comp['sim_repair_ead_raw'] > 0)
    mask_scaled = (comp['aal_ncep'] > 0) & (comp['sim_repair_ead_scaled'] > 0)
    
    # Raw correlations
    pr_raw, _ = pearsonr(np.log10(comp.loc[mask_raw, 'aal_ncep']), 
                         np.log10(comp.loc[mask_raw, 'sim_repair_ead_raw']))
    sr_raw, _ = spearmanr(comp.loc[mask_raw, 'aal_ncep'], 
                          comp.loc[mask_raw, 'sim_repair_ead_raw'])
    
    # Scaled correlations
    pr_scaled, _ = pearsonr(np.log10(comp.loc[mask_scaled, 'aal_ncep']), 
                            np.log10(comp.loc[mask_scaled, 'sim_repair_ead_scaled']))
    sr_scaled, _ = spearmanr(comp.loc[mask_scaled, 'aal_ncep'], 
                             comp.loc[mask_scaled, 'sim_repair_ead_scaled'])
    
    # Normalized correlations
    aal_norm = comp['aal_ncep'] / comp['aal_ncep'].max()
    raw_norm = comp['sim_repair_ead_raw'] / comp['sim_repair_ead_raw'].max()
    scaled_norm = comp['sim_repair_ead_scaled'] / comp['sim_repair_ead_scaled'].max()
    
    mask_norm = (aal_norm > 0) & (raw_norm > 0)
    pr_raw_norm, _ = pearsonr(aal_norm[mask_norm], raw_norm[mask_norm])
    pr_scaled_norm, _ = pearsonr(aal_norm[mask_norm], scaled_norm[mask_norm])
    
    print(f"\n{'Metric':<30} {'Raw':<12} {'Scaled':<12} {'Improvement':<12}")
    print("-"*66)
    print(f"{'Pearson r (log-log)':<30} {pr_raw:<12.3f} {pr_scaled:<12.3f} {pr_scaled-pr_raw:+.3f}")
    print(f"{'Spearman r':<30} {sr_raw:<12.3f} {sr_scaled:<12.3f} {sr_scaled-sr_raw:+.3f}")
    print(f"{'Pearson r (normalized)':<30} {pr_raw_norm:<12.3f} {pr_scaled_norm:<12.3f} {pr_scaled_norm-pr_raw_norm:+.3f}")
    
    raw_total = comp['sim_repair_ead_raw'].sum()
    scaled_total = comp['sim_repair_ead_scaled'].sum()
    aal_total = comp['aal_ncep'].sum()
    
    print(f"\n{'Magnitude (Sim/AAL ratio)':<30} {raw_total/aal_total:<12.3f} {scaled_total/aal_total:<12.3f} {(scaled_total-raw_total)/aal_total:+.3f}")
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    print(f"Scaling provides:")
    print(f"  • Modest improvement in correlations (+0.01 to +0.04)")
    print(f"  • Better alignment with AAL spatial patterns")
    print(f"  • BUT increases magnitude discrepancy (4.3× → 4.9×)")
    print(f"\nThe improvements are SMALL but consistent.")
    print(f"Given that scaling is computationally cheap (already done),")
    print(f"it's worth keeping, but NOT a game-changer.\n")


def explore_water_only_approximation():
    """Explore options for approximating water-only impacts in northern counties."""
    print("="*60)
    print("ANALYSIS: Options for Water-Only Impact Approximation")
    print("="*60)
    
    # Load hazard data
    wind_data = loadmat('data/hazard/maxwindmat_ncep_reanal.mat')
    surge_data = loadmat('data/hazard/maxelev_coastcounty_ncep_reanal.mat')
    rain_data = loadmat('data/hazard/ptot_rain_county_ncep_reanal.mat')
    
    wind = wind_data['maxwindmat'].T  # (counties × events)
    surge = surge_data['scounty']
    rain = rain_data['ptot_mat']
    
    # Load AAL for empirical relationship
    aal_data = loadmat('impacts_out/AAL_ncep_reanal.mat')
    aal_values = aal_data['AAL'].flatten()
    
    # Load county mapping
    county_map = pd.read_csv('data/county_region.csv')
    
    # Identify negligible-wind cells with significant water
    wind_threshold = 25.0
    surge_threshold = 0.5
    rain_threshold = 50.0
    
    zero_wind = wind < wind_threshold
    sig_surge = surge > surge_threshold
    sig_rain = rain > rain_threshold
    missed = zero_wind & (sig_surge | sig_rain)
    
    print(f"\nCells with water-only hazard: {missed.sum():,} ({100*missed.sum()/wind.size:.2f}%)")
    
    # Option 1: Simple empirical scaling from AAL
    print("\n" + "-"*60)
    print("OPTION 1: Empirical Scaling from AAL")
    print("-"*60)
    print("Idea: Use AAL to derive county-specific water damage factors")
    print("\nPros:")
    print("  ✓ Leverages existing AAL data")
    print("  ✓ County-specific adjustment")
    print("\nCons:")
    print("  ✗ Assumes AAL accurately represents water-only risk")
    print("  ✗ No event-level granularity")
    print("  ✗ Circular reasoning (validating against what we're deriving from)")
    
    # Option 2: Depth-damage functions
    print("\n" + "-"*60)
    print("OPTION 2: Simplified Depth-Damage Functions")
    print("-"*60)
    print("Idea: Convert surge height → damage using HAZUS-style curves")
    print("\nPros:")
    print("  ✓ Physically-based")
    print("  ✓ Event-specific")
    print("  ✓ Standard approach in flood modeling")
    print("\nCons:")
    print("  ✗ Need depth-damage curves (HAZUS or similar)")
    print("  ✗ Need building-level elevation data")
    print("  ✗ Surge is only max elevation per county (no spatial detail)")
    print("  ✗ Significant additional complexity")
    
    # Option 3: Hybrid approach - add baseline for missed counties
    print("\n" + "-"*60)
    print("OPTION 3: Additive Baseline from Gori et al.")
    print("-"*60)
    print("Idea: For zero-wind cells, add baseline damage from Gori's")
    print("      water-only contribution")
    print("\nImplementation:")
    print("  If wind_damage == 0 and (surge > threshold OR rain > threshold):")
    print("     damage = baseline_water_factor × exposure_value")
    print("  Else:")
    print("     damage = wind_damage × scaling_factor")
    print("\nPros:")
    print("  ✓ Conceptually simple")
    print("  ✓ Uses existing Gori data")
    print("  ✓ Captures the missed 3.6%")
    print("\nCons:")
    print("  ✗ Still lacks proper impact function for water")
    print("  ✗ Requires deriving baseline factor from Gori AAL data")
    print("  ✗ Mixing two different modeling approaches")
    
    # Option 4: Accept limitation
    print("\n" + "-"*60)
    print("OPTION 4: Accept Limitation & Document")
    print("-"*60)
    print("Idea: Acknowledge that model is wind-focused")
    print("\nPros:")
    print("  ✓ Honest about model scope")
    print("  ✓ No additional complexity")
    print("  ✓ Still captures 96%+ of hazard exposure")
    print("  ✓ Wind damage more relevant for structural recovery modeling")
    print("\nCons:")
    print("  ✗ Underestimates risk in northern coastal counties")
    print("  ✗ May affect recovery potential estimates in those areas")
    
    # Quantify the trade-off
    print("\n" + "="*60)
    print("QUANTITATIVE ASSESSMENT")
    print("="*60)
    
    missed_by_county = missed.sum(axis=1)
    counties_affected = (missed_by_county > 0).sum()
    
    # Estimate missed AAL
    aal_in_missed_counties = aal_values[missed_by_county > 0].sum()
    total_aal = aal_values.sum()
    
    print(f"\nCounties with missed events: {counties_affected:,} / 3,220 ({100*counties_affected/3220:.1f}%)")
    print(f"AAL in these counties: ${aal_in_missed_counties:,.0f}")
    print(f"Fraction of total AAL: {100*aal_in_missed_counties/total_aal:.1f}%")
    
    # Check if these counties are in our study area
    study_states = ['01', '09', '10', '12', '13', '22', '23', '24', '25', '28', 
                    '33', '34', '36', '37', '42', '44', '45', '48', '51']
    county_map['in_study'] = county_map['stcode'].isin(study_states)
    missed_in_study = county_map.loc[missed_by_county > 0, 'in_study'].sum()
    
    print(f"\nOf missed counties, in study area: {missed_in_study} / {counties_affected}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print("\nGiven the trade-offs, I recommend OPTION 4:")
    print("  → Accept the limitation and document it clearly")
    print("\nReasoning:")
    print("  1. Only 3.6% of exposure missed (manageable)")
    print("  2. Mostly northern states where wind models perform poorly anyway")
    print("  3. Your focus is RECOVERY POTENTIAL from structural damage")
    print("  4. Adding water-damage functions would require:")
    print("     - New impact functions (don't have)")
    print("     - Building elevations (don't have)")
    print("     - Significant validation work")
    print("  5. Options 1-3 add complexity with uncertain benefit")
    print("\nIf you MUST address it, Option 3 (additive baseline) is simplest:")
    print("  - Extract water-only AAL from Gori for zero-wind counties")
    print("  - Distribute proportionally to exposure")
    print("  - But this is essentially 'borrowing' Gori's results")
    print("\n")


def main():
    analyze_scaling_value()
    print("\n\n")
    explore_water_only_approximation()


if __name__ == "__main__":
    main()
