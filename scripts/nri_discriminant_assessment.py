"""
NRI construct-validation assessment for the hurricane recovery potential study.

This is NOT criterion validation (no observed recovery ground truth exists). It is
construct validation: we test whether our permits-based capacity and recovery
potential (i) move in the theoretically expected direction relative to established
resilience/vulnerability indices  [convergent validity], and (ii) are not redundant
with them  [discriminant validity].

Reference indices (FEMA National Risk Index, census-tract product rolled up to county):
  RESL_SCORE -> Community Resilience (BRIC-based)
  SOVI_SCORE -> Social Vulnerability
  HRCN_EALT  -> Hurricane expected annual loss (USD)   [exposure control]

Our metrics (analysis sample = positive EARP / EAUA / capacity):
  construction_capacity        building permits per month (CC = permits(12mo)/12)
  earp_months_per_year         annual recovery potential (EARP)
  median_recovery_per_event    per-event recovery potential, paper definition:
                               median over ALL footprint events (incl. no-damage
                               zeros) for capacity>0 counties, from raw pyrecodes JSON

Priority screen = Map B definition: EAUA tercile 3 (highest risk) x CC tercile 1
(lowest capacity).  EAUA = total_weighted_damage_units * FREQ.

Inputs (relative to repo root):
  data/selected_states_counties_with_permits.csv   building-permit data
  data/external/NRI_Table_Counties.csv             FEMA NRI county table
  analysis_output/earp_per_county.csv              written by TC_NA_recovery_analysis_reorganized_complete.py
  analysis_output/county_event_frequency_damage_metrics.csv  written by analyze_event_frequency_damage.py
  data/recovery_potential_per_scenario/            pyrecodes per-scenario JSON outputs (external; not committed)
  data/US_counties.shp                             county boundaries

Outputs (all written to analysis_output/):
  nri_county_merged.csv          one row per county, our metrics + NRI
  nri_correlations.csv           Spearman: our metrics vs NRI (with Pearson-log too)
  nri_partial_correlations.csv   recovery/capacity vs SVI controlling for exposure
  nri_priority_crosstab.csv      67 priority counties vs NRI resilience/SVI terciles
  divergent_counties_SI.csv      priority counties invisible to NRI indices
  fig_mapB_risk_capacity_57.png  Map B regenerated on analysis sample
  fig_capacity_vs_resilience.png
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent                                           # repo root

PERMITS = BASE_DIR / "data" / "selected_states_counties_with_permits.csv"
F_EARP = BASE_DIR / "analysis_output" / "earp_per_county.csv"
F_DAMAGE = BASE_DIR / "analysis_output" / "county_event_frequency_damage_metrics.csv"
SCEN_DIR = BASE_DIR / "data" / "recovery_potential_per_scenario"
OUT_DIR = BASE_DIR / "analysis_output"

CAP_COL = "Average_Building_Permits(12 months)"
REC_KEY = "recovery_potential [months]"
FREQ = 0.00067334  # events/yr, Poisson rate used in the study


def fips5(s):
    return (s.astype(str).str.replace(r"\.0$", "", regex=True)
            .str.replace(r"[^0-9]", "", regex=True).str.zfill(5))


def load_per_event_median():
    """Paper-style per-event recovery: median over ALL footprint events (incl.
    no-damage zeros) for each county; counties with zero capacity (=> inf) dropped."""
    rows = []
    for fp in sorted(SCEN_DIR.glob("*_scaled_recovery_potential.json")):
        with open(fp) as fh:
            for rec in json.load(fh):
                rows.append((str(rec["fips"]), float(rec[REC_KEY])))
    df = pd.DataFrame(rows, columns=["fips", "rec"])
    df["fips"] = fips5(df["fips"])
    finite = df[np.isfinite(df["rec"])]              # drop inf (zero-capacity counties)
    agg = finite.groupby("fips")["rec"].agg(
        median_recovery_per_event="median", n_events="count").reset_index()
    print(f"  counties with finite recovery records: {len(agg)}")
    return agg


def load_nri_county():
    """Official FEMA NRI county table (no tract aggregation)."""
    f = BASE_DIR / "data" / "external" / "NRI_Table_Counties.csv"
    df = pd.read_csv(f, dtype={"STCOFIPS": str})
    df["fips"] = df["STCOFIPS"].str.zfill(5)
    for c in ["RESL_SCORE", "SOVI_SCORE", "HRCN_EALT", "POPULATION"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.rename(columns={"RESL_SCORE": "nri_resilience",
                              "SOVI_SCORE": "nri_social_vuln",
                              "HRCN_EALT": "nri_hurricane_eal",
                              "POPULATION": "nri_population"})[
        ["fips", "nri_population", "nri_resilience", "nri_social_vuln", "nri_hurricane_eal"]]


def load_inputs():
    earp = pd.read_csv(F_EARP); earp["fips"] = fips5(earp["fips"])
    dmg = pd.read_csv(F_DAMAGE); dmg["fips"] = fips5(dmg["fips"])
    dmg["eaua"] = dmg["total_weighted_damage_units"] * FREQ
    perm = pd.read_csv(PERMITS); perm["fips"] = fips5(perm["FIPS"])
    perm["construction_capacity"] = perm[CAP_COL] / 12.0
    return earp, dmg, perm


def tercile(s):
    v = s.where((s > 0) & np.isfinite(s))
    return pd.qcut(v, 3, labels=[1, 2, 3])


def partial_spearman(df, x, y, z):
    """Spearman partial correlation of x,y controlling for z (rank-based)."""
    sub = df[[x, y, z]].replace([np.inf, -np.inf], np.nan).dropna()
    rx, ry, rz = sub[x].rank().values, sub[y].rank().values, sub[z].rank().values

    def resid(a, b):
        B = np.c_[np.ones(len(b)), b]
        coef, *_ = np.linalg.lstsq(B, a, rcond=None)
        return a - B @ coef
    ex, ey = resid(rx, rz), resid(ry, rz)
    r, p = pearsonr(ex, ey)
    return r, p, len(sub)


def main():
    print("Loading raw pyrecodes per-event recovery ...")
    per_event = load_per_event_median()
    earp, dmg, perm = load_inputs()
    nri = load_nri_county()

    m = (earp.merge(dmg[["fips", "eaua", "total_weighted_damage_units", "total_units"]], on="fips", how="outer")
              .merge(perm[["fips", "NAME", "STATE_NAME", "construction_capacity"]], on="fips", how="left")
              .merge(per_event, on="fips", how="left")
              .merge(nri, on="fips", how="left"))

    # analysis sample: positive EARP / EAUA / capacity (matches the paper)
    m["in_sample"] = (m["earp_months_per_year"] > 0) & (m["eaua"] > 0) & (m["construction_capacity"] > 0)
    s = m[m["in_sample"]].copy()

    # priority screen (Map B): EAUA tercile 3 x CC tercile 1
    s["eaua_t"] = tercile(s["eaua"])
    s["cc_t"] = tercile(s["construction_capacity"])
    s["earp_t"] = tercile(s["earp_months_per_year"])
    s["priority"] = (s["eaua_t"] == 3) & (s["cc_t"] == 1)
    m = m.merge(s[["fips", "eaua_t", "cc_t", "earp_t", "priority"]], on="fips", how="left")
    OUT_DIR.mkdir(exist_ok=True)
    m.to_csv(OUT_DIR / "nri_county_merged.csv", index=False)
    print(f"\nAnalysis sample: {len(s)} counties | priority (EAUA t3 x CC t1): {int(s['priority'].sum())}")

    # internal consistency (both statistics, for the record)
    print("\nInternal consistency (analysis sample):")
    for a, b in [("earp_months_per_year", "eaua"), ("earp_months_per_year", "construction_capacity"),
                 ("median_recovery_per_event", "construction_capacity"),
                 ("median_recovery_per_event", "eaua")]:
        d = s[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
        sp = spearmanr(d[a], d[b])[0]
        pe = pearsonr(np.log10(d[a]), np.log10(d[b]))[0]
        print(f"  {a:26s} vs {b:22s}: Spearman={sp:+.3f}  Pearson(log)={pe:+.3f}  n={len(d)}")

    # ---- discriminant correlations: our metrics vs NRI ----
    our = [("construction_capacity", "capacity"),
           ("earp_months_per_year", "EARP (annual recovery potential)"),
           ("median_recovery_per_event", "median per-event recovery potential")]
    nri_cols = [("nri_resilience", "Community Resilience"),
                ("nri_social_vuln", "Social Vulnerability"),
                ("nri_hurricane_eal", "Hurricane EAL (exposure)")]
    rows = []
    for xc, xl in our:
        for yc, yl in nri_cols:
            d = s[[xc, yc]].replace([np.inf, -np.inf], np.nan).dropna()
            r, p = spearmanr(d[xc], d[yc])
            rows.append({"our_metric": xl, "nri_metric": yl,
                         "spearman_r": round(r, 3), "p": f"{p:.1e}", "n": len(d)})
    corr = pd.DataFrame(rows)
    corr.to_csv(OUT_DIR / "nri_correlations.csv", index=False)
    print("\nDiscriminant correlations (Spearman, our metrics vs NRI):")
    print(corr.to_string(index=False))

    # ---- partial correlations: control for exposure ----
    # two exposure controls: our own model exposure (EAUA) and NRI hurricane EAL ($)
    prows = []
    for xc, xl in [("median_recovery_per_event", "median per-event recovery"),
                   ("construction_capacity", "capacity")]:
        d0 = s[[xc, "nri_social_vuln"]].replace([np.inf, -np.inf], np.nan).dropna()
        r0 = spearmanr(d0[xc], d0["nri_social_vuln"])[0]
        r_eaua = partial_spearman(s, xc, "nri_social_vuln", "eaua")[0]
        r_heal = partial_spearman(s, xc, "nri_social_vuln", "nri_hurricane_eal")[0]
        prows.append({"our_metric": xl, "vs": "Social Vulnerability",
                      "spearman_raw": round(r0, 3),
                      "partial_given_EAUA": round(r_eaua, 3),
                      "partial_given_hurricaneEAL": round(r_heal, 3), "n": len(d0)})
    partial = pd.DataFrame(prows)
    partial.to_csv(OUT_DIR / "nri_partial_correlations.csv", index=False)
    print("\nPartial correlation (controlling for hurricane EAL exposure):")
    print(partial.to_string(index=False))

    # ---- priority counties vs NRI terciles ----
    s["resl_t"] = tercile(s["nri_resilience"])
    s["sovi_t"] = tercile(s["nri_social_vuln"])
    pr = s[s["priority"]]
    n_pri = len(pr)
    ct_resl = pd.crosstab(pr["priority"], pr["resl_t"])
    ct_sovi = pd.crosstab(pr["priority"], pr["sovi_t"])
    pd.concat([ct_resl.rename(index={True: "resilience"}),
               ct_sovi.rename(index={True: "social_vuln"})]).to_csv(OUT_DIR / "nri_priority_crosstab.csv")
    print(f"\nPriority counties (n={n_pri}) by NRI Community Resilience tercile (1=low,3=high):")
    print(pr["resl_t"].value_counts().sort_index().to_string())
    print(f"Priority counties by NRI Social Vulnerability tercile (1=low,3=high):")
    print(pr["sovi_t"].value_counts().sort_index().to_string())
    hi_sovi = (pr["sovi_t"] == 3).mean() * 100
    lo_resl = (pr["resl_t"] == 1).mean() * 100
    print(f"\nHeadline: {hi_sovi:.0f}% of priority counties are top-tercile SVI; "
          f"{100-hi_sovi:.0f}% are NOT.")
    print(f"          {lo_resl:.0f}% are bottom-tercile resilience; {100-lo_resl:.0f}% are NOT.")

    # ---- divergent counties: priority but flagged by NEITHER index ----
    s["divergent"] = s["priority"] & ~((s["sovi_t"] == 3) | (s["resl_t"] == 1))
    state_pop = nri.groupby(nri["fips"].str[:2])["nri_population"].sum()
    div = s[s["divergent"]].copy()
    div["pop_pct_state"] = div.apply(
        lambda r: 100 * r["nri_population"] / state_pop[r["fips"][:2]], axis=1)
    div["permits_per_1000units"] = div["construction_capacity"] / div["total_units"] * 1000
    si = div[["fips", "NAME", "STATE_NAME", "total_units", "pop_pct_state",
              "construction_capacity", "permits_per_1000units",
              "nri_social_vuln", "nri_resilience"]].sort_values("total_units")
    si.columns = ["fips", "county", "state", "housing_units", "pop_pct_of_state",
                  "permits_per_month", "permits_per_1000_units", "nri_social_vuln",
                  "nri_resilience"]
    si.to_csv(OUT_DIR / "divergent_counties_SI.csv", index=False)
    print(f"\nDivergent priority counties (flagged by neither index): {len(div)}")
    print(f"  median housing units: {div['total_units'].median():.0f} "
          f"(study median {s['total_units'].median():.0f})")
    print(f"  median permits/1000 units: {div['permits_per_1000units'].median():.3f} "
          f"(study median {(s['construction_capacity']/s['total_units']*1000).median():.3f})")

    # ---- regenerate bivariate Map B (Risk x Capacity) at the 57-county definition ----
    GRID_B = [["#e8e8e8", "#ace4e4", "#5ac8c8"],
              ["#b8d6be", "#90b2b3", "#567994"],
              ["#73ae80", "#5a9178", "#2a5a5b"]]
    COASTAL = ['01', '09', '10', '12', '13', '22', '23', '24', '25', '28',
               '33', '34', '36', '37', '42', '44', '45', '48', '51']
    cty = gpd.read_file(BASE_DIR / "data" / "US_counties.shp")
    cty["fips"] = (cty["STATEFP"] + cty["COUNTYFP"]).str.zfill(5)
    cty = cty[cty["STATEFP"].isin(COASTAL)].merge(
        s[["fips", "eaua_t", "cc_t", "priority", "divergent"]], on="fips", how="left")

    def color_b(r):
        if pd.isna(r["eaua_t"]) or pd.isna(r["cc_t"]):
            return "#d9d9d9"
        return GRID_B[int(4 - r["cc_t"]) - 1][int(r["eaua_t"]) - 1]
    cty["c"] = cty.apply(color_b, axis=1)

    figm, axm = plt.subplots(figsize=(9, 6))
    cty.plot(ax=axm, color=cty["c"], edgecolor="#aaaaaa", linewidth=0.15)
    dv = cty[cty["divergent"] == True]
    if len(dv):
        dv.plot(ax=axm, facecolor="none", edgecolor="#d7191c", linewidth=1.3)
    axm.set_xlim(-107, -65); axm.set_ylim(24, 48); axm.set_aspect("equal"); axm.axis("off")
    axm.set_title("Map B: TC risk (EAUA) x construction capacity, joint-sample terciles\n"
                  "red outline = priority county invisible to NRI SVI/Resilience", fontsize=10)
    figm.tight_layout()
    figm.savefig(OUT_DIR / "fig_mapB_risk_capacity_57.png", dpi=200, bbox_inches="tight")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7, 5.5))
    base = s[~s["priority"]]
    ax.scatter(base["nri_resilience"], base["construction_capacity"], s=14,
               c="#bdbdbd", alpha=0.6, edgecolor="none", label="other counties")
    ax.scatter(pr["nri_resilience"], pr["construction_capacity"], s=30,
               c="#d7191c", alpha=0.85, edgecolor="none", label="priority (high risk / low capacity)")
    ax.set_yscale("log")
    ax.set_xlabel("NRI Community Resilience score")
    ax.set_ylabel("Construction capacity (permits/month, log)")
    ax.set_title("Priority counties vs NRI Community Resilience")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_capacity_vs_resilience.png", dpi=150)
    print("\nWrote merged CSV, correlations, partial correlations, crosstab, figure.")


if __name__ == "__main__":
    main()
