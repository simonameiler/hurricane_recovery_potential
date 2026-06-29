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
  construction_capacity        building permits per month (CC = Average_Building_Permits(12 months))
  earp_months_per_year         annual recovery potential (EARP)
  median_recovery_per_event    per-event recovery potential, paper definition:
                               median over ALL footprint events (incl. no-damage
                               zeros) for capacity>0 counties, from raw pyrecodes JSON

Priority screen = Map B definition: EAUA tercile 3 (highest risk) x CC tercile 1
(lowest capacity).  EAUA = total_weighted_damage_units * FREQ.

Inputs (relative to repo root):
  data/selected_states_counties_with_permits.csv   building-permit data
  data/NRI_Table_Counties.csv                      FEMA NRI county table
  analysis_output/earp_per_county.csv              written by TC_NA_recovery_analysis_reorganized_complete.py
  analysis_output/county_event_frequency_damage_metrics.csv  written by analyze_event_frequency_damage.py
  data/recovery/recovery_potential.csv             consolidated recovery CSV (run run_pyrecodes_light.py)
  data/US_counties.shp                             county boundaries

Outputs (all written to analysis_output/):
  nri_county_merged.csv          one row per county, our metrics + NRI
  nri_risk_comparison.csv        hazard layer: EAUA vs NRI hurricane/coastal-flood EAL
  nri_correlations.csv           Spearman: our metrics vs NRI (with Pearson-log too)
  nri_partial_correlations.csv   recovery/capacity vs SVI controlling for exposure
  nri_priority_crosstab.csv      priority counties vs NRI resilience/SVI terciles
  divergent_counties_SI.csv      priority counties invisible to NRI indices
  table_nri_evaluation.tex       combined SI table (hazard + recovery + partial corr)
  table_divergent_counties.tex   SI table of divergent priority counties
  fig_eaua_vs_nri_eal.png        SI scatter: EAUA vs NRI hurricane (wind) EAL
  fig_recovery_nri_agreement_2x2.png  2x2 agreement maps (EARP/median-recovery x Resilience/SVI)
  fig_recovery_resilience_agreement.png  single panel: median per-event recovery vs Resilience
  fig_capacity_vs_resilience.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent                                           # repo root


def _pick(*candidates):
    """Return the first candidate path that exists, else the first candidate.

    Lets the script run unchanged from the repository layout (data/,
    analysis_output/) and from a flat working folder where the same inputs sit
    next to the script.
    """
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(candidates[0])


PERMITS = _pick(BASE_DIR / "data" / "selected_states_counties_with_permits.csv",
                HERE / "selected_states_counties_with_permits.csv")
F_EARP = _pick(BASE_DIR / "analysis_output" / "earp_per_county.csv",
               HERE / "NRI_GDB_CensusTracts" / "earp_per_county.csv")
F_DAMAGE = _pick(BASE_DIR / "analysis_output" / "county_event_frequency_damage_metrics.csv",
                 HERE / "NRI_GDB_CensusTracts" / "county_event_frequency_damage_metrics.csv")
RECOVERY_CSV = _pick(BASE_DIR / "data" / "recovery" / "recovery_potential.csv",
                     HERE / "recovery_potential.csv")
IMPACT_DIR = BASE_DIR / "data" / "impact" / "per_event"
NRI_COUNTY = _pick(BASE_DIR / "data" / "NRI_Table_Counties.csv",
                   HERE / "NRI_Table_Counties" / "NRI_Table_Counties.csv")
COUNTIES_SHP = _pick(BASE_DIR / "data" / "US_counties.shp",
                     HERE / "US_counties.shp")
OUT_DIR = (BASE_DIR / "analysis_output") if (BASE_DIR / "analysis_output").exists() else HERE

CAP_COL = "Average_Building_Permits(12 months)"
FREQ = 1 / 1500  # events/yr, Poisson rate used in the study (≈0.000667)


def fips5(s):
    return (s.astype(str).str.replace(r"\.0$", "", regex=True)
            .str.replace(r"[^0-9]", "", regex=True).str.zfill(5))


def load_per_event_median():
    """Paper-style per-event recovery: median over ALL footprint events (incl.
    no-damage zeros) for each county; counties with zero capacity (NaN) dropped."""
    df = pd.read_csv(RECOVERY_CSV, dtype={"fips": str})
    df["fips"] = fips5(df["fips"])
    df["rec"] = pd.to_numeric(df["recovery_potential_months"], errors="coerce")
    finite = df[np.isfinite(df["rec"])]              # drop NaN (zero-capacity counties)
    agg = finite.groupby("fips")["rec"].agg(
        median_recovery_per_event="median", n_events="count").reset_index()
    print(f"  counties with finite recovery records: {len(agg)}")
    return agg


def load_nri_county():
    """Official FEMA NRI county table (no tract aggregation).

    HRCN_EALT is the Hurricane peril expected annual loss (essentially wind);
    CFLD_EALT is the separate Coastal Flooding peril EAL (storm surge). We load
    both so the hazard comparison can be checked against a wind-plus-surge match.
    """
    df = pd.read_csv(NRI_COUNTY, dtype={"STCOFIPS": str})
    df["fips"] = df["STCOFIPS"].str.zfill(5)
    nri_cols = ["RESL_SCORE", "SOVI_SCORE", "HRCN_EALT", "CFLD_EALT", "POPULATION"]
    # HRCN_EALB = hurricane building EAL (subset of HRCN_EALT; may not exist in all NRI versions)
    if "HRCN_EALB" in df.columns:
        nri_cols.append("HRCN_EALB")
    for c in nri_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    rename = {"RESL_SCORE": "nri_resilience", "SOVI_SCORE": "nri_social_vuln",
              "HRCN_EALT": "nri_hurricane_eal", "CFLD_EALT": "nri_coastal_flood_eal",
              "POPULATION": "nri_population"}
    if "HRCN_EALB" in df.columns:
        rename["HRCN_EALB"] = "nri_hurricane_eal_building"
    df = df.rename(columns=rename)
    keep = ["fips", "nri_population", "nri_resilience", "nri_social_vuln",
            "nri_hurricane_eal", "nri_coastal_flood_eal"]
    if "nri_hurricane_eal_building" in df.columns:
        keep.append("nri_hurricane_eal_building")
    return df[keep]


def load_inputs():
    earp = pd.read_csv(F_EARP); earp["fips"] = fips5(earp["fips"])
    dmg = pd.read_csv(F_DAMAGE); dmg["fips"] = fips5(dmg["fips"])
    dmg["eaua"] = dmg["total_weighted_damage_units"] * FREQ
    perm = pd.read_csv(PERMITS); perm["fips"] = fips5(perm["FIPS"])
    perm["construction_capacity"] = perm[CAP_COL]
    return earp, dmg, perm


def load_earc() -> pd.DataFrame:
    """Compute Expected Annual Repair Cost (EARC) per county from per_event CSVs.

    EARC_c = Σ_e repair_cost_sum_scaled_{e,c} * FREQ
    """
    files = sorted(IMPACT_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No per_event CSVs in {IMPACT_DIR}")
    print(f"Loading EARC from {len(files)} per_event files ...")
    df = pd.concat([pd.read_csv(f, usecols=["fips", "repair_cost_sum_scaled"]) for f in files],
                   ignore_index=True)
    earc = df.groupby("fips", as_index=False)["repair_cost_sum_scaled"].sum()
    earc["earc"] = earc["repair_cost_sum_scaled"] * FREQ
    earc["fips"] = fips5(earc["fips"])
    return earc[["fips", "earc"]]


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


def _tex_signed(x, dagger=False):
    """Format a correlation with a LaTeX-safe minus and an optional dagger mark."""
    if pd.isna(x):
        return "--"
    s = ("$-$" if x < 0 else "+") + f"{abs(x):.2f}"
    return s + (r"$^{\dagger}$" if dagger else "")


def write_latex_tables(out_dir, risk, corr, partial, si, n_sample):
    """Write the two SI tables: combined NRI evaluation, and divergent counties."""
    cset = corr.set_index(["our_metric", "nri_metric"])
    rrow = {r["comparison"]: r for _, r in risk.iterrows()}
    pset = partial.set_index("our_metric")

    nri_cols = ["Community Resilience", "Social Vulnerability", "Hurricane EAL (exposure)"]
    rec_rows = [("Construction capacity", "capacity"),
                ("EARP (annual recovery potential)", "EARP (annual recovery potential)"),
                ("Median per-event recovery potential", "median per-event recovery potential")]

    L = [r"\begin{table}[htb!]", r"\centering",
         (r"\caption{Evaluation of the recovery-potential model against the FEMA National "
          r"Risk Index (NRI), over the analysis sample of " + str(n_sample) + r" US "
          r"Atlantic-coast counties with positive modelled risk, recovery potential and "
          r"construction capacity. Panel~A compares the modelled hazard layer (expected "
          r"annual units affected, EAUA; expected annual repair cost, EARC) with the "
          r"NRI's independent, historically calibrated monetary risk. Panel~B reports "
          r"rank correlations between the model's quantities and the NRI socio-economic "
          r"layers (convergent and discriminant validity). Panel~C re-estimates the "
          r"association with Social Vulnerability after controlling for exposure. Entries "
          r"are Spearman rank coefficients; Panel~A also gives Pearson coefficients on "
          r"$\log_{10}$-transformed values. All coefficients are significant at $p<0.01$ "
          r"unless marked.}"),
         r"\label{tab:nri_evaluation}",
         r"\begin{tabular}{lrrr}", r"\hline",
         (r"\multicolumn{4}{l}{\textit{Panel A. Hazard layer: modelled physical risk vs "
          r"independent monetary risk}}\\"),
         r" & Spearman & Pearson (log) & $n$ \\", r"\hline"]
    for key in ["EAUA vs NRI hurricane (wind) EAL", "EAUA vs NRI coastal-flood EAL",
                "EAUA vs NRI wind + coastal-flood EAL"]:
        r = rrow.get(key)
        if r is None:
            continue
        L.append(f"{key} & {_tex_signed(r['spearman_r'])} & "
                 f"{_tex_signed(r['pearson_log_r'])} & {int(r['n'])} \\\\")
    # EARC rows (added if available)
    for key in list(rrow.keys()):
        if not key.startswith("EARC"):
            continue
        r = rrow[key]
        L.append(f"{key} & {_tex_signed(r['spearman_r'])} & "
                 f"{_tex_signed(r['pearson_log_r'])} & {int(r['n'])} \\\\")
    L += [r"\hline",
          (r"\multicolumn{4}{l}{\textit{Panel B. Recovery layer vs NRI socio-economic "
           r"indices (Spearman $r$)}}\\"),
          r" & Community & Social & Hurricane \\",
          r" & Resilience & Vulnerability & EAL (exposure) \\", r"\hline"]
    for disp, key in rec_rows:
        vals = [_tex_signed(cset.loc[(key, n), "spearman_r"],
                            dagger=float(cset.loc[(key, n), "p"]) > 0.05) for n in nri_cols]
        L.append(f"{disp} & " + " & ".join(vals) + r" \\")
    L += [r"\hline",
          (r"\multicolumn{4}{l}{\textit{Panel C. Association with Social Vulnerability, "
           r"controlling for exposure (Spearman $r$)}}\\"),
          r" & raw & partial & partial \\",
          r" & & (control EAUA) & (control wind EAL) \\", r"\hline"]
    for disp, key in [("Construction capacity", "capacity"),
                      ("Median per-event recovery potential", "median per-event recovery")]:
        p = pset.loc[key]
        L.append(f"{disp} & {_tex_signed(p['spearman_raw'])} & "
                 f"{_tex_signed(p['partial_given_EAUA'])} & "
                 f"{_tex_signed(p['partial_given_hurricaneEAL'])} \\\\")
    L += [r"\hline", r"\end{tabular}", r"\\[2pt]",
          (r"{\footnotesize $^{\dagger}$ Not significant ($p>0.05$): per-event recovery "
           r"potential is statistically independent of hazard exposure, unlike the annual "
           r"metric.}"),
          r"\end{table}"]
    (out_dir / "table_nri_evaluation.tex").write_text("\n".join(L) + "\n")

    D = [r"\begin{table}[htb!]", r"\centering",
         (r"\caption{Priority counties that socio-economic screening would miss. A "
          r"priority county is one ranked in both the highest tercile of expected "
          r"tropical cyclone (TC) damage and the lowest tercile of construction capacity "
          r"across the analysis sample ($n = 685$ counties); these are places where high "
          r"storm risk coincides with a thin local rebuilding sector. A county is "
          r"considered flagged by a socio-economic index when it falls in that index's "
          r"most concerning tercile, that is, the highest tercile of the FEMA National "
          r"Risk Index (NRI) Social Vulnerability score (SVI) or the lowest tercile of "
          r"the NRI Community Resilience score (Resil.). The 13 counties listed are "
          r"priority counties in neither of those terciles, so a screen built on either "
          r"NRI index would overlook them; they are the counties where the "
          r"building-permit capacity screen adds information beyond socio-economic "
          r"indicators. Counties are sorted by number of housing units. Pop.\ \% of "
          r"state is the percentage of the state population living in the county. Permits "
          r"per month is the construction-capacity proxy (average monthly building "
          r"permits), and permits per 1000 units normalises it by housing stock (sample "
          r"median 0.046). NRI SVI and Community Resilience are county percentile scores "
          r"from 0 to 100; a higher SVI indicates greater social vulnerability and a "
          r"higher Resilience score indicates greater resilience.}"),
         r"\label{tab:divergent_counties}",
         r"\begin{tabular}{llrrrrrr}", r"\hline",
         r"County & State & Housing & Pop.\ \% & Permits & Permits per & NRI & NRI \\",
         r" & & units & of state & per month & 1000 units & SVI & Resil. \\", r"\hline"]
    for _, r in si.iterrows():
        D.append(f"{r['county']} & {r['state']} & {int(round(r['housing_units'])):,} & "
                 f"{r['pop_pct_of_state']:.2f} & {r['permits_per_month']:.2f} & "
                 f"{r['permits_per_1000_units']:.3f} & {int(round(r['nri_social_vuln']))} & "
                 f"{int(round(r['nri_resilience']))} \\\\")
    D += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (out_dir / "table_divergent_counties.tex").write_text("\n".join(D) + "\n")


def main():
    print("Loading raw pyrecodes per-event recovery ...")
    per_event = load_per_event_median()
    earp, dmg, perm = load_inputs()
    nri = load_nri_county()
    earc_df = load_earc()

    m = (earp.merge(dmg[["fips", "eaua", "total_weighted_damage_units", "total_units"]], on="fips", how="outer")
              .merge(perm[["fips", "NAME", "STATE_NAME", "construction_capacity"]], on="fips", how="left")
              .merge(per_event, on="fips", how="left")
              .merge(nri, on="fips", how="left")
              .merge(earc_df, on="fips", how="left"))

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

    # ---- HAZARD LAYER: modelled physical risk vs independent monetary risk ----
    # Our EAUA (expected annual units affected, units/yr) is a physical risk metric;
    # the NRI hurricane EAL is a historically calibrated monetary risk ($) built from
    # entirely separate inputs. Agreement corroborates the hazard layer without sharing
    # data. NRI books storm surge under a separate Coastal Flooding peril, so we also
    # report EAUA against coastal-flood EAL and against wind+coastal-flood as a
    # peril-matching robustness check.
    def corr_pair(x, y):
        d = s[[x, y]].replace([np.inf, -np.inf], np.nan)
        d = d[(d[x] > 0) & (d[y] > 0)].dropna()
        sp, sp_p = spearmanr(d[x], d[y])
        pe = pearsonr(np.log10(d[x]), np.log10(d[y]))[0]
        return sp, pe, sp_p, len(d)

    s["nri_wind_plus_cflood_eal"] = s[["nri_hurricane_eal", "nri_coastal_flood_eal"]].sum(
        axis=1, min_count=1)
    risk_rows = []
    for yc, yl in [("nri_hurricane_eal", "NRI hurricane (wind) EAL"),
                   ("nri_coastal_flood_eal", "NRI coastal-flood EAL"),
                   ("nri_wind_plus_cflood_eal", "NRI wind + coastal-flood EAL")]:
        sp, pe, sp_p, n = corr_pair("eaua", yc)
        risk_rows.append({"metric": "EAUA", "comparison": f"EAUA vs {yl}",
                          "spearman_r": round(sp, 3),
                          "pearson_log_r": round(pe, 3), "p": f"{sp_p:.1e}", "n": n})
    # EARC vs NRI EAL (building EAL preferred; fall back to hurricane total EAL)
    earc_nri_cols = []
    if "nri_hurricane_eal_building" in s.columns:
        earc_nri_cols.append(("nri_hurricane_eal_building", "NRI hurricane building EAL"))
    earc_nri_cols += [("nri_hurricane_eal", "NRI hurricane (wind) EAL"),
                      ("nri_wind_plus_cflood_eal", "NRI wind + coastal-flood EAL")]
    for yc, yl in earc_nri_cols:
        if yc not in s.columns:
            continue
        sp, pe, sp_p, n = corr_pair("earc", yc)
        risk_rows.append({"metric": "EARC", "comparison": f"EARC vs {yl}",
                          "spearman_r": round(sp, 3),
                          "pearson_log_r": round(pe, 3), "p": f"{sp_p:.1e}", "n": n})
    risk = pd.DataFrame(risk_rows)
    risk.to_csv(OUT_DIR / "nri_risk_comparison.csv", index=False)
    print("\nHazard layer (EAUA/EARC vs NRI EAL):")
    print(risk.to_string(index=False))

    # internal consistency (both statistics, for the record)
    print("\nInternal consistency (analysis sample):")
    for a, b in [("earp_months_per_year", "eaua"), ("earp_months_per_year", "construction_capacity"),
                 ("median_recovery_per_event", "construction_capacity"),
                 ("median_recovery_per_event", "eaua")]:
        d = s[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
        sp = spearmanr(d[a], d[b])[0]
        dlog = d[(d[a] > 0) & (d[b] > 0)]
        pe = pearsonr(np.log10(dlog[a]), np.log10(dlog[b]))[0] if len(dlog) >= 3 else float("nan")
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

    # ---- SI tables (combined NRI evaluation + divergent counties) ----
    write_latex_tables(OUT_DIR, risk, corr, partial, si, len(s))
    print("\nWrote SI tables: table_nri_evaluation.tex, table_divergent_counties.tex")

    # ---- SI figure: EAUA vs NRI hurricane (wind) EAL ----
    dfr = s[["NAME", "STATE_NAME", "eaua", "nri_hurricane_eal"]].copy()
    dfr = dfr[(dfr["eaua"] > 0) & (dfr["nri_hurricane_eal"] > 0)].dropna()
    dfr["gap"] = dfr["eaua"].rank(pct=True) - dfr["nri_hurricane_eal"].rank(pct=True)
    figr, axr = plt.subplots(figsize=(6.5, 6))
    axr.scatter(dfr["nri_hurricane_eal"], dfr["eaua"], s=14, c="#9ecae1",
                edgecolor="none", alpha=0.7)
    label = pd.concat([dfr.nlargest(4, "gap"), dfr.nsmallest(4, "gap")])
    axr.scatter(label["nri_hurricane_eal"], label["eaua"], s=24, c="#d7191c", edgecolor="none")
    for _, r in label.iterrows():
        axr.annotate(r["NAME"], (r["nri_hurricane_eal"], r["eaua"]), fontsize=7,
                     xytext=(3, 3), textcoords="offset points", color="#d7191c")
    axr.set_xscale("log"); axr.set_yscale("log")
    axr.set_xlabel("NRI hurricane (wind) expected annual loss (USD)")
    axr.set_ylabel("EAUA (expected annual units affected, units/yr)")
    rho = spearmanr(dfr["eaua"], dfr["nri_hurricane_eal"])[0]
    axr.set_title(f"Modelled physical risk vs NRI monetary risk "
                  f"(Spearman $r={rho:+.2f}$, $n={len(dfr)}$)", fontsize=9)
    figr.tight_layout()
    figr.savefig(OUT_DIR / "fig_eaua_vs_nri_eal.png", dpi=200, bbox_inches="tight")

    # ---- SI figure: 2x2 agreement maps, recovery potential vs NRI socio-economic indices ----
    # Both quantities are oriented as "expected recovery difficulty" via within-sample
    # percentile ranks (n = analysis sample). The mapped value is the recovery-difficulty
    # rank minus the NRI-difficulty rank: red (positive) = our recovery metric flags the
    # county as harder to recover than the NRI score does (physical-capacity blind spots);
    # blue (negative) = the NRI score flags greater difficulty (more vulnerable / less
    # resilient) than our recovery does; white = the two agree.
    COASTAL = ['01', '09', '10', '12', '13', '22', '23', '24', '25', '28',
               '33', '34', '36', '37', '42', '44', '45', '48', '51']
    geo = gpd.read_file(COUNTIES_SHP)
    geo["fips"] = (geo["STATEFP"] + geo["COUNTYFP"]).str.zfill(5)
    geo = geo[geo["STATEFP"].isin(COASTAL)].copy()

    def rank_pct(col, higher_is_harder=True):
        r = s[col].replace([np.inf, -np.inf], np.nan).rank(pct=True)
        return r if higher_is_harder else (1.0 - r)

    # recovery metrics: higher value = slower recovery = harder
    s["rk_earp"] = rank_pct("earp_months_per_year")
    s["rk_medrec"] = rank_pct("median_recovery_per_event")
    # NRI: higher SVI = harder; lower Resilience = harder
    s["rk_svi"] = rank_pct("nri_social_vuln", higher_is_harder=True)
    s["rk_resl"] = rank_pct("nri_resilience", higher_is_harder=False)

    rec_rows = [("rk_earp", "earp_months_per_year", "EARP (annual recovery potential)"),
                ("rk_medrec", "median_recovery_per_event", "Median per-event recovery potential")]
    nri_cols2 = [("rk_resl", "Community Resilience"),
                 ("rk_svi", "Social Vulnerability")]

    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=-1, vmax=1)
    fig2, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    for i, (rk_rec, _, rec_lbl) in enumerate(rec_rows):
        for j, (rk_nri, nri_lbl) in enumerate(nri_cols2):
            ax = axes[i][j]
            s["agree"] = s[rk_rec] - s[rk_nri]
            g = geo.merge(s[["fips", "agree"]], on="fips", how="left")
            g.plot(ax=ax, color="#e9e9e9", edgecolor="#cccccc", linewidth=0.1)   # base / off-sample
            gv = g[g["agree"].notna()]
            gv.plot(ax=ax, column="agree", cmap=cmap, norm=norm,
                    edgecolor="#888888", linewidth=0.1)
            ax.set_xlim(-107, -65); ax.set_ylim(24, 48); ax.set_aspect("equal"); ax.axis("off")
            rho = spearmanr(s[rk_rec], s[rk_nri], nan_policy="omit")[0]
            ax.set_title(f"{rec_lbl}\nvs NRI {nri_lbl}  (agreement $r={rho:+.2f}$)", fontsize=9)
    fig2.subplots_adjust(left=0.02, right=0.88, top=0.90, bottom=0.04, wspace=0.04, hspace=0.14)
    cax = fig2.add_axes([0.90, 0.22, 0.016, 0.56])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig2.colorbar(sm, cax=cax)
    cb.set_label("recovery-difficulty rank  $-$  NRI-difficulty rank\n"
                 "red: model flags worse        blue: NRI flags worse", fontsize=8)
    fig2.suptitle("Agreement between modelled recovery potential and FEMA NRI "
                  "socio-economic indices\n(within-sample percentile ranks, both oriented "
                  "as expected recovery difficulty)", fontsize=11)
    fig2.savefig(OUT_DIR / "fig_recovery_nri_agreement_2x2.png", dpi=200, bbox_inches="tight")

    # ---- SI figure: single panel, median per-event recovery vs NRI Community Resilience ----
    # Same rank-difference construction as the 2x2; the 13 divergent priority counties
    # (Table tab:divergent_counties) are outlined in black.
    s["agree_mr_resl"] = s["rk_medrec"] - s["rk_resl"]
    g1 = geo.merge(s[["fips", "agree_mr_resl", "divergent"]], on="fips", how="left")
    fig1, ax1 = plt.subplots(figsize=(7.5, 6))
    g1.plot(ax=ax1, color="#e9e9e9", edgecolor="#cccccc", linewidth=0.1)   # base / off-sample
    g1[g1["agree_mr_resl"].notna()].plot(ax=ax1, column="agree_mr_resl", cmap=cmap, norm=norm,
                                         edgecolor="#888888", linewidth=0.1)
    dvg = g1[g1["divergent"] == True]
    if len(dvg):
        dvg.plot(ax=ax1, facecolor="none", edgecolor="black", linewidth=1.1)
    ax1.set_xlim(-107, -65); ax1.set_ylim(24, 48); ax1.set_aspect("equal"); ax1.axis("off")
    cax1 = fig1.add_axes([0.84, 0.24, 0.018, 0.52])
    sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm1.set_array([])
    cb1 = fig1.colorbar(sm1, cax=cax1)
    cb1.set_label("recovery-difficulty rank  $-$  resilience-difficulty rank", fontsize=8)
    cb1.ax.text(0.5, 1.03, "red: model flags worse", transform=cb1.ax.transAxes,
                fontsize=8, ha="center", va="bottom")
    cb1.ax.text(0.5, -0.03, "blue: NRI flags worse", transform=cb1.ax.transAxes,
                fontsize=8, ha="center", va="top")
    fig1.savefig(OUT_DIR / "fig_recovery_resilience_agreement.png", dpi=200, bbox_inches="tight")

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
