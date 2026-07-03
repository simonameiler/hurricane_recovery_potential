"""Create the manuscript tables for the historical-storm results section.

Reads the per-event aggregated impact CSVs (data/impact_historical/per_event/),
the historical recovery results (data/recovery_historical/recovery_potential.csv)
and the Huang et al. (2025) county-event table, and writes LaTeX + CSV tables to
tables/:

  - table_laura_main.tex        main-text Laura county table (4 counties)
  - table_laura_SI.tex/.csv     SI full Laura county table (all damaged counties)
  - table_damage_distribution.tex/.csv  FEMA-primary damage evaluation
        (rank correlation + detection share vs OpenFEMA IA verified damage,
        Huang building sample secondary; needs data/fema_ia/ from
        fetch_openfema_damage.py, skipped where absent)
  - table_damage_normalized.tex/.csv    SI robustness: same FEMA comparison
        per exposure unit (county-size control)
  - table_huang_thresholds.tex/.csv  SI Spearman correlations vs Huang filter

Companion to create_manuscript_figures.py / create_validation_figures.py.
Run from the repo root or from scripts/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------- paths
REPO = Path(__file__).resolve().parent.parent
PER_EVENT_DIR = REPO / "data" / "impact_historical" / "per_event"
RECOVERY_CSV = REPO / "data" / "recovery_historical" / "recovery_potential.csv"
HUANG_CSV = REPO / "data" / "huang_recovery_by_county_event.csv"
COUNTIES_DBF = REPO / "data" / "US_counties.dbf"
OUT_DIR = REPO / "tables"
OUT_DIR.mkdir(exist_ok=True)

TAU = {1: 1, 2: 1, 3: 3, 4: 6}  # repair-time weights (months) per damage state

STORMS = {  # ATCF -> (name, year)
    "AL122005": ("Katrina", 2005),
    "AL092012": ("Isaac", 2012),
    "AL092016": ("Hermine", 2016),
    "AL092017": ("Harvey", 2017),
    "AL112017": ("Irma", 2017),
    "AL162017": ("Nate", 2017),
    "AL142018": ("Michael", 2018),
    "AL132020": ("Laura", 2020),
    "AL092021": ("Ida", 2021),
    "AL092022": ("Ian", 2022),
}
FEMA_CSV = REPO / "data" / "fema_ia" / "fema_damage_by_county.csv"


def load_event(atcf: str) -> pd.DataFrame | None:
    files = sorted(PER_EVENT_DIR.glob(f"aggregated_*_{atcf}.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["fips"] = df["fips"].astype(str).str.split(".").str[0].str.zfill(5)
    df["wua_raw"] = sum(df[f"units_DS{k}_raw"] * TAU[k] for k in TAU)
    df["wua_scaled"] = sum(df[f"units_DS{k}_scaled"] * TAU[k] for k in TAU)
    return df


def county_names() -> dict:
    """GEOID -> (name, state_abbr) from the counties shapefile DBF."""
    state_abbr = {
        "01": "AL", "12": "FL", "13": "GA", "22": "LA", "28": "MS",
        "45": "SC", "48": "TX",
    }
    try:
        import geopandas as gpd

        gdf = gpd.read_file(COUNTIES_DBF.with_suffix(".shp"))
        return {
            str(g): (n, state_abbr.get(str(s), str(s)))
            for g, n, s in zip(gdf["GEOID"], gdf["NAME"], gdf["STATEFP"])
        }
    except ImportError:
        from dbfread import DBF

        return {
            str(r["GEOID"]): (r["NAME"], state_abbr.get(str(r["STATEFP"]), ""))
            for r in DBF(COUNTIES_DBF, encoding="utf-8")
        }


def fmt_int(x: float) -> str:
    return f"{int(round(x)):,}"


# ================================================================ Laura tables
recovery = pd.read_csv(RECOVERY_CSV)
recovery["fips"] = recovery["fips"].astype(str).str.zfill(5)
names = county_names()

laura = load_event("AL132020").merge(
    recovery.loc[
        recovery.event_name == "AL132020",
        ["fips", "reconstruction_capacity", "recovery_potential_months"],
    ],
    on="fips",
    how="left",
)
laura = laura[laura["wua_scaled"] > 0].sort_values("wua_scaled", ascending=False)
laura["county"] = laura["fips"].map(lambda f: names.get(f, (f, ""))[0])
laura["state"] = laura["fips"].map(lambda f: names.get(f, (f, ""))[1])

# ---- main-text table: the four counties agreed with SM
MAIN_FIPS = ["22019", "22011", "22079", "22115"]
rows = []
for f in MAIN_FIPS:
    r = laura[laura.fips == f].iloc[0]
    rows.append(
        f"{r.county} & {fmt_int(r.wua_scaled)} & "
        f"{r.reconstruction_capacity:.1f} & {fmt_int(r.recovery_potential_months)} \\\\"
    )
main_tex = (
    "\\begin{table}[htb!]\n\\centering\n"
    "\\caption{Modelled damage, construction capacity and recovery potential for\n"
    "four Louisiana parishes affected by Hurricane Laura (2020). Weighted units\n"
    "affected is the multi-hazard-scaled repair demand,\n"
    "$\\sum_k \\mathrm{units}_{\\mathrm{DS}k} \\times \\tau_k$ with\n"
    "$\\tau = (1, 1, 3, 6)$ months for damage states DS1--DS4. Construction\n"
    "capacity is the average monthly building-permit count. Recovery potential is\n"
    "the simulated time to work off the repair demand at that capacity. The full\n"
    "table for all damaged counties is given in the SI.}\n"
    "\\label{tab:laura_counties}\n"
    "\\begin{tabular}{lrrr}\n\\hline\n"
    "Parish (La.) & Weighted units & Capacity & Recovery potential \\\\\n"
    " & affected & (permits/month) & (months) \\\\\n\\hline\n"
    + "\n".join(rows)
    + "\n\\hline\n\\end{tabular}\n\\end{table}\n"
)
(OUT_DIR / "table_laura_main.tex").write_text(main_tex)

# ---- SI table: all damaged counties
si_rows = []
for _, r in laura.iterrows():
    rec = (
        fmt_int(r.recovery_potential_months)
        if np.isfinite(r.recovery_potential_months)
        else "--"
    )
    cap = f"{r.reconstruction_capacity:.1f}" if r.reconstruction_capacity > 0 else "0.0"
    si_rows.append(
        f"{r.county} & {r.state} & {fmt_int(r.wua_scaled)} & {fmt_int(r.wua_raw)} & "
        f"{r.repair_cost_sum_scaled / 1e6:,.1f} & {cap} & {rec} \\\\"
    )
si_tex = (
    "\\begin{table}[htb!]\n\\centering\n"
    "\\caption{All counties with modelled damage from Hurricane Laura (2020),\n"
    "sorted by scaled weighted units affected (WUA). Raw WUA uses wind-only\n"
    "damage; scaled WUA includes the rain and surge scaling. Repair cost is the\n"
    "scaled residential structural repair cost. Counties with zero permit\n"
    "capacity have undefined recovery potential (--).}\n"
    "\\label{tab:laura_counties_SI}\n"
    "\\begin{tabular}{llrrrrr}\n\\hline\n"
    "County & State & WUA & WUA & Repair cost & Capacity & Recovery \\\\\n"
    " & & (scaled) & (raw) & (million USD) & (permits/mo) & (months) \\\\\n\\hline\n"
    + "\n".join(si_rows)
    + "\n\\hline\n\\end{tabular}\n\\end{table}\n"
)
(OUT_DIR / "table_laura_SI.tex").write_text(si_tex)
laura[
    [
        "fips", "county", "state", "wua_scaled", "wua_raw",
        "repair_cost_sum_raw", "repair_cost_sum_scaled",
        "reconstruction_capacity", "recovery_potential_months",
    ]
].to_csv(OUT_DIR / "table_laura_SI.csv", index=False)

# ==================================================== damage distribution
# FEMA-primary evaluation of the modelled damage distribution.
# For each storm, three statistics against the FEMA-verified real property
# damage of owner households (OpenFEMA HousingAssistanceOwners, all declared
# counties, aggregated by fetch_openfema_damage.py):
#   1. within-storm Spearman of county WUA (scaled) vs FEMA damage over the
#      counties present in both sources (ranking skill);
#   2. detection share: fraction of the storm's total FEMA-verified damage
#      that falls in counties with modelled damage (coverage skill);
#   3. SI robustness: the same Spearman after normalising both sides by the
#      county's exposure units (damage intensity, removes county size).
# The Huang et al. (2025) building sample (n_changed) is the secondary,
# building-level anchor. Spearman is invariant to within-storm scaling, so
# correlations on county values equal correlations on within-storm shares.
huang = pd.read_csv(HUANG_CSV)
huang["GEOID"] = huang["GEOID"].astype(str).str.zfill(5)
fema = None
if FEMA_CSV.exists():
    fema = pd.read_csv(FEMA_CSV, dtype={"fips": str})
    fema["fips"] = fema["fips"].str.zfill(5)
else:
    print(f"note: {FEMA_CSV} missing - run fetch_openfema_damage.py; "
          "FEMA leg skipped")
units = pd.read_csv(REPO / "data" / "exposure_units_by_county.csv",
                    dtype={"fips": str})
units["fips"] = units["fips"].str.zfill(5)

MIN_N = 4  # smallest overlap for which a per-storm rank correlation is shown
rows, norm_rows, pooled_h, pooled_f = [], [], [], []
for atcf, (name, year) in sorted(STORMS.items(), key=lambda kv: kv[1][1]):
    df = load_event(atcf)
    df = df[df["wua_scaled"] > 0] if df is not None else None

    # ---- FEMA leg (primary)
    n_f, rho_f, p_f, det = 0, np.nan, np.nan, np.nan
    n_n, rho_n, p_n = 0, np.nan, np.nan
    f = (fema.loc[fema.atcf_id == atcf, ["fips", "fema_total_damage"]]
         if fema is not None else pd.DataFrame())
    if len(f):
        total_fema = f["fema_total_damage"].sum()
        if df is None:
            det = 0.0
        else:
            ov = df.merge(f, on="fips", how="inner")
            n_f = len(ov)
            det = ov["fema_total_damage"].sum() / total_fema
            if n_f >= MIN_N:
                rho_f, p_f = spearmanr(ov["wua_scaled"],
                                       ov["fema_total_damage"])
            if n_f >= 2:
                pooled_f.append(ov.assign(
                    sm=ov["wua_scaled"] / ov["wua_scaled"].sum(),
                    so=ov["fema_total_damage"] / ov["fema_total_damage"].sum()))
            # normalised (per exposure unit), SI robustness check
            ovn = ov.merge(units, on="fips", how="inner")
            n_n = len(ovn)
            if n_n >= MIN_N:
                rho_n, p_n = spearmanr(
                    ovn["wua_scaled"] / ovn["exposure_units"],
                    ovn["fema_total_damage"] / ovn["exposure_units"])

    # ---- Huang leg (secondary)
    n_h, rho_h, p_h = 0, np.nan, np.nan
    if df is not None:
        h = huang.loc[huang.atcf_id == atcf, ["GEOID", "n_changed"]].rename(
            columns={"GEOID": "fips"})
        ovh = df.merge(h, on="fips", how="inner")
        n_h = len(ovh)
        if n_h >= MIN_N:
            rho_h, p_h = spearmanr(ovh["wua_scaled"], ovh["n_changed"])
        if n_h >= 2:
            pooled_h.append(ovh.assign(
                sm=ovh["wua_scaled"] / ovh["wua_scaled"].sum(),
                so=ovh["n_changed"] / ovh["n_changed"].sum()))

    if len(f) == 0 and n_h == 0:
        continue
    rows.append(dict(storm=name, year=year, atcf=atcf,
                     n_fema=n_f, rho_fema=rho_f, p_fema=p_f, detection=det,
                     n_huang=n_h, rho_huang=rho_h, p_huang=p_h))
    norm_rows.append(dict(storm=name, year=year, n_raw=n_f, rho_raw=rho_f,
                          p_raw=p_f, n_norm=n_n, rho_norm=rho_n, p_norm=p_n))

dist = pd.DataFrame(rows)


def _pooled(pool):
    if not pool:
        return 0, np.nan, np.nan
    d = pd.concat(pool, ignore_index=True)
    rho, p = spearmanr(d["sm"], d["so"])
    return len(d), rho, p


np_f, rp_f, pp_f = _pooled(pooled_f)
np_h, rp_h, pp_h = _pooled(pooled_h)
dist = pd.concat([dist, pd.DataFrame([dict(
    storm="Pooled (within-storm shares)", year="", atcf="",
    n_fema=np_f, rho_fema=rp_f, p_fema=pp_f, detection=np.nan,
    n_huang=np_h, rho_huang=rp_h, p_huang=pp_h)])], ignore_index=True)
dist.to_csv(OUT_DIR / "table_damage_distribution.csv", index=False)


def _p(p):
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def _cell(n, rho, p):
    if n == 0:
        return "-- & -- & --"
    if not np.isfinite(rho):
        return f"{int(n)} & -- & --"
    if abs(rho) == 1.0 and n < 6:
        # scipy's t-approximation returns p=0 at |rho|=1; not meaningful at
        # this sample size, so suppress the p-value
        return f"{int(n)} & ${rho:+.2f}$ & --"
    return f"{int(n)} & ${rho:+.2f}$ & {_p(p)}"


tex_rows = []
for _, r in dist.iterrows():
    label = f"{r.storm} ({r.year})" if r.year != "" else r.storm
    det = f"{100 * r.detection:.0f}\\%" if np.isfinite(r.detection) else "--"
    tex_rows.append(
        f"{label} & {_cell(r.n_fema, r.rho_fema, r.p_fema)} & {det} & "
        f"{_cell(r.n_huang, r.rho_huang, r.p_huang)} \\\\"
    )
dist_tex = (
    "\\begin{table}[htb!]\n\\centering\n"
    "\\caption{Evaluation of the modelled damage distribution against\n"
    "FEMA-verified housing damage (primary) and the Huang et al.\\ (2025)\n"
    "building sample (secondary). For each storm, the scaled weighted units\n"
    "affected (WUA) per county is rank-correlated (Spearman $r_s$, exact\n"
    "two-sided $p$) with the FEMA-verified real property damage of owner\n"
    "households from the OpenFEMA Individual Assistance records, over the\n"
    "counties present in both sources ($n$); correlations are shown for\n"
    "$n \\geq 4$. Detection is the share of the storm's total FEMA-verified\n"
    "damage, summed over all declared counties, that falls in counties with\n"
    "modelled damage. The Huang columns repeat the comparison against the\n"
    "number of assessed damaged buildings per county. The pooled row\n"
    "correlates within-storm shares across all matched county-storm pairs.\n"
    "Hermine produced no modelled damage (detection 0\\%). Nate is not part\n"
    "of the analysis (no reconstructed wind field).}\n"
    "\\label{tab:damage_distribution}\n"
    "\\begin{tabular}{lrrrrrrr}\n\\hline\n"
    " & \\multicolumn{4}{c}{FEMA IA verified damage} & "
    "\\multicolumn{3}{c}{Huang et al.\\ building sample} \\\\\n"
    "Storm & $n$ & $r_s$ & $p$ & Detection & $n$ & $r_s$ & $p$ \\\\\n\\hline\n"
    + "\n".join(tex_rows)
    + "\n\\hline\n\\end{tabular}\n\\end{table}\n"
)
(OUT_DIR / "table_damage_distribution.tex").write_text(dist_tex)

# ---- SI: scaling robustness (wind-only vs scaled WUA against FEMA)
# Detection is identical by construction (the scaling is multiplicative on
# wind damage and cannot create damage where wind produced none), so only
# the rank correlations are tabulated.
sc_rows, sc_pool_r, sc_pool_s = [], [], []
for atcf, (name, year) in sorted(STORMS.items(), key=lambda kv: kv[1][1]):
    if atcf in ("AL092021", "AL092022"):  # wind-only runs: raw == scaled
        continue
    df = load_event(atcf)
    if df is None or fema is None:
        continue
    df = df[df["wua_scaled"] > 0]
    f = fema.loc[fema.atcf_id == atcf, ["fips", "fema_total_damage"]]
    ov = df.merge(f, on="fips", how="inner")
    if len(ov) < MIN_N:
        continue
    rr, pr = spearmanr(ov["wua_raw"], ov["fema_total_damage"])
    rs, ps = spearmanr(ov["wua_scaled"], ov["fema_total_damage"])
    sc_rows.append(dict(storm=name, year=year, n=len(ov),
                        rho_raw=rr, p_raw=pr, rho_scaled=rs, p_scaled=ps))
    sc_pool_r.append(ov.assign(sm=ov["wua_raw"] / ov["wua_raw"].sum(),
                               so=ov["fema_total_damage"] / ov["fema_total_damage"].sum()))
    sc_pool_s.append(ov.assign(sm=ov["wua_scaled"] / ov["wua_scaled"].sum(),
                               so=ov["fema_total_damage"] / ov["fema_total_damage"].sum()))
nr, rr_p, pr_p = _pooled(sc_pool_r)
ns, rs_p, ps_p = _pooled(sc_pool_s)
scal = pd.DataFrame(sc_rows)
scal = pd.concat([scal, pd.DataFrame([dict(
    storm="Pooled (within-storm shares)", year="", n=nr,
    rho_raw=rr_p, p_raw=pr_p, rho_scaled=rs_p, p_scaled=ps_p)])],
    ignore_index=True)
scal.to_csv(OUT_DIR / "table_scaling_robustness.csv", index=False)
tex_rows = []
for _, r in scal.iterrows():
    label = f"{r.storm} ({r.year})" if r.year != "" else r.storm
    tex_rows.append(
        f"{label} & {int(r.n)} & ${r.rho_raw:+.2f}$ & {_p(r.p_raw)} & "
        f"${r.rho_scaled:+.2f}$ & {_p(r.p_scaled)} \\\\"
    )
scal_tex = (
    "\\begin{table}[htb!]\n\\centering\n"
    "\\caption{Insensitivity of the damage evaluation to the rain and surge\n"
    "scaling. Spearman rank correlation of county weighted units affected\n"
    "with the FEMA-verified housing damage, computed with wind-only (raw)\n"
    "and with multi-hazard-scaled damage, for the storms with rain and\n"
    "surge covariates. The detection shares of\n"
    "table~\\ref{tab:damage_distribution} are identical for raw and scaled\n"
    "damage by construction, because the scaling is multiplicative on wind\n"
    "damage and cannot create damage in counties without wind damage.}\n"
    "\\label{tab:scaling_robustness}\n"
    "\\begin{tabular}{lrrrrr}\n\\hline\n"
    " & & \\multicolumn{2}{c}{Wind-only (raw)} & "
    "\\multicolumn{2}{c}{Scaled} \\\\\n"
    "Storm & $n$ & $r_s$ & $p$ & $r_s$ & $p$ \\\\\n\\hline\n"
    + "\n".join(tex_rows)
    + "\n\\hline\n\\end{tabular}\n\\end{table}\n"
)
(OUT_DIR / "table_scaling_robustness.tex").write_text(scal_tex)

# ---- SI: raw vs per-unit-normalised FEMA correlations
norm = pd.DataFrame(norm_rows)
norm.to_csv(OUT_DIR / "table_damage_normalized.csv", index=False)
tex_rows = []
for _, r in norm.iterrows():
    if r.n_raw == 0:
        continue
    tex_rows.append(
        f"{r.storm} ({r.year}) & {_cell(r.n_raw, r.rho_raw, r.p_raw)} & "
        f"{_cell(r.n_norm, r.rho_norm, r.p_norm)} \\\\"
    )
norm_tex = (
    "\\begin{table}[htb!]\n\\centering\n"
    "\\caption{Robustness of the FEMA damage-distribution comparison to\n"
    "county size. Left: Spearman correlation of county WUA with FEMA-verified\n"
    "housing damage (as in the main text). Right: the same correlation after\n"
    "dividing both quantities by the county's residential exposure units, so\n"
    "that damage intensities rather than totals are compared. Counties\n"
    "without exposure-unit data are dropped from the normalised comparison.}\n"
    "\\label{tab:damage_normalized}\n"
    "\\begin{tabular}{lrrrrrr}\n\\hline\n"
    " & \\multicolumn{3}{c}{County totals} & "
    "\\multicolumn{3}{c}{Per exposure unit} \\\\\n"
    "Storm & $n$ & $r_s$ & $p$ & $n$ & $r_s$ & $p$ \\\\\n\\hline\n"
    + "\n".join(tex_rows)
    + "\n\\hline\n\\end{tabular}\n\\end{table}\n"
)
(OUT_DIR / "table_damage_normalized.tex").write_text(norm_tex)

# ================================================================ Huang table
joined = huang.merge(
    recovery, left_on=["GEOID", "atcf_id"], right_on=["fips", "event_name"],
    how="inner",
)

rows = []
for thr in [0, 10, 15, 30]:
    sub = joined[joined["n_changed"] >= thr]
    a = sub.dropna(subset=["reconstruction_capacity", "rebuild_rate"])
    rho1, p1 = spearmanr(a["reconstruction_capacity"], a["rebuild_rate"])
    b = sub.dropna(subset=["recovery_potential_months", "recovery_score"])
    rho2, p2 = spearmanr(b["recovery_potential_months"], b["recovery_score"])
    rows.append(
        dict(threshold=thr, n_capacity=len(a), rho_capacity_rebuild=rho1,
             p_capacity_rebuild=p1, n_recovery=len(b),
             rho_recovery_score=rho2, p_recovery_score=p2)
    )
th = pd.DataFrame(rows)
th.to_csv(OUT_DIR / "table_huang_thresholds.csv", index=False)

tex_rows = [
    f"$\\geq {int(r.threshold)}$ & {int(r.n_capacity)} & "
    f"${r.rho_capacity_rebuild:+.2f}$ & {r.p_capacity_rebuild:.3f} & "
    f"{int(r.n_recovery)} & ${r.rho_recovery_score:+.2f}$ & {r.p_recovery_score:.3f} \\\\"
    for _, r in th.iterrows()
]
th_tex = (
    "\\begin{table}[htb!]\n\\centering\n"
    "\\caption{Sensitivity of the comparison with Huang et al.\\ (2025) to the\n"
    "minimum county sample size (number of assessed changed buildings,\n"
    "$n_{\\mathrm{changed}}$). Spearman rank correlations, pooled across the\n"
    "matched storms, with exact two-sided p-values. Left: construction capacity\n"
    "vs observed rebuild rate (proxy validation). Right: recovery potential vs\n"
    "observed recovery score (integrated-model validation). The primary filter\n"
    "used in the main text is $n_{\\mathrm{changed}} \\geq 10$.}\n"
    "\\label{tab:huang_thresholds}\n"
    "\\begin{tabular}{lrrrrrr}\n\\hline\n"
    " & \\multicolumn{3}{c}{Capacity vs rebuild rate} & "
    "\\multicolumn{3}{c}{Recovery potential vs recovery score} \\\\\n"
    "Filter & $n$ & $r_s$ & $p$ & $n$ & $r_s$ & $p$ \\\\\n\\hline\n"
    + "\n".join(tex_rows)
    + "\n\\hline\n\\end{tabular}\n\\end{table}\n"
)
(OUT_DIR / "table_huang_thresholds.tex").write_text(th_tex)

print(f"Tables written to {OUT_DIR}")
print(dist.to_string(index=False))
print(th.to_string(index=False))
