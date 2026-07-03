"""FEMA-primary damage-distribution evaluation, all storms with model damage.

FEMA leg = model WUA vs FEMA-verified owner real-property damage
(HousingAssistanceOwners), over the model-footprint counties per storm.
Huang leg = secondary. Laura/Michael FEMA data from the earlier verified
pulls (all IA counties); other storms pulled for the model footprint.
"""
import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from dbfread import DBF

R = "/sessions/upbeat-bold-noether/mnt/hurricane_recovery_potential"
OUT = "/sessions/upbeat-bold-noether/mnt/outputs"
TAU = {1: 1, 2: 1, 3: 3, 4: 6}

STORMS = {
    "AL122005": ("Katrina", 2005),
    "AL092012": ("Isaac", 2012),
    "AL092017": ("Harvey", 2017),
    "AL112017": ("Irma", 2017),
    "AL142018": ("Michael", 2018),
    "AL132020": ("Laura", 2020),
    "AL092021": ("Ida", 2021),
    "AL092022": ("Ian", 2022),
}

# ---- model WUA per county per storm
def load_model(atcf):
    fs = glob.glob(f"{R}/data/impact_historical/per_event/aggregated_*_{atcf}.csv")
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["fips"] = df["fips"].astype(str).str.split(".").str[0].str.zfill(5)
    df["wua"] = sum(df[f"units_DS{k}_scaled"] * TAU[k] for k in TAU)
    return df[df["wua"] > 0][["fips", "wua"]]

# ---- county name -> fips (state-specific)
n2f = {}
for r in DBF(f"{R}/data/US_counties.dbf", encoding="utf-8"):
    n2f[(str(r["STATEFP"]), str(r["NAME"]).strip())] = str(r["GEOID"])

def agg(path, statefp, only=None, exclude=None):
    d = pd.read_csv(path)
    if only:
        d = d[d.county.isin(only)]
    if exclude:
        d = d[~d.county.isin(exclude)]
    a = d.groupby("county", as_index=False)["totalDamage"].sum()
    a["fips"] = a["county"].map(lambda n: n2f.get((statefp, n)))
    assert a["fips"].notna().all(), f"unmapped in {path}: {a[a.fips.isna()].county.tolist()}"
    return a[["fips", "county", "totalDamage"]]

# assemble FEMA county totals per storm (model-footprint pulls; Laura/Michael
# from the verified full-IA pulls already in data/fema_ia)
fema_parts = {
    "AL122005": pd.concat([
        agg(f"{OUT}/fema_1603_stt.csv", "22"),
        agg(f"{OUT}/fema_1603_batch2.csv", "22"),
        agg(f"{OUT}/fema_1603_batch3.csv", "22"),
        agg(f"{OUT}/fema_katrina_msal.csv", "28", exclude=["Mobile"]),
        agg(f"{OUT}/fema_katrina_msal.csv", "01", only=["Mobile"]),
    ]),
    "AL092017": agg(f"{OUT}/fema_4332_harvey.csv", "48"),
    "AL112017": pd.concat([agg(f"{OUT}/fema_4337_irma.csv", "12"),
                           agg(f"{OUT}/fema_4337_irma_mdc.csv", "12")]),
    "AL092021": agg(f"{OUT}/fema_4611_ida.csv", "22"),
    "AL092022": pd.concat([
        agg(f"{OUT}/fema_4673_ian.csv", "12", exclude=["Charleston"]),
        agg(f"{OUT}/fema_4673_ian.csv", "45", only=["Charleston"]),
    ]),
}
seed = pd.read_csv(f"{R}/data/fema_ia/fema_damage_by_county.csv", dtype={"fips": str})
seed["fips"] = seed["fips"].str.zfill(5)
for atcf in ["AL132020", "AL142018"]:
    s = seed[seed.atcf_id == atcf]
    fema_parts[atcf] = s.rename(columns={"fema_total_damage": "totalDamage"})[
        ["fips", "county", "totalDamage"]]

huang = pd.read_csv(f"{R}/data/huang_recovery_by_county_event.csv")
huang["GEOID"] = huang["GEOID"].astype(str).str.zfill(5)

rows, pooled_f, pooled_h = [], [], []
detail = {}
for atcf, (name, year) in sorted(STORMS.items(), key=lambda kv: kv[1][1]):
    model = load_model(atcf)
    n_model = len(model)

    # FEMA leg (primary)
    n_f, rho_f, p_f = 0, np.nan, np.nan
    if atcf in fema_parts:
        f = fema_parts[atcf].groupby("fips", as_index=False)["totalDamage"].sum()
        ov = model.merge(f, on="fips", how="inner")
        n_f = len(ov)
        if n_f >= 4:
            rho_f, p_f = spearmanr(ov["wua"], ov["totalDamage"])
        if n_f >= 2:
            pooled_f.append(ov.assign(sm=ov.wua / ov.wua.sum(),
                                      so=ov.totalDamage / ov.totalDamage.sum()))
        fc = fema_parts[atcf].groupby("fips", as_index=False).agg(
            county=("county", "first"), totalDamage=("totalDamage", "sum"))
        detail[name] = model.merge(fc, on="fips", how="outer", indicator=True)

    # Huang leg (secondary)
    h = huang.loc[huang.atcf_id == atcf, ["GEOID", "n_changed"]].rename(
        columns={"GEOID": "fips"})
    ovh = model.merge(h, on="fips", how="inner")
    n_h, rho_h, p_h = len(ovh), np.nan, np.nan
    if n_h >= 4:
        rho_h, p_h = spearmanr(ovh["wua"], ovh["n_changed"])
    if n_h >= 2:
        pooled_h.append(ovh.assign(sm=ovh.wua / ovh.wua.sum(),
                                   so=ovh.n_changed / ovh.n_changed.sum()))

    rows.append(dict(storm=name, year=year, n_model=n_model,
                     n_fema=n_f, rho_fema=rho_f, p_fema=p_f,
                     n_huang=n_h, rho_huang=rho_h, p_huang=p_h))

res = pd.DataFrame(rows)

def pooled(ps):
    d = pd.concat(ps, ignore_index=True)
    rho, p = spearmanr(d.sm, d.so)
    return len(d), rho, p

nf, rf, pf = pooled(pooled_f)
nh, rh, ph = pooled(pooled_h)
print(res.to_string(index=False))
print(f"\nPOOLED within-storm shares: FEMA n={nf} rho={rf:+.3f} p={pf:.2e} | "
      f"Huang n={nh} rho={rh:+.3f} p={ph:.4f}")
res.to_csv(f"{OUT}/fema_primary_results.csv", index=False)

# per-storm detail for the wind-only and counter cases
for name in ["Ida", "Ian", "Katrina", "Harvey"]:
    d = detail[name].sort_values("wua", ascending=False)
    d["fema_share"] = d.totalDamage / d.totalDamage.sum()
    d["model_share"] = d.wua / d.wua.sum()
    print(f"\n=== {name} (model vs FEMA) ===")
    print(d[["fips", "county", "wua", "totalDamage", "model_share", "fema_share"]]
          .to_string(index=False))
