"""Fetch county-level FEMA-verified housing damage for the historical storms.

Pulls the OpenFEMA HousingAssistanceOwners dataset (FEMA-inspected real
property damage of owner-occupied homes, aggregated by zip within disaster
declarations), sums it to county FIPS per storm, and writes
data/fema_ia/fema_damage_by_county.csv with columns
(atcf_id, storm, disaster_numbers, fips, county, state, fema_total_damage,
valid_registrations).

Disaster numbers are discovered from DisasterDeclarationsSummaries by
declaration title and year, keeping only declarations with the Individual
Assistance (IH) program, so no disaster number is hard-coded.

Used by create_historical_tables.py for the damage-distribution comparison
(the FEMA leg is skipped there if this file has not been generated).

Run from the repo root or scripts/. Requires: requests, pandas, dbfread or
geopandas. Sanity checks (against in-session API pulls, 2026-07-01):
Laura/DR-4559 Calcasieu 114,184,167.87; Michael/DR-4399 Bay 118,798,365.85;
Michael/DR-4400 Seminole GA 2,635,778.59.
"""

from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "fema_ia"
OUT_DIR.mkdir(parents=True, exist_ok=True)
COUNTIES_DBF = REPO / "data" / "US_counties.dbf"

BASE = "https://www.fema.gov/api/open"

STORMS = {  # atcf -> (name, landfall year, declaration title)
    "AL122005": ("Katrina", 2005, "HURRICANE KATRINA"),
    "AL092012": ("Isaac", 2012, "HURRICANE ISAAC"),
    "AL092016": ("Hermine", 2016, "HURRICANE HERMINE"),
    "AL092017": ("Harvey", 2017, "HURRICANE HARVEY"),
    "AL112017": ("Irma", 2017, "HURRICANE IRMA"),
    "AL162017": ("Nate", 2017, "HURRICANE NATE"),
    "AL142018": ("Michael", 2018, "HURRICANE MICHAEL"),
    "AL132020": ("Laura", 2020, "HURRICANE LAURA"),
    "AL092021": ("Ida", 2021, "HURRICANE IDA"),
    "AL092022": ("Ian", 2022, "HURRICANE IAN"),
}

STATE_FP = {
    "AL": "01", "FL": "12", "GA": "13", "LA": "22", "MS": "28",
    "NC": "37", "SC": "45", "TX": "48", "VA": "51",
    # Ida remnants produced IA declarations in the Northeast; keep them so
    # coverage is explicit even though the model domain excludes them.
    "NY": "36", "NJ": "34", "PA": "42", "CT": "09",
}

# county-name aliases: OpenFEMA name -> counties-shapefile name
NAME_ALIASES = {
    ("22", "La Salle"): "LaSalle",
}


def get_all(endpoint: str, filt: str, select: str) -> pd.DataFrame:
    """Paged OpenFEMA download (JSON, 10k records per page)."""
    frames, skip = [], 0
    while True:
        r = requests.get(
            f"{BASE}/{endpoint}",
            params={"$filter": filt, "$select": select,
                    "$top": 10000, "$skip": skip},
            timeout=120,
        )
        r.raise_for_status()
        records = r.json()[endpoint.split("/")[-1]]
        if not records:
            break
        frames.append(pd.DataFrame(records))
        if len(records) < 10000:
            break
        skip += 10000
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def county_name_to_fips() -> dict:
    try:
        from dbfread import DBF

        return {
            (str(r["STATEFP"]), str(r["NAME"]).strip()): str(r["GEOID"])
            for r in DBF(COUNTIES_DBF, encoding="utf-8")
        }
    except ImportError:
        import geopandas as gpd

        gdf = gpd.read_file(COUNTIES_DBF.with_suffix(".shp"))
        return {
            (str(s), str(n).strip()): str(g)
            for s, n, g in zip(gdf["STATEFP"], gdf["NAME"], gdf["GEOID"])
        }


def find_disasters(title: str, year: int) -> pd.DataFrame:
    """IH-declared disaster numbers and states for one storm."""
    decl = get_all(
        "v2/DisasterDeclarationsSummaries",
        f"declarationTitle eq '{title}' and fyDeclared ge {year} "
        f"and fyDeclared le {year + 1}",
        "disasterNumber,state,ihProgramDeclared",
    )
    if decl.empty:
        return decl
    ih = decl[decl["ihProgramDeclared"].astype(int) == 1]
    return ih[["disasterNumber", "state"]].drop_duplicates()


def main() -> None:
    n2f = county_name_to_fips()
    rows = []
    for atcf, (name, year, title) in STORMS.items():
        decl = find_disasters(title, year)
        if decl.empty:
            print(f"{name}: no IH-declared disaster found — skipping")
            continue
        print(f"{name}: disasters "
              f"{sorted(decl.disasterNumber.unique().tolist())} "
              f"states {sorted(decl.state.unique().tolist())}")
        for _, d in decl.iterrows():
            dn, st = int(d.disasterNumber), str(d.state)
            statefp = STATE_FP.get(st)
            if statefp is None:
                print(f"  {name} DR-{dn} ({st}): state outside lookup — skipped")
                continue
            df = get_all(
                "v2/HousingAssistanceOwners",
                f"disasterNumber eq {dn}",
                "county,totalDamage,validRegistrations,"
                "femaInspectedDamageBetween1And10000,"
                "femaInspectedDamageBetween10001And20000,"
                "femaInspectedDamageBetween20001And30000,"
                "femaInspectedDamageGreaterThan30000",
            )
            if df.empty:
                print(f"  DR-{dn} ({st}): no HousingAssistanceOwners rows")
                continue
            df["name"] = (
                df["county"]
                .str.replace(r"\s*\((County|Parish)\)", "", regex=True)
                .str.strip()
            )
            agg = df.groupby("name", as_index=False).agg(
                fema_total_damage=("totalDamage", "sum"),
                valid_registrations=("validRegistrations", "sum"),
                band_1_10k=("femaInspectedDamageBetween1And10000", "sum"),
                band_10_20k=("femaInspectedDamageBetween10001And20000", "sum"),
                band_20_30k=("femaInspectedDamageBetween20001And30000", "sum"),
                band_gt30k=("femaInspectedDamageGreaterThan30000", "sum"),
            )
            for _, a in agg.iterrows():
                nm = NAME_ALIASES.get((statefp, a["name"]), a["name"])
                fips = n2f.get((statefp, nm))
                if fips is None:
                    print(f"  DR-{dn}: unmapped county '{a['name']}' ({st})")
                    continue
                rows.append(
                    dict(atcf_id=atcf, storm=name, disaster_number=dn,
                         fips=fips, county=nm, state=st,
                         fema_total_damage=round(a["fema_total_damage"], 2),
                         valid_registrations=int(a["valid_registrations"]),
                         band_1_10k=int(a["band_1_10k"]),
                         band_10_20k=int(a["band_10_20k"]),
                         band_20_30k=int(a["band_20_30k"]),
                         band_gt30k=int(a["band_gt30k"]))
                )

    out = pd.DataFrame(rows)
    # a county can appear under two declarations of one storm; keep the sum
    out = out.groupby(
        ["atcf_id", "storm", "fips", "county", "state"], as_index=False
    ).agg(
        disaster_numbers=("disaster_number",
                          lambda s: "+".join(str(x) for x in sorted(set(s)))),
        fema_total_damage=("fema_total_damage", "sum"),
        valid_registrations=("valid_registrations", "sum"),
        band_1_10k=("band_1_10k", "sum"),
        band_10_20k=("band_10_20k", "sum"),
        band_20_30k=("band_20_30k", "sum"),
        band_gt30k=("band_gt30k", "sum"),
    )
    out_path = OUT_DIR / "fema_damage_by_county.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} county-storm rows -> {out_path}")

    # sanity checks against values pulled interactively on 2026-07-01
    checks = [
        ("AL132020", "22019", 114184167.87),
        ("AL142018", "12005", 118798365.85),
        ("AL142018", "13253", 2635778.59),
    ]
    for atcf, fips, expected in checks:
        got = out.loc[(out.atcf_id == atcf) & (out.fips == fips),
                      "fema_total_damage"]
        status = "OK" if (len(got) and abs(got.iloc[0] - expected) < 1.0) \
            else f"MISMATCH (got {got.tolist()})"
        print(f"  check {atcf}/{fips}: expected {expected:,.2f} -> {status}")


if __name__ == "__main__":
    main()
