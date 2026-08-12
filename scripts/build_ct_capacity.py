#!/usr/bin/env python3
"""
ct_capacity_crosswalk.py
------------------------
Assign construction capacity to Connecticut's 8 legacy counties.

The Building Permits Survey reports Connecticut on the nine planning regions
(FIPS 09110-09190) that replaced counties as county equivalents, while the
exposure and hazard in this study use the legacy county FIPS (09001-09015).
Nothing joins, so all 8 Connecticut counties are dropped by the capacity filter.

Method
------
Planning regions and legacy counties are both unions of Connecticut's towns, so
the two geographies overlap without nesting. This script derives the overlap
from the study's own exposure:

  1. read the 879 Connecticut census tracts from the NRI 2025 geodatabase, whose
     county field already carries the planning-region FIPS;
  2. locate each of the 1,040,159 Connecticut exposure buildings in a tract, and
     so in a planning region, while keeping its legacy county code;
  3. cross-tabulate housing units by legacy county and planning region;
  4. apportion each region's permits to counties in proportion to housing units.

  C_county = sum_over_regions ( units[county, region] / units[region] ) * C_region

ASSUMPTION, stated plainly: permit activity within a planning region is taken to
be distributed in proportion to the housing stock. That is an assumption, not a
measurement. It can be removed by using the BPS place-level (municipal) file and
aggregating towns directly to legacy counties, which needs no weighting at all.

Verification built in: the apportioned county capacities must sum to the sum of
the nine regional values, and each county's units must sum to its exposure total.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely import from_wkb, STRtree, points as mkpoints

R_EARTH = 6378137.0          # NRI geodatabase is EPSG:3857


def ct_tracts(gdb: str) -> tuple[pd.DataFrame, np.ndarray]:
    meta, _idx, geom, fields = pyogrio.raw.read(
        gdb, layer="NRI_CensusTracts",
        columns=["STCOFIPS", "TRACTFIPS", "COUNTY", "STATEFIPS"],
        where="STATEFIPS = '09'", read_geometry=True)
    T = pd.DataFrame({n: a for n, a in zip(list(meta["fields"]), fields)})
    return T, from_wkb(geom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    T, polys = ct_tracts(os.path.join(a.root, "NRI_GDB_CensusTracts/NRI_GDB_CensusTracts.gdb"))
    print(f"Connecticut tracts: {len(T)} in {T.STCOFIPS.nunique()} planning regions")

    exp = pd.read_hdf(os.path.join(a.root, "repro_data/data/exposure/states/connecticut_exposure.hdf5"),
                      "exposures")
    pts = from_wkb(exp.geometry.values)
    x = R_EARTH * np.radians(shapely.get_x(pts))
    y = R_EARTH * np.log(np.tan(np.pi / 4 + np.radians(shapely.get_y(pts)) / 2))
    hit = STRtree(polys).query(mkpoints(x, y), predicate="within")
    gi = (pd.DataFrame({"pi": hit[0], "gi": hit[1]})
            .drop_duplicates("pi").set_index("pi").gi.reindex(range(len(exp))))
    n_unmatched = int(gi.isna().sum())
    print(f"buildings located in a tract: {len(exp) - n_unmatched:,} of {len(exp):,} "
          f"({100*n_unmatched/len(exp):.3f} % unmatched)")

    m = pd.DataFrame({"fips": "09" + exp.ccode.astype(str).str.zfill(3),
                      "units": exp.NumberOfUnits.values,
                      "region": np.where(gi.notna(), T.STCOFIPS.values[gi.fillna(0).astype(int)], None)})
    X = (m.dropna(subset=["region"])
           .pivot_table(index="fips", columns="region", values="units", aggfunc="sum", fill_value=0.0))

    p = pd.read_csv(os.path.join(a.root, "repro_data/data/selected_states_counties_with_permits.csv"), dtype=str)
    col = [c for c in p.columns if c.startswith("Average_Building_Permits")][0]
    p["cap"] = pd.to_numeric(p[col], errors="coerce")
    p["fips"] = p.FIPS.str.zfill(5)
    cap_reg = p.set_index("fips").cap.reindex(X.columns)
    if cap_reg.isna().any():
        raise SystemExit(f"missing regional capacity for {list(cap_reg.index[cap_reg.isna()])}")

    share = X.div(X.sum(axis=0), axis=1)                    # county share of each region's units
    cap_county = (share * cap_reg).sum(axis=1).rename("construction_capacity")

    out = pd.DataFrame({"fips": cap_county.index, "construction_capacity": cap_county.values,
                        "exposure_units": X.sum(axis=1).values,
                        "n_regions_overlapped": (X > 0).sum(axis=1).values})
    X.to_csv(os.path.join(a.out, "ct_units_by_county_and_region.csv"))

    # Augmented permits table, for consumers that read the permit CSV directly
    # (notebooks/probabilistic_analysis.ipynb). The nine Connecticut planning
    # region rows are replaced by the eight legacy counties so that every
    # consumer sees one consistent county geography. The raw download is left
    # untouched.
    CT_NAMES = {"09001": "Fairfield", "09003": "Hartford", "09005": "Litchfield",
                "09007": "Middlesex", "09009": "New Haven", "09011": "New London",
                "09013": "Tolland", "09015": "Windham"}
    aug = p[~(p.fips.str.startswith("09"))].copy()
    rows = []
    for f, c in cap_county.items():
        rows.append({"STATE_NAME": "Connecticut", "STATEFP": "9", "COUNTYFP": f[2:],
                     "FIPS": f.lstrip("0"), "NAME": CT_NAMES.get(f, f),
                     "1_Unit": "", "2_Units": "", "3_and_4_Units": "",
                     "5_Units_or_More": "", col: f"{c:.9g}", "fips": f})
    aug = pd.concat([aug, pd.DataFrame(rows)], ignore_index=True)
    aug = aug.sort_values("fips").drop(columns=["cap", "fips"])
    aug_path = os.path.join(a.out, "selected_states_counties_with_permits_ctfix.csv")
    aug.to_csv(aug_path, index=False)
    print(f"\nwrote {aug_path}: {len(aug)} rows "
          f"({len(p)} original, 9 planning regions replaced by 8 counties)")

    # ---------------- NRI county scores for the legacy counties
    # The NRI 2025 release also reports Connecticut on the planning regions, so the
    # eight legacy counties have no Social Vulnerability or Community Resilience
    # score. Unlike capacity these are 0-100 index scores, so they are averaged
    # rather than apportioned, weighted by each county's distribution of housing
    # units across the regions it overlaps. Using the region county-scores rather
    # than tract scores keeps Connecticut on the same scale as every other county;
    # the NRI tract scores are a monotone rescaling of the county scores.
    nri = pd.read_csv(REPO / "data" / "NRI_Table_Counties.csv", low_memory=False)
    nri["fips"] = nri.STCOFIPS.astype(str).str.zfill(5)
    reg = nri.set_index("fips")[["RESL_SCORE", "SOVI_SCORE"]].astype(float).reindex(X.columns)
    w_row = X.div(X.sum(axis=1), axis=0)          # county's share across regions
    scores = pd.DataFrame({c: w_row.mul(reg[c], axis=1).sum(axis=1) for c in reg.columns})
    scores.index.name = "fips"
    scores.reset_index().to_csv(os.path.join(a.out, "ct_county_nri_scores.csv"), index=False)

    # ---------------- verification
    print("\nverification")
    for c in reg.columns:
        inside = scores[c].between(reg[c].min(), reg[c].max()).all()
        state_c = np.average(scores[c], weights=X.sum(axis=1))
        state_r = np.average(reg[c], weights=X.sum(axis=0))
        print(f"  {c}: derived values inside region range: {inside} | "
              f"unit-weighted state mean counties {state_c:.4f} vs regions {state_r:.4f} "
              f"(diff {state_c-state_r:.1e})")
    print(f"  sum of apportioned county capacity : {cap_county.sum():.4f}")
    print(f"  sum of the nine regional values    : {cap_reg.sum():.4f}")
    print(f"  difference                         : {cap_county.sum()-cap_reg.sum():.2e}")
    ref = pd.read_csv(os.path.join(a.root, "repro_data/analysis_output/county_exposed_housing_units.csv"),
                      dtype={"FIPS": str})
    ref = ref.set_index("FIPS").exposed_units
    d = (X.sum(axis=1) - ref.reindex(X.index)).abs()
    print(f"  max |units difference| vs exposure table: {d.max():.1f} "
          f"(expected small, from the {n_unmatched} unlocated buildings)")

    print("\nConnecticut county capacity, permitted units per month")
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    main()
