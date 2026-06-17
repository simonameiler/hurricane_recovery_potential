#!/usr/bin/env python3
"""
compare_surge_on_off.py
-----------------------
Compare damage-state (DS) unit counts under the committed surge-OFF scaling vs a
surge-ON scaling, WITHOUT re-running CLIMADA. Starts from the per-building
wind-only loss (rel_wind), re-applies each scaling, re-discretizes to DS, and
re-aggregates to county-event. Mirrors modules/impact_utils.py exactly.

Pipeline reproduced (impact_utils.py):
  - lower_threshold = 0.005 applied to wind loss BEFORE scaling; zeros stay zero
  - compound scaling, k = 1:  D = 1 - (1 - loss)^Scaling
  - Hazus DS cutoffs (compute_damage_state): DS0<0.02, DS1<0.05, DS2<0.10, DS3<0.50, DS4>=0.50
  - DS unit counts weighted by NumberOfUnits

OUTPUT columns (per event_name, fips), matching your per_event schema plus _on:
  units_DS{1..4}_raw     wind-only        (Scaling = 1)
  units_DS{1..4}_off     committed        (wind+rain  scaling)
  units_DS{1..4}_on      surge-on         (wind+rain+surge scaling)

VALIDATION: if a per_event reference dir is given, the script checks that its
_off counts reproduce your committed units_DS*_scaled (they must, to trust _on).
"""
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------------------- paths (edit)
IMPACTS_WIND  = Path("data/impacts_wind")          # event_id=XXXX/part-*.parquet  [id, rel_wind]
EXPOSURE_HDF5 = Path("data/exposure/NA_coast_exposure.hdf5")   # id must match IMPACTS_WIND
COUNTY_REGION = Path("data/county_region.csv")     # county_index, fips, ...
SCALING_OFF   = Path("data/scaling_relative.npz")              # committed (surge off)
SCALING_ON    = Path("data/scaling_relative_SURGE_ON.npz")
PER_EVENT_REF = Path("")                            # optional: dir of committed per_event CSVs for validation
OUT_CSV       = Path("data/ds_compare_surge_on_off.csv")

LOWER_THRESHOLD = 0.005
DS_EDGES = [0.02, 0.05, 0.10, 0.50]                 # -> DS0..DS4

# --------------------------------------------------------------- exposure map
exp = pd.read_hdf(EXPOSURE_HDF5)
if "fips" not in exp.columns:
    exp["fips"] = exp["stcode"].astype(int) * 1000 + exp["ccode"].astype(int)
exp["fips"] = exp["fips"].astype(int)
cr = pd.read_csv(COUNTY_REGION)
fips_to_col = dict(zip(cr["fips"].astype(int), cr["county_index"].astype(int)))
exp["cidx"] = exp["fips"].map(fips_to_col)
id_map = exp.set_index("id")[["fips", "cidx", "NumberOfUnits"]]

S_off = np.load(SCALING_OFF)["Scaling"]
S_on  = np.load(SCALING_ON)["Scaling"]

def scaled(loss, scaling):                          # zeros stay zero
    loss = np.where(loss >= LOWER_THRESHOLD, loss, 0.0)
    out = 1.0 - np.power(1.0 - np.clip(loss, 0.0, 1.0), scaling)
    return np.where(loss >= LOWER_THRESHOLD, out, 0.0)

def ds_unit_counts(df, tag):
    df = df.assign(ds=np.digitize(df["d"].to_numpy(), DS_EDGES))
    out = {}
    for k in (1, 2, 3, 4):
        out[f"units_DS{k}_{tag}"] = df.loc[df.ds == k].groupby("fips")["NumberOfUnits"].sum()
    return pd.DataFrame(out)

rows = []
for ed in sorted(IMPACTS_WIND.glob("event_id=*")):
    e = int(str(ed.name).split("=")[1]); erow = e - 1
    if not (0 <= erow < S_off.shape[0]):
        continue
    df = pd.concat([pd.read_parquet(p) for p in ed.glob("*.parquet")], ignore_index=True)
    df = df.join(id_map, on="id").dropna(subset=["cidx"])
    df["cidx"] = df["cidx"].astype(int)
    loss = df["rel_wind"].to_numpy(float)
    res_e = None
    for tag, S in (("raw", None), ("off", S_off), ("on", S_on)):
        sc = np.ones(len(df)) if S is None else S[erow, df["cidx"].to_numpy()]
        part = ds_unit_counts(df.assign(d=scaled(loss, sc)), tag)
        res_e = part if res_e is None else res_e.join(part, how="outer")
    res_e = res_e.fillna(0.0); res_e["event_name"] = e
    rows.append(res_e.reset_index())

res = pd.concat(rows, ignore_index=True).fillna(0.0)
res.to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV}  ({len(res)} county-event rows)")

# --------------------------------------------------------------- summary: off vs on
for k in (2, 3, 4):
    off = res[f"units_DS{k}_off"].sum(); on = res[f"units_DS{k}_on"].sum()
    print(f"DS{k}: off={off:,.0f}  on={on:,.0f}  change={100*(on-off)/off if off else 0:+.3f}%")
ch = res[(res.filter(like="_on").values != res.filter(like="_off").values).any(axis=1)]
print(f"county-event pairs changed by surge: {len(ch)} / {len(res)} "
      f"({100*len(ch)/len(res):.3f}%); counties affected: {ch['fips'].nunique() if len(ch) else 0}")

# --------------------------------------------------------------- optional validation vs committed per_event
if str(PER_EVENT_REF):
    ref = pd.concat([pd.read_csv(f) for f in Path(PER_EVENT_REF).glob("*.csv")], ignore_index=True)
    m = res.merge(ref, on=["event_name", "fips"], suffixes=("", "_ref"))
    ok = all(np.allclose(m[f"units_DS{k}_off"], m[f"units_DS{k}_scaled"], atol=1e-6) for k in (1, 2, 3, 4))
    print("VALIDATION _off reproduces committed units_DS*_scaled:", ok)
