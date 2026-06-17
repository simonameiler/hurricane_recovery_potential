#!/usr/bin/env python3
"""
diff_per_event_surge.py
-----------------------
Compare county-aggregated DS unit counts between the surge-OFF and surge-ON runs,
using the per_event output schema:
    event_index,event_name,stcode,ccode,fips,
    units_DS1_raw,units_DS1_scaled, ... units_DS4_raw,units_DS4_scaled,
    repair_cost_sum_raw,repair_cost_sum_scaled

Both runs come from the SAME impact computation (one calc_state_impact pass with
--scaling-npz + --scaling-npz-on), so the only difference is the scaling.
Compares the *_scaled columns (surge-off vs surge-on damage).
"""
from pathlib import Path
import numpy as np
import pandas as pd

# surge-OFF and surge-ON per_event dirs (from the dual-export run)
PER_EVENT_OFF = Path("/home/groups/bakerjw/smeiler/climada_data/data/results/hrp_impacts_out_recheck/per_event")
PER_EVENT_ON  = Path("/home/groups/bakerjw/smeiler/climada_data/data/results/hrp_impacts_out_SURGE_ON/per_event")
OUT_CSV       = Path("/home/users/smeiler/repos/hurricane_recovery_potential/data/ds_surge_diff_by_county_event.csv")

KEYS = ["event_name", "fips"]
DS = [f"units_DS{k}_scaled" for k in (1, 2, 3, 4)]

def load(d):
    files = list(Path(d).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No per_event CSVs found in {d}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df[KEYS + DS].copy()

off = load(PER_EVENT_OFF).rename(columns={c: c + "_off" for c in DS})
on  = load(PER_EVENT_ON ).rename(columns={c: c + "_on"  for c in DS})
m = off.merge(on, on=KEYS, how="outer").fillna(0.0)

for k in (1, 2, 3, 4):
    m[f"dDS{k}"] = m[f"units_DS{k}_scaled_on"] - m[f"units_DS{k}_scaled_off"]
m.to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV}  ({len(m)} county-event rows)")

print("\n--- totals (surge off -> on) ---")
for k in (2, 3, 4):                       # DS2-4 drive recovery demand
    o = m[f"units_DS{k}_scaled_off"].sum(); n = m[f"units_DS{k}_scaled_on"].sum()
    print(f"DS{k}: off={o:,.0f}  on={n:,.0f}  change={100*(n-o)/o if o else 0:+.3f}%")

changed = m[(m[[f"dDS{k}" for k in (1, 2, 3, 4)]].abs() > 1e-9).any(axis=1)]
print(f"\ncounty-event pairs changed: {len(changed)} / {len(m)} ({100*len(changed)/len(m):.3f}%)")
print(f"counties affected: {changed['fips'].nunique() if len(changed) else 0}")
if len(changed):
    top = (changed.assign(absshift=changed[[f'dDS{k}' for k in (2, 3, 4)]].abs().sum(axis=1))
                  .groupby('fips')['absshift'].sum().sort_values(ascending=False).head(15))
    print("\ntop 15 counties by DS2-4 unit shift:")
    print(top.to_string())
