#!/usr/bin/env python3
"""
diff_per_event_surge.py
-----------------------
Compare county-aggregated DS unit counts between the surge-OFF and surge-ON runs,
using the per_event output schema:
    event_index,event_name,stcode,ccode,fips,
    units_DS1_raw,units_DS1_scaled, ... units_DS4_raw,units_DS4_scaled,
    repair_cost_sum_raw,repair_cost_sum_scaled

Both runs must come from the SAME impact computation (one calc_state_impact pass
with --scaling-npz + --scaling-npz-on), so the only difference is the scaling.

SANITY: turning surge on can only move units to HIGHER damage states (and may pull
a few from DS0 into DS1 as they cross 2%). So on the common key set, total DS1-4
and DS4 specifically can only INCREASE or stay equal -- never decrease, and DS4
can never lose units. A drop is impossible and signals mismatched coverage (e.g.
an incomplete run) rather than a surge effect; the script flags it.
"""
from pathlib import Path
import numpy as np
import pandas as pd

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
    return df[KEYS + DS].copy(), len(files)

off, nfo = load(PER_EVENT_OFF)
on,  nfn = load(PER_EVENT_ON)

# ---- coverage report ----
print("=== coverage ===")
for name, df, nf in (("OFF", off, nfo), ("ON", on, nfn)):
    print(f"{name}: files={nf}  rows={len(df)}  events={df['event_name'].nunique()}  "
          f"counties={df['fips'].nunique()}  totalDS1-4={df[DS].to_numpy().sum():,.0f}")

ko = set(map(tuple, off[KEYS].values)); kn = set(map(tuple, on[KEYS].values))
common = ko & kn
print(f"keys: off={len(ko)}  on={len(kn)}  common={len(common)}  "
      f"off-only={len(ko - kn)}  on-only={len(kn - ko)}")
if ko != kn:
    print("!! COVERAGE MISMATCH: the two runs do not cover the same county-event set.")
    print("   Comparison restricted to the COMMON keys below; investigate the missing rows")
    print("   (incomplete array? failed tasks? off/on dirs swapped?).")

# ---- restrict to common keys (apples to apples) ----
off_c = off.rename(columns={c: c + "_off" for c in DS})
on_c  = on.rename(columns={c: c + "_on"  for c in DS})
m = off_c.merge(on_c, on=KEYS, how="inner")
for k in (1, 2, 3, 4):
    m[f"dDS{k}"] = m[f"units_DS{k}_scaled_on"] - m[f"units_DS{k}_scaled_off"]
m.to_csv(OUT_CSV, index=False)
print(f"\nwrote {OUT_CSV}  ({len(m)} common county-event rows)")

# ---- sanity check: on >= off expected (surge only adds/promotes damage) ----
tot_off = sum(m[f"units_DS{k}_scaled_off"].sum() for k in (1, 2, 3, 4))
tot_on  = sum(m[f"units_DS{k}_scaled_on"].sum()  for k in (1, 2, 3, 4))
ds4_off = m["units_DS4_scaled_off"].sum(); ds4_on = m["units_DS4_scaled_on"].sum()
print(f"\n=== sanity (common keys; on must be >= off) ===")
print(f"total DS1-4: off={tot_off:,.0f}  on={tot_on:,.0f}  ({100*(tot_on-tot_off)/tot_off if tot_off else 0:+.3f}%)")
print(f"DS4 only:    off={ds4_off:,.0f}  on={ds4_on:,.0f}  ({100*(ds4_on-ds4_off)/ds4_off if ds4_off else 0:+.3f}%)")
if tot_on + 1e-6 < tot_off or ds4_on + 1e-6 < ds4_off:
    print("!! IMPOSSIBLE: surge reduced damaged units / DS4. Surge can only add or promote.")
    print("   This is a coverage or pairing problem (incomplete run, swapped dirs, or")
    print("   off/on built from different impact computations), NOT a surge effect.")

# ---- expected migration: DS1 down, DS4 up ----
print("\n=== DS migration (surge off -> on, common keys) ===")
for k in (1, 2, 3, 4):
    o = m[f"units_DS{k}_scaled_off"].sum(); n = m[f"units_DS{k}_scaled_on"].sum()
    print(f"DS{k}: off={o:,.0f}  on={n:,.0f}  change={100*(n-o)/o if o else 0:+.3f}%")

changed = m[(m[[f"dDS{k}" for k in (1, 2, 3, 4)]].abs() > 1e-9).any(axis=1)]
print(f"\ncounty-event pairs changed: {len(changed)} / {len(m)} "
      f"({100*len(changed)/len(m) if len(m) else 0:.3f}%); "
      f"counties affected: {changed['fips'].nunique() if len(changed) else 0}")
if len(changed):
    top = (changed.assign(absshift=changed[[f'dDS{k}' for k in (1, 2, 3, 4)]].abs().sum(axis=1))
                  .groupby('fips')['absshift'].sum().sort_values(ascending=False).head(15))
    print("\ntop 15 counties by DS1-4 unit shift:")
    print(top.to_string())
