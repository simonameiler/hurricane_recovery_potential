"""
pyrecodes light — simplified per-county hurricane housing-recovery simulator.

This is a lightweight ("pyrecodes light") stand-in for a full pyrecodes
discrete-event, resource-constrained recovery simulation.  Instead of
simulating recovery component-by-component, it estimates each county's
housing recovery time analytically as the total housing repair demand
(housing-unit-months) divided by that county's monthly residential
construction capacity (derived from building-permit data), floored by the
longest single damage-state repair time.

It is the PRODUCER of the per-event recovery JSONs that the rest of the
pipeline consumes.  It slots in between Stage 4 (impacts) and Stage 6
(EARP):

    impacts_out/by_event/scaled/{event}_scaled.csv          (Stage 4 output)
          │   + data/selected_states_counties_with_permits.csv
          ▼
    scripts/run_pyrecodes_light.py                          (THIS SCRIPT)
          ▼
    data/recovery_potential_per_scenario/
            {event}_scaled_recovery_potential.json
          ▼
    scripts/compute_recovery_potential.py                   (Stage 6 — EARP)

Method (per county c, per event e)
----------------------------------
    demand_c   = Σ_ds  units_DS{ds}_scaled,c · repair_time[ds]   (unit-months)
    capacity_c = Average_Building_Permits(12 months)_c · modifier (units/month)
    floor_c    = max repair_time[ds] over damage states with > 0 damaged units
    recovery_c = max( floor_c, demand_c / capacity_c )           (months)

Counties whose construction capacity is zero or absent are assigned an
infinite recovery time (json `Infinity`); Stage 6 / the distribution
scripts already treat these as NaN.

repair_time (months per damaged housing unit), HAZUS-based:
    DS1 (Slight)    = 1
    DS2 (Moderate)  = 1
    DS3 (Extensive) = 3
    DS4 (Complete)  = 6

Run with:
    conda activate climada_env && python scripts/run_pyrecodes_light.py

Inputs:
    impacts_out/by_event/scaled/*_scaled.csv
        Per-event scaled damage (Stage 4). Columns used:
        event_name, fips, units_DS1_scaled, units_DS2_scaled,
        units_DS3_scaled, units_DS4_scaled
    data/selected_states_counties_with_permits.csv
        Building-permit reference table. Columns used:
        FIPS, Average_Building_Permits(12 months)

Output:
    data/recovery_potential_per_scenario/{event}_scaled_recovery_potential.json
        One JSON list per event; each record:
        {"event", "fips", "reconstruction_capacity",
         "recovery_potential [months]"}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

DEFAULT_IMPACTS_DIR = BASE_DIR / "impacts_out" / "by_event" / "scaled"
DEFAULT_PERMIT_FILE = BASE_DIR / "data" / "selected_states_counties_with_permits.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "recovery_potential_per_scenario"

# Damage-state column → repair time in months per damaged housing unit (HAZUS).
REPAIR_TIME = {
    "units_DS1_scaled": 1,
    "units_DS2_scaled": 1,
    "units_DS3_scaled": 3,
    "units_DS4_scaled": 6,
}

PERMIT_CAPACITY_COLUMN = "Average_Building_Permits(12 months)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(value: str) -> float:
    """Parse a CSV cell to float, treating empty/invalid values as 0.0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_construction_capacity(permit_file: Path, modifier: float) -> dict[str, float]:
    """
    Build {fips: monthly_construction_capacity} from the permit reference table.

    Capacity is the average monthly residential building permits issued in the
    county, scaled by `modifier`. Counties with missing permit values get 0.
    FIPS codes are kept as raw strings to match the damage CSVs (both files
    store unpadded FIPS, e.g. "1005", "12071").
    """
    capacity: dict[str, float] = {}
    with open(permit_file, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fips = row["FIPS"]
            capacity[fips] = _to_float(row.get(PERMIT_CAPACITY_COLUMN, "")) * modifier
    print(f"Loaded construction capacity for {len(capacity):,} counties "
          f"(modifier = {modifier})")
    return capacity


def housing_unit_month_demand(row: dict) -> float:
    """Total repair demand (housing-unit-months) for one county-event row."""
    return sum(_to_float(row[ds]) * months for ds, months in REPAIR_TIME.items())


def minimal_recovery_time(row: dict) -> float:
    """Longest single-damage-state repair time among damage states present.

    Acts as a floor: even with unlimited capacity, a unit in damage state ds
    still needs repair_time[ds] months to repair.
    """
    return max((months for ds, months in REPAIR_TIME.items() if _to_float(row[ds]) > 0),
               default=0)


def recovery_potential_for_event(damage_rows: list[dict],
                                  capacity: dict[str, float]) -> list[dict]:
    """Compute per-county recovery potential for a single event's damage rows."""
    output = []
    for row in damage_rows:
        fips = row["fips"]
        county_capacity = capacity.get(fips, 0.0)
        demand = housing_unit_month_demand(row)
        floor = minimal_recovery_time(row)

        if county_capacity <= 0:
            # No (or unknown) construction capacity → cannot recover.
            recovery = math.inf
        else:
            recovery = max(floor, demand / county_capacity)

        output.append({
            "event": row["event_name"],
            "fips": fips,
            "reconstruction_capacity": county_capacity,
            "recovery_potential [months]": recovery,
        })
    return output


def read_damage_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impacts-dir", type=Path, default=DEFAULT_IMPACTS_DIR,
                        help="Directory of per-event scaled damage CSVs "
                             "(default: impacts_out/by_event/scaled)")
    parser.add_argument("--permit-file", type=Path, default=DEFAULT_PERMIT_FILE,
                        help="Building-permit reference CSV "
                             "(default: data/selected_states_counties_with_permits.csv)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Where to write per-event recovery JSONs "
                             "(default: data/recovery_potential_per_scenario)")
    parser.add_argument("--construction-capacity-modifier", type=float, default=1.0,
                        help="Scale factor applied to monthly construction "
                             "capacity (default: 1.0)")
    args = parser.parse_args()

    print("=" * 60)
    print("pyrecodes light — per-county hurricane recovery simulator")
    print("=" * 60)

    damage_files = sorted(args.impacts_dir.glob("*_scaled.csv"))
    if not damage_files:
        raise FileNotFoundError(
            f"No '*_scaled.csv' files found in {args.impacts_dir}. "
            "Run the impact stage (Stage 4) first."
        )
    print(f"Found {len(damage_files)} per-event damage files in {args.impacts_dir}")

    capacity = load_construction_capacity(args.permit_file,
                                          args.construction_capacity_modifier)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_records = 0
    for idx, damage_file in enumerate(damage_files, 1):
        if idx % 500 == 0:
            print(f"  {idx}/{len(damage_files)} events …")
        damage_rows = read_damage_csv(damage_file)
        records = recovery_potential_for_event(damage_rows, capacity)
        n_records += len(records)

        out_path = args.output_dir / f"{damage_file.stem}_recovery_potential.json"
        with open(out_path, "w") as fh:
            json.dump(records, fh, indent=4)

    print(f"\nWrote {len(damage_files)} event files "
          f"({n_records:,} county-event records) → {args.output_dir}")
    print("Next: python scripts/compute_recovery_potential.py  (Stage 6 — EARP)")


if __name__ == "__main__":
    main()
