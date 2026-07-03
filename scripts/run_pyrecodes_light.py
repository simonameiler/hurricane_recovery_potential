#!/usr/bin/env python3
"""Run the pyrecodes-light recovery simulation on the probabilistic event set.

Thin command-line wrapper around modules.recovery_utils.compute_recovery_potential.
Reads the per-event impact CSVs (data/impact/per_event/) and writes one
consolidated recovery table (data/recovery/recovery_potential.csv) with columns
event_name, fips, reconstruction_capacity, recovery_potential_months
(NaN for zero capacity). See modules/recovery_utils.py for the method.
"""
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from modules.recovery_utils import compute_recovery_potential

DEFAULT_PER_EVENT_DIR = BASE_DIR / "data" / "impact" / "per_event"
DEFAULT_PERMIT_FILE = BASE_DIR / "data" / "selected_states_counties_with_permits.csv"
DEFAULT_OUT_CSV = BASE_DIR / "data" / "recovery" / "recovery_potential.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-event-dir", type=Path, default=DEFAULT_PER_EVENT_DIR)
    ap.add_argument("--permit-file", type=Path, default=DEFAULT_PERMIT_FILE)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--capacity-modifier", type=float, default=1.0)
    a = ap.parse_args()

    compute_recovery_potential(a.per_event_dir, a.permit_file, a.out_csv,
                               a.capacity_modifier)


if __name__ == "__main__":
    main()
