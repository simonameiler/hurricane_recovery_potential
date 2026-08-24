#!/usr/bin/env python3
"""Run the recovery-burden step (Stage 5) for the scaling_strength=0.0
(wind-only) specification, using the shadow per-event CSV built by
build_scaling_sensitivity_specs.py. scaling_strength=1.0 needs no rerun: it
is exactly the repository's existing committed
analysis_output/earp_per_county.csv and data/recovery/recovery_potential.csv.

This is the g=0.0 leg of the multi-hazard-scaling value-add comparison
(Figure S9's EARB panel, notebooks/probabilistic_analysis.ipynb); see
scripts/compute_scaling_valueadd_recovery.py, which reads this script's
output.

Writes:
  analysis_output/scaling_sensitivity/g0.0/recovery_potential.csv
  analysis_output/scaling_sensitivity/g0.0/earp_per_county.csv

Does not touch the manuscript's own analysis_output/*.csv
(scaling_strength=1.0) or data/recovery/recovery_potential.csv at all.
"""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from modules.recovery_utils import compute_recovery_potential
import scripts.compute_recovery_potential as compute_recovery_potential_script

PERMIT_FILE = REPO / "data" / "selected_states_counties_with_permits_ctfix.csv"
OUT_DIR = REPO / "analysis_output" / "scaling_sensitivity" / "g0.0"
PER_EVENT_DIR = REPO / "data" / "impact_sensitivity" / "g0.0" / "per_event"


def main():
    if not PER_EVENT_DIR.exists():
        raise FileNotFoundError(
            f"{PER_EVENT_DIR} not found; run scripts/build_scaling_sensitivity_specs.py first"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    recovery_csv = OUT_DIR / "recovery_potential.csv"
    compute_recovery_potential(PER_EVENT_DIR, PERMIT_FILE, recovery_csv)

    earp_csv = OUT_DIR / "earp_per_county.csv"
    compute_recovery_potential_script.main(recovery_csv, earp_csv)

    print(f"\nWrote {recovery_csv.name}, {earp_csv.name} -> {OUT_DIR}")
    print("Done. scaling_strength=1.0 uses the existing committed "
          "analysis_output/earp_per_county.csv directly (no rerun).")


if __name__ == "__main__":
    main()
