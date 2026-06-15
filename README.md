# Hurricane Recovery Potential

Probabilistic tropical-cyclone damage (CLIMADA) + multi-hazard scaling + pyrecodes recovery
simulations for US Atlantic-coast counties. Reproduces all figures and reported numbers in
the manuscript on **Expected Annual Recovery Potential (EARP)**.

---

## Environment

```bash
conda env create -f environment.yml
conda activate climada_env
```

---

## External inputs (not committed — document paths locally)

| Item | Location in repo | Notes |
|------|-----------------|-------|
| Gori TC wind-field `.mat` files | `climada_data/hazard/tropical_cyclone/gori/` | ~5018 events; Gori et al. (2025) |
| Raw ACS/Census housing CSVs | `data/exposure/states/` | one CSV per coastal state |
| pyrecodes per-event recovery JSONs | `data/recovery_potential_per_scenario/` | `{event_id}_scaled_recovery_potential.json` per event |
| FEMA NRI county geodatabase | `data/external/NRI_GDB_CensusTracts.gdb` | 612 MB; download from FEMA |

Small reference tables already committed: `data/US_counties.shp`, `data/county_region.csv`,
`data/selected_states_counties_with_permits.csv`, `data/external/NRI_Table_Counties.csv`,
`data/hazard/max{windmat,elev_coastcounty,ptot_rain_county}_ncep_reanal.mat`.

---

## Pipeline

### Stage 1 — Exposure

Build CLIMADA `Exposures` HDF5 files from raw ACS housing CSVs.

```bash
# cluster (SLURM)
sbatch batch/sbatch_make_state_exposures.sh   # → data/exposure/states/*.hdf5
sbatch batch/sbatch_make_NA_exposure.sh       # → data/exposure/NA_coast_exposure.hdf5
# local
python scripts/make_state_exposures.py
python scripts/make_NA_exposure.py
```

### Stage 2 — Hazard

Build CLIMADA `TropCyclone` hazard from Gori wind-field `.mat` files.

```bash
# cluster array (recommended; ~5018 events split into chunks)
sbatch batch/sbatch_make_haz_gori_array.sh
python scripts/concat_haz_gori_chunks.py \
    --pattern 'tc_ncep_reanal_chunk*.hdf5' \
    --output <climada_data>/hazard/tropical_cyclone/gori/tc_ncep_reanal.hdf5
# local (single run)
python scripts/make_haz_gori.py
```

### Stage 3 — Multi-hazard scaling

Compute county-level scaling factors from wind + rainfall + surge matrices.

```bash
python scripts/multihazard_scaling_relative.py \
    --wind  data/hazard/maxwindmat_ncep_reanal.mat \
    --rain  data/hazard/ptot_rain_county_ncep_reanal.mat \
    --surge data/hazard/maxelev_coastcounty_ncep_reanal.mat \
    --county-region data/county_region.csv \
    --out data/scaling_relative.npz
```

Output: `data/scaling_relative.npz` (consumed by Stage 4).

### Stage 4 — Impacts

Run CLIMADA impact calculation per state/event; merge to per-event scaled CSVs.

```bash
# cluster array (recommended)
bash batch/submit_calc_impacts_per_chunk.sh
sbatch batch/sbatch_combine_aggregated_outputs.sh
# local (per state, then merge)
python scripts/calc_state_impact.py --state <STATE> ...
python scripts/combine_aggregated_outputs.py
```

Outputs: `impacts_out/by_event/scaled/{event_id}_scaled.csv` per event.

Extract housing unit counts per county (needed by Stage 5):

```bash
python scripts/extract_exposure_units_by_county.py
# → analysis_output/county_exposed_housing_units.csv
```

### Stage 5 — Metrics / distributions

```bash
python scripts/analyze_event_frequency_damage.py
# → analysis_output/county_event_frequency_damage_metrics.csv
#   (EAUA inputs for all figure scripts)

python scripts/compare_median_vs_max_events.py
# → analysis_output/median_vs_max_event_comparison.csv
#   (per-event median/max metrics; consumed by Figs S5, S6)

python scripts/analyze_recovery_distributions.py
# → analysis_output/county_distribution_metrics.csv
#   (skewness metrics; consumed by Fig S8)

# Optional — EAD vs Gori AAL validation (potential supplementary figure):
python scripts/analyze_impacts.py
python scripts/create_validation_figures.py
```

### Stage 6 — Recovery (EARP)

```bash
python scripts/TC_NA_recovery_analysis_reorganized_complete.py
# → analysis_output/earp_per_county.csv
```

### Stage 7 — Manuscript figures

```bash
# Fig 2 (annual_3panel), Fig 3 (recovery_drivers_annual_vs_median),
# Fig S1 (na_coast_hazard_overview), Fig S2 (event_350 / event_4347 3-panel),
# Fig S5 (median_event_3panel), Fig S6 (max_event_3panel),
# Fig S7 (recovery_drivers_annual_max), Fig S8 (skewness_maps):
python scripts/create_manuscript_figures.py
# → analysis_output/figures/{stem}.png + .pdf  (300 dpi)

# Fig 4 (bivariate maps — risk × capacity, risk × recovery potential):
python scripts/create_bivariate_maps.py
# → analysis_output/bivariate_map_{A,B}_*.{png,pdf}
# → analysis_output/bivariate_maps_combined.{png,pdf}
```

The Spearman ρ values reported in the text (annual EARP vs EAUA and vs CC;
per-event median recovery vs EAUA and vs CC) are computed and annotated
directly inside `create_manuscript_figures.py`.

### Stage 8 — NRI construct validation

```bash
python scripts/nri_discriminant_assessment.py
# → analysis_output/nri_county_merged.csv
# → analysis_output/nri_correlations.csv
# → analysis_output/nri_partial_correlations.csv
# → analysis_output/nri_priority_crosstab.csv
# → analysis_output/divergent_counties_SI.csv
# → analysis_output/fig_capacity_vs_resilience.png
```

Requires `data/external/NRI_Table_Counties.csv` (already committed).

---

## Module dependencies

| Module | Imported by |
|--------|------------|
| `modules/exposure_utils.py` | `make_NA_exposure.py`, `make_state_exposures.py` |
| `modules/hazard_utils.py` | `make_haz_gori.py`, `make_haz_gori_chunks.py` |
| `modules/impact_utils.py` | `calc_state_impact.py`, `combine_aggregated_outputs.py` |
| `modules/impfunc_utils.py` | `calc_state_impact.py` |

---

## Figure → script map

| Manuscript item | Script | Output file |
|-----------------|--------|-------------|
| Fig 2 (annual triptych) | `create_manuscript_figures.py` | `figures/annual_3panel.{png,pdf}` |
| Fig 3 (driver scatter) | `create_manuscript_figures.py` | `figures/recovery_drivers_annual_vs_median.{png,pdf}` |
| Fig 4 (bivariate maps) | `create_bivariate_maps.py` | `bivariate_map_{A,B}_*.{png,pdf}` |
| Fig S1 (hazard overview) | `create_manuscript_figures.py` | `figures/na_coast_hazard_overview.{png,pdf}` |
| Fig S2 (event 350 / 4347) | `create_manuscript_figures.py` | `figures/event_{350,4347}_3panel_map.{png,pdf}` |
| Fig S5 (median event) | `create_manuscript_figures.py` | `figures/median_event_3panel.{png,pdf}` |
| Fig S6 (max event) | `create_manuscript_figures.py` | `figures/max_event_3panel.{png,pdf}` |
| Fig S7 (annual + max drivers) | `create_manuscript_figures.py` | `figures/recovery_drivers_annual_max.{png,pdf}` |
| Fig S8 (skewness maps) | `create_manuscript_figures.py` | `figures/skewness_maps.{png,pdf}` |
| Spearman ρ (text) | `create_manuscript_figures.py` | annotated on Fig 3 / S7 panels |
| NRI construct validation | `nri_discriminant_assessment.py` | `analysis_output/nri_*.csv` |
