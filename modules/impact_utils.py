import os
import json
import h5py
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as dt
import copy as cp
from shapely.geometry import Point, shape
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.io import loadmat, whosmat
from pathlib import Path
from typing import Iterable, Optional, Tuple

# import CLIMADA modules:
from climada.hazard import Centroids, Hazard
from climada.entity.exposures import Exposures
from climada.engine import ImpactCalc
from climada.entity.impact_funcs import ImpactFuncSet, ImpactFunc
import climada.util.coordinates as u_coord
from climada.util import ureg

import scripts.impact_funcs as ifunc

KMH_TO_MS = (1.0 * ureg.km / ureg.hour).to(ureg.meter / ureg.second).magnitude

def assign_damage_state(impact):
    """
    Assigns a building damage state based on the expected loss ratio (impact).

    The loss ratio represents structural building damage, defined as the ratio of 
    estimated repair cost to replacement cost, with values ranging from 0 (no damage) 
    to 1 (complete loss). Thresholds follow the definitions from Hazus Hurricane Model 
    Technical Manual, Section 8.1.4.3 (Loss of Use for Residential Buildings), Hazus 5.1, July 2022.

    Parameters:
        impact (float): Structural building damage (loss ratio), between 0 and 1.

    Returns:
        str: Damage state label.
    """
    if impact < 0.02:
        return 'DS0: None'       # 0–2% loss
    elif impact < 0.05:
        return 'DS1: Slight'     # 2–5% loss
    elif impact < 0.10:
        return 'DS2: Moderate'   # 5–10% loss
    elif impact < 0.50:
        return 'DS3: Extensive'  # 10–50% loss
    else:
        return 'DS4: Complete'   # 50–100%+ loss
    

def _generate_building_json(exposure_gdf, general_info_path, loss_info_path):
    """
    Generate two JSON files: one for general building information and one for damage information,
    wrapped in the "Buildings" -> "Building" -> {ID: ...} structure.

    Parameters:
        exposure_gdf (GeoDataFrame): GeoDataFrame containing building data.
        general_info_path (str): Path to save the general building information JSON.
        loss_info_path (str): Path to save the damage information JSON.

    Returns:
        tuple: Two dictionaries containing general information and loss information.
    """
    general_building_data = {}
    loss_building_data = {}

    for _, row in exposure_gdf.iterrows():
        building_id = str(row['id'])

        # General building info nested under "GeneralInformation"
        general_building_data[building_id] = {
            "GeneralInformation": {
                "PlanArea": row['PlanArea'],
                "NumberOfStories": row['NumberOfStories'],
                "MedianYearBuilt": row['MedianYearBuilt'],
                "ReplacementCost": row['ReplacementCost'],
                "StructureType": row['StructureType'],
                "NumberOfUnits": row['NumberOfUnits'],
                "CensusBlock": row['CensusBlock'],
                "CensusTract": row['CensusTract'],
                "FootprintID": row['FootprintID'],
                "OccupancyClass": row['OccupancyClass'],
                "Footprint": row['footprint_geometry'],
                "Latitude": row['geometry'].y,
                "Longitude": row['geometry'].x,
                "type": "Building"
            }
        }

        # Loss info
        loss_building_data[building_id] = {
            "LossRatio": row['LossRatio'],
            "DamageState": row['DamageState'],
            "RepairCost": row['RepairCost']
        }

    # Wrap the dictionaries with the desired nesting
    general_info = {
        "Buildings": {
            "Building": general_building_data
        }
    }

    loss_info = {
        "Buildings": {
            "Building": loss_building_data
        }
    }

    # Write to JSON
    with open(general_info_path, "w") as f:
        json.dump(general_info, f, indent=4)

    with open(loss_info_path, "w") as f:
        json.dump(loss_info, f, indent=4)

    return general_info, loss_info

def process_and_write_each_event(exp, imp, output_dir, base_filename):
    """
    For each event in the Impact object, filter affected buildings and write to JSON files.

    Parameters
    ----------
    exp : Exposures
        CLIMADA Exposures object (with .gdf attribute).
    imp : Impact
        CLIMADA Impact object with computed impacts and event names.
    output_dir : str
        Directory to save output JSON files.
    base_filename : str
        Base name for the output files (e.g., 'FL2023').
    """
    os.makedirs(output_dir, exist_ok=True)
    imp_dense = imp.imp_mat.todense()

    for i, name in enumerate(imp.event_name):
        imp_vals = imp_dense[i].A1
        repair_cost = imp_vals * exp.gdf['ReplacementCost']

        # Filter only affected buildings
        affected = exp.gdf[repair_cost > 0].copy()
        if affected.empty:
            continue

        # Add relevant columns
        affected['RepairCost'] = repair_cost[repair_cost > 0]
        affected['LossRatio'] = imp_vals[repair_cost > 0]
        affected['DamageState'] = affected['LossRatio'].apply(assign_damage_state)

        # Drop clutter from previous runs if present
        drop_cols = [col for col in affected.columns if col.startswith('RepairCost_') or col.startswith('DS_') or col.startswith('imp_')]
        affected.drop(columns=drop_cols, errors='ignore', inplace=True)

        # Construct file paths
        gen_path = os.path.join(output_dir, f"{base_filename}_{name}_general.json")
        loss_path = os.path.join(output_dir, f"{base_filename}_{name}_loss.json")

        # Write to JSON
        _generate_building_json(affected, gen_path, loss_path)

    
def attach_all_event_impacts_to_exposure(exp, imp):
    """
    Add LossRatio, RepairCost, and DamageState columns to the exposure GeoDataFrame for each event.

    Parameters
    ----------
    exp : Exposures
        CLIMADA Exposures object (with .gdf).
    imp : Impact
        CLIMADA Impact object with .imp_mat and .event_name.

    Returns
    -------
    Exposures
        Updated Exposures object with additional columns in .gdf:
        - LossRatio_<event_name>
        - RepairCost_<event_name>
        - DS_<event_name>
    """
    imp_dense = imp.imp_mat.todense()

    for i, name in enumerate(imp.event_name):
        loss_ratio = imp_dense[i].A1
        exp.gdf[f'LossRatio_{name}'] = loss_ratio
        exp.gdf[f'RepairCost_{name}'] = loss_ratio * exp.gdf['ReplacementCost']
        exp.gdf[f'DS_{name}'] = exp.gdf[f'LossRatio_{name}'].apply(assign_damage_state)

    return exp

def compute_damage_state_metrics(
    exp,
    haz,
    ds_states=range(1,5)
):
    """
    For each event in haz.event_name, compute and append to exp.gdf:
      - imp_ds{n}_{event}      (exceedance values for n=1..4)
      - prob_ds{n}_{event}     (slice‐by‐slice probabilities for n=0..4)
      - LossRatio_{event}      (expected mean damage ratio)
      - RepairCost_{event}     (LossRatio * replacement cost)
      - most_probable_ds_{event} (index 0–4 of the highest prob_ds)
    """

    # 1) build a dict of full (n_events × n_exposures) matrices for each DS
    ds_mats = {}
    for ds in ds_states:
        impf   = ifunc.load_impact_func_set_ds(ds)
        impobj = ImpactCalc(exp, impf, haz).impact(save_mat=True)
        # convert to a numpy array shape (n_events, n_exposures)
        ds_mats[ds] = impobj.imp_mat.todense().A

    # pre‐define the MDR mapping
    mdr_map = {0:0.0, 1:0.02, 2:0.10, 3:0.50, 4:1.00}

    # 2) for each event, slice out its row from every DS matrix and attach columns
    for i, name in enumerate(haz.event_name):
        # attach the raw exceedances
        for ds in ds_states:
            exp.gdf[f'imp_ds{ds}_{name}'] = ds_mats[ds][i, :]

        # compute slice‐by‐slice probabilities 
        exp.gdf[f'prob_ds0_{name}'] = 1 - exp.gdf[f'imp_ds1_{name}']
        for ds in ds_states[:-1]:
            exp.gdf[f'prob_ds{ds}_{name}'] = (
                exp.gdf[f'imp_ds{ds}_{name}'] 
                - exp.gdf[f'imp_ds{ds+1}_{name}']
            )
        # the “complete” probability is just imp_ds4
        exp.gdf[f'prob_ds4_{name}'] = exp.gdf[f'imp_ds4_{name}']

        # expected mean damage ratio = sum_j p_j * MDR_j
        lr_col = f'LossRatio_{name}'
        exp.gdf[lr_col] = sum(
            mdr_map[j] * exp.gdf[f'prob_ds{j}_{name}']
            for j in mdr_map
        )

        # expected repair cost
        exp.gdf[f'RepairCost_{name}'] = (
            exp.gdf[lr_col] * exp.gdf['ReplacementCost']
        )

        # most probable DS by argmax over the prob_ds columns
        prob_cols = [f'prob_ds{j}_{name}' for j in range(5)]
        exp.gdf[f'DS_{name}'] = (
            exp.gdf[prob_cols].values.argmax(axis=1)
        )

    return exp

def compute_and_store_event_impacts_parquet(exp, imp, output_dir, base_filename="FL2023"):
    """
    For each event in the Impact object, compute impacts, store results as compressed Parquet files,
    and log metadata.

    Parameters
    ----------
    exp : Exposures
        CLIMADA Exposures object (with .gdf).
    imp : Impact
        CLIMADA Impact object with computed impacts and event names.
    output_dir : str
        Directory to save output Parquet files and metadata.
    base_filename : str
        Prefix for output files (e.g., FL2023).
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    for i, name in enumerate(imp.event_name):
        imp_sparse = imp.imp_mat.getrow(i).tocoo()
        if imp_sparse.nnz == 0:
            continue

        affected_ids = imp_sparse.col
        loss_ratios = imp_sparse.data

        affected = exp.gdf.iloc[affected_ids].copy()
        affected['LossRatio'] = loss_ratios
        affected['RepairCost'] = affected['LossRatio'] * affected['ReplacementCost']
        affected['DamageState'] = affected['LossRatio'].apply(assign_damage_state)

        # Select only relevant output
        out_df = affected[['id', 'LossRatio', 'RepairCost', 'DamageState']].copy()
        file_name = f"{base_filename}_impact_{name}.parquet"
        out_path = os.path.join(output_dir, file_name)

        out_df.to_parquet(out_path, index=False, compression='snappy')

        metadata.append({
            "event_index": i,
            "event_name": name,
            "n_affected": out_df.shape[0],
            "file": file_name
        })

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    return metadata_df

def merge_stored_impacts_parquet(exp, impact_dir):
    """
    Load and merge per-event Parquet files stored in impact_dir into the exposure DataFrame.

    Parameters
    ----------
    exp : Exposures
        CLIMADA Exposures object to update.
    impact_dir : str
        Path to the directory containing metadata.csv and per-event Parquet files.

    Returns
    -------
    Exposures
        Updated CLIMADA Exposures object with merged impact data.
    """
    metadata_path = os.path.join(impact_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata file at: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    for _, row in metadata.iterrows():
        file_path = os.path.join(impact_dir, row["file"])
        df = pd.read_parquet(file_path)

        # Rename impact columns to be event-specific BEFORE merging
        df = df.rename(columns={
            "LossRatio": f"LossRatio_{row['event_name']}",
            "RepairCost": f"RepairCost_{row['event_name']}",
            "DamageState": f"DS_{row['event_name']}"
        })

        # Merge on building ID; keep all rows in the exposure dataset
        res_df = exp.gdf.merge(df, on="id", how="left", validate="one_to_one")

    return res_df

def process_and_write_probabilistic_events(
    exp,
    haz,
    output_dir,
    base_filename,
    ds_states=range(1, 5),               # DS1–DS4 only
    replacement_cost_col='ReplacementCost'
):
    """
    1) Calls compute_damage_state_metrics (DS1–DS4),
    2) For each event in haz.event_name, filters affected buildings (>0 cost),
       assigns a discrete DS label based on expected_mdd, then writes out
       your general + loss JSONs.

    Returns the updated exp.
    """
    exp = compute_damage_state_metrics(exp, haz, ds_states, replacement_cost_col)
    os.makedirs(output_dir, exist_ok=True)

    for idx, name in enumerate(haz.event_name):
        loss_ratio = exp.gdf['expected_mdd']
        repair_cost = exp.gdf['probabilistic_repair_cost']

        affected = exp.gdf[repair_cost > 0].copy()
        if affected.empty:
            continue

        # assign a DS label using your assign_damage_state()
        affected['DamageState'] = affected['expected_mdd'].apply(assign_damage_state)
        affected['LossRatio']   = loss_ratio[repair_cost > 0]
        affected['RepairCost']  = repair_cost[repair_cost > 0]

        # drop DS- and imp- columns we no longer need in the JSON
        drop_cols = [c for c in affected.columns if c.startswith('imp_ds') or c.startswith('prob_ds')]
        affected.drop(columns=drop_cols, errors='ignore', inplace=True)

        gen_path  = Path(output_dir) / f"{base_filename}_{name}_general.json"
        loss_path = Path(output_dir) / f"{base_filename}_{name}_loss.json"

        generate_building_json(affected, str(gen_path), str(loss_path))

    return exp