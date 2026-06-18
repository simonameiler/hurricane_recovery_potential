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
from typing import Optional, Tuple

# import CLIMADA modules:
from climada.hazard import Centroids, Hazard
from climada.entity.exposures import Exposures
from climada.engine import ImpactCalc
from climada.entity.impact_funcs import ImpactFuncSet, ImpactFunc
import climada.util.coordinates as u_coord
from climada.util import ureg

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


def compute_scaled_loss(loss_arr, scaling_arr, kscale=1.0, mode="compound"):
    """Compute scaled loss according to mode. Pulled out for testability.

    Behaves like the previous in-function compute_scaled_loss_local.
    """
    if mode == "compound":
        s_eff = 1.0 + kscale * (scaling_arr - 1.0)
        s_eff = np.maximum(s_eff, 0.0)
        return 1.0 - np.power(1.0 - np.clip(loss_arr, 0.0, 1.0), s_eff)
    elif mode == "multiply":
        return np.clip(loss_arr * scaling_arr, 0.0, 1.0)
    else:
        raise ValueError("Unknown scale_mode")


def ds_code(ds_str):
    """Return numeric DS code from DS label string like 'DS2: Moderate'.

    Returns -1 on parse failure.
    """
    try:
        return int(str(ds_str).split(':')[0].strip().replace('DS', ''))
    except Exception:
        return -1


def infer_event_name_from_filename(path: Path) -> str:
    """Infer an event name from a file stem like 'aggregated_12_event_name.csv'.

    Falls back to the stem if it can't parse an 'aggregated_' pattern.
    """
    stem = path.stem
    if stem.startswith("aggregated_"):
        remainder = stem[len("aggregated_"):]
        parts = remainder.split("_")
        if len(parts) >= 2:
            return "_".join(parts[1:])
    return stem


def sanitize_event_name(name: str) -> str:
    """Sanitize a string for filenames: allow letters, numbers, dot, underscore, dash.

    Collapses repeated underscores and strips leading/trailing underscores.
    """
    import re

    s = str(name)
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "event"

def export_state_and_county_results_all_events(
    exp,
    imp,
    scaling_npz_path,
    county_region_path=None,
    out_dir="./data/impact",
    k: float = 1.0,
    scale_mode: str = "compound",
    lower_threshold: float = 0.005,
):
    """
    Produce county-event aggregated CSVs (per_event/) for all events.

    Parameters
    ----------
    exp, imp: CLIMADA Exposures and Impact for the state(s)
    scaling_npz_path: path to NPZ with Scaling and county_index
    county_region_path: optional mapping CSV for county_index -> stcode/ccode or fips
    out_dir: output directory
    k, scale_mode: scaling options
    """
    os.makedirs(out_dir, exist_ok=True)
    per_event_dir = Path(out_dir) / "per_event"
    per_event_dir.mkdir(parents=True, exist_ok=True)

    # load scaling
    scaling_data = np.load(scaling_npz_path)
    if "Scaling" not in scaling_data:
        raise KeyError("Scaling matrix not found in NPZ file")
    scaling_matrix = scaling_data["Scaling"]
    county_indices = scaling_data["county_index"]

    # build county_index -> fips if provided
    county_index_to_fips = {}
    if county_region_path:
        crm = pd.read_csv(county_region_path)
        if set(["stcode", "ccode"]).issubset(crm.columns):
            crm = crm.assign(stcode_str=crm["stcode"].astype(str), ccode_str=crm["ccode"].astype(str).str.zfill(3))
            crm["fips"] = crm["stcode_str"] + crm["ccode_str"]
        elif "fips" not in crm.columns:
            raise KeyError("county_region CSV must contain 'county_index' and ('stcode'+'ccode') or 'fips' columns")
        for _, r in crm.iterrows():
            county_index_to_fips[int(r["county_index"])] = str(r["fips"]).zfill(5)

    N = len(exp.gdf)
    E = len(imp.event_name)

    # prepare full_all per-building DataFrame
    full_all = exp.gdf.copy().reset_index(drop=True)
    if "stcode" in full_all.columns and "ccode" in full_all.columns:
        full_all["fips"] = full_all["stcode"].astype(str) + full_all["ccode"].astype(str).str.zfill(3)
    else:
        full_all["fips"] = None

    # aggregated rows collect
    agg_rows = []

    imp_csr = imp.imp_mat.tocsr()

    # compute_scaled_loss is provided as a top-level helper to improve testability

    for ev_idx, ev_name in enumerate(imp.event_name):
        row = imp_csr.getrow(ev_idx)
        if row.nnz == 0:
            continue
        coo = row.tocoo()
        affected_idx = coo.col.astype(int)
        loss_vals = coo.data.astype(float)
        # apply lower threshold: treat tiny fractional losses as zero
        loss_vals_thresh = np.where(loss_vals >= lower_threshold, loss_vals, 0.0)

        # compute scaling per affected
        fips_vals = full_all.loc[affected_idx, "fips"].to_numpy()
        scaling_for_build = np.ones(len(fips_vals), dtype=float)

        # Determine which row in the scaling matrix corresponds to this event.
        # Scaling NPZ typically indexes events by track number (e.g. '0001' -> row 0).
        event_row_idx = None
        try:
            # if event names are numeric (e.g., '0312'), map to 0-based index
            event_id_int = int(str(ev_name))
            event_row_idx = int(event_id_int) - 1
        except Exception:
            # fallback: if ev_name is not numeric, we cannot map reliably
            event_row_idx = None

        if event_row_idx is not None and not (0 <= event_row_idx < scaling_matrix.shape[0]):
            # scaling matrix has no row for this event (e.g., missing tracks); skip scaling
            print(f"Scaling: no scaling row for event '{ev_name}' (row {event_row_idx}) — using 1.0")
            event_row_idx = None
        if county_index_to_fips:
            fips_to_index = {v: k for k, v in county_index_to_fips.items()}
            for i, f in enumerate(fips_vals):
                if f is None or (isinstance(f, float) and np.isnan(f)):
                    scaling_for_build[i] = 1.0
                    continue
                ci = fips_to_index.get(str(f).zfill(5))
                if ci is None:
                    scaling_for_build[i] = 1.0
                else:
                    if ci in county_indices:
                        pos = int(np.where(county_indices == ci)[0][0])
                        if event_row_idx is not None:
                            scaling_for_build[i] = float(scaling_matrix[event_row_idx, pos])
                        else:
                            scaling_for_build[i] = 1.0
        else:
            for i, f in enumerate(fips_vals):
                try:
                    ci = int(f)
                except Exception:
                    scaling_for_build[i] = 1.0
                    continue
                if ci in county_indices:
                    pos = int(np.where(county_indices == ci)[0][0])
                    if event_row_idx is not None:
                        scaling_for_build[i] = float(scaling_matrix[event_row_idx, pos])
                    else:
                        scaling_for_build[i] = 1.0

        # compute scaled values from thresholded raw losses
        scaled_vals = compute_scaled_loss(loss_vals_thresh, scaling_for_build, kscale=k, mode=scale_mode)
        # ensure thresholded raws remain zero after scaling
        scaled_vals = np.where(loss_vals_thresh >= lower_threshold, scaled_vals, 0.0)

        # write columns to full_all
        raw_col = f"loss_raw_ev{ev_idx}"
        scaled_col = f"loss_scaled_ev{ev_idx}"
        repair_raw_col = f"repair_raw_ev{ev_idx}"
        repair_scaled_col = f"repair_scaled_ev{ev_idx}"
        for c in (raw_col, scaled_col, repair_raw_col, repair_scaled_col):
            if c not in full_all.columns:
                full_all[c] = 0.0
        # write thresholded raw and scaled values into the per-building dataframe
        full_all.loc[affected_idx, raw_col] = loss_vals_thresh
        full_all.loc[affected_idx, repair_raw_col] = loss_vals_thresh * full_all.loc[affected_idx, "ReplacementCost"]
        full_all.loc[affected_idx, scaled_col] = scaled_vals
        full_all.loc[affected_idx, repair_scaled_col] = scaled_vals * full_all.loc[affected_idx, "ReplacementCost"]

        # aggregation per county for this event
        aff = pd.DataFrame({
            "stcode": full_all.loc[affected_idx, "stcode"].values,
            "ccode": full_all.loc[affected_idx, "ccode"].values,
            "fips": full_all.loc[affected_idx, "fips"].values,
            "loss_raw": loss_vals_thresh,
            "loss_scaled": scaled_vals,
            "repair_raw": loss_vals_thresh * full_all.loc[affected_idx, "ReplacementCost"].values,
            "repair_scaled": scaled_vals * full_all.loc[affected_idx, "ReplacementCost"].values,
            "NumberOfUnits": full_all.loc[affected_idx, "NumberOfUnits"].values,
        })

        # only keep affected rows with at least one non-zero impact
        aff = aff[(aff["loss_raw"] > 0.0) | (aff["loss_scaled"] > 0.0)].copy()
        if aff.empty:
            # nothing to aggregate for this event
            continue

        # assign DS strings and numeric DS codes for robust aggregation
        aff["DS_raw"] = aff["loss_raw"].apply(assign_damage_state)
        aff["DS_scaled"] = aff["loss_scaled"].apply(assign_damage_state)

        # use top-level ds_code helper
        aff["DS_raw_code"] = aff["DS_raw"].apply(ds_code)
        aff["DS_scaled_code"] = aff["DS_scaled"].apply(ds_code)

        # build per-event aggregated rows (and write per-event aggregated file)
        agg_groups_event = []
        grp = aff.groupby(["stcode", "ccode"]) if "stcode" in aff.columns and "ccode" in aff.columns else [((None,None), aff)]
        for (stc, cc), g in grp:
            rowd = {
                "event_index": ev_idx,
                "event_name": ev_name,
                "stcode": stc,
                "ccode": cc,
                "fips": (g["fips"].iloc[0] if "fips" in g.columns else None),
            }
            for dsn in [1,2,3,4]:
                if "NumberOfUnits" in g.columns:
                    rowd[f"units_DS{dsn}_raw"] = int(g.loc[g["DS_raw_code"] == dsn, "NumberOfUnits"].sum())
                    rowd[f"units_DS{dsn}_scaled"] = int(g.loc[g["DS_scaled_code"] == dsn, "NumberOfUnits"].sum())
                else:
                    rowd[f"units_DS{dsn}_raw"] = int((g["DS_raw_code"] == dsn).sum())
                    rowd[f"units_DS{dsn}_scaled"] = int((g["DS_scaled_code"] == dsn).sum())
            rowd["repair_cost_sum_raw"] = float(g["repair_raw"].sum())
            rowd["repair_cost_sum_scaled"] = float(g["repair_scaled"].sum())
            agg_groups_event.append(rowd)
            agg_rows.append(rowd)

        # write per-event aggregated CSV(s) split by state: aggregated_{stcode}_{ev_name}.csv
        if agg_groups_event:
            agg_event_df = pd.DataFrame(agg_groups_event)
            if 'stcode' in agg_event_df.columns:
                for st in agg_event_df['stcode'].dropna().unique():
                    subset = agg_event_df[agg_event_df['stcode'] == st]
                    per_event_path = per_event_dir / f"aggregated_{str(st)}_{ev_name}.csv"
                    subset.to_csv(per_event_path, index=False)
            else:
                per_event_path = per_event_dir / f"aggregated_all_{ev_name}.csv"
                agg_event_df.to_csv(per_event_path, index=False)

    if agg_rows:
        return full_all, pd.DataFrame(agg_rows)
    return full_all, None