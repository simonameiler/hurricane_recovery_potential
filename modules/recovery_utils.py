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

# def load_exposure_from_csv(csv_path):
#     """
#     Load exposure data from a CSV and return a CLIMADA Exposures object.

#     Parameters
#     ----------
#     csv_path : str
#         Path to the exposure inventory CSV file.

#     Returns
#     -------
#     Exposures
#         CLIMADA-compatible Exposures object with geometry and required fields.
#     """
#     # Load the CSV and drop unwanted index columns
#     df = pd.read_csv(csv_path, index_col=False)

#     # Create geometry
#     df['geometry'] = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]

#     # Convert to GeoDataFrame
#     gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

#     # Rename to match CLIMADA requirements
#     gdf = gdf.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
#     gdf['value'] = 1

#     # Create and return the Exposures object
#     return Exposures(gdf)

def load_exposure_from_csv(csv_paths):
    """
    Load exposure data from multiple CSV files and return a combined CLIMADA Exposures object,
    with added 'County' and 'State' columns inferred from the file path, and unique building IDs.

    Parameters
    ----------
    csv_paths : list of str
        List of paths to exposure inventory CSV files.

    Returns
    -------
    Exposures
        Combined CLIMADA-compatible Exposures object with 'County' and 'State' info and unique 'id's.
    """
    gdf_list = []

    for csv_path in csv_paths:
        # Infer state and county name from path
        path_obj = Path(csv_path)
        state = path_obj.parent.name
        filename = path_obj.name
        county = filename.split('_')[0]  # assumes {County}_Inventory.csv

        # Load CSV
        df = pd.read_csv(csv_path, index_col=False)

        # Skip files without required columns
        if not {'Longitude', 'Latitude'}.issubset(df.columns):
            print(f"Skipping {csv_path} (missing Latitude/Longitude)")
            continue

        # Drop existing ID column if it exists
        if 'id' in df.columns:
            df = df.drop(columns='id')

        # Create geometry
        df['geometry'] = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]

        # Add county and state info
        df['County'] = county
        df['State'] = state

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

        # Rename columns to match CLIMADA expectations
        gdf = gdf.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
        gdf['value'] = 1

        gdf_list.append(gdf)

    if not gdf_list:
        raise ValueError("No valid exposure files loaded (all missing Latitude/Longitude or empty).")

    # Combine all GeoDataFrames
    combined_gdf = pd.concat(gdf_list, ignore_index=True)

    # Assign unique IDs
    combined_gdf['id'] = combined_gdf.index + 1

    # Return as Exposures object
    return Exposures(combined_gdf)


def filter_bem_exposure_by_counties(exposure, admin2_boundary_path, counties):
    """
    Filter an exposure object for a given list of counties based on admin2 boundaries.
    
    Parameters:
        exposure (Exposure): The CLIMADA exposure object to be filtered.
        admin2_boundary_path (str): Path to the shapefile/GeoJSON containing admin2 boundaries.
        counties (list): List of county names (e.g., ["Lee", "Collier"]).
    
    Returns:
        Exposure: A new exposure object filtered for the given counties.
    """
    # Load the admin2 boundaries
    admin2_boundaries = gpd.read_file(admin2_boundary_path)
    
    # Ensure the GeoDataFrame is in WGS84 (latitude/longitude)
    admin2_boundaries = admin2_boundaries.to_crs("EPSG:4326")
    
    # Perform a spatial join between the exposure GeoDataFrame and admin2 boundaries
    exp_adm2_matched = gpd.sjoin(
        exposure.gdf,
        admin2_boundaries[["NAME_2", "GID_2", "geometry"]],
        how="left",
        predicate="within"
    )
    
    # Filter the matched GeoDataFrame for the specified counties
    exp_adm2_filtered = exp_adm2_matched[exp_adm2_matched["NAME_2"].isin(counties)]
    exp_adm2_filtered.pop("index_right")
    
    # Create a new exposure object with the filtered GeoDataFrame
    #filtered_exposure = cp.deepcopy(exposure)
    #filtered_exposure.gdf = exp_adm2_filtered
    filtered_exposure = Exposures(exp_adm2_filtered)
    
    return filtered_exposure

def load_reask_hazard(file_list, haz_dir):
    """
    Creates a hazard set from a list of NetCDF file names, appending each hazard event.
    
    Parameters:
    -----------
    file_list : list of str
        List of NetCDF file names (with paths) to process.
    haz_dir : str or Path
        Directory containing the NetCDF files.

    Returns:
    --------
    hazard_set : climada.Hazard
        A hazard object containing all the events from the files.
    """
    
    hazard_set = None  # Initialize an empty hazard set to append to
    
    for idx, file_name in enumerate(file_list):
        # Extract the year (second part of the filename) and convert it to a timestamp
        year_str = file_name.split('_')[1]  # Extract year (e.g., '2005')
        time_value = pd.Timestamp(f"{year_str}-01-01")  # Create a timestamp for the year

        # Open the dataset
        dataset = xr.open_dataset(haz_dir / file_name)

        # Extract the relevant metadata from the variable
        variable = dataset["FT_50pct_3-second_kph"]

        # Add 'time' dimension using the extracted year
        dataset = dataset.expand_dims(time=[time_value])

        # Now pass the modified dataset to from_xarray_raster
        hazard = Hazard.from_xarray_raster(
            dataset,
            hazard_type="TC",  # 'TC' for Tropical Cyclone
            intensity_unit="m/s",  # 'km/h'
            intensity="FT_50pct_3-second_kph"
        )
        
        # unit conversion intensiy kph to ms
        hazard.intensity = hazard.intensity*KMH_TO_MS
        
        # Set event_name and event_id explicitly (convert to NumPy arrays)
        hazard.event_name = np.array([variable.sid])  # Convert to NumPy array
        hazard.event_id = np.array([idx + 1])  # Convert to NumPy array for event_id
        
        # re-initialize centroids w/ normalized longitued
        hazard.centroids = Centroids(lat=hazard.centroids.lat, 
                                     lon=u_coord.lon_normalize(hazard.centroids.lon))
        
        # Append the hazard to the set
        if hazard_set is None:
            hazard_set = hazard
        else:
            hazard_set.append(hazard)
    
    return hazard_set

def load_reask_geotiffs(file_list, haz_dir):
    """
    Load Reask GeoTIFF wind fields with xarray and build a CLIMADA Hazard object.

    Parameters
    ----------
    file_list : list of str
        Filenames of GeoTIFF files.
    haz_dir : str or Path
        Directory containing the GeoTIFF files.

    Returns
    -------
    climada.hazard.Hazard
        A Hazard object with one event per GeoTIFF file.
    """
    hazard_set = None

    for idx, file_name in enumerate(file_list):
        file_path = os.path.join(haz_dir, file_name)
        parts = file_name.replace(".tif", "").split("_")
        year = parts[1]
        sid = parts[2]
        storm_name = parts[0].title()
        time_value = pd.Timestamp(f"{year}-01-01")

        # Load GeoTIFF using xarray with rasterio engine
        ds = xr.open_dataset(file_path, engine="rasterio")
        data = ds["band_data"].squeeze()  # shape: (y, x)

        # Extract lat/lon grids
        lat = data.y.values
        lon = data.x.values
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Flatten and mask NaNs
        intensity = data.values
        mask = ~np.isnan(intensity)
        val_flat = intensity[mask]
        lat_flat = lat_grid[mask]
        lon_flat = lon_grid[mask]
        lon_flat = u_coord.lon_normalize(lon_flat)

        # Build centroids and hazard
        centroids = Centroids(lat=lat_flat, lon=lon_flat)
        hazard = Hazard()
        hazard.intensity = sparse.csr_matrix(val_flat[None, :])
        hazard.fraction = sparse.csr_matrix(np.ones_like(val_flat)[None, :])
        hazard.event_id = np.array([idx + 1])
        hazard.event_name = np.array([f"{storm_name}_{year}"])
        hazard.date = np.array([time_value])
        hazard.frequency = np.array([1.0])
        hazard.centroids = centroids
        hazard.units = "m/s"
        hazard.haz_type = "TC"

        if hazard_set is None:
            hazard_set = hazard
        else:
            hazard_set.append(hazard)

    return hazard_set

# ---------- IO: read lon/lat/wind from either v7.3 (HDF5) or v5 MAT ----------
def _read_mat_arrays(fp: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (lon, lat, wind) with shape (ny, nx, nt).
    Variables: PLONG_SAVE, PLAT_SAVE, wind_grid (m/s).
    """
    def _ensure_3d(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        a = np.squeeze(a)
        if a.ndim == 2:
            a = a[:, :, None]
        if a.ndim != 3:
            raise ValueError(f"{fp.name}: expected 2D/3D arrays; got {a.shape}")
        return a

    if h5py.is_hdf5(fp):
        with h5py.File(fp, "r") as f:
            def g(name):
                if name not in f:
                    raise KeyError(f"{name} not found in {fp.name}")
                return np.array(f[name])
            lon = g("PLONG_SAVE")
            lat = g("PLAT_SAVE")
            wind = g("wind_grid")
    else:
        md = loadmat(fp, squeeze_me=False, struct_as_record=False)
        def g(name):
            if name not in md:
                raise KeyError(f"{name} not found in {fp.name}")
            return md[name]
        lon = g("PLONG_SAVE")
        lat = g("PLAT_SAVE")
        wind = g("wind_grid")

    lon = _ensure_3d(lon)
    lat = _ensure_3d(lat)
    wind = _ensure_3d(wind)

    if lon.shape != lat.shape or lat.shape != wind.shape:
        raise ValueError(f"{fp.name}: lon{lon.shape}, lat{lat.shape}, wind{wind.shape} differ.")

    return lon, lat, wind


# ---------- Grid: build one common lon/lat grid for all events ----------
def _build_target_grid(
    lon_list: list[np.ndarray],
    lat_list: list[np.ndarray],
    resolution_deg: float,
    pad_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create regular 1D lon/lat vectors covering all storms/timesteps.
    Uses unwrap to handle dateline and normalizes to [-180, 180].
    """
    # Conservative bounds from all files
    lon_min = float(np.nanmin([np.nanmin(a) for a in lon_list]))
    lon_max = float(np.nanmax([np.nanmax(a) for a in lon_list]))
    lat_min = float(np.nanmin([np.nanmin(a) for a in lat_list]))
    lat_max = float(np.nanmax([np.nanmax(a) for a in lat_list]))

    # unwrap lon span to avoid crossing issues
    lon_pair = np.array([lon_min, lon_max], dtype=float)
    lon_unw  = np.unwrap(np.deg2rad(lon_pair), discont=np.deg2rad(180.0))
    lon_min_u, lon_max_u = np.rad2deg([lon_unw.min(), lon_unw.max()])

    lon_vec = np.arange(lon_min_u - pad_deg, lon_max_u + pad_deg + 1e-12, resolution_deg)
    lat_vec = np.arange(lat_min   - pad_deg, lat_max   + pad_deg   + 1e-12, resolution_deg)

    # normalize back to [-180, 180]
    lon_vec = u_coord.lon_normalize(lon_vec)
    return lon_vec, lat_vec


# ---------- Core: bin every timestep to nearest cell and take per-cell max ----------
def _accumulate_max_to_grid(
    lon3d: np.ndarray, lat3d: np.ndarray, wind3d: np.ndarray,
    lon_vec: np.ndarray, lat_vec: np.ndarray
) -> np.ndarray:
    """
    Map all timesteps of a moving grid to a fixed (lat_vec, lon_vec) grid
    via nearest-cell binning, taking the per-cell maximum over time.

    Returns (Ny, Nx) array.
    """
    Ny, Nx = len(lat_vec), len(lon_vec)
    out = np.zeros((Ny, Nx), dtype=float)

    # Precompute bin edges (midpoints) for nearest-cell assignment
    lon_edges = np.empty(Nx + 1)
    lat_edges = np.empty(Ny + 1)
    lon_edges[1:-1] = 0.5 * (lon_vec[1:] + lon_vec[:-1])
    lat_edges[1:-1] = 0.5 * (lat_vec[1:] + lat_vec[:-1])
    lon_edges[0]    = lon_vec[0]  - (lon_vec[1]  - lon_vec[0])  / 2.0
    lon_edges[-1]   = lon_vec[-1] + (lon_vec[-1] - lon_vec[-2]) / 2.0
    lat_edges[0]    = lat_vec[0]  - (lat_vec[1]  - lat_vec[0])  / 2.0
    lat_edges[-1]   = lat_vec[-1] + (lat_vec[-1] - lat_vec[-2]) / 2.0

    nt = wind3d.shape[2]
    flat_out = out.ravel()
    for t in range(nt):
        lon = np.asarray(lon3d[:, :, t])
        lat = np.asarray(lat3d[:, :, t])
        w   = np.asarray(wind3d[:, :, t])

        m = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(w)
        if not np.any(m):
            continue

        lon_f = lon[m].ravel()
        lat_f = lat[m].ravel()
        w_f   = w[m].ravel()

        # Assign to nearest cell via bin edges
        j = np.searchsorted(lon_edges, lon_f, side="right") - 1  # x index
        i = np.searchsorted(lat_edges, lat_f, side="right") - 1  # y index

        # Clamp to grid
        j = np.clip(j, 0, Nx - 1)
        i = np.clip(i, 0, Ny - 1)

        # Reduce by maximum
        flat_idx = i * Nx + j
        np.maximum.at(flat_out, flat_idx, w_f)

    return out


# ---------- Public: one CLIMADA event per .mat file (per-cell max over time) ----------
def load_gori_wind_hazard_max(
    mat_dir: str | Path,
    file_names: Optional[Iterable[str]] = None,
    *,
    resolution_deg: float = 0.25,
    pad_deg: float = 1.0,
    hazard_type: str = "TC"
) -> Hazard:
    """
    Minimal loader: one CLIMADA event per .mat file.
    Each event intensity = per-gridcell MAX wind (m/s) over all timesteps.
    """
    mat_dir = Path(mat_dir)
    files = (sorted(p for p in mat_dir.glob("*.mat") if p.is_file())
             if file_names is None else [mat_dir / fn for fn in file_names])
    if not files:
        raise FileNotFoundError(f"No .mat files found in {mat_dir}")

    # One fixed grid covering all storms/timesteps
    lon_list, lat_list = [], []
    for fp in files:
        lon, lat, _ = _read_mat_arrays(fp)
        lon_list.append(lon)
        lat_list.append(lat)
    lon_vec, lat_vec = _build_target_grid(lon_list, lat_list, resolution_deg, pad_deg)

    # Build max footprint per file (binning; no interpolation)
    event_arrays = []
    event_names = []
    for fp in files:
        lon, lat, wind = _read_mat_arrays(fp)   # (ny, nx, nt)
        ev_grid = _accumulate_max_to_grid(lon, lat, wind, lon_vec, lat_vec)
        event_arrays.append(ev_grid)
        event_names.append(fp.stem)

    # Stack and build Hazard (avoid 'event' as a dim name here!)
    intensity_stack = np.stack(event_arrays, axis=0)  # (n_events, n_lat, n_lon)

    dset = xr.Dataset(
        data_vars=dict(
            intensity=(["event_id", "latitude", "longitude"], intensity_stack)
        ),
        coords=dict(
            event_id=np.arange(intensity_stack.shape[0]),
            latitude=lat_vec,
            longitude=lon_vec,
        ),
        attrs=dict(description="TC 10m sustained wind (m/s), per-storm max over time (binned)"),
    )

    hazard = Hazard.from_xarray_raster(
        dset,
        hazard_type=hazard_type,
        intensity_unit="m/s",
        intensity="intensity",
        # Map CLIMADA's expected names to our actual coordinate dims
        coordinate_vars=dict(event="event_id", latitude="latitude", longitude="longitude"),
    )

    # Names & placeholder dates (must match number of events)
    hazard.event_name = list(event_names)
    hazard.date = np.ones((len(event_names),), dtype=int)

    # Normalize longitudes on centroids
    hazard.centroids = hazard.centroids.__class__(
        lat=hazard.centroids.lat,
        lon=u_coord.lon_normalize(hazard.centroids.lon),
    )
    return hazard


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
    

def generate_building_json(exposure_gdf, general_info_path, loss_info_path):
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
        generate_building_json(affected, gen_path, loss_path)

    
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

# def assign_damage_state(loss_ratio):
#     """Simple mapping of damage state based on loss ratio thresholds."""
#     if loss_ratio < 0.01:
#         return "None"
#     elif loss_ratio < 0.1:
#         return "Slight"
#     elif loss_ratio < 0.3:
#         return "Moderate"
#     elif loss_ratio < 0.6:
#         return "Extensive"
#     else:
#         return "Complete"

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


# def compute_damage_state_metrics(
#     exp,
#     haz,
#     ds_states=range(1,5),
#     replacement_cost_col='ReplacementCost'
# ):
#     """
#     Attach to exp.gdf for DS1–DS4:
#       - imp_ds1…imp_ds4
#       - prob_ds0…prob_ds4
#       - 'LossRatio'   (mean damage ratio)
#       - 'RepairCost'  (expected repair cost)
#       - 'most_probable_ds' (index of max probability)
#     """

#     # 1) run ImpactCalc for each DS
#     for ds in ds_states:
#         impf = ifunc.load_impact_func_set_ds(ds)
#         imp  = ImpactCalc(exp, impf, haz).impact(save_mat=True)
#         arr  = imp.imp_mat.todense().A1
#         exp.gdf[f'imp_ds{ds}'] = arr

#     # 2) slice-by-slice probabilities
#     exp.gdf['prob_ds0'] = 1 - exp.gdf['imp_ds1']
#     for ds in ds_states[:-1]:
#         exp.gdf[f'prob_ds{ds}'] = (
#             exp.gdf[f'imp_ds{ds}'] - exp.gdf[f'imp_ds{ds+1}']
#         )
#     exp.gdf['prob_ds4'] = exp.gdf['imp_ds4']

#     # 3) expected mean damage ratio (MDR)
#     mdr_map = {0:0.0,1:0.02,2:0.10,3:0.50,4:1.00}
#     exp.gdf['expected_mdd'] = sum(
#         mdr_map[i] * exp.gdf[f'prob_ds{i}'] for i in mdr_map
#     )

#     # 4) probabilistic repair cost
#     exp.gdf['probabilistic_repair_cost'] = (
#         exp.gdf['expected_mdd'] * exp.gdf[replacement_cost_col]
#     )

#     # 5) rename and drop intermediate columns
#     exp.gdf['LossRatio']  = exp.gdf['expected_mdd']
#     exp.gdf['RepairCost'] = exp.gdf['probabilistic_repair_cost']
#     exp.gdf.drop(columns=['expected_mdd','probabilistic_repair_cost'],
#                   errors=True, inplace=True)

#     # 6) most probable DS index
#     def most_probable_ds(row):
#         probs = [row[f'prob_ds{i}'] for i in range(5)]
#         return int(np.argmax(probs))
#     exp.gdf['most_probable_ds'] = exp.gdf.apply(most_probable_ds, axis=1)

#     return exp


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

# def generate_building_json(exposure_gdf, general_info_path, loss_info_path):
#     """
#     Generate two JSON files: one for general building information and one for damage information.

#     Parameters:
#         exposure_gdf (GeoDataFrame): GeoDataFrame containing building data.
#         general_info_path (str): Path to save the general building information JSON.
#         loss_info_path (str): Path to save the damage information JSON.

#     Returns:
#         tuple: Two dictionaries containing general information and loss information.
#     """
#     general_info = {}
#     loss_info = {}

#     for _, row in exposure_gdf.iterrows():
#         building_id = str(row['id'])  # Ensure it's a string for JSON keys

#         # Build general information (excluding value, impf_TC, centr_TC)
#         general_info[building_id] = {
#             "PlanArea": row['PlanArea'],
#             "NumberOfStories": row['NumberOfStories'],
#             "MedianYearBuilt": row['MedianYearBuilt'],
#             "ReplacementCost": row['ReplacementCost'],
#             "StructureType": row['StructureType'],
#             "NumberOfUnits": row['NumberOfUnits'],
#             "CensusBlock": row['CensusBlock'],
#             "CensusTract": row['CensusTract'],
#             "FootprintID": row['FootprintID'],
#             "OccupancyClass": row['OccupancyClass'],
#             "footprint_geometry": row['Footprint'],
#             "Latitude": row['geometry'].y,
#             "Longitude": row['geometry'].x,
#             "type": "Building"
#             }

#         # Build damage (loss) info
#         loss_info[building_id] = {
#             "LossRatio": row['LossRatio'],
#             "DamageState": row['DamageState'],
#             "RepairCost": row['RepairCost']
#         }

#     # Save the JSONs
#     with open(general_info_path, "w") as f:
#         json.dump(general_info, f, indent=4)

#     with open(loss_info_path, "w") as f:
#         json.dump(loss_info, f, indent=4)

#     return general_info, loss_info