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

def load_exposure_from_csv(csv_paths):
    """
    Load exposure data from multiple CSV files and return a combined CLIMADA Exposures object,
    with added 'County' and 'State' columns inferred from the file path, and unique building IDs.
    Also adds 'stcode' (2-digit state FIPS) and 'ccode' (3-digit county FIPS) from mapping file.

    Parameters
    ----------
    csv_paths : list of str
        List of paths to exposure inventory CSV files.

    Returns
    -------
    Exposures
        Combined CLIMADA-compatible Exposures object with state/county info, FIPS codes, and unique IDs.
    """
    # Load US counties shapefile for robust FIPS assignment
    counties_shp = Path(__file__).parent.parent / "data" / "US_counties.shp"
    counties = gpd.read_file(counties_shp)[['STATEFP','COUNTYFP','GEOID','NAME','STATE_NAME','geometry']]
    if counties.crs is None:
        counties = counties.set_crs('EPSG:4326')
    else:
        counties = counties.to_crs('EPSG:4326')
    counties.sindex  # build spatial index
    gdf_list = []

    for csv_path in csv_paths:
        # Skip incomplete_counties.csv files, but check if they have content
        path_obj = Path(csv_path)
        if path_obj.name == "incomplete_counties.csv":
            # Check if file has data beyond header
            df_check = pd.read_csv(csv_path)
            if len(df_check) > 0:
                print(f"Warning: {csv_path} contains {len(df_check)} entries that will be ignored")
            continue

        # Infer state and county name from path
        state = path_obj.parent.name
        filename = path_obj.name
        county = filename.split('_')[0]  # assumes {County}_Inventory.csv

        # Load CSV
        df = pd.read_csv(csv_path, index_col=False)

        # Skip files without required columns
        required_cols = {'Longitude', 'Latitude', 'ReplacementCost', 'StructureType', 'NumberOfUnits'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            print(f"Skipping {csv_path} (missing columns: {missing})")
            continue

        # Keep only specified columns plus required geo columns
        df = df[['Longitude', 'Latitude', 'ReplacementCost', 'StructureType', 'NumberOfUnits']]

        # Drop existing ID column if it exists (should not exist after column selection, but check anyway)
        if 'id' in df.columns:
            df = df.drop(columns='id')

        # Create geometry
        df['geometry'] = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]

        # Add county and state info
        df['County'] = county
        df['State'] = state


        # Assign stcode/ccode by spatial join to counties
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        joined = gpd.sjoin(gdf, counties, how='left', predicate='within')
        # Assign stcode and ccode from joined columns
        gdf['stcode'] = joined['STATEFP']
        gdf['ccode'] = joined['COUNTYFP']
        # Optionally warn if any points did not match a county
        n_unmatched = gdf['stcode'].isna().sum()
        if n_unmatched > 0:
            print(f"Warning: {n_unmatched} exposure points did not match any county polygon.")

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
