import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

# import CLIMADA modules:
from climada.entity.exposures import Exposures

from modules.exposure_utils import load_exposure_from_csv, state_fp_map

data_dir = Path('/home/groups/bakerjw/smeiler/climada_data/data')
exp_dir = data_dir / 'exposure' / 'building_inventory_NAcoast'

state_list = list(state_fp_map.keys())

# Collect CSV files from all state subdirectories
csv_files = []
for state in state_list:
    state_dir = exp_dir / state
    state_csvs = [str(f) for f in state_dir.glob('*.csv')]
    csv_files.extend(state_csvs)

# Step 2: Load and combine exposures
exp = load_exposure_from_csv(csv_files)
exp.write_hdf5(data_dir / 'exposure' / 'NA_coast_exposure.hdf5')