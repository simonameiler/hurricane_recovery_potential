import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

# import CLIMADA modules:
from climada.entity.exposures import Exposures

from modules.exposure_utils import load_exposure_from_csv

#exp_dir = Path("/home/groups/bakerjw/smeiler/climada_data/data/exposure")

data_dir = Path('/Users/simonameiler/Documents/work/03_code/repos/hurricane_recovery_potential/data')
exp_dir = data_dir / 'exposure'
county_fp = data_dir / 'US_counties.shp'

# Load data
counties = gpd.read_file(county_fp)

# Example mapping of state names to STATEFP codes (add all you need!)
state_fp_map = {
    'Alabama': '01',
    'Connecticut': '09',
    'Delaware': '10',
    'Florida': '12',
    'Georgia': '13',
    'Louisiana': '22',
    'Maine': '23',
    'Maryland': '24',
    'Massachusetts': '25',
    'Mississippi': '28',
    'NewHampshire': '33',
    'NewJersey': '34',
    'NewYork': '36',
    'NorthCarolina': '37',
    'Pennsylvania': '42',
    'RhodeIsland': '44',
    'SouthCarolina': '45',
    'Texas': '48',
    'Virginia': '51'
}

state_list = list(state_fp_map.keys())

# Collect CSV files from all state subdirectories
csv_files = []
for state in state_list:
    state_dir = exp_dir / state
    state_csvs = [str(f) for f in state_dir.glob('*.csv')]
    csv_files.extend(state_csvs)

# Step 2: Load and combine exposures
exp = load_exposure_from_csv(csv_files)
exp.write_hdf5(exp_dir / 'NA_coast_exposure.hdf5')