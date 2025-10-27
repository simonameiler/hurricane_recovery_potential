import os
from pathlib import Path
import pandas as pd
import geopandas as gpd

# import CLIMADA modules:
from climada.entity.exposures import Exposures
from modules.exposure_utils import load_exposure_from_csv, state_fp_map

def process_state(state_name):
    """Process exposure data for a single state and save to HDF5.
    
    Parameters
    ----------
    state_name : str
        Name of the state to process (must match directory name)
    """
    data_dir = Path('/home/groups/bakerjw/smeiler/climada_data/data')
    exp_dir = data_dir / 'exposure' / 'building_inventory_NAcoast'
    out_dir = data_dir / 'exposure' / 'states'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of CSV files for this state
    state_dir = exp_dir / state_name
    state_csvs = [str(f) for f in state_dir.glob('*.csv')]
    
    if not state_csvs:
        print(f"No CSV files found for {state_name}")
        return
    
    print(f"\nProcessing {state_name}...")
    print(f"Found {len(state_csvs)} CSV files")
    
    # Load and combine exposures for this state
    exp = load_exposure_from_csv(state_csvs)
    
    # Save to HDF5
    out_file = out_dir / f"{state_name.lower()}_exposure.hdf5"
    print(f"Saving to {out_file}")
    exp.write_hdf5(out_file)
    
    # Clean up memory
    import gc
    del exp
    gc.collect()
    
    print(f"Completed {state_name}")

def main():
    """Process all states in parallel using multiprocessing."""
    from multiprocessing import Pool
    from functools import partial
    
    # Get list of all states
    state_list = list(state_fp_map.keys())
    print(f"Processing {len(state_list)} states: {', '.join(state_list)}")
    
    # Process each state
    with Pool() as pool:
        pool.map(process_state, state_list)

if __name__ == "__main__":
    main()