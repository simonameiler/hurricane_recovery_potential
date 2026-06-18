import os
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from shapely.ops import unary_union

# import CLIMADA modules:
from climada.hazard import Centroids, Hazard
from climada.entity.exposures import Exposures
from climada.entity.impact_funcs import ImpactFuncSet, ImpactFunc
from climada.engine import ImpactCalc
from climada.util import SYSTEM_DIR

from modules.impfunc_utils import IMPF_SET_TC_CAPRA, DICT_PAGER_TCIMPF_CAPRA
from modules.impact_utils import export_state_and_county_results_all_events


data_dir_default = Path('/home/groups/bakerjw/smeiler/climada_data/data')
haz_rel_path = Path('hazard') / 'tropical_cyclone' / 'gori'
exp_rel_path = Path('exposure') / 'states'

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCALING_NPZ = str(BASE_DIR / 'data' / 'scaling_relative.npz')
DEFAULT_COUNTY_REGION = str(BASE_DIR / 'data' / 'county_region.csv')
DEFAULT_OUT_DIR = str(BASE_DIR / 'data' / 'impact')


def calc_state_impact(
    state_name: str,
    data_dir: Path = data_dir_default,
    haz_file_name: str = 'tc_ncep_reanal.hdf5',
    scaling_npz_path: str = DEFAULT_SCALING_NPZ,
    county_region_path: str = DEFAULT_COUNTY_REGION,
    out_dir: str = DEFAULT_OUT_DIR,
    k: float = 1.0,
    scale_mode: str = 'compound',
    lower_threshold: float = 0.005,
):
    """Calculate tropical cyclone impact for a given state (surge-ON scaling).

    Parameters
    ----------
    state_name : str
        Name of the state to process (must match exposure file name; without suffix).
    data_dir : Path
        Base data directory containing hazard and exposure subfolders.
    haz_file_name : str
        Name of the hazard HDF5 file inside the hazard/gori directory.
    scaling_npz_path : str
        Path to scaling NPZ file (data/scaling_relative.npz, surge-ON by default).
    county_region_path : str
        Path to county_region CSV mapping.
    out_dir : str
        Output directory for impacts (data/impact/).
    k, scale_mode, lower_threshold :
        Scaling parameters forwarded to exporter.
    """
    print(f"\nCalculating impact for {state_name}...")

    # Load hazard data
    haz_dir = Path(data_dir) / haz_rel_path
    haz_file = haz_dir / haz_file_name
    print(f"Loading hazard from {haz_file}")
    haz = Hazard.from_hdf5(haz_file)

    # Load exposure data
    exp_dir = Path(data_dir) / exp_rel_path
    exp_file = exp_dir / f"{state_name.lower()}_exposure.hdf5"
    print(f"Loading exposure from {exp_file}")
    exp = Exposures.from_hdf5(exp_file)

    # attach impact function mapping column expected by ImpactCalc/impf loader
    exp.gdf['impf_TC'] = exp.gdf.apply(lambda row: DICT_PAGER_TCIMPF_CAPRA[row.StructureType], axis=1)

    # Load impact functions
    impf_set_tc = IMPF_SET_TC_CAPRA

    print("Calculating impact...")
    imp = ImpactCalc(exp, impf_set_tc, haz).impact(save_mat=True)

    print(f"Exporting results to {out_dir}")
    export_state_and_county_results_all_events(
        exp=exp,
        imp=imp,
        scaling_npz_path=scaling_npz_path,
        county_region_path=county_region_path,
        out_dir=out_dir,
        k=k,
        scale_mode=scale_mode,
        lower_threshold=lower_threshold,
    )


def _list_state_names_from_exposure_dir(data_dir: Path = data_dir_default) -> list:
    exp_dir = Path(data_dir) / exp_rel_path
    if not exp_dir.exists():
        raise FileNotFoundError(f"Exposure states directory not found: {exp_dir}")
    names = []
    for p in sorted(exp_dir.glob("*_exposure.hdf5")):
        stem = p.stem
        if stem.endswith("_exposure"):
            names.append(stem[: -len("_exposure")])
    return names


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Calculate state impacts (single state or all states)')
    parser.add_argument('--state', help='State name to process (e.g., Florida)')
    parser.add_argument('--data-dir', default=str(data_dir_default), help='Base data directory')
    parser.add_argument('--haz-file', default='tc_ncep_reanal.hdf5', help='Hazard file name inside hazard/gori')
    parser.add_argument('--scaling-npz', default=DEFAULT_SCALING_NPZ)
    parser.add_argument('--county-region', default=DEFAULT_COUNTY_REGION)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--k', type=float, default=1.0)
    parser.add_argument('--scale-mode', default='compound')
    parser.add_argument('--lower-threshold', type=float, default=0.005)
    parser.add_argument('--all', action='store_true', help='Process all states found in exposure/states directory')

    args = parser.parse_args()

    common = dict(
        data_dir=Path(args.data_dir),
        haz_file_name=args.haz_file,
        scaling_npz_path=args.scaling_npz,
        county_region_path=args.county_region,
        out_dir=args.out_dir,
        k=args.k,
        scale_mode=args.scale_mode,
        lower_threshold=args.lower_threshold,
    )

    if args.state:
        calc_state_impact(state_name=args.state, **common)
    else:
        # if --all specified or no state provided, run for every exposure file
        state_names = _list_state_names_from_exposure_dir(Path(args.data_dir))
        for st in state_names:
            calc_state_impact(state_name=st, **common)
