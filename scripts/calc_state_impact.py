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


def calc_state_impact(
    state_name: str,
    data_dir: Path = data_dir_default,
    haz_file_name: str = 'tc_ncep_reanal.hdf5',
    scaling_npz_path: str = '/home/users/smeiler/repos/hurricane_recovery_potential/data/scaling_relative.npz',
    county_region_path: str = '/home/users/smeiler/repos/hurricane_recovery_potential/data/county_region.csv',
    out_dir: str = '/home/groups/bakerjw/smeiler/climada_data/data/results/hrp_impacts_out',
    k: float = 1.0,
    scale_mode: str = 'compound',
    lower_threshold: float = 0.005,
    scaling_npz_on_path: str = None,
    out_dir_on: str = None,
):
    """Calculate tropical cyclone impact for a given state.

    The impact (CLIMADA wind damage) is computed ONCE. If a second scaling matrix
    is supplied via ``scaling_npz_on_path`` (and ``out_dir_on``), the exporter is
    run a second time on the same impact object, so the surge-ON results are
    produced without recomputing the wind impact.

    Parameters
    ----------
    state_name : str
        Name of the state to process (must match exposure file name; without suffix).
    data_dir : Path
        Base data directory containing hazard and exposure subfolders.
    haz_file_name : str
        Name of the hazard HDF5 file inside the hazard/gori directory.
    scaling_npz_path : str
        Path to scaling NPZ file (surge-OFF, committed).
    county_region_path : str
        Path to county_region CSV mapping.
    out_dir : str
        Output directory for impacts (surge-OFF).
    k, scale_mode, lower_threshold :
        Scaling parameters forwarded to exporter.
    scaling_npz_on_path : str, optional
        Path to a second scaling NPZ (surge-ON). If given, a second export is run.
    out_dir_on : str, optional
        Output directory for the surge-ON export. Required if scaling_npz_on_path is set.
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

    # Impact calculation (computed ONCE; reused for both exports)
    print("Calculating impact...")
    imp = ImpactCalc(exp, impf_set_tc, haz).impact(save_mat=True)

    # Export 1: surge-OFF (committed scaling)
    print(f"Exporting surge-OFF results to {out_dir}")
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

    # Export 2: surge-ON (optional, same impact object, no recompute)
    if scaling_npz_on_path:
        if not out_dir_on:
            raise ValueError("out_dir_on must be set when scaling_npz_on_path is provided.")
        print(f"Exporting surge-ON results to {out_dir_on}")
        export_state_and_county_results_all_events(
            exp=exp,
            imp=imp,
            scaling_npz_path=scaling_npz_on_path,
            county_region_path=county_region_path,
            out_dir=out_dir_on,
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
    parser.add_argument('--scaling-npz', default='/home/users/smeiler/repos/hurricane_recovery_potential/data/scaling_relative.npz')
    parser.add_argument('--county-region', default='/home/users/smeiler/repos/hurricane_recovery_potential/data/county_region.csv')
    parser.add_argument('--out-dir', default='/home/groups/bakerjw/smeiler/climada_data/data/results/hrp_impacts_out')
    parser.add_argument('--k', type=float, default=1.0)
    parser.add_argument('--scale-mode', default='compound')
    parser.add_argument('--lower-threshold', type=float, default=0.005)
    parser.add_argument('--scaling-npz-on', default=None,
                        help='Optional second scaling NPZ (surge-ON). If set, a second export is run.')
    parser.add_argument('--out-dir-on', default=None,
                        help='Output directory for the surge-ON export (required with --scaling-npz-on).')
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
        scaling_npz_on_path=args.scaling_npz_on,
        out_dir_on=args.out_dir_on,
    )

    if args.state:
        calc_state_impact(state_name=args.state, **common)
    else:
        # if --all specified or no state provided, run for every exposure file
        state_names = _list_state_names_from_exposure_dir(Path(args.data_dir))
        to_run = state_names
        for st in to_run:
            calc_state_impact(state_name=st, **common)
