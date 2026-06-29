from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List

import numpy as np
from scipy.io import loadmat, whosmat
from scipy.interpolate import RegularGridInterpolator
from scipy import sparse
from datetime import date

from climada.hazard.base import Hazard
from climada.hazard.centroids.centr import Centroids
import climada.util.coordinates as u_coord

# ---------- (optional) quick inspector ----------
def list_mat_vars(path: str | Path):
    """Print (name, shape, dtype) for all variables in a .mat file."""
    print(whosmat(str(path)))

# ---------- read wind .mat (v5 or v7.3) ----------
def _read_wind_mat(fp: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lon, lat, wind) as (ny, nx, nt). Variables: PLONG_SAVE, PLAT_SAVE, wind_grid (m/s)."""
    def _ensure_3d(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a); a = np.squeeze(a)
        if a.ndim == 2: a = a[:, :, None]
        if a.ndim != 3: raise ValueError(f"{fp.name}: expected 2D/3D arrays; got {a.shape}")
        return a

    md = loadmat(fp, squeeze_me=False, struct_as_record=False)
    def g(n):
        if n not in md: raise KeyError(f"{n} not found in {fp.name}")
        return md[n]
    lon = g("PLONG_SAVE"); lat = g("PLAT_SAVE"); wind = g("wind_grid")

    lon = _ensure_3d(lon); lat = _ensure_3d(lat); wind = _ensure_3d(wind)
    if lon.shape != lat.shape or lat.shape != wind.shape:
        raise ValueError(f"{fp.name}: lon{lon.shape}, lat{lat.shape}, wind{wind.shape} differ.")
    return lon, lat, wind


# ---------- build one common output grid ----------
def _build_target_grid(
    lon_list: List[np.ndarray],
    lat_list: List[np.ndarray],
    resolution_deg: float,
    pad_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    lon_min = float(np.nanmin([np.nanmin(a) for a in lon_list]))
    lon_max = float(np.nanmax([np.nanmax(a) for a in lon_list]))
    lat_min = float(np.nanmin([np.nanmin(a) for a in lat_list]))
    lat_max = float(np.nanmax([np.nanmax(a) for a in lat_list]))

    # unwrap lon span to avoid dateline issues
    lon_pair = np.array([lon_min, lon_max], float)
    lon_unw = np.unwrap(np.deg2rad(lon_pair), discont=np.deg2rad(180.0))
    lon_min_u, lon_max_u = np.rad2deg([lon_unw.min(), lon_unw.max()])

    lon_vec = np.arange(lon_min_u - pad_deg, lon_max_u + pad_deg + 1e-12, resolution_deg)
    lat_vec = np.arange(lat_min   - pad_deg, lat_max   + pad_deg   + 1e-12, resolution_deg)

    lon_vec = u_coord.lon_normalize(lon_vec)
    return lon_vec, lat_vec

def _nearest_resample_one(lon2d, lat2d, wind2d, lon_vec, lat_vec):
    """Nearest-cell resample to fixed grid (robust fallback, no Qhull)."""
    Ny, Nx = len(lat_vec), len(lon_vec)
    out = np.zeros((Ny, Nx), float)
    # bin edges (midpoints)
    lon_edges = np.empty(Nx + 1); lat_edges = np.empty(Ny + 1)
    lon_edges[1:-1] = 0.5 * (lon_vec[1:] + lon_vec[:-1])
    lat_edges[1:-1] = 0.5 * (lat_vec[1:] + lat_vec[:-1])
    lon_edges[0]  = lon_vec[0]  - (lon_vec[1]  - lon_vec[0])  / 2
    lon_edges[-1] = lon_vec[-1] + (lon_vec[-1] - lon_vec[-2]) / 2
    lat_edges[0]  = lat_vec[0]  - (lat_vec[1]  - lat_vec[0])  / 2
    lat_edges[-1] = lat_vec[-1] + (lat_vec[-1] - lat_vec[-2]) / 2

    m = np.isfinite(lon2d) & np.isfinite(lat2d) & np.isfinite(wind2d)
    if not np.any(m):
        return out
    lon_f = lon2d[m].ravel(); lat_f = lat2d[m].ravel(); w_f = wind2d[m].ravel()
    j = np.searchsorted(lon_edges, lon_f, side="right") - 1
    i = np.searchsorted(lat_edges, lat_f, side="right") - 1
    j = np.clip(j, 0, Nx - 1); i = np.clip(i, 0, Ny - 1)
    flat = out.ravel(); idx = i * Nx + j
    np.maximum.at(flat, idx, w_f)
    return out

def _interp_timestep_bilinear(lon2d, lat2d, wind2d, lon_vec, lat_vec):
    """
    Robust bilinear remap from rectilinear-ish 2D meshes to fixed lon_vec/lat_vec.
    Enforces strict monotonic axes via sort + unique; falls back to nearest if needed.
    """
    # 1) representative 1D axes from 2D meshes
    lon_axis = np.nanmean(lon2d, axis=0)   # (nx,)
    lat_axis = np.nanmean(lat2d, axis=1)   # (ny,)
    # drop NaNs
    lon_axis = lon_axis[np.isfinite(lon_axis)]
    lat_axis = lat_axis[np.isfinite(lat_axis)]
    if lon_axis.size < 2 or lat_axis.size < 2:
        return _nearest_resample_one(lon2d, lat2d, wind2d, lon_vec, lat_vec)

    # 2) sort axes and reorder the field
    j_sort = np.argsort(lon_axis)
    i_sort = np.argsort(lat_axis)
    lon_sorted = lon_axis[j_sort]
    lat_sorted = lat_axis[i_sort]
    w_sorted = wind2d[np.ix_(i_sort, j_sort)]

    # 3) enforce strictly increasing by deduplicating
    lon_unique, j_keep = np.unique(lon_sorted, return_index=True)
    lat_unique, i_keep = np.unique(lat_sorted, return_index=True)
    w_unique = w_sorted[np.ix_(i_keep, j_keep)]

    # guard: need at least 2 points per axis
    if lon_unique.size < 2 or lat_unique.size < 2:
        return _nearest_resample_one(lon2d, lat2d, wind2d, lon_vec, lat_vec)

    # 4) build interpolator
    try:
        rgi = RegularGridInterpolator(
            (lat_unique, lon_unique), w_unique,
            bounds_error=False, fill_value=0.0
        )
        LONt, LATt = np.meshgrid(lon_vec, lat_vec)
        out = rgi(np.column_stack((LATt.ravel(), LONt.ravel()))).reshape(LATt.shape)
        return np.nan_to_num(out, nan=0.0)
    except Exception:
        # any residual monotonicity/value issues → safe fallback
        return _nearest_resample_one(lon2d, lat2d, wind2d, lon_vec, lat_vec)


# ---------- build sparse intensity (events x centroids) ----------
def _stack_events_to_csr(event_grids: List[np.ndarray]) -> sparse.csr_matrix:
    rows = []; cols = []; vals = []
    for e_idx, grid in enumerate(event_grids):
        flat = grid.ravel()
        nz = flat > 0.0
        if not np.any(nz): continue
        cols_e = np.nonzero(nz)[0]
        rows_e = np.full(cols_e.size, e_idx, dtype=int)
        rows.append(rows_e); cols.append(cols_e); vals.append(flat[nz])
    if rows:
        rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    else:
        rows = np.array([], int); cols = np.array([], int); vals = np.array([], float)
    n_events = len(event_grids); n_cells = event_grids[0].size if n_events else 0
    return sparse.coo_matrix((vals, (rows, cols)), shape=(n_events, n_cells)).tocsr()


# ---------- read dates for the specific event IDs you loaded ----------
def _read_track_dates_for_ids(track_mat_path: str | Path, event_ids: List[int]) -> Dict[int, int]:
    """
    Return {event_id -> ordinal} using first valid (year, month, day) per event.
    Uses lower-case names present in your track file: year100, month100, day100.
    """
    md = loadmat(track_mat_path, squeeze_me=False, struct_as_record=False)
    L = {k.lower(): k for k in md.keys()}

    year = np.squeeze(np.asarray(md[L["year100"]]))   # (N,1) or (N,T)
    mon  = np.squeeze(np.asarray(md[L["month100"]]))  # (N,T)
    day  = np.squeeze(np.asarray(md[L["day100"]]))    # (N,T)

    if year.ndim == 1: year = year[:, None]
    if mon.ndim  == 1: mon  = mon[:, None]
    if day.ndim  == 1: day  = day[:, None]

    ord_map: Dict[int, int] = {}
    for eid in event_ids:
        idx = eid - 1  # filenames 0001 -> row 0
        y_row = year[idx, :]
        m_row = mon[idx, :]
        d_row = day[idx, :]
        mask = np.isfinite(y_row) & np.isfinite(m_row) & np.isfinite(d_row)
        if not np.any(mask):
            ord_map[eid] = 1
            continue
        t0 = int(np.argmax(mask))
        y = int(y_row[t0]); m = int(m_row[t0]); d = int(d_row[t0])
        try:
            ord_map[eid] = date(y, m, d).toordinal()
        except ValueError:
            ord_map[eid] = 1
    return ord_map

def _read_catalog_freq(track_mat_path: str | Path) -> float:
    md = loadmat(track_mat_path, squeeze_me=False, struct_as_record=False)
    keys = {k.lower(): k for k in md.keys()}
    if "freq" not in keys:
        raise KeyError(f"'freq' not found in {Path(track_mat_path).name}")
    return float(np.asarray(md[keys["freq"]]).squeeze())

def _read_total_events(track_mat_path: str | Path) -> int:
    md = loadmat(track_mat_path, squeeze_me=False, struct_as_record=False)
    keys = {k.lower(): k for k in md.keys()}
    # Prefer an (N, T) array to get N directly
    for nm in ("month100", "day100", "lat100", "lon100"):
        if nm in keys:
            arr = np.asarray(md[keys[nm]]).squeeze()
            return int(arr.shape[0]) if arr.ndim >= 2 else int(arr.size)
    # Fallback: year100 is often (N, 1)
    if "year100" in keys:
        return int(np.asarray(md[keys["year100"]]).squeeze().shape[0])
    raise RuntimeError("Could not determine N_total from track file.")


# ---------- main: build CLIMADA Hazard ----------
# ---------- read blended (Fuad) per-storm .mat ----------
def _read_blended_wind_mat(fp: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lon3d, lat3d, wind3d) as (ny, nx, nt) from a blended Fuad .mat file.

    Variables: PLONG_high, PLAT_high, U_merge, V_merge.
    Wind speed = sqrt(U_merge**2 + V_merge**2), m/s.
    """
    md = loadmat(str(fp), squeeze_me=False, struct_as_record=False)
    def g(n):
        if n not in md:
            raise KeyError(f"{n} not found in {fp.name}")
        return np.asarray(md[n], dtype=float)
    lon = g("PLONG_high")
    lat = g("PLAT_high")
    U   = g("U_merge")
    V   = g("V_merge")
    wind = np.sqrt(U**2 + V**2)
    for arr, nm in ((lon, "PLONG_high"), (lat, "PLAT_high"), (wind, "wind_speed")):
        if arr.ndim != 3:
            raise ValueError(f"{fp.name}: expected 3D array for {nm}, got {arr.shape}")
    if lon.shape != lat.shape or lat.shape != wind.shape:
        raise ValueError(f"{fp.name}: shape mismatch lon{lon.shape}, lat{lat.shape}, wind{wind.shape}")
    return lon, lat, wind


def _read_track_dates_for_atcf(
    track_mat_path: str | Path, atcf_ids: List[str]
) -> Dict[str, int]:
    """Return {atcf_id -> date_ordinal} by looking up each ID in stname100.

    Uses the first Best Track timestep with non-zero year/month/day.
    """
    md = loadmat(str(track_mat_path), squeeze_me=False, struct_as_record=False)
    L = {k.lower(): k for k in md.keys()}

    stnames_raw = np.asarray(md[L["stname100"]]).ravel()  # (N,)

    def _extract_name(x) -> str:
        if isinstance(x, np.ndarray):
            return str(x.ravel()[0]).strip()
        return str(x).strip()

    name_to_row: Dict[str, int] = {_extract_name(s): i for i, s in enumerate(stnames_raw)}

    year_arr = np.asarray(md[L["year100"]])   # (N, T)
    mon_arr  = np.asarray(md[L["month100"]])
    day_arr  = np.asarray(md[L["day100"]])

    if year_arr.ndim == 1:
        year_arr = year_arr[:, None]
        mon_arr  = mon_arr[:, None]
        day_arr  = day_arr[:, None]

    ord_map: Dict[str, int] = {}
    for atcf_id in atcf_ids:
        row = name_to_row.get(atcf_id)
        if row is None:
            ord_map[atcf_id] = 1
            continue
        y_row = year_arr[row, :].astype(float)
        m_row = mon_arr[row, :].astype(float)
        d_row = day_arr[row, :].astype(float)
        mask = (y_row > 0) & (m_row > 0) & (d_row > 0)
        if not np.any(mask):
            ord_map[atcf_id] = 1
            continue
        t0 = int(np.argmax(mask))
        try:
            ord_map[atcf_id] = date(int(y_row[t0]), int(m_row[t0]), int(d_row[t0])).toordinal()
        except ValueError:
            ord_map[atcf_id] = 1
    return ord_map


# ---------- main: build CLIMADA Hazard from blended (Fuad) files ----------
def load_tc_hazard_from_blended_mats(
    blended_dir: str | Path,
    track_mat_path: str | Path,
    atcf_ids: List[str],
    *,
    resolution_deg: float = 0.25,
    pad_deg: float = 1.0,
    haz_type: str = "TC",
) -> Tuple["Hazard", List[str]]:
    """Build a CLIMADA Hazard from per-storm blended wind-field .mat files.

    Files are named <ATCF_ID>.mat (e.g., AL142018.mat) and contain:
      PLONG_high, PLAT_high : (ny, nx, nt) storm-centred lon/lat grids
      U_merge, V_merge      : (ny, nx, nt) blended u/v wind, m/s, 10 m 1-min sustained

    Parameters
    ----------
    blended_dir      : directory holding per-storm .mat files
    track_mat_path   : Best Track .mat for storm dates (stname100, year100, month100, day100)
    atcf_ids         : ordered list of ATCF IDs to load (e.g. ['AL142018', 'AL132020', ...])
    resolution_deg   : target grid resolution — must match the probabilistic hazard (0.25°)
    pad_deg          : padding around the union footprint

    Returns
    -------
    haz     : CLIMADA Hazard; frequency = 1.0 for every event; event_name = ATCF ID
    missing : ATCF IDs requested but not found in blended_dir (skipped)
    """
    blended_dir     = Path(blended_dir)
    track_mat_path  = Path(track_mat_path)

    found:   List[Tuple[str, Path]] = []
    missing: List[str]              = []
    for aid in atcf_ids:
        fp = blended_dir / f"{aid}.mat"
        if fp.exists():
            found.append((aid, fp))
        else:
            missing.append(aid)

    if missing:
        print(f"  [load_tc_hazard_from_blended_mats] MISSING from blended dir: {missing}")
    if not found:
        raise FileNotFoundError(f"No matching .mat files in {blended_dir}")

    # shared target grid across all storms
    lon_list, lat_list = [], []
    for _aid, fp in found:
        lon, lat, _ = _read_blended_wind_mat(fp)
        lon_list.append(lon)
        lat_list.append(lat)
    lon_vec, lat_vec = _build_target_grid(lon_list, lat_list, resolution_deg, pad_deg)

    # per-storm max footprints
    event_grids:    List[np.ndarray] = []
    event_atcf_ids: List[str]        = []

    for aid, fp in found:
        lon3d, lat3d, wind3d = _read_blended_wind_mat(fp)
        nt = wind3d.shape[2]
        regr_max = None
        for t in range(nt):
            w_t = _interp_timestep_bilinear(
                lon3d[:, :, t], lat3d[:, :, t], wind3d[:, :, t],
                lon_vec, lat_vec,
            )
            regr_max = w_t if regr_max is None else np.maximum(regr_max, w_t)
        event_grids.append(regr_max)
        event_atcf_ids.append(aid)

    # dates from track file via ATCF ID lookup
    ord_map     = _read_track_dates_for_atcf(track_mat_path, event_atcf_ids)
    event_dates = np.array([ord_map.get(aid, 1) for aid in event_atcf_ids], dtype=int)

    # centroids
    LON, LAT   = np.meshgrid(lon_vec, lat_vec)
    centroids  = Centroids(lat=LAT.ravel(), lon=u_coord.lon_normalize(LON.ravel()))

    intensity_csr = _stack_events_to_csr(event_grids)
    n_events      = len(event_atcf_ids)

    haz = Hazard(
        haz_type=haz_type,
        units="m/s",
        centroids=centroids,
        event_id=np.arange(1, n_events + 1, dtype=int),
        frequency=np.ones(n_events, dtype=float),   # historical: each event occurs once per year
        frequency_unit="1/year",
        event_name=list(event_atcf_ids),
        date=event_dates,
        intensity=intensity_csr,
    )
    haz.centroids = Centroids(
        lat=haz.centroids.lat, lon=u_coord.lon_normalize(haz.centroids.lon)
    )

    print(
        f"  [load_tc_hazard_from_blended_mats] {n_events} events loaded, "
        f"{len(lon_vec)*len(lat_vec):,} centroids at {resolution_deg}°"
    )
    return haz, missing


def load_tc_hazard_from_wind_mats(
    mat_dir: str | Path,
    track_mat_path: str | Path,
    *,
    resolution_deg: float = 0.25,
    pad_deg: float = 1.0,
    haz_type: str = "TC",
) -> Hazard:
    """
    One event per numbered wind .mat (e.g., 0001.mat), footprint = per-cell max wind over time.
    - units: m/s (10 m sustained, already in wind_grid)
    - frequency: per-event = catalog 'freq' / total number of storms (as if full catalog loaded)
    - date: first valid (year, month, day) from track .mat for the loaded event IDs
    - fraction: left empty -> interpreted by CLIMADA as 1 everywhere
    """
    mat_dir = Path(mat_dir)
    track_mat_path = Path(track_mat_path)

    files = sorted(p for p in mat_dir.glob("*.mat") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .mat files found in {mat_dir}")

    # shared output grid across all events
    lon_list, lat_list = [], []
    for fp in files:
        lon, lat, _ = _read_wind_mat(fp)
        lon_list.append(lon); lat_list.append(lat)
    lon_vec, lat_vec = _build_target_grid(lon_list, lat_list, resolution_deg, pad_deg)

    # per-event max footprints
    event_grids: List[np.ndarray] = []
    event_names: List[str] = []
    event_ids: List[int] = []
    for fp in files:
        # read this storm's moving grids
        lon, lat, wind = _read_wind_mat(fp)   # (ny, nx, nt)

        # per-storm max over time on the fixed grid
        regr_max = None
        for t in range(wind.shape[2]):
            w_t = _interp_timestep_bilinear(
                lon[:, :, t], lat[:, :, t], wind[:, :, t],
                lon_vec, lat_vec
            )
            regr_max = w_t if regr_max is None else np.maximum(regr_max, w_t)

        event_grids.append(regr_max)
        stem = fp.stem
        event_names.append(stem)
        try:
            event_ids.append(int(stem))
        except ValueError:
            event_ids.append(len(event_ids) + 1)

    # dates for the loaded event IDs
    ord_map = _read_track_dates_for_ids(track_mat_path, event_ids)
    event_dates = np.array([ord_map.get(eid, 1) for eid in event_ids], dtype=int)

    # Frequency: set as if the full catalog were loaded
    catalog_freq = _read_catalog_freq(track_mat_path)   # events per year (scalar)
    n_total      = _read_total_events(track_mat_path)   # e.g., 5018
    per_event    = catalog_freq / float(n_total)        # constant for every catalog storm
    freq         = np.full(len(event_ids), per_event, dtype=float)

    # centroids from fixed grid
    LON, LAT = np.meshgrid(lon_vec, lat_vec)
    centroids = Centroids(lat=LAT.ravel(), lon=u_coord.lon_normalize(LON.ravel()))

    # intensity sparse matrix (events x centroids)
    intensity_csr = _stack_events_to_csr(event_grids)

    haz = Hazard(
        haz_type=haz_type,
        units="m/s",
        centroids=centroids,
        event_id=np.array(event_ids, dtype=int),
        frequency=freq,
        frequency_unit="1/year",
        event_name=list(event_names),
        date=event_dates,
        intensity=intensity_csr,
        # fraction left empty -> interpreted as all ones
    )

    # ensure normalized longitudes
    haz.centroids = Centroids(lat=haz.centroids.lat, lon=u_coord.lon_normalize(haz.centroids.lon))
    return haz
