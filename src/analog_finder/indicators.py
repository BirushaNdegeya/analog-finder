"""Geospatial indicator computation — 8 indicators across 4 criteria.

Each public function returns a 2-D NumPy array of pixel-level values.
``compute_site_indicators`` aggregates those arrays into 1-D sample
distributions (one per indicator), ready for Wasserstein scoring.

Indicator mapping (internal ID → criterion):
  Topography   : slope_deg_arr, roughness_std  → "pente", "rugosite"
  Geology      : basalt_fraction, iron_oxide_index, clay_mineral_index
                 → "basalte", "oxydes_fer", "argiles"
  Environment  : bare_soil_fraction, annual_precip  → "sol_nu", "aridite"
  ISRU         : pit_cavity_density, accessible_basalt
                 → "cavites", "basalte_accessible"
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.morphology import reconstruction

from . import config as C


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _window_px(window_m: float, res_m: float) -> int:
    """Convert a metric window size to a pixel count (minimum 3 pixels)."""
    return max(3, int(round(window_m / res_m)))


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    """Replace NaN / Inf pixels with the array mean (in-place copy).

    Required before gradient / filter operations that do not handle NaNs.
    If the entire array is invalid, it is returned unchanged.
    """
    arr = arr.astype("float32").copy()
    invalid = ~np.isfinite(arr)
    if invalid.any() and not invalid.all():
        arr[invalid] = float(np.nanmean(arr[~invalid]))
    return arr


# ---------------------------------------------------------------------------
# Elementary spatial operations
# ---------------------------------------------------------------------------

def slope_deg(dem: np.ndarray, res_m: float) -> np.ndarray:
    """Compute terrain slope in degrees from a DEM (metres, uniform spacing).

    Parameters
    ----------
    dem : 2-D float32 array — elevation in metres.
    res_m : pixel size in metres (must be the same in both directions).

    Returns
    -------
    2-D float32 array — slope angle in degrees [0°, 90°].
    """
    gy, gx = np.gradient(dem, res_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def local_std(arr: np.ndarray, window_px: int) -> np.ndarray:
    """Local standard deviation of *arr* computed over a square sliding window.

    Uses the identity Var = E[X²] − E[X]² and a uniform (box) filter, which
    runs in O(n) regardless of window size.

    Parameters
    ----------
    arr : 2-D float32 array.
    window_px : side length of the square kernel in pixels (odd preferred).

    Returns
    -------
    2-D float32 array — local std, non-negative (numerical clipping applied).
    """
    mean = uniform_filter(arr, window_px, mode="nearest")
    mean_sq = uniform_filter(arr ** 2, window_px, mode="nearest")
    return np.sqrt(np.clip(mean_sq - mean ** 2, 0.0, None))


def local_fraction(mask: np.ndarray, window_px: int) -> np.ndarray:
    """Fraction of *True* (or non-zero) pixels in a sliding square window.

    Parameters
    ----------
    mask : 2-D boolean or float32 array — positive values count as 1.
    window_px : side length of the square kernel in pixels.

    Returns
    -------
    2-D float32 array — values in [0, 1].
    """
    return uniform_filter(mask.astype("float32"), window_px, mode="nearest")


def closed_depressions(dem: np.ndarray, min_depth_m: float = C.PIT_MIN_DEPTH_M) -> np.ndarray:
    """Detect closed depressions (pit / skylight candidates) in a DEM.

    Uses morphological reconstruction (flood-fill from the border) to compute
    a filled DEM; pixels where filled − original ≥ *min_depth_m* are
    classified as closed depressions.

    Parameters
    ----------
    dem : 2-D float32 array — elevation in metres (NaN already filled).
    min_depth_m : minimum depression depth to flag (metres).

    Returns
    -------
    2-D boolean array — True where a closed depression is detected.
    """
    seed = dem.copy()
    seed[1:-1, 1:-1] = dem.max()
    filled = reconstruction(seed, dem, method="erosion")
    return (filled - dem) >= min_depth_m


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------

def sample(arr: np.ndarray, n: int = C.MAX_SAMPLES, seed: int = 0) -> np.ndarray:
    """Draw a flat 1-D random subsample of finite pixel values.

    Parameters
    ----------
    arr : array of any shape — NaN / Inf values are excluded.
    n : maximum sample size.
    seed : random seed for reproducibility.

    Returns
    -------
    1-D float32 array — length ≤ *n*.
    """
    v = np.asarray(arr, dtype="float32").ravel()
    v = v[np.isfinite(v)]
    if v.size > n:
        v = np.random.default_rng(seed).choice(v, n, replace=False)
    return v


# ---------------------------------------------------------------------------
# Composite indicator sets
# ---------------------------------------------------------------------------

def dem_indicators(dem: np.ndarray, res_m: float = C.COMMON_RES_M) -> dict:
    """Compute slope and roughness distributions from a DEM.

    Parameters
    ----------
    dem : 2-D float32 array — elevation in metres.
    res_m : pixel resolution in metres.

    Returns
    -------
    {"pente": 1-D array, "rugosite": 1-D array}
    """
    slope = slope_deg(_fill_nan(dem), res_m)
    roughness = local_std(slope, _window_px(C.ROUGHNESS_WINDOW_M, res_m))
    return {
        "pente": sample(slope),
        "rugosite": sample(roughness),
    }


def compute_site_indicators(
    dem100: np.ndarray | None = None,
    dem30: np.ndarray | None = None,
    s2: dict | None = None,
    basalt_mask: np.ndarray | None = None,
    precip: np.ndarray | None = None,
) -> dict:
    """Compute all available indicators for a single terrestrial site.

    Each indicator is returned as a flat 1-D float32 sample array suitable
    for Wasserstein distance computation.  Indicators whose input data are
    missing are simply omitted from the output dict (they are **not** set
    to 0.0 or any other placeholder value).

    Parameters
    ----------
    dem100 : DEM resampled to 100 m — used for slope and roughness.
    dem30 : DEM at native 30 m — used for pit detection (better resolution).
    s2 : Sentinel-2 band dictionary {band_name: 2-D float32 reflectance}.
         Expected bands: B02, B04, B08, B11, B12 (already in [0, 1]).
    basalt_mask : binary raster — 1 where basalt is mapped, 0 elsewhere.
    precip : annual precipitation raster in mm yr⁻¹.

    Returns
    -------
    dict[str, np.ndarray] — {indicator_id: 1-D sample array}
    """
    out: dict[str, np.ndarray] = {}
    win = _window_px(C.LOCAL_FRACTION_WINDOW_M, C.COMMON_RES_M)

    # --- Topography ---------------------------------------------------------
    if dem100 is not None:
        out.update(dem_indicators(dem100, C.COMMON_RES_M))

    # Pit / skylight density — computed at native DEM resolution (30 m)
    if dem30 is not None:
        pits = closed_depressions(_fill_nan(dem30))
        out["cavites"] = sample(local_fraction(pits, _window_px(C.LOCAL_FRACTION_WINDOW_M, 30)))

    # --- Geology & Environment (Sentinel-2) ---------------------------------
    sol_nu: np.ndarray | None = None
    if s2 is not None:
        valid = np.isfinite(s2["B04"]) & np.isfinite(s2["B08"])

        # Bare soil fraction — NDVI < threshold
        ndvi = (s2["B08"] - s2["B04"]) / (s2["B08"] + s2["B04"] + 1e-6)
        sol_nu = local_fraction((ndvi < C.NDVI_BARE_THRESHOLD) & valid, win)
        sol_nu[~valid] = np.nan
        out["sol_nu"] = sample(sol_nu)

        # Iron-oxide index: Red / Blue (B04 / B02) — elevated on basalt / iron soils
        out["oxydes_fer"] = sample(s2["B04"] / (s2["B02"] + 1e-6))

        # Clay mineral index: SWIR1 / SWIR2 (B11 / B12)
        out["argiles"] = sample(s2["B11"] / (s2["B12"] + 1e-6))

    # --- Environment — annual precipitation ---------------------------------
    if precip is not None:
        # Lower precipitation → higher aridity → closer to lunar/Martian surface
        out["aridite"] = sample(precip)

    # --- ISRU — basalt cover and accessible basalt --------------------------
    if basalt_mask is not None:
        bas = local_fraction(basalt_mask > 0, win)
        out["basalte"] = sample(bas)

        # Accessible basalt = basalt fraction × bare-soil fraction
        # (assumes basalt exposed only where there is no vegetation cover)
        if sol_nu is not None and bas.shape == sol_nu.shape:
            out["basalte_accessible"] = sample(bas * sol_nu)

    return out


# ---------------------------------------------------------------------------
# Multi-temporal differential indicator computation (Mode B)
# ---------------------------------------------------------------------------

def compute_differential_indicators(
    dem_t0: np.ndarray | None = None,
    dem_t1: np.ndarray | None = None,
    s2_t0: dict | None = None,
    s2_t1: dict | None = None,
    basalt_t0: np.ndarray | None = None,
    basalt_t1: np.ndarray | None = None,
    precip_t0: np.ndarray | None = None,
    precip_t1: np.ndarray | None = None,
) -> dict:
    """Compute indicator distributions at T0 and T1, plus pixel-level differencing.

    Returns
    -------
    {
        "t0": dict of 1-D indicator sample distributions at baseline,
        "t1": dict of 1-D indicator sample distributions at surveillance,
        "pixel_diffs": dict of 1-D pixel-level signed differentials (T1 - T0),
    }
    """
    inds_t0 = compute_site_indicators(dem100=dem_t0, s2=s2_t0, basalt_mask=basalt_t0, precip=precip_t0)
    inds_t1 = compute_site_indicators(dem100=dem_t1, s2=s2_t1, basalt_mask=basalt_t1, precip=precip_t1)

    pixel_diffs: dict[str, np.ndarray] = {}

    # Pixel-level topographic slope shift
    if dem_t0 is not None and dem_t1 is not None and dem_t0.shape == dem_t1.shape:
        slope0 = slope_deg(_fill_nan(dem_t0), C.COMMON_RES_M)
        slope1 = slope_deg(_fill_nan(dem_t1), C.COMMON_RES_M)
        pixel_diffs["delta_pente"] = sample(slope1 - slope0)
        pixel_diffs["delta_dem"] = sample(dem_t1 - dem_t0)

    # Pixel-level vegetation change (Delta NDVI)
    if s2_t0 is not None and s2_t1 is not None:
        valid0 = np.isfinite(s2_t0["B04"]) & np.isfinite(s2_t0["B08"])
        valid1 = np.isfinite(s2_t1["B04"]) & np.isfinite(s2_t1["B08"])
        common_valid = valid0 & valid1

        ndvi0 = (s2_t0["B08"] - s2_t0["B04"]) / (s2_t0["B08"] + s2_t0["B04"] + 1e-6)
        ndvi1 = (s2_t1["B08"] - s2_t1["B04"]) / (s2_t1["B08"] + s2_t1["B04"] + 1e-6)

        delta_ndvi = ndvi1 - ndvi0
        delta_ndvi[~common_valid] = np.nan
        pixel_diffs["delta_ndvi"] = sample(delta_ndvi)

    return {
        "t0": inds_t0,
        "t1": inds_t1,
        "pixel_diffs": pixel_diffs,
    }
