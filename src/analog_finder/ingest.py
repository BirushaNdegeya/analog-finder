"""Geospatial data ingestion — Planetary Computer STAC + local rasters.

Retrieves and preprocesses:
  - Copernicus DEM GLO-30 (30 m global DEM, resampled to target resolution)
  - Sentinel-2 L2A cloud-free median composite (SCL-masked, 6 bands)
  - Local GeoTIFF rasters: basalt masks, annual precipitation

IMPORTANT — verify before the event
------------------------------------
- DEM collection name  : ``cop-dem-glo-30``  (confirmed working Nov 2026)
- S2 collection name   : ``sentinel-2-l2a``   (confirmed working Nov 2026)
- S2 reflectance scale : DN values in Planetary Computer baseline ≥ 04.00
  follow the formula ``(DN − 1000) / 10 000``.  Confirm that ``odc-stac``
  does **not** apply its own automatic rescaling before calling
  ``to_reflectance()``.  If it does, pass raw DN directly and skip the call.
- Band names expected  : B02, B04, B08, B11, B12, SCL — these are confirmed
  for the ``sentinel-2-l2a`` collection on Planetary Computer (Nov 2026).
"""
from __future__ import annotations

import warnings
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
import rasterio.errors

# Suppress harmless rasterio in-memory reprojection warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

from . import config as C


# ---------------------------------------------------------------------------
# STAC catalog helpers
# ---------------------------------------------------------------------------

def utm_epsg(lon: float, lat: float) -> int:
    """Return the EPSG code of the UTM zone that contains (lon, lat)."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _catalog():
    """Open the Microsoft Planetary Computer STAC catalog with signed URLs."""
    import planetary_computer
    import pystac_client
    return pystac_client.Client.open(C.STAC_URL, modifier=planetary_computer.sign_inplace)


# ---------------------------------------------------------------------------
# DEM ingestion
# ---------------------------------------------------------------------------

def load_dem(bbox: tuple[float, float, float, float], res_m: float = C.COMMON_RES_M) -> np.ndarray:
    """Download and stack the Copernicus DEM GLO-30 for a bounding box.

    The DEM is reprojected to the local UTM zone and resampled to *res_m*
    using bilinear interpolation.

    Parameters
    ----------
    bbox : (west, south, east, north) in EPSG:4326 decimal degrees.
    res_m : target pixel resolution in metres.

    Returns
    -------
    2-D float32 array — elevation in metres.

    Raises
    ------
    ValueError : if no DEM tiles are found for the bounding box.
    """
    import odc.stac
    items = list(_catalog().search(collections=[C.DEM_COLLECTION], bbox=bbox).items())
    if not items:
        raise ValueError(f"No DEM tiles found for bbox {bbox}.")
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    ds = odc.stac.load(
        items,
        bands=["data"],
        bbox=bbox,
        crs=f"EPSG:{utm_epsg(lon, lat)}",
        resolution=res_m,
        resampling="bilinear",
        chunks={},
    )
    return ds["data"].max("time", skipna=True).compute().values.astype("float32")


# ---------------------------------------------------------------------------
# Sentinel-2 reflectance helpers
# ---------------------------------------------------------------------------

def to_reflectance(dn: np.ndarray) -> np.ndarray:
    """Convert Sentinel-2 L2A integer DN to surface reflectance.

    Formula for Sentinel-2 processing baseline ≥ 04.00 (applied by ESA
    since January 2022): ``reflectance = (DN − 1000) / 10 000``

    Values are clipped to [0.0, 1.0] to avoid negative reflectances on
    dark surfaces (e.g., fresh basalt at night-time or in shadow), which
    are physically meaningless and would distort Wasserstein distances.

    .. warning::
       If your version of ``odc-stac`` already applies automatic DN→
       reflectance conversion, do **not** call this function a second time.
    """
    reflectance = (dn.astype("float32") - 1000.0) / 10_000.0
    return np.clip(reflectance, 0.0, 1.0)


def load_s2_composite(
    bbox: tuple[float, float, float, float],
    date_range: str = C.S2_DATE_RANGE,
    res_m: float = C.COMMON_RES_M,
) -> dict[str, np.ndarray]:
    """Build a cloud-free Sentinel-2 L2A median composite for a bounding box.

    The Scene Classification Layer (SCL) is used to mask:
      - 0 (No data), 1 (Saturated / Defective), 3 (Shadow),
      - 8 (Medium-probability cloud), 9 (High-probability cloud),
      - 10 (Cirrus), 11 (Snow/Ice).

    The composite is a pixel-wise median across all valid acquisitions in
    *date_range*, then converted to surface reflectance.

    Parameters
    ----------
    bbox : (west, south, east, north) in EPSG:4326.
    date_range : ISO 8601 interval string, e.g. ``"2023-01-01/2024-01-01"``.
    res_m : output pixel resolution in metres.

    Returns
    -------
    dict[str, np.ndarray] — {band_name: 2-D float32 reflectance in [0, 1]}
    Keys: "B02", "B04", "B08", "B11", "B12"

    Raises
    ------
    ValueError : if no Sentinel-2 scenes are found.

    Notes
    -----
    - DEM and Sentinel-2 may have different grid sizes if they are loaded at
      different resolutions or for a bbox whose aspect ratio does not evenly
      divide the resolution.  Always verify ``arr.shape`` before combining.
    - ``odc.stac.load`` resamples all bands to the requested *res_m*;
      B11 and B12 (20 m native) are bilinearly upsampled if *res_m* = 100 m.
    """
    import odc.stac
    items = list(
        _catalog().search(
            collections=[C.S2_COLLECTION],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": C.S2_MAX_CLOUD}},
        ).items()
    )
    if not items:
        raise ValueError(f"No Sentinel-2 scenes found for bbox {bbox} in {date_range}.")
    # Keep up to 6 least cloudy scenes to optimize bandwidth, memory, and speed
    items = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))[:6]
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    bands = ["B02", "B04", "B08", "B11", "B12", "SCL"]
    ds = odc.stac.load(
        items,
        bands=bands,
        bbox=bbox,
        crs=f"EPSG:{utm_epsg(lon, lat)}",
        resolution=res_m,
        groupby="solar_day",
        chunks={},
    )
    # Mask cloudy / invalid pixels (SCL-based)
    invalid = ds["SCL"].isin([0, 1, 3, 8, 9, 10, 11])
    out: dict[str, np.ndarray] = {}
    for b in bands[:-1]:  # exclude SCL from output
        refl_dn = ds[b].astype("float32").where(~invalid)
        median_dn = refl_dn.median("time", skipna=True).compute().values
        out[b] = to_reflectance(median_dn)
    return out


# ---------------------------------------------------------------------------
# Local raster loader
# ---------------------------------------------------------------------------

def load_local_raster(
    path: str,
    bbox: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> np.ndarray | None:
    """Read a local GeoTIFF raster (EPSG:4326) cropped to *bbox* and resampled to *shape*.

    Parameters
    ----------
    path : path to the GeoTIFF file.
    bbox : (west, south, east, north) in EPSG:4326.
    shape : (rows, cols) of the desired output array.

    Returns
    -------
    2-D float32 array, or ``None`` if the file is missing.

    Notes
    -----
    Uses ``boundless=True`` so that bounding boxes partially outside the
    raster extent are filled with the raster's nodata value rather than
    raising an exception.
    """
    try:
        with rasterio.open(path) as src:
            window = from_bounds(*bbox, transform=src.transform)
            return src.read(
                1,
                window=window,
                out_shape=shape,
                resampling=Resampling.nearest,
                boundless=True,
            ).astype("float32")
    except (rasterio.errors.RasterioIOError, FileNotFoundError):
        print(f"[WARNING] Local raster not found: {path}")
        return None


# ---------------------------------------------------------------------------
# Multi-temporal ingestion (Mode B)
# ---------------------------------------------------------------------------

def load_s2_time_pair(
    bbox: tuple[float, float, float, float],
    date_t0: str,
    date_t1: str,
    res_m: float = C.COMMON_RES_M,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fetch paired Sentinel-2 composites for baseline T0 and surveillance T1.

    Guarantees spatial alignment and common pixel resolution across time slices.

    Parameters
    ----------
    bbox : (west, south, east, north) in EPSG:4326.
    date_t0 : ISO date range string for baseline, e.g. '2021-01-01/2021-05-15'.
    date_t1 : ISO date range string for observation, e.g. '2021-06-01/2021-09-30'.
    res_m : spatial grid resolution in metres.

    Returns
    -------
    (composite_t0, composite_t1) : tuple of band dictionaries.
    """
    s2_t0 = load_s2_composite(bbox, date_range=date_t0, res_m=res_m)
    s2_t1 = load_s2_composite(bbox, date_range=date_t1, res_m=res_m)
    return s2_t0, s2_t1


def load_dem_differential(
    dem_t0_path: str,
    dem_t1_path: str,
    bbox: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load two multi-epoch DEMs and compute the elevation change raster (ΔDEM = T1 - T0).

    Used for volcanic edifice collapse, caldera volume change, or major landslide scarps.

    Parameters
    ----------
    dem_t0_path : path to baseline pre-event GeoTIFF.
    dem_t1_path : path to surveillance post-event GeoTIFF.
    bbox : bounding box in EPSG:4326.
    shape : (rows, cols) target raster shape.

    Returns
    -------
    (dem_t0, dem_t1, delta_dem) as float32 arrays, or None if either raster is missing.
    """
    dem_t0 = load_local_raster(dem_t0_path, bbox, shape)
    dem_t1 = load_local_raster(dem_t1_path, bbox, shape)
    if dem_t0 is None or dem_t1 is None:
        return None

    delta_dem = dem_t1 - dem_t0
    return dem_t0, dem_t1, delta_dem
