"""Planetary STAC Ingestion Engine & Cross-Body Data Bridge.

Provides programmatic access to altimetry (DEM) and optical/multispectral
imagery across Earth, Moon, and Mars using open STAC (SpatioTemporal Asset Catalog)
APIs and Cloud-Optimized GeoTIFFs (COGs).

Data Providers:
  - Earth : Microsoft Planetary Computer (Copernicus DEM GLO-30 & Sentinel-2 L2A)
  - Moon  : USGS Astrogeology STAC (Kaguya TC DTMs, LOLA altimetry, Orthoimages)
  - Mars  : USGS Astrogeology STAC (MRO CTX DTMs, HiRISE meter-scale DTMs, THEMIS IR)
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import numpy as np
import rasterio
from rasterio.windows import Window

from . import config as C


# ---------------------------------------------------------------------------
# STAC Search Functions
# ---------------------------------------------------------------------------

def search_planetary_stac(
    body: str = "moon",
    bbox: Optional[List[float]] = None,
    collection: Optional[str] = None,
    limit: int = 5,
    timeout_sec: int = 10,
) -> List[Dict[str, Any]]:
    """Search lunar and Martian datasets via USGS Astrogeology STAC API.

    Parameters
    ----------
    body : 'moon' or 'mars'
    bbox : [min_lon, min_lat, max_lon, max_lat]
    collection : specific STAC collection ID (optional)
    limit : maximum number of items to return

    Returns
    -------
    List of parsed item dicts with direct COG asset hrefs.
    """
    body_lower = body.lower()
    target_collection = collection
    if not target_collection:
        if body_lower == "moon":
            target_collection = C.LUNAR_DEM_COLLECTION
        elif body_lower == "mars":
            target_collection = C.MARS_DEM_COLLECTION

    payload: Dict[str, Any] = {"limit": limit}
    if target_collection:
        payload["collections"] = [target_collection]
    if bbox and len(bbox) == 4:
        payload["bbox"] = bbox

    url = f"{C.PLANETARY_STAC_URL}/search"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "AnalogFinder/1.1 (GeoCongoAI)"},
    )

    items_out = []
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            features = data.get("features", [])
            for f in features:
                assets = f.get("assets", {})
                # Resolve DEM and Optical asset URLs
                dem_href = None
                for key in ["dtm", "dem", "geoid_adjusted_dem", "data"]:
                    if key in assets:
                        dem_href = assets[key].get("href")
                        break

                optical_href = None
                for key in ["orthoimage", "orthoimage_left", "orthoimage_right", "browse"]:
                    if key in assets:
                        optical_href = assets[key].get("href")
                        break

                items_out.append({
                    "id": f.get("id"),
                    "body": body_lower,
                    "collection": f.get("collection"),
                    "bbox": f.get("bbox"),
                    "dem_href": dem_href,
                    "optical_href": optical_href,
                    "all_assets": list(assets.keys()),
                })
    except Exception as exc:
        print(f"[WARNING] Planetary STAC query failed ({body}): {exc}")

    return items_out


def search_earth_stac(
    bbox: List[float],
    data_type: str = "dem",
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Search terrestrial datasets via Microsoft Planetary Computer STAC."""
    try:
        from .ingest import _catalog
        coll = C.DEM_COLLECTION if data_type == "dem" else C.S2_COLLECTION
        search = _catalog().search(collections=[coll], bbox=bbox, limit=limit)
        items = list(search.items())
        return [
            {
                "id": item.id,
                "body": "earth",
                "collection": coll,
                "bbox": item.bbox,
                "assets": list(item.assets.keys()),
                "properties": {
                    k: item.properties.get(k)
                    for k in ["datetime", "eo:cloud_cover"]
                    if k in item.properties
                },
            }
            for item in items
        ]
    except Exception as exc:
        print(f"[WARNING] Earth STAC query failed: {exc}")
        return []


def search_unified_stac(
    body: str = "moon",
    bbox: Optional[List[float]] = None,
    data_type: str = "dem",
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Unified bridge function searching Earth, Moon, or Mars STAC catalogs."""
    b = body.lower()
    if b == "earth":
        bb = bbox or [0.0, 0.0, 1.0, 1.0]
        return search_earth_stac(bb, data_type=data_type, limit=limit)
    else:
        return search_planetary_stac(body=b, bbox=bbox, limit=limit)


# ---------------------------------------------------------------------------
# Streaming COG Window Extraction (HTTP Range Requests)
# ---------------------------------------------------------------------------

def stream_cog_window(
    href: str,
    window_dim: int = 250,
) -> Optional[np.ndarray]:
    """Stream a subset window from a remote Cloud-Optimized GeoTIFF via HTTP Range requests.

    Does not download the full multi-gigabyte raster. Reads only the required window.
    """
    try:
        with rasterio.open(href) as src:
            w = min(src.width, window_dim)
            h = min(src.height, window_dim)
            # Read central window
            c_x = max(0, (src.width - w) // 2)
            c_y = max(0, (src.height - h) // 2)
            window = Window(c_x, c_y, w, h)
            data = src.read(1, window=window).astype("float32")
            # Replace nodata with NaN
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            return data
    except Exception as exc:
        print(f"[WARNING] Failed to stream COG window from {href}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Automatic Target DEM & Optical Loader (No Local File Required)
# ---------------------------------------------------------------------------

def load_planetary_target_dem(tcfg: Dict[str, Any]) -> Optional[np.ndarray]:
    """Load target DEM: checks local file first, then seamlessly streams from planetary STAC.

    The pipeline is NEVER blocked:
      1. If local GeoTIFF exists at tcfg['dem_raster'], load it.
      2. If missing, automatically query USGS Astrogeology STAC for the target's bbox.
      3. Stream the COG DTM window remotely.
      4. If offline or unavailable, return None gracefully.
    """
    # 1. Local file fallback check
    local_path = C.ROOT / tcfg.get("dem_raster", "")
    if local_path.exists() and local_path.is_file():
        try:
            with rasterio.open(local_path) as src:
                print(f"[INFO] Using local target DEM: {local_path}")
                return src.read(1).astype("float32")
        except Exception as exc:
            print(f"[WARNING] Could not read local DEM: {exc} — switching to STAC.")

    # 2. Automated STAC streaming
    body = tcfg.get("body", "moon")
    bbox = tcfg.get("bbox", [-57.0, 13.0, -55.0, 15.0])
    collection = tcfg.get("stac_collection")

    print(f"[INFO] Local DEM absent. Querying USGS Planetary STAC for {tcfg.get('name')} ({body})...")
    items = search_planetary_stac(body=body, bbox=bbox, collection=collection, limit=3)

    for item in items:
        dem_href = item.get("dem_href")
        if dem_href:
            print(f"[INFO] Streaming Cloud-Optimized GeoTIFF from {dem_href[:80]}...")
            data = stream_cog_window(dem_href)
            if data is not None and np.isfinite(data).any():
                print(f"[SUCCESS] Streamed planetary elevation data: shape {data.shape}, "
                      f"elevation range [{np.nanmin(data):.1f}m, {np.nanmax(data):.1f}m].")
                return data

    print(f"[WARNING] No streamable planetary DEM found for {tcfg.get('name')}. Altimetry indicators skipped.")
    return None
