# Analog Finder (GeoCongo AI)

> **NASA Space Apps Challenge 2026** — *Identify Earth Locations that Analog the Permanent Moon Base Locations and Mars*

An open-source scientific framework and interactive decision-support tool that identifies and ranks Earth sites based on their geomorphological, environmental, and geological resemblance to target planetary base locations (e.g., **Marius Hills**, Moon).

---

## 1. Project Overview & Scientific Challenge

Future permanent lunar bases and Martian exploration outposts require rigorous mission simulation and rover testing on Earth. However, no terrestrial site is a 100% duplicate of a planetary body. 

**Analog Finder** provides an objective, transparent, and reproducible methodology to assess:
- **Where an analogy holds**: specific criteria where the site closely mimics the planetary target.
- **Where it falls short**: non-reproduced or partially reproduced parameters.
- **What can be validated on site**: targeted testing domains (rover mobility, lava tube exploration, EVA operational logistics, ISRU regolith testing).

### Multi-Criteria Framework
The assessment evaluates **4 core criteria** across **8 quantitative indicators**:
1. **Topography**: Terrain slope and surface roughness (100 m baseline grid).
2. **Geology**: Basalt spatial fraction, iron oxides spectral index, clay minerals index.
3. **Environment**: Bare ground fraction (NDVI thresholding) and aridity (annual precipitation).
4. **In-Situ Resources (ISRU)**: Skylight / lava tube cavity density and accessible surface basalt.

---

## 2. Repository Structure

```
Analog-Finder/
├── config/                  # Configuration files
│   ├── sites.yaml           # Terrestrial candidate sites metadata & bounding boxes
│   └── targets.yaml         # Planetary target specifications (Marius Hills)
├── src/
│   └── analog_finder/       # Core Python package
│       ├── __init__.py
│       └── config.py        # System paths, indicator definitions & thresholds
├── data/
│   ├── raw/                 # Raw satellite rasters, DEMs & geological masks (git ignored)
│   └── processed/           # Processed similarity rankings and validation results
├── requirements.txt         # Pinned software dependencies
├── .gitignore               # Clean repository exclusion rules
└── README.md                # Project documentation
```

---

## 3. Expert Data Collection Protocol

Multidisciplinary data collection is conducted collaboratively across the **GeoCongo AI** research team:

| Team Member | Assigned Sites / Target | Data Types to Collect & Process | Target Resolution | Primary Data Sources & Portals |
| :--- | :--- | :--- | :--- | :--- |
| **Gérard** | **Marius Hills (Moon Target)** | LOLA metric DEM, Geologic Map, Climate baseline (null/vacuum), Pit skylight inventory | 100 m | [NASA Planetary Geosciences Data (PGDA)](https://pgda.gsfc.nasa.gov/search?quicksearchcat=89), LROC/SELENE |
| **Nicolas** | **Virunga**, **Atacama**, **Harrat Khaybar** | Georeferenced geological maps, Basalt masks, WorldClim/ERA5 precipitation, Cave/Lava tube surveys | 100 m | NASA PGDA, Copernicus DEM, Local Geological surveys |
| **Diego** | **Mount Suswa**, **Death Valley**, **Hawaii** | Georeferenced geology, Basalt masks, Climate records, Volcanic collapse depressions | 100 m | NASA PGDA, USGS 3DEP, Hawaiian Volcano Observatory |
| **Bosworth** | **Lava Beds**, **Lanzarote**, **Congo Rainforest** (Negative Control) | Geological mapping, Weather station series, Cloud-free Sentinel-2 composites | 100 m | [NPS Geodiversity Atlas](https://www.nps.gov/articles/nps-geodiversity-atlas-lava-beds-national-monument-california.htm), [Lanzarote Geology](https://www.researchgate.net/figure/Simplified-geological-map-of-Lanzarote-Island-The-top-left-inset-displays-the-location_fig1_318350212), [CHELSA Climate](https://envicloud.wsl.ch/#/?bucket=https%3A%2F%2Fos.zhdk.cloud.switch.ch%2Fchelsa01%2F&prefix=chelsa_canaryclim%2Fcanaries%2Fclimatologies%2F) |

### Expected Data Deposit Schema
- `data/raw/targets/marius_hills_dem.tif` (Planetary metric DEM, EPSG:4326 / Equirectangular)
- `data/raw/<site_id>/basalt_mask.tif` (Georeferenced binary basalt raster, EPSG:4326)
- `data/raw/climate/annual_precip.tif` (Global or regional annual precipitation raster, EPSG:4326)

---

## 4. Setup & Environment Installation

### Requirements
- Python 3.10+
- GDAL / PROJ compatible geospatial environment

```bash
# Clone the repository
git clone https://github.com/GeoCongo-AI/analog-finder.git
cd analog-finder

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
