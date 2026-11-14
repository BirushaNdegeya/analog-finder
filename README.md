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

> [!NOTE]
> On some Ubuntu systems `ensurepip` is not available. In that case install `virtualenv` first:
> `pip install --user virtualenv && ~/.local/bin/virtualenv -p /usr/bin/python3 .venv`

---

## 5. Scoring Methodology

### Wasserstein Similarity Score

For each indicator $k$ and candidate site $s$, the similarity score is computed as:

$$S_k(s) = 1 - \frac{W_1\bigl(\hat{P}_{k,\text{target}},\; \hat{P}_{k,s}\bigr)}{W_{k,\text{max}}}$$

Where:
- $W_1(P, Q)$ is the **first Wasserstein distance** (Earth Mover's Distance) between the target empirical distribution and the site distribution.
- $W_{k,\text{max}} = \max_{s'} W_1(\hat{P}_{k,\text{target}}, \hat{P}_{k,s'})$ normalises the score **relative to the candidate panel**.
- $S = 1$ means the site's distribution is identical to the target. $S = 0$ means it is the most dissimilar site in the panel.

### Criterion Aggregation

Indicator scores within each criterion are averaged (missing indicators are ignored, not set to zero). A criterion with **no data** is set to `null` and **excluded** from the global score aggregation.

The global similarity score is a **weighted average** of the four criterion scores, with user-adjustable weights via the dashboard sidebar.

### Confidence & Thresholds

| Score threshold | Interpretation |
|:---|:---|
| ≥ 0.70 | **Strong analog** — distribution closely matches the target |
| 0.40–0.70 | **Partial analog** — overlap with significant differences |
| < 0.40 | **Weak / non-analog** — distribution is incompatible |

> [!IMPORTANT]
> **Scores are relative to the panel.** Adding or removing candidate sites changes all scores simultaneously. A score of 0.80 does not mean "80% Mars-compatible" — it means this site is **closer to the target than 80% of the current panel**.

> [!WARNING]
> **Scientific limitations:** Temperature extremes, liquid water / ice, electrostatic lunar dust, 1/6 g gravity, and hard-vacuum conditions are **not evaluated** by this tool. These are listed in each site's validity card as "Never reproducible on Earth".

---

## 6. Geospatial Data Sources & Indicator Computation

### Satellite & Planetary Data Sources

| Data Product | Collection / Dataset | Access | Used For |
|:---|:---|:---|:---|
| **Copernicus DEM GLO-30** | `cop-dem-glo-30` | [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/cop-dem-glo-30) | Slope, Roughness, Pit detection |
| **Sentinel-2 L2A** | `sentinel-2-l2a` | Microsoft Planetary Computer | Bare soil, Iron oxides, Clay minerals, Accessible basalt |
| **LOLA / LROC DEM** (Moon) | NASA PGDA | [pgda.gsfc.nasa.gov](https://pgda.gsfc.nasa.gov/search?quicksearchcat=89) | Marius Hills target slope & roughness |
| **CHELSA / WorldClim** | Annual precipitation | [envicloud.wsl.ch](https://envicloud.wsl.ch) | Aridity indicator |
| **Local basalt masks** | Georeferenced geological maps | NPS, ResearchGate, national surveys | Basalt fraction, Accessible basalt |

### Indicator Computation Pipeline

```
DEM (30 m)  ──→  resample to 100 m  ──→  slope_deg()  ──→  "pente"
                                     ──→  local_std()  ──→  "rugosite"
            ──→  native 30 m         ──→  closed_depressions()  ──→  "cavites"

Sentinel-2 L2A  ──→  SCL cloud mask  ──→  to_reflectance()
               ──→  NDVI < 0.15      ──→  local_fraction()  ──→  "sol_nu"
               ──→  B04 / B02        ──→  "oxydes_fer"
               ──→  B11 / B12        ──→  "argiles"

Basalt mask  ──→  local_fraction()  ──→  "basalte"
             × sol_nu               ──→  "basalte_accessible"

Precipitation  ──→  "aridite"
```

> [!NOTE]
> DEM and Sentinel-2 arrays may have slightly different grid dimensions due to bounding-box and resolution rounding. The pipeline verifies shape compatibility before computing `basalte_accessible`. When shapes differ, `basalte_accessible` is skipped rather than producing incorrect results.

> [!TIP]
> The Sentinel-2 reflectance formula `(DN − 1000) / 10 000` applies to **ESA Baseline ≥ 04.00** (January 2022 onward). Values are clipped to `[0.0, 1.0]` to suppress physically meaningless negative reflectances on dark basalt in shadow.

---

## 7. Execution & Decision-Support Dashboard

### 1. Execute Pipeline

```bash
# Option A: Synthetic mode (for testing pipeline and UI without network or external rasters)
python -m analog_finder.pipeline --synthetic

# Option B: Real satellite mode (queries Planetary Computer STAC + local rasters)
python -m analog_finder.pipeline
```

The pipeline writes results to `data/processed/scores.json`, containing:
- Candidate sites ranked with Wasserstein similarity across the 8 indicators.
- Per-criterion scores (`topography`, `geology`, `environment`, `resources`).
- Global similarity score weighted dynamically.
- Scientific validity card with 5 classification buckets.
- Spearman sensitivity validation metrics and negative control ranking.

### 2. Launch the Streamlit Interactive Dashboard

```bash
streamlit run app/streamlit_app.py
```

#### Dashboard Features:
1. **Interactive Global Map & Ranking**:
   - Georeferenced view of candidate sites color-coded by similarity tier (Green: $\ge 0.70$, Yellow: $0.40 - 0.70$, Red: $< 0.40$).
   - Symbolic lunar/Martian target reference coordinates.
   - Dynamic weight sliders in the sidebar enabling custom mission scenario tuning.
2. **Site Detail & Scientific Validity Card**:
   - Radar chart profile comparing multi-criteria performance.
   - Qualitative analog validity card categorizing reproduced vs. non-reproduced parameters.
   - Recommended testable use cases (e.g. mobility, drill testing, EVA logistics).
   - Median measured indicator values vs. target values.
3. **Validation Tab**:
   - Automated rank verification for known lunar analogs.
   - Negative control verification (confirming non-analogs like rainforest rank last).
   - Dirichlet Monte Carlo sensitivity analysis ($\rho$ stability score).
