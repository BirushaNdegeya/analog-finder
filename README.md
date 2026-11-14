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

### Core Principle: One Algorithmic Engine, Dual Societal Reading

> **Analog Finder is neither strictly a space tool nor strictly a humanitarian hazard platform. It is a unified geospatial optimal transport engine.**  
> What is compared (a terrestrial site vs a lunar target, or a terrestrial site against itself across time) is merely a configuration parameter. The mathematical core remains identical.

```
Geospatial Rasters ──→ Indicator Distributions ──→ Optimal Transport (Wasserstein W₁)
                              │
       ┌──────────────────────┴──────────────────────┐
       ▼                                             ▼
Mode A: Planetary Analog                      Mode B: Temporal Geohazard
"Does this Earth site resemble the Moon?"    "Has this slope moved dangerously between T₀ & T₁?"
- Space mission design & rover testing        - Bukavu urban landslides & gully erosion
- Planetary geological training               - Nyiragongo & Virunga volcanic deformation
```

For the comprehensive technical and socio-economic whitepaper, see [docs/DUAL_USE_ARCHITECTURE.md](docs/DUAL_USE_ARCHITECTURE.md).

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
│   ├── targets.yaml         # Planetary target specifications (Marius Hills)
│   └── temporal_pairs.yaml  # Geohazard multi-temporal monitoring pairs (Virunga, Bukavu)
├── src/
│   └── analog_finder/       # Core Python package
│       ├── __init__.py
│       ├── config.py        # System paths, indicator definitions & criticality thresholds
│       ├── indicators.py    # Geospatial indicator extraction algorithms
│       ├── ingest.py        # STAC satellite data ingestion & raster loaders
│       ├── schemas.py       # Pydantic contracts for Mode A (analog) & Mode B (temporal)
│       ├── scoring.py       # Mode A Wasserstein similarity engine
│       ├── temporal.py      # Mode B deformation energy, patch heatmaps & significance
│       ├── validation.py    # Ranking validation & sensitivity analysis
│       └── pipeline.py      # Execution pipeline
├── data/
│   ├── raw/                 # Raw satellite rasters, DEMs & geological masks (git ignored)
│   └── processed/           # Processed rankings (scores.json) and changes (changes.json)
├── tests/                   # Complete automated test suite (30 unit tests)
├── requirements.txt         # Pinned software dependencies
├── pyproject.toml           # Standard build & editable installation configuration
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

### 5.2 Mode B — Temporal Deformation Energy & Geohazard Criticality

In temporal surveillance mode, the system assesses whether a specific territory has undergone anomalous physical alteration between baseline period $T_0$ and surveillance period $T_1$:

$$\Delta_{\text{crit}}(k) = \frac{W_1\bigl(\hat{P}_{k,T_0},\; \hat{P}_{k,T_1}\bigr)}{W_{k,\text{scale}}}$$

Where $W_{k,\text{scale}}$ represents the expected physical threshold indicating hazardous geomorphic displacement (e.g. $15^\circ$ slope shift, $60\%$ vegetation loss).

#### 1. Hypothesis Testing (Filtering Sensor Noise)
To prevent false alarms caused by sensor noise or atmospheric fluctuations, every temporal indicator undergoes a two-sample **Kolmogorov-Smirnov test**:
- $H_0$: $\hat{P}_{k,T_0} = \hat{P}_{k,T_1}$ (no true ground deformation)
- Reject $H_0$ if $p\text{-value} < \alpha$ ($\alpha = 0.05$). Indicators failing this significance test are labeled stable.

#### 2. Resolving the Spatial Blind Spot: 2D Patch-Wise Wasserstein Heatmaps
A global 1D distribution can conceal localized mass movements (e.g., an active landslide toe or volcanic fracture). To maintain spatial localization, the engine splits multi-temporal rasters into sliding metric tiles ($500\,\text{m} \times 500\,\text{m}$) and computes the Earth Mover's Distance patch-by-patch, generating a **georeferenced 2D critical deformation heatmap**.

#### 3. Standardized 4-Tier Early Warning Alert Scale

| Criticality Score | Alert Level | Geomorphic Interpretation | Recommended Municipal / Field Action |
|:---|:---|:---|:---|
| **< 0.20** | 🟢 **STABLE** | Normal seasonal variation | Routine baseline observation |
| **0.20 – 0.40** | 🟡 **WATCH** | Minor creeping surface drift | Bi-annual InSAR / drone surveillance |
| **0.40 – 0.65** | 🟠 **WARNING** | Substantial slope or land degradation | Restrict construction permits; drainage stabilization |
| **≥ 0.65** | 🔴 **CRITICAL** | Severe structural collapse risk / active fissure | Immediate ground inspection & evacuation readiness |

---

## 6. Geospatial Data Sources & Indicator Computation

### Satellite & Planetary Data Sources (Universal STAC Integration)

Analog Finder provides a unified programmatic gateway across Earth, Moon, and Mars using open STAC (SpatioTemporal Asset Catalog) APIs and Cloud-Optimized GeoTIFFs (COGs):

| Astre | Type de donnée | Collection STAC | Résolution | Format & Accès |
| :--- | :--- | :--- | :--- | :--- |
| 🌍 **Terre** | **MNT (DEM)** | `cop-dem-glo-30` (Copernicus) | 30 m | COG via Planetary Computer STAC |
| 🌍 **Terre** | **Optique** | `sentinel-2-l2a` (12 bandes) | 10 m – 20 m | COG via Planetary Computer STAC |
| 🌕 **Lune** | **MNT (DEM)** | `kaguya_terrain_camera_usgs_dtms_v2_equi` | **26.2 m** | COG via USGS Astrogeology STAC (AWS S3 public) |
| 🌕 **Lune** | **Optique** | `kaguya_terrain_camera` (Orthos) & LROC WAC | 10 m – 100 m | COG via USGS Astrogeology STAC |
| 🔴 **Mars** | **MNT (DEM)** | `mro_ctx_controlled_usgs_dtms` & HiRISE | **20 m** / **1 m** | COG via USGS Astrogeology STAC (AWS S3 public) |
| 🔴 **Mars** | **Optique** | `mo_themis_controlled_mosaics` & CTX Ortho | 6 m – 100 m | COG via USGS Astrogeology STAC |

> [!NOTE]
> **Automated Planetary Fallback**: If local target DEMs (e.g. `data/raw/targets/marius_hills_dem.tif`) are not present on disk, the pipeline automatically queries the USGS Astrogeology STAC API and streams the required window over HTTP via rasterio. Pipeline execution is **never blocked**.

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

### 6.2 Multi-Temporal Differencing (Mode B)

For geohazard monitoring, the ingestion pipeline loads paired multi-temporal scenes across identical bounding boxes and resolutions:

1. **Paired Sentinel-2 L2A Composites (`load_s2_time_pair`)**:
   - Generates median cloud-free composites for baseline $T_0$ and surveillance $T_1$.
   - Computes pixel-aligned differential rasters: $\Delta\text{NDVI} = \text{NDVI}_{T_1} - \text{NDVI}_{T_0}$ (vegetation stripping and gully expansion).
2. **Multi-Epoch DEM Differencing (`load_dem_differential`)**:
   - Compares multi-temporal elevation models (e.g. drone photogrammetry or post-event LiDAR):
     $$\Delta\text{DEM} = \text{DEM}_{T_1} - \text{DEM}_{T_0} \quad\text{and}\quad \Delta\text{Slope} = \text{Slope}_{T_1} - \text{Slope}_{T_0}$$
   - Accurately tracks scarp steepening, landslide volume displacements, and volcanic lava flow thickening.

---

## 7. Execution, Decision-Support Dashboard & REST API

### 7.1 Execute Dual-Mode Pipeline

The pipeline operates in three modes: `analog` (planetary simulation), `temporal` (geohazard deformation), or `all`:

```bash
# Synthetic demo mode (safe offline run — fast and fully populated)
python -m analog_finder.pipeline --mode all --synthetic

# Mode A only: Planetary analog scoring (writes data/processed/scores.json)
python -m analog_finder.pipeline --mode analog --synthetic

# Mode B only: Geohazard risk monitoring (writes data/processed/changes.json)
python -m analog_finder.pipeline --mode temporal --synthetic

# Real satellite mode (Planetary Computer STAC + local rasters)
python -m analog_finder.pipeline --mode all
```

Output artifacts:
- `data/processed/scores.json`: Candidate sites ranked by Wasserstein distance relative to the planetary target.
- `data/processed/changes.json`: Multi-temporal deformation indices, KS-test significance, and 4-tier alert status for geohazard zones.

### 7.2 Launch the Streamlit Interactive Dashboard

```bash
streamlit run app/streamlit_app.py
```

The interface features an **Operational Focus** selector in the sidebar:

#### 🚀 Persona 1: Mode A — Planetary Mission Analogs
1. **Interactive Global Map & Ranking**: Georeferenced candidate sites color-coded by similarity tier (Green: $\ge 0.70$, Yellow: $0.40 - 0.70$, Red: $< 0.40$) relative to the Moon target.
2. **Site Detail & Scientific Validity Card**: Multi-criteria radar chart, physical indicator medians vs target, and actionable mission testing domains (rover mobility, EVA ergonomics, ISRU regolith).
3. **Scientific Validation Tab**: Benchmark verification of known lunar analogs, negative control exclusion, and Monte Carlo Dirichlet sensitivity ($\rho$).

#### 🌋 Persona 2: Mode B — Geohazard & Volcanic Risk Monitor
1. **Surveillance Map & Alert Tiers**: Active monitoring map centered on the East African Rift (Bukavu & Virunga) with color-coded 4-tier alert badges (🔴 Critical, 🟠 Warning, 🟡 Watch, 🟢 Stable).
2. **Zone Criticality & Decision Card**: Detailed deformation radar, two-sample Kolmogorov-Smirnov significance testing table ($p$-values), and actionable civil protection mandates (drainage reinforcement, building moratoria, sensor deployment).
3. **2D Spatial Patch Differencing Simulation**: Localized sliding-window patch heatmap resolving the spatial blind spot and mapping exact deformation peaks ($W_1$).

### 7.3 Launch the FastAPI REST Backend

Analog Finder exposes an enterprise-grade REST API powering external GIS clients, municipal early-warning dashboards, and mission planning systems:

```bash
uvicorn analog_finder.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive documentation and OpenAPI Swagger UI:
- **Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

#### Key Endpoints:
- `GET /v1/health` : Engine status, version, and active mathematical transport modules.
- `GET /v1/sites` : Catalog of terrestrial analog candidates (`sites.yaml`).
- `GET /v1/targets` : Catalog of lunar and Martian targets (`targets.yaml`).
- `GET /v1/temporal-pairs` : Catalog of monitored geohazard zones (`temporal_pairs.yaml`).
- `GET /v1/indicators` : Complete definitions, physical units, and criteria mappings.
- `POST /v1/analyze` : Unified analysis engine accepting either `mode="analog"` or `mode="temporal"`.

#### Cross-Body Planetary Data Bridge Endpoints:
- `GET /v1/planetary/bodies` : List supported bodies (Earth, Moon, Mars) and open STAC endpoints.
- `GET /v1/planetary/search` : Programmatic search across Earth, Lunar (Kaguya/LOLA), and Martian (CTX/HiRISE) STAC catalogs.
- `GET /v1/planetary/sample` : Streams remote COG windows via HTTP Range requests and returns real-time topographic metrics (elevation range, mean slope, roughness).

---

## 8. Scientific Analog Validity Framework

No terrestrial environment is an exact replica of the Moon or Mars. Presenting a site as a "universal analog" is scientifically invalid and potentially hazardous for space mission planning. 

Analog Finder enforces an explicit **Validity Card** categorizing site parameters into five transparent tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ANALOG VALIDITY CARD                            │
├──────────────────────────┬─────────────────────────────────────────────┤
│ 1. Reproduced            │ Score ≥ 0.70 (Distribution closely matches) │
│ 2. Partially Reproduced  │ 0.40 ≤ Score < 0.70 (Overlap with variance) │
│ 3. Not Reproduced        │ Score < 0.40 (Significant divergence)       │
│ 4. Not Evaluated         │ Missing data or no baseline established     │
│ 5. Never Reproducible    │ Fundamental physical divergence on Earth   │
└──────────────────────────┴─────────────────────────────────────────────┘
```

### Actionable Mission Testing Domains
Based on reproduced criteria, Analog Finder suggests appropriate experimental activities:
- **Rover Mobility & Traction Testing**: Requires topographic fidelity (slope and roughness match).
- **Lava Tube Exploration & Cavity Mapping**: Requires volcanic cavity and collapse structure presence.
- **EVA Suit Ergonomics & Traverse Logistics**: Applicable in hyper-arid, bare volcanic terrain.
- **ISRU & Regolith Excavation**: Requires high surface basalt concentration and minimal organic soil cover.

---

## 9. Environmental Divergence & Limitations

### Parameters Never Reproducible on Earth
The following planetary surface conditions **cannot** be replicated naturally anywhere on Earth and require specialized laboratory chambers:
- **Surface Gravity**: Earth ($1.0\,g$) vs. Lunar ($0.166\,g$, $1/6\,g$) vs. Mars ($0.38\,g$).
- **Atmospheric Pressure**: Earth ($\approx 1013\,\text{hPa}$) vs. Moon ($10^{-12}\,\text{hPa}$, hard vacuum) vs. Mars ($\approx 6\,\text{hPa}$, $95\%\,\text{CO}_2$).
- **Radiation Environment**: Earth is shielded by a thick atmosphere and intrinsic magnetosphere; the Moon and Mars receive lethal unattenuated galactic cosmic rays (GCR) and solar particle events (SPE).
- **Regolith Physical Properties**: Terrestrial sands and rocks are weathered by wind and water; lunar regolith particles are pristine, highly abrasive, jagged glass-rich shards with electrostatic surface charges.

### Relative Scoring Caveat
Scores are normalized by $W_{k,\text{max}}$, meaning they reflect **relative ranking within the candidate panel**, not an absolute fidelity metric. Expanding the panel to include additional extreme sites will rescale intermediate numerical scores.

---

## 10. Development Lifecycle & Commit History

This project was built following software engineering best practices with an incremental, progressive release schedule:

### Phase 1 — Core Planetary Analogy Foundations (`main`)
1. **Commit 1 (`feat(core)`)**: Project architecture, `.gitignore`, dependency specifications, metadata schemas (`sites.yaml`, `targets.yaml`), and data protocol.
2. **Commit 2 (`feat(scoring)`)**: Wasserstein similarity distance engine, criterion aggregation, confidence metrics, and scientific validation suite (`spearmanr`, ranking tests).
3. **Commit 3 (`feat(indicators)`)**: Geospatial indicators (gradient slope, local standard deviation roughness, morphological cavity detection, multi-spectral mineral indices) and Planetary Computer STAC ingestion.
4. **Commit 4 (`feat(dashboard)`)**: End-to-end execution pipeline and interactive Streamlit decision-support dashboard with Folium georeferencing and Plotly radar profiles.
5. **Commit 5 (`chore(release)`)**: Expert data deposit directory scaffolding, sourced target baselines, and complete scientific documentation.

### Phase 2 — Dual-Mode Engine & Geohazard Surveillance (`feature/dual-mode-api`)
1. **Commit 1 (`feat(api-core)`)**: Pydantic schemas (`schemas.py`), temporal pairs configuration (`temporal_pairs.yaml`), and 4-tier alert scale (`config.py`).
2. **Commit 2 (`feat(temporal-engine)`)**: Temporal Wasserstein deformation energy, Kolmogorov-Smirnov significance testing, and 2D patch-wise localized heatmaps (`temporal.py`).
3. **Commit 3 (`feat(ingest-temporal)`)**: Paired multi-temporal Sentinel-2 composite loader and multi-epoch DEM differential engine (`ingest.py`, `indicators.py`).
4. **Commit 4 (`feat(fastapi-backend)`)**: Unified REST API (`app.py`), multi-mode pipeline execution, and pre-computed geohazard surveillance results (`changes.json`).
5. **Commit 5 (`feat(dual-ui)`)**: Dual-persona Streamlit interface (Planetary Mission vs Geohazard Monitor) and comprehensive socio-economic whitepaper (`docs/DUAL_USE_ARCHITECTURE.md`).

---

## 11. Expert Data Deposit Protocol

Team members depositing newly georeferenced rasters should follow this protocol:

### Raster Requirements
- **Projection**: `EPSG:4326` (WGS 84 geographic coordinates)
- **Target Resolution**: 100 m / pixel ($\approx 0.000898^\circ$ at equator)
- **Format**: Cloud-Optimized GeoTIFF (`.tif`)
- **NoData Value**: Set explicitly in GeoTIFF metadata

### Deposit Locations
```
data/raw/
├── targets/
│   └── marius_hills_dem.tif        # (Gérard: Moon target DEM)
├── virunga/
│   └── basalt_mask.tif             # (Nicolas: Binary mask 1=basalt, 0=other)
├── atacama/
│   └── basalt_mask.tif             # (Nicolas: Geological mask)
├── harrat_khaybar/
│   └── basalt_mask.tif             # (Nicolas: Volcanic field mask)
├── suswa/
│   └── basalt_mask.tif             # (Diego: Caldera & lava field mask)
├── vallee_mort/
│   └── basalt_mask.tif             # (Diego: Arid terrain mask)
├── hawaii/
│   └── basalt_mask.tif             # (Diego: Mauna Loa basalt mask)
├── lava_beds/
│   └── basalt_mask.tif             # (Bosworth: Lava tube national monument mask)
├── lanzarote/
│   └── basalt_mask.tif             # (Bosworth: Volcanic island basalt mask)
├── foret_congo/
│   └── basalt_mask.tif             # (Bosworth: Negative control zero-mask)
└── climate/
    └── annual_precip.tif           # (Global or multi-region CHELSA/WorldClim raster)
```

### Re-running the Full Scientific Pipeline
Once rasters are deposited:
```bash
# Execute real satellite ingestion and local raster fusion
python -m analog_finder.pipeline

# Restart the Streamlit dashboard
streamlit run app/streamlit_app.py
```

---

## 12. Team & Acknowledgments

**Team GeoCongo AI** — NASA Space Apps Challenge 2026:
- **Gérard Cubaka Bisimwa** — Lead Software & Geospatial Engineer, Lunar Target (Marius Hills)
- **Nicolas** — Geological Cartography & Planetary Analog Specialist (Virunga, Atacama, Harrat Khaybar)
- **Diego** — Remote Sensing & Volcanology Specialist (Mount Suswa, Death Valley, Hawaii)
- **Bosworth** — Data Integration & Validation Analyst (Lava Beds, Lanzarote, Congo Rainforest)

### References & Portals
- NASA Planetary Geosciences Data Access: [pgda.gsfc.nasa.gov](https://pgda.gsfc.nasa.gov/search?quicksearchcat=89)
- Copernicus Open Access & Microsoft Planetary Computer: [planetarycomputer.microsoft.com](https://planetarycomputer.microsoft.com)
- National Park Service Geodiversity Atlas: [nps.gov](https://www.nps.gov/articles/nps-geodiversity-atlas-lava-beds-national-monument-california.htm)
- CHELSA High-Resolution Climate Data: [envicloud.wsl.ch](https://envicloud.wsl.ch)
