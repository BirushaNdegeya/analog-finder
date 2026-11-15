"""Configuration and scientific parameters for Analog Finder.

All adjustable model parameters, paths, and thresholds are defined here.
NASA Space Apps Challenge 2026 - GeoCongo AI
"""
from pathlib import Path

# --- Project Paths -----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TEMPORAL_PAIRS_FILE = CONFIG_DIR / "temporal_pairs.yaml"
CHANGES_FILE = PROCESSED_DIR / "changes.json"

# --- Geospatial Data Ingestion Endpoints & Defaults --------------------------
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEM_COLLECTION = "cop-dem-glo-30"
S2_COLLECTION = "sentinel-2-l2a"
S2_DATE_RANGE = "2025-01-01/2025-12-31"
S2_MAX_CLOUD = 30
TERRACLIMATE_COLLECTION = "terraclimate"

# Planetary STAC APIs (USGS Astrogeology - Moon & Mars)
PLANETARY_STAC_URL = "https://stac.astrogeology.usgs.gov/api"
LUNAR_DEM_COLLECTION = "kaguya_terrain_camera_usgs_dtms_v2_equi"
MARS_DEM_COLLECTION = "mro_ctx_controlled_usgs_dtms"

# --- Spatial Analysis Scales -------------------------------------------------
COMMON_RES_M = 100            # Common grid cell resolution in meters (slope, roughness)
ROUGHNESS_WINDOW_M = 1000     # Moving standard deviation window (meters)
LOCAL_FRACTION_WINDOW_M = 1000
PIT_MIN_DEPTH_M = 5.0         # Minimum depression depth for pit skylight candidate (meters)
NDVI_BARE_THRESHOLD = 0.2     # Bare ground threshold on Sentinel-2 NDVI
MAX_SAMPLES = 5000            # Sample size for empirical distribution estimation

# --- Scientific Criteria & Indicators ---------------------------------------
CRITERIA = {
    "topographie": ["pente", "rugosite"],
    "geologie": ["basalte", "oxydes_fer", "argiles"],
    "environnement": ["sol_nu", "aridite"],
    "ressources": ["cavites", "basalte_accessible"],
}

CRITERIA_LABELS = {
    "topographie": "Topography",
    "geologie": "Geology",
    "environnement": "Environment",
    "ressources": "In-Situ Resources",
}

# Geohazard Risk Criteria (In-Situ Resources excluded as irrelevant to disaster monitoring)
TEMPORAL_CRITERIA = ["topographie", "geologie", "environnement"]
TEMPORAL_CRITERIA_LABELS = {
    "topographie": "Topography (Slope & Scarp)",
    "geologie": "Geology (Lava & Mineral Alteration)",
    "environnement": "Environment (Bare Soil & Vegetation Loss)",
}
TEMPORAL_DEFAULT_WEIGHTS = {c: 1.0 / len(TEMPORAL_CRITERIA) for c in TEMPORAL_CRITERIA}

INDICATOR_LABELS = {
    "pente": "Slope",
    "rugosite": "Roughness",
    "basalte": "Basalt Fraction",
    "oxydes_fer": "Iron Oxides Index",
    "argiles": "Clay Minerals Index",
    "sol_nu": "Bare Ground Fraction",
    "aridite": "Annual Precipitation",
    "cavites": "Cavities & Skylights",
    "basalte_accessible": "Accessible Basalt",
}

INDICATORS = [i for inds in CRITERIA.values() for i in inds]
DEFAULT_WEIGHTS = {c: 1 / len(CRITERIA) for c in CRITERIA}

# --- Scientific Score Thresholds ---------------------------------------------
STRONG = 0.7   # Score >= 0.70: Strong planetary resemblance
WEAK = 0.4     # Score < 0.40: Low resemblance / unsuited analog

# --- Units and Validity Metadata ---------------------------------------------
UNITS = {
    "pente": "°",
    "rugosite": "°",
    "basalte": "fraction",
    "oxydes_fer": "Red/Blue ratio",
    "argiles": "SWIR1/SWIR2 ratio",
    "sol_nu": "fraction",
    "aridite": "mm/year",
    "cavites": "areal fraction",
    "basalte_accessible": "fraction",
}

USAGES = {
    "topographie": "Rover mobility, terrain navigation, landing stability",
    "geologie": "Surface regolith mechanics, spectral sensor calibration, geological tools",
    "environnement": "Dust & arid operations, extreme EVA simulation in barren terrain",
    "ressources": "Subterranean habitat exploration (lava tubes), ISRU regolith testing",
}

NEVER_REPRODUCED = [
    "Reduced lunar gravity (1/6 g) or Martian gravity (3/8 g)",
    "Hard lunar vacuum and hypervelocity micrometeoroids",
    "Unshielded solar & galactic cosmic radiation",
    "Atmospheric density differences (Mars low-pressure CO2 atmosphere)",
]

NOT_EVALUATED = [
    "Diurnal temperature swings and polar illumination",
    "Subsurface ice, permafrost, and volatile stability",
    "Electrostatic regolith dust behavior",
    "Astrobiology and strict planetary protection constraints",
]

# --- Temporal Criticality Thresholds & Alerts (Mode B) -----------------------
CRITICALITY_CRITICAL = 0.65   # Severe structural rupture / rapid ground displacement
CRITICALITY_WARNING = 0.40    # Significant anomalous geomorphic deformation
CRITICALITY_WATCH = 0.20      # Mild seasonal or creeping surface drift
SIGNIFICANCE_ALPHA = 0.05     # Statistical p-value rejection threshold

TEMPORAL_ACTIONS = {
    "critical": "Immediate field inspection, sensor deployment, and evacuation preparedness in high-slope zones",
    "warning": "Restricted construction zoning, high-frequency InSAR monitoring, and drainage stabilization",
    "watch": "Routine geotechnical surveillance and quarterly multi-spectral satellite updates",
    "stable": "Normal territorial maintenance and ongoing baseline environmental observation",
}
