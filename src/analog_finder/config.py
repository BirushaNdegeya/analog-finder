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

# --- Geospatial Data Ingestion Endpoints & Defaults --------------------------
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEM_COLLECTION = "cop-dem-glo-30"
S2_COLLECTION = "sentinel-2-l2a"
S2_DATE_RANGE = "2025-01-01/2025-12-31"
S2_MAX_CLOUD = 30

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
