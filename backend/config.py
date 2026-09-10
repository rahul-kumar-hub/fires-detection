from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "model1" / "model_package.joblib"


# ============================================================
# FIRMS
# ============================================================

# NASA FIRMS MAP_KEY should NOT be hard-coded here.
# We will enter/load it safely later.

FIRMS_WFS_URL = "https://firms.modaps.eosdis.nasa.gov/mapserver/wfs"
FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


# Live FIRMS satellite sources used by your working notebook
FIRMS_SOURCES = [
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
]

LIVE_SEARCH_RADIUS_KM = 10.0


# ============================================================
# OSM
# ============================================================

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

# Same radius used in the working pipeline
OSM_RADIUS_KM = 25

# Distance used when an OSM feature is not found
NOT_FOUND_DISTANCE = 1e9


# ============================================================
# PERSISTENCE
# ============================================================

PERSISTENCE_RADIUS_KM = 1.0
PERSISTENCE_DAYS = 30


# ============================================================
# DYNAMIC WORLD
# ============================================================

DYNAMIC_WORLD_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"

# Same search window used in the working notebook
DYNAMIC_WORLD_DAYS_BEFORE = 15
DYNAMIC_WORLD_DAYS_AFTER = 15


# ============================================================
# MACHINE LEARNING
# ============================================================

EXPECTED_FEATURE_COUNT = 74


# These are the class labels used by the trained system.
CLASS_MAPPING = {
    0: "Other",
    1: "Industrial Fire",
    2: "Gas Flare",
    3: "Agricultural Fire",
    4: "Mining Activity",
    5: "Wildfire",
}