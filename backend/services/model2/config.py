from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"

MODEL2_DIR = PROJECT_ROOT / "models" / "model2"

OUTPUT_DIR = DATA_DIR


# ============================================================
# MODEL 2 ARTIFACTS
# ============================================================

MODEL_PATH = (
    MODEL2_DIR /
    "firms_2025_final_production_random_forest.joblib"
)

IMPUTER_PATH = (
    MODEL2_DIR /
    "firms_2025_final_production_imputer.joblib"
)

FEATURE_SCHEMA_PATH = (
    MODEL2_DIR /
    "firms_2025_final_model_feature_columns.csv"
)


# ============================================================
# MODEL 2 EVIDENCE DATA
# ============================================================

INDUSTRIAL_PATH = (
    DATA_DIR /
    "firms_industrial_evidence_indicators_2025_corrected.csv"
)

MINING_PATH = (
    DATA_DIR /
    "firms_coal_mine_distance_evidence_2025.csv"
)

AGRICULTURE_PATH = (
    DATA_DIR /
    "firms_agriculture_evidence_indicators_2025.csv"
)

WILDFIRE_PATH = (
    DATA_DIR /
    "firms_wildfire_evidence_2025.csv"
)

FLARE_PATH = (
    DATA_DIR /
    "Flare-Volume-Estimates-by-individual-Flare-Location-2012-2025.xlsx"
)


# ============================================================
# OUTPUT FILES
# ============================================================

PREDICTION_OUTPUT = (
    DATA_DIR /
    "firms_2026_test_point_prediction.csv"
)

FEATURE_OUTPUT = (
    DATA_DIR /
    "firms_2026_test_point_137_features.csv"
)


# ============================================================
# NASA FIRMS
# ============================================================

FIRMS_SOURCES = [
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
    "VIIRS_SNPP_NRT",
]

FIRMS_SP_SOURCES = [
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_SP",
    "VIIRS_SNPP_SP",
]

CURRENT_DAYS = 5

HISTORICAL_DAYS = 30

EVENT_RADIUS_KM = 1.0

EVENT_TIME_HOURS = 48.0

FIRMS_BBOX_DEG = 0.05


# ============================================================
# OPENSTREETMAP
# ============================================================

OSM_RADIUS_M = 5000

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


# ============================================================
# GOOGLE DYNAMIC WORLD
# ============================================================

EE_PROJECT = "planar-depth-508020-q9"

DW_AREA_M = 300

DW_SEARCH_BEFORE_DAYS = 120

DW_SEARCH_AFTER_DAYS = 3

DW_BANDS = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]

DW_CLASS_NAMES = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "flooded_vegetation",
    4: "crops",
    5: "shrub_and_scrub",
    6: "built",
    7: "bare",
    8: "snow_and_ice",
}


# ============================================================
# MODEL EXPECTATIONS
# ============================================================

EXPECTED_MODEL_FEATURE_COUNT = 137

CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# PIPELINE COMPATIBILITY ALIASES
# ============================================================

FLARE_EVIDENCE_FILE = FLARE_PATH
INDUSTRIAL_EVIDENCE_FILE = INDUSTRIAL_PATH
MINING_EVIDENCE_FILE = MINING_PATH
AGRICULTURE_EVIDENCE_FILE = AGRICULTURE_PATH
WILDFIRE_EVIDENCE_FILE = WILDFIRE_PATH


# ============================================================
# MODEL 2 CLASS MAPPING
# ============================================================

MODEL2_CLASS_MAPPING = {
    0: "Industrial Fire",
    1: "Gas Flare",
    2: "Agricultural Fire",
    3: "Mining Activity",
    4: "Wildfire",
}


# ============================================================
# MODEL FEATURE SCHEMA
# ============================================================

import joblib
import pandas as pd


MODEL_FEATURES = (
    pd.read_csv(
        FEATURE_SCHEMA_PATH,
        header=None,
    )
    .iloc[:, 0]
    .dropna()
    .astype(str)
    .tolist()
)

# The production model expects exactly 137 features.
# The schema CSV contains one extra header entry.
if len(MODEL_FEATURES) == EXPECTED_MODEL_FEATURE_COUNT + 1:
    MODEL_FEATURES = MODEL_FEATURES[1:]


if len(MODEL_FEATURES) != EXPECTED_MODEL_FEATURE_COUNT:
    raise RuntimeError(
        f"Expected "
        f"{EXPECTED_MODEL_FEATURE_COUNT} model features, "
        f"got {len(MODEL_FEATURES)}"
    )


# ============================================================
# MODEL LOADERS
# ============================================================

def load_model2_model():
    return joblib.load(
        MODEL_PATH
    )


def load_model2_imputer():
    return joblib.load(
        IMPUTER_PATH
    )
