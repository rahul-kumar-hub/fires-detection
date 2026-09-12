from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"

# ============================================================
# MODEL 2 DIRECTORY
# ============================================================
#
# IMPORTANT:
# You have placed the three Kaggle production artifacts directly
# inside:
#
# backend/services/model2/
#
# Therefore we load them from this directory.
#
# DO NOT move them to another folder.
# ============================================================

MODEL2_DIR = Path(__file__).resolve().parent


# ============================================================
# MODEL 2 PRODUCTION ARTIFACTS
# ============================================================

MODEL_PATH = (
    MODEL2_DIR
    / "firms_2025_final_production_random_forest.joblib"
)

IMPUTER_PATH = (
    MODEL2_DIR
    / "firms_2025_final_production_imputer.joblib"
)

FEATURE_SCHEMA_PATH = (
    MODEL2_DIR
    / "firms_2025_final_model_feature_columns.csv"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = DATA_DIR


# ============================================================
# EVIDENCE DATA FILES
# ============================================================

INDUSTRIAL_PATH = (
    DATA_DIR
    / "firms_industrial_evidence_indicators_2025_corrected.csv"
)

MINING_PATH = (
    DATA_DIR
    / "firms_coal_mine_distance_evidence_2025.csv"
)

AGRICULTURE_PATH = (
    DATA_DIR
    / "firms_agriculture_evidence_indicators_2025.csv"
)

WILDFIRE_PATH = (
    DATA_DIR
    / "firms_wildfire_evidence_2025.csv"
)

FLARE_PATH = (
    DATA_DIR
    / "Flare-Volume-Estimates-by-individual-Flare-Location-2012-2025.xlsx"
)


# ============================================================
# OUTPUT FILES
# ============================================================

PREDICTION_OUTPUT = (
    DATA_DIR
    / "firms_2026_test_point_prediction.csv"
)

FEATURE_OUTPUT = (
    DATA_DIR
    / "firms_2026_test_point_137_features.csv"
)


# ============================================================
# NASA FIRMS CONFIGURATION
# ============================================================

FIRMS_SOURCES = [
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
    "VIIRS_SNPP_NRT",
]


# SP fallback sources
FIRMS_SP_SOURCES = [
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_SP",
    "VIIRS_SNPP_SP",
]


# Current FIRMS search window
CURRENT_DAYS = 5


# Historical FIRMS search window
HISTORICAL_DAYS = 30


# Local event radius
EVENT_RADIUS_KM = 1.0


# Local event temporal window
EVENT_TIME_HOURS = 48.0


# FIRMS bounding box
FIRMS_BBOX_DEG = 0.05


# ============================================================
# OPENSTREETMAP CONFIGURATION
# ============================================================

OSM_RADIUS_M = 5000


OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


# ============================================================
# GOOGLE EARTH ENGINE / DYNAMIC WORLD
# ============================================================
#
# IMPORTANT:
#
# This is the same Earth Engine project used by the Kaggle
# Model 2 production pipeline.
#
# Do NOT use:
#
# planar-depth-508020-q9
#
# because that was the project causing the previous 403 error.
# ============================================================

EE_PROJECT = "possible-haven-507714-f0"


# Dynamic World analysis area
DW_AREA_M = 300


# Dynamic World search window
DW_SEARCH_BEFORE_DAYS = 120

DW_SEARCH_AFTER_DAYS = 3


# Dynamic World probability bands
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


# Dynamic World class mapping
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


# Kaggle production confidence threshold
CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# MODEL 2 CLASS MAPPING
# ============================================================

MODEL2_CLASS_MAPPING = {
    0: "Industrial",
    1: "Gas",
    2: "Agriculture",
    3: "Mining",
    4: "Wildfire",
}


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

FLARE_EVIDENCE_FILE = FLARE_PATH

INDUSTRIAL_EVIDENCE_FILE = INDUSTRIAL_PATH

MINING_EVIDENCE_FILE = MINING_PATH

AGRICULTURE_EVIDENCE_FILE = AGRICULTURE_PATH

WILDFIRE_EVIDENCE_FILE = WILDFIRE_PATH


# ============================================================
# LOAD EXACT 137-FEATURE MODEL SCHEMA
# ============================================================

if not FEATURE_SCHEMA_PATH.exists():

    raise FileNotFoundError(
        "Model 2 feature schema not found:\n"
        f"{FEATURE_SCHEMA_PATH}"
    )


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


# ------------------------------------------------------------
# Handle possible CSV header
# ------------------------------------------------------------

if (
    len(MODEL_FEATURES)
    == EXPECTED_MODEL_FEATURE_COUNT + 1
):

    MODEL_FEATURES = MODEL_FEATURES[1:]


# ------------------------------------------------------------
# Validate feature count
# ------------------------------------------------------------

if (
    len(MODEL_FEATURES)
    != EXPECTED_MODEL_FEATURE_COUNT
):

    raise RuntimeError(
        "Invalid Model 2 feature schema.\n"
        f"Expected: {EXPECTED_MODEL_FEATURE_COUNT}\n"
        f"Found: {len(MODEL_FEATURES)}"
    )


# ============================================================
# MODEL LOADER
# ============================================================

def load_model2_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Model 2 Random Forest not found:\n"
            f"{MODEL_PATH}"
        )

    return joblib.load(
        MODEL_PATH
    )


# ============================================================
# IMPUTER LOADER
# ============================================================

def load_model2_imputer():

    if not IMPUTER_PATH.exists():

        raise FileNotFoundError(
            "Model 2 production imputer not found:\n"
            f"{IMPUTER_PATH}"
        )

    return joblib.load(
        IMPUTER_PATH
    )