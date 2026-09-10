# src/features.py

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config import EXPECTED_FEATURE_COUNT


# ============================================================
# DYNAMIC WORLD FEATURES
# ============================================================

DW_FEATURES = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
    "dw_date_difference_days",
    "month",
    "dw_max_probability",
    "dw_available",
]


# ============================================================
# ASSEMBLE NON-DYNAMIC-WORLD FEATURES
# ============================================================

def assemble_feature_dict(
    firms_features: dict,
    persistence_features: dict,
    osm_features: dict,
) -> dict:
    """
    Assemble all non-Dynamic-World features.

    Dynamic World features are added separately by
    add_dynamic_world_features().
    """

    feature_dict = {}

    # --------------------------------------------------------
    # FIRMS raw features
    # --------------------------------------------------------

    required_firms_features = [
        "latitude",
        "longitude",
        "brightness",
        "scan",
        "track",
        "bright_t31",
        "frp",
        "frp_log",
        "confidence_low",
        "confidence_nominal",
        "confidence_high",
        "hour_sin",
        "hour_cos",
    ]

    missing_firms = [
        feature
        for feature in required_firms_features
        if feature not in firms_features
    ]

    if missing_firms:
        raise RuntimeError(
            "Required FIRMS features are missing:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_firms
            )
        )

    # Exact FIRMS raw features from the 74-feature schema
    feature_dict["latitude"] = firms_features["latitude"]
    feature_dict["longitude"] = firms_features["longitude"]
    feature_dict["brightness"] = firms_features["brightness"]
    feature_dict["scan"] = firms_features["scan"]
    feature_dict["track"] = firms_features["track"]
    feature_dict["bright_t31"] = firms_features["bright_t31"]
    feature_dict["frp"] = firms_features["frp"]

    # --------------------------------------------------------
    # OSM features
    # --------------------------------------------------------

    if not isinstance(osm_features, dict):
        raise TypeError(
            "osm_features must be a dictionary."
        )

    feature_dict.update(
        osm_features
    )

    # --------------------------------------------------------
    # Persistence features
    # --------------------------------------------------------

    required_persistence_features = [
        "detections_7_days",
        "detections_30_days",
        "night_ratio",
        "persistence_score",
    ]

    missing_persistence = [
        feature
        for feature in required_persistence_features
        if feature not in persistence_features
    ]

    if missing_persistence:
        raise RuntimeError(
            "Required persistence features are missing:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_persistence
            )
        )

    feature_dict.update(
        persistence_features
    )

    # --------------------------------------------------------
    # FIRMS engineered features
    # --------------------------------------------------------

    feature_dict["frp_log"] = (
        firms_features["frp_log"]
    )

    feature_dict["confidence_low"] = (
        firms_features["confidence_low"]
    )

    feature_dict["confidence_nominal"] = (
        firms_features["confidence_nominal"]
    )

    feature_dict["confidence_high"] = (
        firms_features["confidence_high"]
    )

    feature_dict["hour_sin"] = (
        firms_features["hour_sin"]
    )

    feature_dict["hour_cos"] = (
        firms_features["hour_cos"]
    )

    return feature_dict


# ============================================================
# ADD DYNAMIC WORLD FEATURES
# ============================================================

def add_dynamic_world_features(
    feature_dict: dict,
    dynamic_world_features: dict,
) -> dict:
    """
    Add the 13 Dynamic World-related features.

    Missing required DW features are treated as an error,
    matching Cell 2B/2C readiness checks.
    """

    if not isinstance(dynamic_world_features, dict):
        raise TypeError(
            "dynamic_world_features must be a dictionary."
        )

    result = dict(
        feature_dict
    )

    missing = [
        feature
        for feature in DW_FEATURES
        if feature not in dynamic_world_features
    ]

    if missing:
        raise RuntimeError(
            "Required Dynamic World features are missing:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing
            )
        )

    for feature in DW_FEATURES:
        result[feature] = (
            dynamic_world_features[feature]
        )

    return result


# ============================================================
# VALIDATE FEATURE SET
# ============================================================

def validate_feature_names(
    feature_dict: dict,
    model_features,
) -> bool:
    """
    Verify that every required model feature exists.

    Extra keys are allowed because Cell 2C ultimately
    selects only ml_features.
    """

    missing = [
        feature
        for feature in model_features
        if feature not in feature_dict
    ]

    if missing:
        raise RuntimeError(
            "Required model features are missing:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing
            )
        )

    return True


# ============================================================
# CREATE RAW MODEL INPUT
# ============================================================

def create_raw_model_input(
    feature_dict: dict,
    model_features,
) -> pd.DataFrame:
    """
    Reproduce Cell 2C Section 4:

        X_live_raw = pd.DataFrame(
            [[feature_dict[f] for f in ml_features]],
            columns=ml_features
        )

    This guarantees the exact training feature order.
    """

    validate_feature_names(
        feature_dict=feature_dict,
        model_features=model_features,
    )

    model_features = list(
        model_features
    )

    X_live_raw = pd.DataFrame(
        [
            [
                feature_dict[feature]
                for feature in model_features
            ]
        ],
        columns=model_features,
    )

    return X_live_raw


# ============================================================
# VALIDATE RAW INPUT
# ============================================================

def validate_raw_model_input(
    X_live_raw: pd.DataFrame,
) -> dict:
    """
    Inspect the raw model vector before imputation.

    Returns counts/lists of NaN and infinite features.
    """

    nan_features = [
        feature
        for feature in X_live_raw.columns
        if pd.isna(
            X_live_raw.loc[0, feature]
        )
    ]

    inf_features = [
        feature
        for feature in X_live_raw.columns
        if np.isinf(
            X_live_raw.loc[0, feature]
        )
    ]

    return {
        "nan_features": nan_features,
        "inf_features": inf_features,
        "nan_count": len(nan_features),
        "inf_count": len(inf_features),
    }


# ============================================================
# APPLY TRAINING IMPUTATION
# ============================================================

def apply_training_imputation(
    X_live_raw: pd.DataFrame,
    train_medians,
) -> pd.DataFrame:
    """
    Apply the exact training-compatible imputation:

        fillna(train_medians).fillna(0)

    This is intentionally performed only at the final
    model-input stage.
    """

    X_live = (
        X_live_raw
        .fillna(train_medians)
        .fillna(0)
    )

    return X_live


# ============================================================
# VALIDATE FINAL MODEL INPUT
# ============================================================

def validate_final_model_input(
    X_live: pd.DataFrame,
    expected_feature_count: int = EXPECTED_FEATURE_COUNT,
) -> bool:
    """
    Final Cell 2C-style validation:

        - exact feature count
        - no NaN
        - no infinite values
    """

    # --------------------------------------------------------
    # Feature count
    # --------------------------------------------------------

    if X_live.shape[1] != expected_feature_count:
        raise RuntimeError(
            "Incorrect model feature count: "
            f"{X_live.shape[1]}. "
            f"Expected {expected_feature_count}."
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    try:
        values = X_live.to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Model input contains non-numeric values."
        ) from exc

    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    remaining_nan = int(
        np.isnan(values).sum()
    )

    if remaining_nan != 0:
        raise RuntimeError(
            "Final model input still contains "
            f"{remaining_nan} NaN value(s)."
        )

    # --------------------------------------------------------
    # Inf
    # --------------------------------------------------------

    remaining_inf = int(
        np.isinf(values).sum()
    )

    if remaining_inf != 0:
        raise RuntimeError(
            "Final model input still contains "
            f"{remaining_inf} infinite value(s)."
        )

    return True


# ============================================================
# COMPLETE MODEL INPUT PREPARATION
# ============================================================

def prepare_model_input(
    feature_dict: dict,
    model_features,
    train_medians,
) -> pd.DataFrame:
    """
    Complete Cell 2C-compatible model-input preparation.

    Steps:

        1. Verify all model features exist.
        2. Create exact ml_features-order DataFrame.
        3. Apply train_medians.
        4. Apply zero fallback.
        5. Verify exactly 74 finite features.
    """

    model_features = list(
        model_features
    )

    # --------------------------------------------------------
    # 1. Create exact raw input
    # --------------------------------------------------------

    X_live_raw = create_raw_model_input(
        feature_dict=feature_dict,
        model_features=model_features,
    )

    # --------------------------------------------------------
    # 2. Verify expected schema
    # --------------------------------------------------------

    if len(model_features) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Expected exactly "
            f"{EXPECTED_FEATURE_COUNT} model features, "
            f"but received {len(model_features)}."
        )

    # --------------------------------------------------------
    # 3. Apply exact training imputation
    # --------------------------------------------------------

    X_live = apply_training_imputation(
        X_live_raw=X_live_raw,
        train_medians=train_medians,
    )

    # --------------------------------------------------------
    # 4. Final validation
    # --------------------------------------------------------

    validate_final_model_input(
        X_live=X_live,
        expected_feature_count=EXPECTED_FEATURE_COUNT,
    )

    return X_live