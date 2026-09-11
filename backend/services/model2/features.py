from __future__ import annotations

import numpy as np
import pandas as pd


ALIASES = {
    "event_frp_mean":
        "event_mean_frp",

    "event_frp_median":
        "event_median_frp",

    "event_frp_max":
        "event_max_frp",

    "event_frp_std":
        "event_std_frp",

    "event_occupied_blocks_500m":
        "event_500m_block_count",

    "event_occupied_blocks_1km":
        "event_1km_block_count",

    "event_duration":
        "event_duration_hours",
}


CRITICAL_EVENT_FEATURES = [
    "event_detection_count",
    "event_active_days",
    "event_mean_frp",
    "event_median_frp",
    "event_max_frp",
    "event_std_frp",
    "event_mean_latitude",
    "event_mean_longitude",
    "event_min_latitude",
    "event_max_latitude",
    "event_min_longitude",
    "event_max_longitude",
    "event_500m_block_count",
    "event_1km_block_count",
    "event_duration_hours",
    "event_duration_days",
    "event_detections_per_active_day",
    "event_detections_per_hour",
]


def build_dynamic_world_base_features(
    dw_df: pd.DataFrame,
    bands: list[str],
) -> dict:

    first_valid_dw = None

    if len(dw_df):

        valid_mask = (
            dw_df["status"]
            ==
            "OK"
        )

        if valid_mask.any():

            first_valid_dw = (
                dw_df[
                    valid_mask
                ]
                .iloc[0]
            )

    dw_base_features = {}

    for band in bands:

        key = f"dw_{band}"

        if (
            first_valid_dw is not None
            and
            key in first_valid_dw.index
        ):
            dw_base_features[
                key
            ] = first_valid_dw[key]

        else:
            dw_base_features[
                key
            ] = np.nan

    dw_base_features[
        "dw_dominant_probability"
    ] = (
        first_valid_dw[
            "dw_dominant_probability"
        ]
        if first_valid_dw is not None
        else np.nan
    )

    dw_base_features[
        "dw_dominant_class"
    ] = (
        first_valid_dw[
            "dw_dominant_class"
        ]
        if first_valid_dw is not None
        else np.nan
    )

    return dw_base_features


def combine_features(
    event_features: dict,
    flare_features: dict,
    industrial_features: dict,
    temporal_features: dict,
    mining_features: dict,
    agriculture_features: dict,
    wildfire_features: dict,
    dw_event_features: dict,
    dw_base_features: dict,
    osm_nearest: dict,
    agriculture_osm_distance,
    osm_categories: list[str],
) -> dict:

    features = {}

    features.update(
        event_features
    )

    features.update(
        flare_features
    )

    features.update(
        industrial_features
    )

    features.update(
        temporal_features
    )

    features.update(
        mining_features
    )

    features.update(
        agriculture_features
    )

    features.update(
        wildfire_features
    )

    features.update(
        dw_event_features
    )

    features.update(
        dw_base_features
    )

    for category in osm_categories:

        features[
            f"distance_to_nearest_{category}_m"
        ] = osm_nearest.get(
            category,
            np.nan,
        )

    features[
        "distance_to_nearest_agriculture_m"
    ] = agriculture_osm_distance

    for old, new in ALIASES.items():

        if (
            old in features
            and
            new not in features
        ):
            features[new] = features[old]

    return features


def build_model_matrix(
    features: dict,
    model_features: list[str],
) -> pd.DataFrame:

    raw_df = pd.DataFrame(
        [features]
    )

    X_test = raw_df.reindex(
        columns=model_features
    )

    X_test = X_test.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X_test.shape != (
        1,
        len(model_features),
    ):
        raise RuntimeError(
            f"Expected "
            f"(1,{len(model_features)}), "
            f"got {X_test.shape}"
        )

    return X_test


def feature_quality(
    X_test: pd.DataFrame,
    model_features: list[str],
) -> dict:

    missing_mask = (
        X_test.iloc[0]
        .isna()
    )

    missing_features = (
        X_test.columns[
            missing_mask
        ]
        .tolist()
    )

    available_features = (
        len(model_features)
        -
        len(missing_features)
    )

    coverage = (
        available_features /
        len(model_features)
    )

    critical_missing = [
        feature
        for feature in CRITICAL_EVENT_FEATURES
        if (
            feature in model_features
            and
            pd.isna(
                X_test.iloc[0][feature]
            )
        )
    ]

    if critical_missing:
        raise RuntimeError(
            "Critical FIRMS event features "
            "missing: "
            +
            ", ".join(
                critical_missing
            )
        )

    return {
        "missing_features":
            missing_features,

        "available_features":
            available_features,

        "coverage":
            coverage,

        "critical_missing":
            critical_missing,
    }


def apply_imputer(
    X_test: pd.DataFrame,
    imputer,
) -> pd.DataFrame:

    X_imputed = imputer.transform(
        X_test
    )

    return pd.DataFrame(
        X_imputed,
        columns=X_test.columns,
        index=X_test.index,
    )
