from __future__ import annotations

import numpy as np
import pandas as pd

from .config import WILDFIRE_PATH
from .evidence import aggregate_nearest_evidence, find_column


WILDFIRE_DISTANCE_COLUMNS = [
    "distance_to_nearest_flare_m",
    "distance_to_nearest_oil_well_m",
    "distance_to_nearest_mineshaft_m",
    "distance_to_nearest_adit_m",
    "distance_to_nearest_gasometer_m",
    "distance_to_nearest_industrial_m",
    "distance_to_nearest_quarry_m",
    "distance_to_nearest_farmland_m",
    "distance_to_nearest_farmyard_m",
    "distance_to_nearest_orchard_m",
    "distance_to_nearest_vineyard_m",
    "distance_to_nearest_plant_nursery_m",
    "distance_to_nearest_greenhouse_m",
    "distance_to_nearest_allotment_m",
    "distance_to_nearest_forest_m",
    "distance_to_nearest_scrub_m",
    "distance_to_nearest_grassland_m",
    "distance_to_nearest_heath_m",
    "distance_to_nearest_agriculture_m",
    "distance_to_nearest_natural_vegetation_m",
]


def load_wildfire_evidence() -> pd.DataFrame:
    if not WILDFIRE_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        WILDFIRE_PATH
    )


def build_wildfire_features(
    wildfire_df: pd.DataFrame,
    event_df: pd.DataFrame,
) -> dict:

    raw = aggregate_nearest_evidence(
        wildfire_df,
        event_df,
        columns_min=WILDFIRE_DISTANCE_COLUMNS,
    )

    wildfire_features = {}

    for column in WILDFIRE_DISTANCE_COLUMNS:
        wildfire_features[
            column + "_min"
        ] = raw.get(
            column,
            np.nan,
        )

    natural_context_column = find_column(
        wildfire_df,
        exact="natural_landcover_context",
    )

    natural_strong_column = find_column(
        wildfire_df,
        exact="natural_vegetation_strong",
    )

    natural_moderate_column = find_column(
        wildfire_df,
        exact="natural_vegetation_moderate",
    )

    natural_columns = [
        column
        for column in [
            natural_context_column,
            natural_strong_column,
            natural_moderate_column,
        ]
        if column is not None
    ]

    natural_raw = aggregate_nearest_evidence(
        wildfire_df,
        event_df,
        columns_max=natural_columns,
    )

    wildfire_features[
        "natural_landcover_context_max"
    ] = (
        natural_raw.get(
            natural_context_column,
            np.nan,
        )
        if natural_context_column
        else np.nan
    )

    wildfire_features[
        "natural_vegetation_strong_max"
    ] = (
        natural_raw.get(
            natural_strong_column,
            np.nan,
        )
        if natural_strong_column
        else np.nan
    )

    wildfire_features[
        "natural_vegetation_moderate_max"
    ] = (
        natural_raw.get(
            natural_moderate_column,
            np.nan,
        )
        if natural_moderate_column
        else np.nan
    )

    return wildfire_features
