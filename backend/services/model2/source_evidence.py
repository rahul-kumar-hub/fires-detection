from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evidence import aggregate_nearest_evidence, find_column


def load_evidence(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def build_industrial_features(
    industrial_df: pd.DataFrame,
    event_df: pd.DataFrame,
) -> dict[str, float]:
    columns_min = [
        "steel_distance_m",
        "cement_distance_m",
        "wri_power_distance_m",
        "fertilizer_distance_m",
        "refinery_petro_distance_m",
    ]

    columns_max = [
        "industrial_source_count_500m",
        "industrial_source_count_1km",
        "industrial_evidence_score",
    ]

    raw = aggregate_nearest_evidence(
        industrial_df,
        event_df,
        columns_min=columns_min,
        columns_max=columns_max,
    )

    return {
        "event_min_steel_distance_m": raw.get(
            "steel_distance_m", np.nan
        ),
        "event_min_cement_distance_m": raw.get(
            "cement_distance_m", np.nan
        ),
        "event_min_wri_power_distance_m": raw.get(
            "wri_power_distance_m", np.nan
        ),
        "event_min_fertilizer_distance_m": raw.get(
            "fertilizer_distance_m", np.nan
        ),
        "event_min_refinery_petro_distance_m": raw.get(
            "refinery_petro_distance_m", np.nan
        ),
        "event_max_industrial_source_count_500m": raw.get(
            "industrial_source_count_500m", np.nan
        ),
        "event_max_industrial_source_count_1km": raw.get(
            "industrial_source_count_1km", np.nan
        ),
        "event_max_industrial_evidence_score": raw.get(
            "industrial_evidence_score", np.nan
        ),
    }


def build_mining_features(
    mining_df: pd.DataFrame,
    event_df: pd.DataFrame,
) -> dict[str, float]:
    known_columns = [
        "nearest_coal_mine_distance_m",
        "coal_mine_distance_m",
        "event_min_nearest_coal_mine_distance_m",
        "nearest_mine_distance_m",
    ]

    distance_column = next(
        (
            column
            for column in known_columns
            if column in mining_df.columns
        ),
        None,
    )

    if distance_column is None:
        for column in mining_df.columns:
            name = str(column).lower()

            if (
                "coal" in name
                and "mine" in name
                and "distance" in name
            ):
                distance_column = column
                break

    raw = aggregate_nearest_evidence(
        mining_df,
        event_df,
        columns_min=(
            [distance_column]
            if distance_column
            else []
        ),
    )

    distance_m = (
        raw.get(distance_column, np.nan)
        if distance_column
        else np.nan
    )

    return {
        "event_min_nearest_coal_mine_distance_m": distance_m,
        "event_min_nearest_coal_mine_distance_km": (
            distance_m / 1000.0
            if pd.notna(distance_m)
            else np.nan
        ),
    }


def build_agriculture_features(
    agriculture_df: pd.DataFrame,
    event_df: pd.DataFrame,
    agri_osm_distance: float,
) -> dict[str, float]:
    score_column = None

    for column in agriculture_df.columns:
        name = str(column).lower()

        if "agri" in name and "score" in name:
            score_column = column
            break

    raw = aggregate_nearest_evidence(
        agriculture_df,
        event_df,
        columns_max=(
            [score_column]
            if score_column
            else []
        ),
    )

    return {
        "agriculture_evidence_score_max": (
            raw.get(score_column, np.nan)
            if score_column
            else np.nan
        ),
        "agri_osm_distance_m_min": agri_osm_distance,
    }


def build_wildfire_features(
    wildfire_df: pd.DataFrame,
    event_df: pd.DataFrame,
) -> dict[str, float]:
    distance_columns = [
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

    raw = aggregate_nearest_evidence(
        wildfire_df,
        event_df,
        columns_min=distance_columns,
    )

    result = {
        f"{column}_min": raw.get(
            column,
            np.nan,
        )
        for column in distance_columns
    }

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

    result[
        "natural_landcover_context_max"
    ] = (
        natural_raw.get(
            natural_context_column,
            np.nan,
        )
        if natural_context_column
        else np.nan
    )

    result[
        "natural_vegetation_strong_max"
    ] = (
        natural_raw.get(
            natural_strong_column,
            np.nan,
        )
        if natural_strong_column
        else np.nan
    )

    result[
        "natural_vegetation_moderate_max"
    ] = (
        natural_raw.get(
            natural_moderate_column,
            np.nan,
        )
        if natural_moderate_column
        else np.nan
    )

    return result
