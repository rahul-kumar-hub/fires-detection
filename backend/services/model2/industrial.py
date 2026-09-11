from __future__ import annotations

import pandas as pd

from .config import INDUSTRIAL_PATH
from .evidence import aggregate_nearest_evidence


def load_industrial_evidence() -> pd.DataFrame:
    if not INDUSTRIAL_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        INDUSTRIAL_PATH
    )


def build_industrial_features(
    industrial_df: pd.DataFrame,
    event_df: pd.DataFrame,
) -> dict:

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
        "event_min_steel_distance_m":
            raw["steel_distance_m"],

        "event_min_cement_distance_m":
            raw["cement_distance_m"],

        "event_min_wri_power_distance_m":
            raw["wri_power_distance_m"],

        "event_min_fertilizer_distance_m":
            raw["fertilizer_distance_m"],

        "event_min_refinery_petro_distance_m":
            raw["refinery_petro_distance_m"],

        "event_max_industrial_source_count_500m":
            raw["industrial_source_count_500m"],

        "event_max_industrial_source_count_1km":
            raw["industrial_source_count_1km"],

        "event_max_industrial_evidence_score":
            raw["industrial_evidence_score"],
    }
