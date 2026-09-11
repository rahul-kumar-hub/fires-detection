from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MINING_PATH
from .evidence import aggregate_nearest_evidence


def load_mining_evidence() -> pd.DataFrame:
    if not MINING_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        MINING_PATH
    )


def build_mining_features(
    mining_df: pd.DataFrame,
    event_df: pd.DataFrame,
) -> dict:

    mine_distance_column = None

    known_mine_columns = [
        "nearest_coal_mine_distance_m",
        "coal_mine_distance_m",
        "event_min_nearest_coal_mine_distance_m",
        "nearest_mine_distance_m",
    ]

    for candidate in known_mine_columns:
        if candidate in mining_df.columns:
            mine_distance_column = candidate
            break

    if mine_distance_column is None:
        for column in mining_df.columns:
            column_lower = str(column).lower()

            if (
                "coal" in column_lower
                and
                "mine" in column_lower
                and
                "distance" in column_lower
            ):
                mine_distance_column = column
                break

    raw = aggregate_nearest_evidence(
        mining_df,
        event_df,
        columns_min=(
            [mine_distance_column]
            if mine_distance_column
            else []
        ),
    )

    mine_distance_m = (
        raw.get(
            mine_distance_column,
            np.nan,
        )
        if mine_distance_column
        else np.nan
    )

    return {
        "event_min_nearest_coal_mine_distance_m":
            mine_distance_m,

        "event_min_nearest_coal_mine_distance_km":
            (
                mine_distance_m / 1000
                if pd.notna(mine_distance_m)
                else np.nan
            ),
    }
