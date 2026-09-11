from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AGRICULTURE_PATH
from .evidence import aggregate_nearest_evidence


def load_agriculture_evidence() -> pd.DataFrame:
    if not AGRICULTURE_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        AGRICULTURE_PATH
    )


def build_agriculture_features(
    agriculture_df: pd.DataFrame,
    event_df: pd.DataFrame,
    agri_osm_distance,
) -> dict:

    agri_score_column = None

    for column in agriculture_df.columns:
        column_lower = str(column).lower()

        if (
            "agri" in column_lower
            and
            "score" in column_lower
        ):
            agri_score_column = column
            break

    raw = aggregate_nearest_evidence(
        agriculture_df,
        event_df,
        columns_max=(
            [agri_score_column]
            if agri_score_column
            else []
        ),
    )

    agriculture_score = (
        raw.get(
            agri_score_column,
            np.nan,
        )
        if agri_score_column
        else np.nan
    )

    return {
        "agriculture_evidence_score_max":
            agriculture_score,

        "agri_osm_distance_m_min":
            agri_osm_distance,
    }
