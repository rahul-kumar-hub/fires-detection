from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EVENT_RADIUS_KM, EVENT_TIME_HOURS


def haversine_m(
    lat1,
    lon1,
    lat2,
    lon2,
):
    R = 6371000.0

    lat1 = np.radians(
        np.asarray(lat1, dtype=float)
    )

    lat2 = np.radians(
        np.asarray(lat2, dtype=float)
    )

    dlat = lat2 - lat1

    dlon = np.radians(
        np.asarray(lon2, dtype=float)
        -
        np.asarray(lon1, dtype=float)
    )

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        *
        R
        *
        np.arcsin(
            np.sqrt(a)
        )
    )


def build_event_context(
    firms_current: pd.DataFrame,
    selected: pd.Series,
    selected_datetime: pd.Timestamp,
) -> pd.DataFrame:

    firms_current = firms_current.copy()

    firms_current[
        "event_distance_m"
    ] = haversine_m(
        selected["latitude"],
        selected["longitude"],
        firms_current["latitude"].values,
        firms_current["longitude"].values,
    )

    firms_current[
        "event_time_difference_hours"
    ] = (
        (
            firms_current["acq_datetime"]
            -
            selected_datetime
        )
        .abs()
        .dt.total_seconds()
        /
        3600.0
    )

    event_df = firms_current[
        (
            firms_current[
                "event_distance_m"
            ]
            <=
            EVENT_RADIUS_KM * 1000
        )
        &
        (
            firms_current[
                "event_time_difference_hours"
            ]
            <=
            EVENT_TIME_HOURS
        )
    ].copy()

    if len(event_df) == 0:
        event_df = pd.DataFrame(
            [selected]
        )

    event_df = (
        event_df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return event_df
