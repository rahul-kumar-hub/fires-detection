from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_historical_firms(
    historical_df: pd.DataFrame,
    selected: pd.Series,
    selected_datetime: pd.Timestamp,
):
    if historical_df is None or historical_df.empty:
        return pd.DataFrame()

    historical_df = historical_df.copy()

    for column in [
        "latitude",
        "longitude",
        "frp",
    ]:
        if column in historical_df.columns:
            historical_df[column] = pd.to_numeric(
                historical_df[column],
                errors="coerce",
            )

    historical_df["acq_date"] = pd.to_datetime(
        historical_df["acq_date"],
        errors="coerce",
    )

    historical_df = historical_df.dropna(
        subset=[
            "latitude",
            "longitude",
            "acq_datetime",
        ]
    ).copy()

    history_start = (
        selected_datetime
        -
        pd.Timedelta(days=29)
    )

    historical_df = historical_df[
        (
            historical_df["acq_datetime"]
            >=
            history_start
        )
        &
        (
            historical_df["acq_datetime"]
            <=
            selected_datetime
        )
    ].copy()

    if historical_df.empty:
        return historical_df

    historical_df[
        "distance_to_selected_m"
    ] = haversine_m(
        selected["latitude"],
        selected["longitude"],
        historical_df["latitude"].values,
        historical_df["longitude"].values,
    )

    historical_df = historical_df[
        historical_df["distance_to_selected_m"] > 100
    ].copy()

    identity_columns = [
        column
        for column in [
            "latitude",
            "longitude",
            "acq_datetime",
            "frp",
        ]
        if column in historical_df.columns
    ]

    if identity_columns:
        historical_df = (
            historical_df
            .drop_duplicates(
                subset=identity_columns
            )
        )

    return historical_df


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
