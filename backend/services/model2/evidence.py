from __future__ import annotations

import numpy as np
import pandas as pd

from .events import haversine_m


def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return np.nan

        return float(value)

    except Exception:
        return np.nan


def find_column(
    df: pd.DataFrame,
    exact: str | None = None,
    contains: list[str] | None = None,
):
    if df is None:
        return None

    columns = list(df.columns)

    if exact:
        target = str(exact).strip().lower()

        for column in columns:
            if (
                str(column).strip().lower()
                ==
                target
            ):
                return column

    if contains:
        for column in columns:
            column_lower = str(column).lower()

            if all(
                str(item).lower() in column_lower
                for item in contains
            ):
                return column

    return None


def aggregate_nearest_evidence(
    df: pd.DataFrame,
    event_df: pd.DataFrame,
    columns_min: list[str] | None = None,
    columns_max: list[str] | None = None,
) -> dict:

    columns_min = columns_min or []
    columns_max = columns_max or []

    result = {}

    for column in columns_min:
        result[column] = np.nan

    for column in columns_max:
        result[column] = np.nan

    if df is None or len(df) == 0:
        return result

    lat_col = (
        find_column(
            df,
            exact="latitude",
        )
        or
        find_column(
            df,
            exact="lat",
        )
    )

    lon_col = (
        find_column(
            df,
            exact="longitude",
        )
        or
        find_column(
            df,
            exact="lon",
        )
    )

    if (
        lat_col is None
        or lon_col is None
    ):
        return result

    wanted_columns = [
        column
        for column in (
            columns_min +
            columns_max
        )
        if column in df.columns
    ]

    source = df[
        [
            lat_col,
            lon_col,
            *wanted_columns,
        ]
    ].copy()

    source["_lat"] = pd.to_numeric(
        source[lat_col],
        errors="coerce",
    )

    source["_lon"] = pd.to_numeric(
        source[lon_col],
        errors="coerce",
    )

    source = source.dropna(
        subset=[
            "_lat",
            "_lon",
        ]
    )

    if len(source) == 0:
        return result

    source_lat = source["_lat"].to_numpy()

    source_lon = source["_lon"].to_numpy()

    values_min = {
        column: []
        for column in columns_min
    }

    values_max = {
        column: []
        for column in columns_max
    }

    event_latitudes = (
        event_df[
            "latitude"
        ].to_numpy()
    )

    event_longitudes = (
        event_df[
            "longitude"
        ].to_numpy()
    )

    for event_lat, event_lon in zip(
        event_latitudes,
        event_longitudes,
    ):

        distances = haversine_m(
            event_lat,
            event_lon,
            source_lat,
            source_lon,
        )

        nearest_idx = int(
            np.argmin(
                distances
            )
        )

        evidence_row = source.iloc[
            nearest_idx
        ]

        for column in columns_min:

            if column in evidence_row.index:

                value = safe_float(
                    evidence_row[column]
                )

                if pd.notna(value):
                    values_min[column].append(
                        value
                    )

        for column in columns_max:

            if column in evidence_row.index:

                value = safe_float(
                    evidence_row[column]
                )

                if pd.notna(value):
                    values_max[column].append(
                        value
                    )

    for column in columns_min:

        if values_min[column]:
            result[column] = min(
                values_min[column]
            )

    for column in columns_max:

        if values_max[column]:
            result[column] = max(
                values_max[column]
            )

    return result
