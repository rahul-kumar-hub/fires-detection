from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FLARE_PATH
from .events import haversine_m


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


def build_flare_features(
    selected: pd.Series,
) -> dict:

    flare_features = {
        "nearest_flare_distance_km": np.nan,
        "nearest_flare_2025_activity": np.nan,

        "active_flare_count_within_1km": 0,
        "has_active_flare_within_1km": 0,

        "active_flare_count_within_2km": 0,
        "has_active_flare_within_2km": 0,

        "active_flare_count_within_5km": 0,
        "has_active_flare_within_5km": 0,

        "active_flare_count_within_10km": 0,
        "has_active_flare_within_10km": 0,

        "nearest_gas_flare_distance_km": np.nan,
        "nearest_gas_flare_2025_activity": np.nan,

        "gas_flare_count_within_1km": 0,
        "has_gas_flare_within_1km": 0,

        "gas_flare_count_within_2km": 0,
        "has_gas_flare_within_2km": 0,

        "gas_flare_count_within_5km": 0,
        "has_gas_flare_within_5km": 0,

        "gas_flare_count_within_10km": 0,
        "has_gas_flare_within_10km": 0,

        "nearest_flare_is_gas": 0,
        "nearest_flare_is_oil": 0,
        "nearest_flare_is_unknown": 0,
    }

    try:
        flare_df = pd.read_excel(
            FLARE_PATH,
            sheet_name=(
                "2012-2025-Flare-Volume-Estimate"
            ),
        )

        flare_df.columns = [
            str(column).strip()
            for column in flare_df.columns
        ]

        flare_lat_col = (
            find_column(
                flare_df,
                exact="Latitude",
            )
            or
            find_column(
                flare_df,
                contains=["lat"],
            )
        )

        flare_lon_col = (
            find_column(
                flare_df,
                exact="Longitude",
            )
            or
            find_column(
                flare_df,
                contains=["lon"],
            )
        )

        if (
            flare_lat_col is None
            or flare_lon_col is None
        ):
            return flare_features

        flare_df["_lat"] = pd.to_numeric(
            flare_df[flare_lat_col],
            errors="coerce",
        )

        flare_df["_lon"] = pd.to_numeric(
            flare_df[flare_lon_col],
            errors="coerce",
        )

        country_col = find_column(
            flare_df,
            exact="Country",
        )

        if country_col:
            india = flare_df[
                flare_df[country_col]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                "india"
            ].copy()

        else:
            india = flare_df[
                (
                    flare_df["_lat"] >= 6
                )
                &
                (
                    flare_df["_lat"] <= 38
                )
                &
                (
                    flare_df["_lon"] >= 67
                )
                &
                (
                    flare_df["_lon"] <= 98
                )
            ].copy()

        activity_col = None

        for column in india.columns:
            column_lower = str(column).lower()

            if (
                "2025" in column_lower
                and
                (
                    "volume" in column_lower
                    or
                    "flare" in column_lower
                )
            ):
                activity_col = column
                break

        if activity_col:
            india["_activity"] = pd.to_numeric(
                india[activity_col],
                errors="coerce",
            ).fillna(0)

        else:
            india["_activity"] = 0.0

        india = india[
            india["_activity"] > 0
        ].copy()

        if len(india) == 0:
            print(
                "No active India 2025 flares."
            )
            return flare_features

        india["_distance_m"] = haversine_m(
            selected["latitude"],
            selected["longitude"],
            india["_lat"].values,
            india["_lon"].values,
        )

        nearest = (
            india
            .sort_values("_distance_m")
            .iloc[0]
        )

        flare_features[
            "nearest_flare_distance_km"
        ] = (
            nearest["_distance_m"] /
            1000
        )

        flare_features[
            "nearest_flare_2025_activity"
        ] = nearest["_activity"]

        for radius in [1, 2, 5, 10]:

            mask = (
                india["_distance_m"]
                <=
                radius * 1000
            )

            flare_features[
                f"active_flare_count_within_{radius}km"
            ] = int(mask.sum())

            flare_features[
                f"has_active_flare_within_{radius}km"
            ] = int(mask.any())

        field_col = None

        for column in india.columns:
            column_lower = str(column).lower()

            if (
                "field" in column_lower
                and
                "type" in column_lower
            ):
                field_col = column
                break

        if field_col:
            nearest_type = str(
                nearest[field_col]
            ).lower()
        else:
            nearest_type = ""

        flare_features[
            "nearest_flare_is_gas"
        ] = int(
            "gas" in nearest_type
        )

        flare_features[
            "nearest_flare_is_oil"
        ] = int(
            "oil" in nearest_type
        )

        flare_features[
            "nearest_flare_is_unknown"
        ] = int(
            "gas" not in nearest_type
            and
            "oil" not in nearest_type
        )

        if field_col:

            gas = india[
                india[field_col]
                .astype(str)
                .str.lower()
                .str.contains(
                    "gas",
                    na=False,
                )
            ].copy()

        else:
            gas = pd.DataFrame()

        if len(gas):

            gas["_distance_m"] = haversine_m(
                selected["latitude"],
                selected["longitude"],
                gas["_lat"].values,
                gas["_lon"].values,
            )

            nearest_gas = (
                gas
                .sort_values("_distance_m")
                .iloc[0]
            )

            flare_features[
                "nearest_gas_flare_distance_km"
            ] = (
                nearest_gas["_distance_m"] /
                1000
            )

            flare_features[
                "nearest_gas_flare_2025_activity"
            ] = nearest_gas["_activity"]

            for radius in [1, 2, 5, 10]:

                mask = (
                    gas["_distance_m"]
                    <=
                    radius * 1000
                )

                flare_features[
                    f"gas_flare_count_within_{radius}km"
                ] = int(mask.sum())

                flare_features[
                    f"has_gas_flare_within_{radius}km"
                ] = int(mask.any())

        print(
            "Active India 2025 flares:",
            len(india),
        )

    except Exception as exc:
        print(
            "⚠ Flare evidence error:",
            exc,
        )

    return flare_features
