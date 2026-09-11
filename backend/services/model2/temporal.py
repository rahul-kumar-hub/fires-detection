from __future__ import annotations

import numpy as np
import pandas as pd

from .events import haversine_m


def build_temporal_features(
    historical_df: pd.DataFrame,
    selected: pd.Series,
    selected_datetime: pd.Timestamp,
    history_available: bool,
) -> dict:

    temporal_features = {}

    if history_available:

        if len(historical_df):

            historical_df = historical_df.copy()

            historical_df[
                "hours_before_selected"
            ] = (
                (
                    selected_datetime
                    -
                    historical_df[
                        "acq_datetime"
                    ]
                )
                .dt.total_seconds()
                /
                3600
            )

            historical_df[
                "days_before_selected"
            ] = (
                historical_df[
                    "hours_before_selected"
                ]
                /
                24
            )

            lat_step_500 = (
                0.5 / 111.32
            )

            selected_lat500 = np.floor(
                selected["latitude"]
                /
                lat_step_500
            )

            selected_lon500 = np.floor(
                selected["longitude"]
                /
                (
                    0.5
                    /
                    (
                        111.32
                        *
                        np.cos(
                            np.radians(
                                selected["latitude"]
                            )
                        )
                    )
                )
            )

            historical_df["_lat500"] = np.floor(
                historical_df[
                    "latitude"
                ]
                /
                lat_step_500
            )

            historical_df["_lon500"] = np.floor(
                historical_df[
                    "longitude"
                ]
                /
                (
                    0.5
                    /
                    (
                        111.32
                        *
                        np.cos(
                            np.radians(
                                historical_df[
                                    "latitude"
                                ]
                            )
                        )
                    )
                )
            )

            same_cell = (
                (
                    historical_df[
                        "_lat500"
                    ]
                    ==
                    selected_lat500
                )
                &
                (
                    historical_df[
                        "_lon500"
                    ]
                    ==
                    selected_lon500
                )
            )

            nearby_1km = (
                historical_df[
                    "distance_to_selected_m"
                ]
                <=
                1000
            )

        else:
            same_cell = pd.Series(
                dtype=bool
            )

            nearby_1km = pd.Series(
                dtype=bool
            )

        for days in [3, 7, 30]:

            if len(historical_df):

                mask = (
                    historical_df[
                        "days_before_selected"
                    ]
                    <=
                    days
                )

                same_count = int(
                    (
                        same_cell
                        &
                        mask
                    ).sum()
                )

                nearby_count = int(
                    (
                        nearby_1km
                        &
                        mask
                    ).sum()
                )

            else:

                same_count = 0
                nearby_count = 0

            temporal_features[
                f"same_cell_fire_count_{days}d_max"
            ] = same_count

            temporal_features[
                f"nearby_fire_count_{days}d_1km_max"
            ] = nearby_count

            temporal_features[
                f"event_max_same_cell_fire_count_{days}d"
            ] = same_count

            temporal_features[
                f"event_max_nearby_fire_count_{days}d_1km"
            ] = nearby_count

    else:

        for days in [3, 7, 30]:

            temporal_features[
                f"same_cell_fire_count_{days}d_max"
            ] = np.nan

            temporal_features[
                f"nearby_fire_count_{days}d_1km_max"
            ] = np.nan

            temporal_features[
                f"event_max_same_cell_fire_count_{days}d"
            ] = np.nan

            temporal_features[
                f"event_max_nearby_fire_count_{days}d_1km"
            ] = np.nan

    return temporal_features
