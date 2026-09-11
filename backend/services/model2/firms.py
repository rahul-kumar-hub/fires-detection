from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import requests

from .config import (
    FIRMS_BBOX_DEG,
    FIRMS_SOURCES,
    FIRMS_SP_SOURCES,
)


def make_acq_datetime(df: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(
        df["acq_date"],
        errors="coerce",
    )

    time_numeric = pd.to_numeric(
        df["acq_time"],
        errors="coerce",
    )

    time_string = (
        time_numeric
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(4)
    )

    return pd.to_datetime(
        date.dt.strftime("%Y-%m-%d")
        + " "
        + time_string,
        format="%Y-%m-%d %H%M",
        errors="coerce",
    )


def firms_request(
    map_key: str,
    source: str,
    bbox_string: str,
    days: int = 5,
    date: str | None = None,
):
    if date is None:
        url = (
            "https://firms.modaps.eosdis.nasa.gov/"
            "api/area/csv/"
            f"{map_key}/"
            f"{source}/"
            f"{bbox_string}/"
            f"{days}"
        )
    else:
        url = (
            "https://firms.modaps.eosdis.nasa.gov/"
            "api/area/csv/"
            f"{map_key}/"
            f"{source}/"
            f"{bbox_string}/"
            f"{days}/"
            f"{date}"
        )

    try:
        response = requests.get(
            url,
            timeout=60,
        )

        if response.status_code != 200:
            return None, response.status_code

        if not response.text.strip():
            return pd.DataFrame(), 200

        return (
            pd.read_csv(
                StringIO(response.text)
            ),
            200,
        )

    except Exception:
        return None, None


def build_bbox(
    latitude: float,
    longitude: float,
) -> str:
    return (
        f"{longitude - FIRMS_BBOX_DEG},"
        f"{latitude - FIRMS_BBOX_DEG},"
        f"{longitude + FIRMS_BBOX_DEG},"
        f"{latitude + FIRMS_BBOX_DEG}"
    )


def fetch_current_firms(
    map_key: str,
    latitude: float,
    longitude: float,
    current_days: int,
):
    bbox = build_bbox(
        latitude,
        longitude,
    )

    frames = []
    successful_requests = 0

    for source in FIRMS_SOURCES:
        df_source, status_code = firms_request(
            map_key=map_key,
            source=source,
            bbox_string=bbox,
            days=current_days,
        )

        if (
            status_code != 200
            or df_source is None
        ):
            continue

        successful_requests += 1

        if len(df_source):
            df_source["firms_source_api"] = source
            frames.append(df_source)

    if successful_requests == 0:
        raise RuntimeError(
            "All current FIRMS requests failed."
        )

    if not frames:
        raise RuntimeError(
            "No FIRMS detection found within "
            "the requested area during the current period."
        )

    firms_current = pd.concat(
        frames,
        ignore_index=True,
    )

    numeric_columns = [
        "latitude",
        "longitude",
        "frp",
        "bright_ti4",
        "bright_ti5",
        "scan",
        "track",
    ]

    for column in numeric_columns:
        if column in firms_current.columns:
            firms_current[column] = pd.to_numeric(
                firms_current[column],
                errors="coerce",
            )

    firms_current["acq_date"] = pd.to_datetime(
        firms_current["acq_date"],
        errors="coerce",
    )

    firms_current["acq_datetime"] = make_acq_datetime(
        firms_current
    )

    firms_current = (
        firms_current
        .dropna(
            subset=[
                "latitude",
                "longitude",
                "acq_datetime",
            ]
        )
        .copy()
    )

    firms_current["distance_m_from_input"] = haversine_m(
        latitude,
        longitude,
        firms_current["latitude"].values,
        firms_current["longitude"].values,
    )

    firms_current = (
        firms_current
        .sort_values(
            [
                "distance_m_from_input",
                "acq_datetime",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return firms_current, bbox, successful_requests


def fetch_historical_firms(
    map_key: str,
    bbox: str,
    selected_date: pd.Timestamp,
):
    chunk_starts = [
        selected_date - pd.Timedelta(days=29),
        selected_date - pd.Timedelta(days=24),
        selected_date - pd.Timedelta(days=19),
        selected_date - pd.Timedelta(days=14),
        selected_date - pd.Timedelta(days=9),
        selected_date - pd.Timedelta(days=4),
    ]

    frames = []
    successful_requests = 0
    total_requests = 0

    for source in FIRMS_SOURCES:
        for start_date in chunk_starts:
            total_requests += 1

            df_hist, status_code = firms_request(
                map_key=map_key,
                source=source,
                bbox_string=bbox,
                days=5,
                date=start_date.strftime("%Y-%m-%d"),
            )

            if (
                status_code == 200
                and df_hist is not None
            ):
                successful_requests += 1

                if len(df_hist):
                    df_hist["firms_source_api"] = source
                    frames.append(df_hist)

    if successful_requests == 0:
        for source in FIRMS_SP_SOURCES:
            for start_date in chunk_starts:
                total_requests += 1

                df_hist, status_code = firms_request(
                    map_key=map_key,
                    source=source,
                    bbox_string=bbox,
                    days=5,
                    date=start_date.strftime("%Y-%m-%d"),
                )

                if (
                    status_code == 200
                    and df_hist is not None
                ):
                    successful_requests += 1

                    if len(df_hist):
                        df_hist["firms_source_api"] = source
                        frames.append(df_hist)

    if frames:
        historical_df = pd.concat(
            frames,
            ignore_index=True,
        )
    else:
        historical_df = pd.DataFrame()

    if len(historical_df):
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

        historical_df["acq_datetime"] = make_acq_datetime(
            historical_df
        )

        historical_df = historical_df.dropna(
            subset=[
                "latitude",
                "longitude",
                "acq_datetime",
            ]
        ).copy()

    return (
        historical_df,
        successful_requests,
        total_requests,
    )


def haversine_m(
    lat1,
    lon1,
    lat2,
    lon2,
):
    radius_m = 6371000.0

    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))

    dlat = lat2 - lat1
    dlon = np.radians(
        np.asarray(lon2, dtype=float)
        - np.asarray(lon1, dtype=float)
    )

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        * radius_m
        * np.arcsin(np.sqrt(a))
    )
