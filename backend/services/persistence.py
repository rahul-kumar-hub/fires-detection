# src/persistence.py

from __future__ import annotations

import math
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import requests

from backend.config import (
    FIRMS_AREA_URL,
    FIRMS_SOURCES,
    PERSISTENCE_RADIUS_KM,
)
from backend.services.firms import normalize_firms_columns


# ============================================================
# TRAINING PERSISTENCE REFERENCE VALUES
# ============================================================

# These are the original training-data P95 values used by
# the working inference pipeline.
P95_7 = 18.0
P95_30 = 68.0

REQUEST_TIMEOUT = 90


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

EARTH_RADIUS_KM = 6371.0088


def haversine_vectorized(
    lat1: float,
    lon1: float,
    lat2,
    lon2,
) -> np.ndarray:
    """
    Vectorized Haversine distance.

    Returns distance in kilometres.
    """

    lat1_rad = np.radians(float(lat1))
    lon1_rad = np.radians(float(lon1))

    lat2_rad = np.radians(
        np.asarray(lat2, dtype=float)
    )
    lon2_rad = np.radians(
        np.asarray(lon2, dtype=float)
    )

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(dlon / 2.0) ** 2
    )

    a = np.clip(a, 0.0, 1.0)

    return (
        2.0
        * EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(a))
    )


# ============================================================
# BUILD PERSISTENCE BBOX
# ============================================================

def make_persistence_bbox(
    target_lat: float,
    target_lon: float,
    radius_km: float,
) -> str:
    """
    Build the same small spatial BBOX used by Cell 2A-P.

    Cell 2A-P uses:
        radius / 111.0
        radius / (111.0 * abs(cos(latitude)))

    Returned format:
        west,south,east,north
    """

    lat_delta = radius_km / 111.0

    cos_lat = max(
        abs(
            np.cos(
                np.radians(target_lat)
            )
        ),
        0.01,
    )

    lon_delta = (
        radius_km
        / (111.0 * cos_lat)
    )

    south = max(
        -90.0,
        target_lat - lat_delta,
    )

    north = min(
        90.0,
        target_lat + lat_delta,
    )

    west = max(
        -180.0,
        target_lon - lon_delta,
    )

    east = min(
        180.0,
        target_lon + lon_delta,
    )

    return (
        f"{west:.6f},"
        f"{south:.6f},"
        f"{east:.6f},"
        f"{north:.6f}"
    )


# ============================================================
# FETCH ONE VALID FIRMS HISTORY CHUNK
# ============================================================

def fetch_firms_history_chunk(
    map_key: str,
    source: str,
    start_date: pd.Timestamp,
    day_count: int,
    target_lat: float,
    target_lon: float,
) -> pd.DataFrame:
    """
    Fetch at most 5 days from NASA FIRMS.

    This follows Cell 2A-P exactly:
        DAY_RANGE must be between 1 and 5.
    """

    if not (
        1 <= day_count <= 5
    ):
        raise ValueError(
            "FIRMS DAY_RANGE must be between 1 and 5."
        )

    bbox = make_persistence_bbox(
        target_lat=target_lat,
        target_lon=target_lon,
        radius_km=PERSISTENCE_RADIUS_KM,
    )

    url = (
        f"{FIRMS_AREA_URL}/"
        f"{map_key}/"
        f"{source}/"
        f"{bbox}/"
        f"{day_count}/"
        f"{start_date.strftime('%Y-%m-%d')}"
    )

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    text = response.text.strip()

    if not text:
        return pd.DataFrame()

    if text.lower().startswith("<html"):
        raise RuntimeError(
            "FIRMS returned HTML instead of CSV."
        )

    df = pd.read_csv(
        StringIO(text)
    )

    if df.empty:
        return df

    return df


# ============================================================
# BUILD EXACT 30-DAY DATE WINDOWS
# ============================================================

def build_date_windows(
    target_date: pd.Timestamp,
):
    """
    Build Cell 2A-P's date windows:

        target_date - 29 days
        through
        target_date

    Each request is <= 5 days.
    """

    history_start = (
        target_date
        - pd.Timedelta(days=29)
    )

    history_end = target_date

    date_windows = []

    window_start = history_start

    while window_start <= history_end:

        remaining_days = (
            history_end
            - window_start
        ).days + 1

        day_count = min(
            5,
            remaining_days,
        )

        date_windows.append(
            (
                window_start,
                day_count,
            )
        )

        window_start = (
            window_start
            + pd.Timedelta(
                days=day_count
            )
        )

    return (
        history_start,
        history_end,
        date_windows,
    )


# ============================================================
# PARSE FIRMS ACQUISITION DATETIME
# ============================================================

def parse_firms_datetime(
    row,
):
    """
    Reproduce the Cell 2A-P acquisition datetime logic.

    FIRMS acquisition time is converted into a UTC-aware
    pandas Timestamp.
    """

    date_value = row["acq_date"]

    if pd.isna(date_value):
        return pd.NaT

    raw_time = str(
        row.get(
            "acq_time",
            "",
        )
    ).strip()

    if "." in raw_time:
        raw_time = raw_time.split(".")[0]

    raw_time = raw_time.zfill(4)

    try:

        hour = int(
            raw_time[:2]
        )

        minute = int(
            raw_time[2:4]
        )

        if hour > 23 or minute > 59:
            return pd.NaT

        return pd.Timestamp(
            year=date_value.year,
            month=date_value.month,
            day=date_value.day,
            hour=hour,
            minute=minute,
            tz="UTC",
        )

    except Exception:
        return pd.NaT


# ============================================================
# DOWNLOAD COMPLETE 30-DAY HISTORY
# ============================================================

def fetch_persistence_history(
    map_key: str,
    hotspot_lat: float,
    hotspot_lon: float,
    target_datetime,
) -> pd.DataFrame:
    """
    Fetch the exact history needed for persistence.

    Matches Cell 2A-P:
        - 30-day history
        - valid 1-5 day API chunks
        - all three VIIRS sources
        - 1 km spatial request around target hotspot
    """

    target_datetime = pd.Timestamp(
        target_datetime
    )

    if target_datetime.tzinfo is None:
        target_datetime = (
            target_datetime.tz_localize("UTC")
        )
    else:
        target_datetime = (
            target_datetime.tz_convert("UTC")
        )

    target_date = (
        target_datetime
        .tz_convert(None)
        .normalize()
    )

    (
        history_start,
        history_end,
        date_windows,
    ) = build_date_windows(
        target_date
    )

    print("\n" + "=" * 60)
    print(
        "CALCULATING FIRMS PERSISTENCE"
    )
    print("=" * 60)

    print(
        f"\n30-day period: "
        f"{history_start.strftime('%Y-%m-%d')}"
        f" -> "
        f"{history_end.strftime('%Y-%m-%d')}"
    )

    print(
        f"FIRMS API chunks required: "
        f"{len(date_windows)} per source"
    )

    history_frames = []

    for source in FIRMS_SOURCES:

        print(
            f"\nSource: {source}"
        )

        for chunk_number, (
            chunk_start,
            day_count,
        ) in enumerate(
            date_windows,
            start=1,
        ):

            chunk_end = (
                chunk_start
                + pd.Timedelta(
                    days=day_count - 1
                )
            )

            print(
                f"  Chunk {chunk_number:02d}: "
                f"{chunk_start.strftime('%Y-%m-%d')}"
                f" -> "
                f"{chunk_end.strftime('%Y-%m-%d')}"
                f" ({day_count} days)"
            )

            try:

                chunk_df = (
                    fetch_firms_history_chunk(
                        map_key=map_key,
                        source=source,
                        start_date=chunk_start,
                        day_count=day_count,
                        target_lat=float(
                            hotspot_lat
                        ),
                        target_lon=float(
                            hotspot_lon
                        ),
                    )
                )

                if chunk_df.empty:

                    print(
                        "    - No records"
                    )

                    continue

                chunk_df[
                    "firms_source"
                ] = source

                history_frames.append(
                    chunk_df
                )

                print(
                    f"    ✓ {len(chunk_df)} records"
                )

            except Exception as exc:

                print(
                    f"    ! Request failed: {exc}"
                )

    if not history_frames:
        return pd.DataFrame()

    firms_history = pd.concat(
        history_frames,
        ignore_index=True,
    )

    # Match the common normalization used by Cell 2A.
    firms_history = normalize_firms_columns(
        firms_history
    )

    print(
        f"\nCombined history records: "
        f"{len(firms_history)}"
    )

    # ========================================================
    # CLEAN HISTORY
    # ========================================================

    for column in [
        "latitude",
        "longitude",
    ]:

        if column in firms_history.columns:

            firms_history[column] = pd.to_numeric(
                firms_history[column],
                errors="coerce",
            )

    if "acq_date" not in firms_history.columns:
        raise RuntimeError(
            "FIRMS history does not contain acq_date."
        )

    if "acq_time" not in firms_history.columns:
        raise RuntimeError(
            "FIRMS history does not contain acq_time."
        )

    firms_history["acq_date"] = (
        pd.to_datetime(
            firms_history["acq_date"],
            errors="coerce",
        )
    )

    firms_history = firms_history.dropna(
        subset=[
            "latitude",
            "longitude",
            "acq_date",
        ]
    ).copy()

    # ========================================================
    # NORMALIZE ACQUISITION TIME
    # ========================================================

    firms_history["acq_time"] = (
        firms_history["acq_time"]
        .astype(str)
        .str.replace(
            ".0",
            "",
            regex=False,
        )
        .str.zfill(4)
    )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    dedup_cols = [
        "latitude",
        "longitude",
        "acq_date",
        "acq_time",
        "satellite",
        "instrument",
    ]

    dedup_cols = [
        column
        for column in dedup_cols
        if column in firms_history.columns
    ]

    firms_history = (
        firms_history
        .drop_duplicates(
            subset=dedup_cols
        )
        .reset_index(drop=True)
    )

    print(
        f"Records after duplicate removal: "
        f"{len(firms_history)}"
    )

    # ========================================================
    # BUILD EXACT FIRMS DATETIME
    # ========================================================

    firms_history[
        "datetime"
    ] = firms_history.apply(
        parse_firms_datetime,
        axis=1,
    )

    firms_history = firms_history.dropna(
        subset=["datetime"]
    ).copy()

    return firms_history


# ============================================================
# CALCULATE PERSISTENCE
# ============================================================

def calculate_persistence(
    map_key: str,
    hotspot_lat: float,
    hotspot_lon: float,
    hotspot_datetime,
):
    """
    Calculate the four persistence features required
    by the 74-feature Random Forest model.

    Matches Cell 2A-P:
        detections_7_days
        detections_30_days
        night_ratio
        persistence_score
    """

    # --------------------------------------------------------
    # Normalize target datetime to UTC
    # --------------------------------------------------------

    target_datetime = pd.Timestamp(
        hotspot_datetime
    )

    if target_datetime.tzinfo is None:

        target_datetime = (
            target_datetime.tz_localize("UTC")
        )

    else:

        target_datetime = (
            target_datetime.tz_convert("UTC")
        )

    # --------------------------------------------------------
    # Fetch history
    # --------------------------------------------------------

    firms_history = fetch_persistence_history(
        map_key=map_key,
        hotspot_lat=float(hotspot_lat),
        hotspot_lon=float(hotspot_lon),
        target_datetime=target_datetime,
    )

    # --------------------------------------------------------
    # No history available
    # --------------------------------------------------------

    if firms_history.empty:

        detections_7_days = 0
        detections_30_days = 0
        night_ratio = 0.0

    else:

        # ====================================================
        # SPATIAL FILTER — SAME 1 KM AS TRAINING
        # ====================================================

        firms_history[
            "distance_km"
        ] = haversine_vectorized(
            float(hotspot_lat),
            float(hotspot_lon),
            firms_history[
                "latitude"
            ].values,
            firms_history[
                "longitude"
            ].values,
        )

        nearby_history = firms_history[
            firms_history["distance_km"]
            <= PERSISTENCE_RADIUS_KM
        ].copy()

        print(
            f"\nRecords within "
            f"{PERSISTENCE_RADIUS_KM:.1f} km: "
            f"{len(nearby_history)}"
        )

        # ====================================================
        # STRICTLY PREVIOUS DETECTIONS
        # ====================================================

        previous_history = nearby_history[
            nearby_history["datetime"]
            < target_datetime
        ].copy()

        # ====================================================
        # EXACT WINDOWS
        # ====================================================

        window_30_start = (
            target_datetime
            - pd.Timedelta(days=30)
        )

        window_7_start = (
            target_datetime
            - pd.Timedelta(days=7)
        )

        previous_30 = previous_history[
            previous_history["datetime"]
            >= window_30_start
        ].copy()

        previous_7 = previous_history[
            previous_history["datetime"]
            >= window_7_start
        ].copy()

        # ====================================================
        # COUNTS
        # ====================================================

        detections_7_days = int(
            len(previous_7)
        )

        detections_30_days = int(
            len(previous_30)
        )

        # ====================================================
        # NIGHT RATIO
        # ====================================================

        if len(previous_30) > 0:

            if "daynight" in previous_30.columns:

                night_values = (
                    previous_30[
                        "daynight"
                    ]
                    .astype(str)
                    .str.upper()
                    .str.strip()
                )

                night_ratio = float(
                    (
                        night_values == "N"
                    ).mean()
                )

            else:

                night_ratio = 0.0

        else:

            night_ratio = 0.0

    # ========================================================
    # TRAINING P95 VALUES
    # ========================================================

    training_p95_7 = float(
        P95_7
    )

    training_p95_30 = float(
        P95_30
    )

    if training_p95_7 <= 0:
        training_p95_7 = 1.0

    if training_p95_30 <= 0:
        training_p95_30 = 1.0

    # ========================================================
    # PERSISTENCE SCORE
    # ========================================================

    score_7 = (
        detections_7_days
        / training_p95_7
    )

    score_30 = (
        detections_30_days
        / training_p95_30
    )

    score_7 = np.clip(
        score_7,
        0,
        1,
    )

    score_30 = np.clip(
        score_30,
        0,
        1,
    )

    persistence_score = (
        0.6 * score_7
        + 0.4 * score_30
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n" + "=" * 60)
    print(
        "CORRECTED FIRMS PERSISTENCE"
    )
    print("=" * 60)

    print(
        f"Previous 7-day detections : "
        f"{detections_7_days}"
    )

    print(
        f"Previous 30-day detections: "
        f"{detections_30_days}"
    )

    print(
        f"Night ratio               : "
        f"{night_ratio:.6f}"
    )

    print(
        f"Training P95 7            : "
        f"{training_p95_7:.6f}"
    )

    print(
        f"Training P95 30           : "
        f"{training_p95_30:.6f}"
    )

    print(
        f"Normalized 7-day score    : "
        f"{score_7:.6f}"
    )

    print(
        f"Normalized 30-day score   : "
        f"{score_30:.6f}"
    )

    print(
        f"Persistence score         : "
        f"{persistence_score:.6f}"
    )

    # ========================================================
    # RETURN EXACT FOUR FEATURES
    # ========================================================

    return {
        "detections_7_days": float(
            detections_7_days
        ),
        "detections_30_days": float(
            detections_30_days
        ),
        "night_ratio": float(
            night_ratio
        ),
        "persistence_score": float(
            persistence_score
        ),
    }