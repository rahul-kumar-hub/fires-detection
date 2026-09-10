# src/firms.py

from __future__ import annotations

import math
from datetime import datetime, timezone
from io import StringIO
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests

from backend.config import (
    FIRMS_AREA_URL,
    FIRMS_WFS_URL,
    FIRMS_SOURCES,
    LIVE_SEARCH_RADIUS_KM,
)


# ============================================================
# Constants
# ============================================================

EARTH_RADIUS_KM = 6371.0088

# Same as Cell 2A
PERSISTENCE_RADIUS_KM = 1.0

REQUEST_TIMEOUT = 60

# WFS layers used by the NASA FIRMS South Asia WFS endpoint
WFS_LAYER_MAP = {
    "VIIRS_SNPP_NRT": "ms:fires_snpp_7days",
    "VIIRS_NOAA20_NRT": "ms:fires_noaa20_7days",
    "VIIRS_NOAA21_NRT": "ms:fires_noaa21_7days",
}

REQUIRED_LIVE_FIELDS = [
    "latitude",
    "longitude",
    "brightness",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_t31",
    "frp",
    "daynight",
]


# ============================================================
# Basic helpers
# ============================================================

def haversine_km(
    lat1: float,
    lon1: float,
    lat2,
    lon2,
) -> np.ndarray:
    """
    Vectorized Haversine distance.

    Returns distance in kilometres.
    """
    lat1 = np.radians(float(lat1))
    lon1 = np.radians(float(lon1))

    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    a = np.clip(a, 0.0, 1.0)

    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def make_bbox(
    lat: float,
    lon: float,
    radius_km: float,
) -> Tuple[float, float, float, float]:
    """
    Return bbox as:

        west, south, east, north

    Same basic logic as Cell 2A.
    """
    lat_delta = radius_km / EARTH_RADIUS_KM

    lon_radius = radius_km / (
        EARTH_RADIUS_KM * max(math.cos(math.radians(lat)), 1e-12)
    )

    south = max(-90.0, lat - math.degrees(lat_delta))
    north = min(90.0, lat + math.degrees(lat_delta))
    west = lon - math.degrees(lon_radius)
    east = lon + math.degrees(lon_radius)

    # Normalize longitude into [-180, 180]
    if west < -180:
        west += 360
    if east > 180:
        east -= 360

    return west, south, east, north


def normalize_firms_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize FIRMS column names exactly in the spirit of Cell 2A.

    Important mappings:
        bright_ti4 -> brightness
        bright_ti5 -> bright_t31
    """
    df = df.copy()

    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    rename_map = {
        "bright_ti4": "brightness",
        "bright_ti5": "bright_t31",
    }

    df = df.rename(columns=rename_map)

    return df


def parse_hotspot_datetime(
    acq_date,
    acq_time,
) -> datetime:
    """
    Convert FIRMS acq_date + acq_time into a UTC datetime.

    FIRMS acq_time is generally HHMM or HHMMSS.
    """
    date_str = str(acq_date).strip()
    time_str = str(acq_time).strip()

    # Remove decimal artifacts such as 2015.0 / 1234.0
    if time_str.endswith(".0"):
        time_str = time_str[:-2]

    time_str = time_str.zfill(4)

    if len(time_str) == 4:
        fmt = "%Y-%m-%d%H%M"
        combined = f"{date_str}{time_str}"

    elif len(time_str) == 6:
        fmt = "%Y-%m-%d%H%M%S"
        combined = f"{date_str}{time_str}"

    else:
        raise ValueError(
            f"Unsupported FIRMS acq_time format: {acq_time!r}"
        )

    return datetime.strptime(
        combined,
        fmt,
    ).replace(tzinfo=timezone.utc)


# ============================================================
# WFS discovery
# ============================================================

def fetch_firms_wfs(
    map_key: str,
    source: str,
    bbox: Tuple[float, float, float, float],
) -> pd.DataFrame:
    """
    Fetch the latest 7-day FIRMS detections using the WFS endpoint.

    This is the important part missing from the old firms.py.

    Endpoint:
        https://firms.modaps.eosdis.nasa.gov/mapserver/wfs/South_Asia/{MAP_KEY}/
    """
    if source not in WFS_LAYER_MAP:
        raise ValueError(
            f"Unsupported FIRMS source: {source}"
        )

    west, south, east, north = bbox

    url = f"{FIRMS_WFS_URL}/South_Asia/{map_key}/"

    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAME": WFS_LAYER_MAP[source],
        "OUTPUTFORMAT": "csv",
        "SRSNAME": "urn:ogc:def:crs:EPSG::4326",
        # Cell 2A uses:
        # south,west,north,east
        "BBOX": f"{south},{west},{north},{east}",
        "COUNT": 1000,
        "STARTINDEX": 0,
    }

    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    text = response.text

    if not text.strip():
        return pd.DataFrame()

    df = pd.read_csv(StringIO(text))

    if df.empty:
        return df

    df = normalize_firms_columns(df)

    # Ensure coordinates are numeric
    if "latitude" in df.columns:
        df["latitude"] = pd.to_numeric(
            df["latitude"],
            errors="coerce",
        )

    if "longitude" in df.columns:
        df["longitude"] = pd.to_numeric(
            df["longitude"],
            errors="coerce",
        )

    df = df.dropna(
        subset=["latitude", "longitude"]
    ).reset_index(drop=True)

    return df


def discover_wfs_candidates(
    lat: float,
    lon: float,
    map_key: str,
    radius_km: float = LIVE_SEARCH_RADIUS_KM,
) -> pd.DataFrame:
    """
    Search all three FIRMS VIIRS sources through WFS
    and return candidates inside radius_km.

    Each source is tagged in the resulting dataframe.
    """
    bbox = make_bbox(
        lat,
        lon,
        radius_km,
    )

    candidates = []

    for source in FIRMS_SOURCES:

        try:
            df = fetch_firms_wfs(
                map_key=map_key,
                source=source,
                bbox=bbox,
            )

        except requests.RequestException as exc:
            print(
                f"WFS request failed for {source}: {exc}"
            )
            continue

        except Exception as exc:
            print(
                f"WFS processing failed for {source}: {exc}"
            )
            continue

        if df.empty:
            continue

        df = df.copy()
        df["firms_source"] = source

        df["distance_km"] = haversine_km(
            lat,
            lon,
            df["latitude"].to_numpy(),
            df["longitude"].to_numpy(),
        )

        df = df[
            df["distance_km"] <= radius_km
        ].copy()

        if not df.empty:
            candidates.append(df)

    if not candidates:
        return pd.DataFrame()

    result = pd.concat(
        candidates,
        ignore_index=True,
    )

    result = result.sort_values(
        "distance_km",
        ascending=True,
    ).reset_index(drop=True)

    return result


# ============================================================
# Area API exact-date lookup
# ============================================================

def fetch_firms_area_exact_date(
    map_key: str,
    source: str,
    hotspot_lat: float,
    hotspot_lon: float,
    date_str: str,
) -> pd.DataFrame:
    """
    Query the FIRMS Area API for the exact acquisition date
    around the WFS-discovered hotspot.

    Cell 2A uses a 2 km bbox and a 1-day interval.
    """
    bbox = make_bbox(
        hotspot_lat,
        hotspot_lon,
        2.0,
    )

    west, south, east, north = bbox

    bbox_string = (
        f"{west},{south},{east},{north}"
    )

    url = (
        f"{FIRMS_AREA_URL}/"
        f"{map_key}/"
        f"{source}/"
        f"{bbox_string}/"
        f"1/"
        f"{date_str}"
    )

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    text = response.text

    if not text.strip():
        return pd.DataFrame()

    df = pd.read_csv(
        StringIO(text)
    )

    if df.empty:
        return df

    return normalize_firms_columns(df)


def select_matching_area_detection(
    area_df: pd.DataFrame,
    wfs_lat: float,
    wfs_lon: float,
) -> pd.Series | None:
    """
    Match Area API results to the WFS hotspot.

    Preferred rule:
        records within 1 km

    Fallback:
        closest Area API record.
    """
    if area_df.empty:
        return None

    df = area_df.copy()

    if "latitude" not in df.columns:
        return None

    if "longitude" not in df.columns:
        return None

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["latitude", "longitude"]
    ).reset_index(drop=True)

    if df.empty:
        return None

    df["_distance_km"] = haversine_km(
        wfs_lat,
        wfs_lon,
        df["latitude"].to_numpy(),
        df["longitude"].to_numpy(),
    )

    within_1km = df[
        df["_distance_km"] <= PERSISTENCE_RADIUS_KM
    ]

    if not within_1km.empty:
        idx = within_1km["_distance_km"].idxmin()
    else:
        idx = df["_distance_km"].idxmin()

    row = df.loc[idx].copy()

    row["area_match_distance_km"] = float(
        row["_distance_km"]
    )

    row = row.drop(
        labels=["_distance_km"],
        errors="ignore",
    )

    return row


# ============================================================
# Complete live hotspot discovery
# ============================================================

def find_live_hotspot(
    lat: float,
    lon: float,
    map_key: str,
    search_radius_km: float = LIVE_SEARCH_RADIUS_KM,
) -> Dict:
    """
    Exact Cell 2A-style live hotspot workflow.

    1. Search the three VIIRS FIRMS WFS layers for the
       latest 7-day detections.
    2. Keep detections within search_radius_km.
    3. Select the closest WFS hotspot.
    4. Use that hotspot's source and acquisition date.
    5. Query the Area API for that exact date.
    6. Match the Area API record to the WFS hotspot.
    7. Verify all required FIRMS fields.
    8. Return the complete live FIRMS record.
    """

    if not (
        -90.0 <= float(lat) <= 90.0
    ):
        raise ValueError(
            f"Latitude out of range: {lat}"
        )

    if not (
        -180.0 <= float(lon) <= 180.0
    ):
        raise ValueError(
            f"Longitude out of range: {lon}"
        )

    # --------------------------------------------------------
    # Step 1: WFS 7-day discovery
    # --------------------------------------------------------

    candidates = discover_wfs_candidates(
        lat=float(lat),
        lon=float(lon),
        map_key=map_key,
        radius_km=search_radius_km,
    )

    if candidates.empty:
        raise RuntimeError(
            "No live NASA FIRMS VIIRS hotspot found "
            f"within {search_radius_km:.1f} km "
            f"of ({lat}, {lon})."
        )

    # --------------------------------------------------------
    # Step 2: Closest WFS hotspot
    # --------------------------------------------------------

    wfs_row = candidates.iloc[0].copy()

    wfs_lat = float(
        wfs_row["latitude"]
    )

    wfs_lon = float(
        wfs_row["longitude"]
    )

    source = str(
        wfs_row["firms_source"]
    )

    # --------------------------------------------------------
    # Step 3: Determine date for Area API
    # --------------------------------------------------------

    if "acq_date" not in wfs_row.index:
        raise RuntimeError(
            "WFS hotspot does not contain acq_date."
        )

    date_str = str(
        wfs_row["acq_date"]
    ).strip()

    # --------------------------------------------------------
    # Step 4: Exact-date Area API lookup
    # --------------------------------------------------------

    try:
        area_df = fetch_firms_area_exact_date(
            map_key=map_key,
            source=source,
            hotspot_lat=wfs_lat,
            hotspot_lon=wfs_lon,
            date_str=date_str,
        )

    except requests.RequestException as exc:
        raise RuntimeError(
            "NASA FIRMS Area API request failed "
            f"for source {source}, date {date_str}: {exc}"
        ) from exc

    # --------------------------------------------------------
    # Step 5: Match Area record to WFS hotspot
    # --------------------------------------------------------

    area_row = select_matching_area_detection(
        area_df=area_df,
        wfs_lat=wfs_lat,
        wfs_lon=wfs_lon,
    )

    if area_row is None:
        raise RuntimeError(
            "NASA FIRMS WFS found a live hotspot, but "
            "the exact-date Area API returned no matching "
            "detection."
        )

    # --------------------------------------------------------
    # Step 6: Verify complete FIRMS record
    # --------------------------------------------------------

    missing_fields = [
        col
        for col in REQUIRED_LIVE_FIELDS
        if col not in area_row.index
    ]

    if missing_fields:
        raise RuntimeError(
            "Live FIRMS Area record is missing required "
            f"fields: {missing_fields}"
        )

    # --------------------------------------------------------
    # Step 7: Build clean live record
    # --------------------------------------------------------

    live = area_row.to_dict()

    # Numeric FIRMS fields
    numeric_fields = [
        "latitude",
        "longitude",
        "brightness",
        "scan",
        "track",
        "bright_t31",
        "frp",
    ]

    for col in numeric_fields:
        live[col] = pd.to_numeric(
            live[col],
            errors="coerce",
        )

    # Check required numeric values
    numeric_required = [
        "latitude",
        "longitude",
        "brightness",
        "scan",
        "track",
        "bright_t31",
        "frp",
    ]

    bad_numeric = [
        col
        for col in numeric_required
        if pd.isna(live[col])
    ]

    if bad_numeric:
        raise RuntimeError(
            "Live FIRMS record contains invalid numeric "
            f"values in: {bad_numeric}"
        )

    # Normalize strings
    live["satellite"] = str(
        live["satellite"]
    ).strip()

    live["instrument"] = str(
        live["instrument"]
    ).strip()

    live["confidence"] = str(
        live["confidence"]
    ).strip().lower()

    live["version"] = str(
        live["version"]
    ).strip()

    live["daynight"] = str(
        live["daynight"]
    ).strip().lower()

    # --------------------------------------------------------
    # Step 8: Parse exact hotspot UTC datetime
    # --------------------------------------------------------

    hotspot_datetime = parse_hotspot_datetime(
        live["acq_date"],
        live["acq_time"],
    )

    # --------------------------------------------------------
    # Step 9: Add metadata used later
    # --------------------------------------------------------

    live["firms_source"] = source

    live["wfs_distance_km"] = float(
        wfs_row["distance_km"]
    )

    live["area_match_distance_km"] = float(
        area_row.get(
            "area_match_distance_km",
            np.nan,
        )
    )

    live["hotspot_datetime"] = hotspot_datetime

    return live


# ============================================================
# FIRMS-derived ML features
# ============================================================

def build_firms_features(
    live_hotspot: Dict,
) -> Dict:
    """
    Build FIRMS-derived engineered features used by the
    trained 74-feature model.

    Exact engineered features from Cell 2A:

        frp_log
        confidence_low
        confidence_nominal
        confidence_high
        hour_sin
        hour_cos
    """

    frp = float(
        live_hotspot["frp"]
    )

    hotspot_datetime = live_hotspot[
        "hotspot_datetime"
    ]

    # --------------------------------------------------------
    # FRP log
    # --------------------------------------------------------

    frp_log = np.log1p(
        max(frp, 0.0)
    )

    # --------------------------------------------------------
    # Confidence one-hot
    # --------------------------------------------------------

    confidence = str(
        live_hotspot["confidence"]
    ).strip().lower()

    confidence_low = int(
        confidence == "l"
    )

    confidence_nominal = int(
        confidence == "n"
    )

    confidence_high = int(
        confidence == "h"
    )

    # --------------------------------------------------------
    # Hour cyclic encoding
    # --------------------------------------------------------

    hour = (
        hotspot_datetime.hour
        + hotspot_datetime.minute / 60.0
        + hotspot_datetime.second / 3600.0
    )

    hour_angle = (
        2.0 * np.pi * hour / 24.0
    )

    hour_sin = np.sin(
        hour_angle
    )

    hour_cos = np.cos(
        hour_angle
    )

    return {
        "frp_log": float(frp_log),
        "confidence_low": confidence_low,
        "confidence_nominal": confidence_nominal,
        "confidence_high": confidence_high,
        "hour_sin": float(hour_sin),
        "hour_cos": float(hour_cos),
    }


# ============================================================
# Convenience function
# ============================================================

def get_live_firms_features(
    lat: float,
    lon: float,
    map_key: str,
    search_radius_km: float = LIVE_SEARCH_RADIUS_KM,
):
    """
    Convenience wrapper.

    Returns:
        live_hotspot, firms_features
    """

    live_hotspot = find_live_hotspot(
        lat=lat,
        lon=lon,
        map_key=map_key,
        search_radius_km=search_radius_km,
    )

    firms_features = build_firms_features(
        live_hotspot
    )

    return live_hotspot, firms_features