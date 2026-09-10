# src/osm.py

from __future__ import annotations

import math
import time

import numpy as np
import requests


# ============================================================
# CONFIGURATION
# ============================================================

OSM_RADIUS_KM = 25.0
NOT_FOUND_DISTANCE = 1e9

EARTH_RADIUS_KM = 6371.0088

# Same servers as the working Cell 2A.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


# ============================================================
# EXACT OSM FEATURE / TAG MAPPING
# ============================================================

# This is the important correction.
# These are the exact mappings used by Cell 2A.

OSM_FEATURE_TAGS = {

    "flare": [
        ("man_made", "flare")
    ],

    "oil_well": [
        ("man_made", "petroleum_well")
    ],

    "mineshaft": [
        ("man_made", "mineshaft")
    ],

    "adit": [
        ("man_made", "adit")
    ],

    "gasometer": [
        ("man_made", "gasometer")
    ],

    "industrial": [
        ("landuse", "industrial")
    ],

    "quarry": [
        ("landuse", "quarry")
    ],

    "farmland": [
        ("landuse", "farmland")
    ],

    "farmyard": [
        ("landuse", "farmyard")
    ],

    "orchard": [
        ("landuse", "orchard")
    ],

    "vineyard": [
        ("landuse", "vineyard")
    ],

    "plant_nursery": [
        ("landuse", "plant_nursery")
    ],

    "greenhouse": [
        ("building", "greenhouse")
    ],

    "allotment": [
        ("landuse", "allotments")
    ],

    "forest": [
        ("landuse", "forest")
    ],

    "scrub": [
        ("natural", "scrub")
    ],

    "grassland": [
        ("natural", "grassland")
    ],

    "heath": [
        ("natural", "heath")
    ],

    "agriculture": [
        ("landuse", "agricultural")
    ],

    "natural_vegetation": [
        ("natural", "vegetation")
    ],
}


# ============================================================
# HAVERSINE
# ============================================================

def haversine_vectorized(
    lat1: float,
    lon1: float,
    lat2,
    lon2,
) -> np.ndarray:
    """
    Vectorized Haversine distance.

    Returns kilometres.
    """

    lat1 = np.radians(float(lat1))
    lon1 = np.radians(float(lon1))

    lat2 = np.radians(
        np.asarray(
            lat2,
            dtype=float,
        )
    )

    lon2 = np.radians(
        np.asarray(
            lon2,
            dtype=float,
        )
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    a = np.clip(
        a,
        0.0,
        1.0,
    )

    return (
        2.0
        * EARTH_RADIUS_KM
        * np.arcsin(
            np.sqrt(a)
        )
    )


# ============================================================
# BBOX
# ============================================================

def make_bbox(
    lat: float,
    lon: float,
    radius_km: float,
):
    """
    Return:

        west, south, east, north
    """

    lat_delta = (
        radius_km
        / EARTH_RADIUS_KM
    )

    cos_lat = max(
        abs(
            math.cos(
                math.radians(lat)
            )
        ),
        0.01,
    )

    lon_delta = (
        radius_km
        / (
            EARTH_RADIUS_KM
            * cos_lat
        )
    )

    south = max(
        -90.0,
        lat
        - math.degrees(lat_delta),
    )

    north = min(
        90.0,
        lat
        + math.degrees(lat_delta),
    )

    west = max(
        -180.0,
        lon
        - math.degrees(lon_delta),
    )

    east = min(
        180.0,
        lon
        + math.degrees(lon_delta),
    )

    return (
        west,
        south,
        east,
        north,
    )


# ============================================================
# TARGETED OVERPASS QUERY
# ============================================================

def build_overpass_query(
    south: float,
    west: float,
    north: float,
    east: float,
) -> str:
    """
    Build the same targeted OSM query used by Cell 2A.

    For every feature we query:

        node
        way
        relation

    using its exact OSM key/value pair.
    """

    query_parts = []

    osm_tags = [

        ("man_made", "flare"),
        ("man_made", "petroleum_well"),
        ("man_made", "mineshaft"),
        ("man_made", "adit"),
        ("man_made", "gasometer"),

        ("landuse", "industrial"),
        ("landuse", "quarry"),
        ("landuse", "farmland"),
        ("landuse", "farmyard"),
        ("landuse", "orchard"),
        ("landuse", "vineyard"),
        ("landuse", "plant_nursery"),

        ("building", "greenhouse"),

        ("landuse", "allotments"),

        ("landuse", "forest"),

        ("natural", "scrub"),
        ("natural", "grassland"),
        ("natural", "heath"),

        ("landuse", "agricultural"),

        ("natural", "vegetation"),
    ]

    for key, value in osm_tags:

        query_parts.append(
            f'node["{key}"="{value}"]'
            f"({south},{west},{north},{east});"
        )

        query_parts.append(
            f'way["{key}"="{value}"]'
            f"({south},{west},{north},{east});"
        )

        query_parts.append(
            f'relation["{key}"="{value}"]'
            f"({south},{west},{north},{east});"
        )

    return f"""
[out:json][timeout:120];
(
{''.join(query_parts)}
);
out center;
"""


# ============================================================
# OVERPASS FAILOVER
# ============================================================

def query_overpass_with_failover(
    query: str,
    servers=None,
    max_attempts: int = 2,
    wait_seconds: int = 3,
):
    """
    Query Overpass with retries and server failover.

    Matches the approach used by Cell 2A.
    """

    if servers is None:
        servers = OVERPASS_URLS

    headers = {
        "User-Agent":
            "FireSourceClassification-Kaggle/1.0"
    }

    last_error = None

    for server in servers:

        print(
            "\nTrying Overpass server:"
            f"\n{server}"
        )

        for attempt in range(
            1,
            max_attempts + 1,
        ):

            try:

                response = requests.post(
                    server,
                    data=query.encode(
                        "utf-8"
                    ),
                    headers=headers,
                    timeout=180,
                )

                # ------------------------------------------------
                # Rate-limit handling
                # ------------------------------------------------

                if response.status_code == 429:

                    print(
                        f"  429 rate limited "
                        f"(attempt {attempt})"
                    )

                    last_error = (
                        f"429 from {server}"
                    )

                    if attempt < max_attempts:

                        time.sleep(
                            wait_seconds
                            * attempt
                        )

                    continue

                response.raise_for_status()

                data = response.json()

                print(
                    "  ✓ OSM query successful"
                )

                return data

            except requests.RequestException as exc:

                last_error = exc

                print(
                    f"  Attempt {attempt} "
                    f"failed: {exc}"
                )

                if attempt < max_attempts:

                    time.sleep(
                        wait_seconds
                        * attempt
                    )

            except Exception as exc:

                last_error = exc

                print(
                    f"  Unexpected error: "
                    f"{exc}"
                )

                if attempt < max_attempts:

                    time.sleep(
                        wait_seconds
                        * attempt
                    )

    raise RuntimeError(
        "All Overpass servers failed.\n"
        f"Last error: {last_error}"
    )


# ============================================================
# EXTRACT OSM OBJECTS
# ============================================================

def extract_osm_records(
    osm_data: dict,
):
    """
    Convert Overpass elements into records containing:

        type
        id
        lat
        lon
        tags
    """

    osm_records = []

    for element in osm_data.get(
        "elements",
        [],
    ):

        element_type = element.get(
            "type"
        )

        element_id = element.get(
            "id"
        )

        tags = element.get(
            "tags",
            {},
        )

        object_lat = None
        object_lon = None

        # --------------------------------------------------------
        # Node
        # --------------------------------------------------------

        if element_type == "node":

            object_lat = element.get(
                "lat"
            )

            object_lon = element.get(
                "lon"
            )

        # --------------------------------------------------------
        # Way / Relation center
        # --------------------------------------------------------

        elif "center" in element:

            object_lat = (
                element["center"].get(
                    "lat"
                )
            )

            object_lon = (
                element["center"].get(
                    "lon"
                )
            )

        if (
            object_lat is None
            or object_lon is None
        ):
            continue

        osm_records.append(
            {
                "type":
                    element_type,
                "id":
                    element_id,
                "lat":
                    float(
                        object_lat
                    ),
                "lon":
                    float(
                        object_lon
                    ),
                "tags":
                    tags,
            }
        )

    return osm_records


# ============================================================
# TAG MATCHING
# ============================================================

def osm_tag_matches(
    tags: dict,
    tag_pairs,
) -> bool:
    """
    Check whether an OSM object's tags match any of
    the requested key/value pairs.
    """

    for key, value in tag_pairs:

        actual = str(
            tags.get(
                key,
                "",
            )
        ).lower()

        expected = str(
            value
        ).lower()

        if actual == expected:
            return True

    return False


# ============================================================
# CALCULATE OSM DISTANCES
# ============================================================

def calculate_osm_distances(
    live_hotspot_lat: float,
    live_hotspot_lon: float,
    osm_records,
):
    """
    Calculate nearest distance for all 20 OSM features.

    Missing features use NOT_FOUND_DISTANCE.
    """

    osm_distance_dict = {}

    for (
        feature_name,
        tag_pairs,
    ) in OSM_FEATURE_TAGS.items():

        matching_points = []

        for record in osm_records:

            if osm_tag_matches(
                record["tags"],
                tag_pairs,
            ):

                matching_points.append(
                    (
                        record["lat"],
                        record["lon"],
                    )
                )

        if not matching_points:

            distance_m = (
                NOT_FOUND_DISTANCE
            )

        else:

            points = np.asarray(
                matching_points,
                dtype=float,
            )

            distances_km = (
                haversine_vectorized(
                    float(
                        live_hotspot_lat
                    ),
                    float(
                        live_hotspot_lon
                    ),
                    points[:, 0],
                    points[:, 1],
                )
            )

            distance_m = float(
                distances_km.min()
                * 1000.0
            )

        osm_distance_dict[
            f"distance_to_nearest_"
            f"{feature_name}_m"
        ] = distance_m

    return osm_distance_dict


# ============================================================
# LOG FEATURES
# ============================================================

def add_osm_log_features(
    osm_distance_dict: dict,
):
    """
    Create log1p-transformed OSM distance features.
    """

    result = dict(
        osm_distance_dict
    )

    for base_feature in list(
        osm_distance_dict.keys()
    ):

        if not base_feature.endswith(
            "_m"
        ):
            continue

        log_feature = (
            base_feature[:-2]
            + "_log"
        )

        value = float(
            osm_distance_dict[
                base_feature
            ]
        )

        result[log_feature] = (
            float(
                np.log1p(value)
            )
        )

    return result


# ============================================================
# SAFE MINIMUM
# ============================================================

def safe_min(
    osm_distance_dict: dict,
    feature_names,
) -> float:
    """
    Return the minimum valid OSM distance.

    NOT_FOUND_DISTANCE is ignored when another valid
    feature exists.
    """

    values = []

    for name in feature_names:

        key = (
            f"distance_to_nearest_"
            f"{name}_m"
        )

        if key not in osm_distance_dict:
            continue

        value = float(
            osm_distance_dict[key]
        )

        if (
            np.isfinite(value)
            and value
            < NOT_FOUND_DISTANCE
        ):
            values.append(value)

    if values:
        return float(
            min(values)
        )

    return float(
        NOT_FOUND_DISTANCE
    )


# ============================================================
# MINIMUM-DISTANCE FEATURES
# ============================================================

def build_minimum_distance_features(
    osm_distance_dict: dict,
):
    """
    Exact grouped minimum features used in training
    and Cell 2A.
    """

    return {

        "min_fossil_distance_m":
            safe_min(
                osm_distance_dict,
                [
                    "flare",
                    "oil_well",
                    "gasometer",
                ],
            ),

        "min_mining_distance_m":
            safe_min(
                osm_distance_dict,
                [
                    "mineshaft",
                    "adit",
                    "quarry",
                ],
            ),

        "min_agriculture_distance_m":
            safe_min(
                osm_distance_dict,
                [
                    "farmland",
                    "farmyard",
                    "orchard",
                    "vineyard",
                    "plant_nursery",
                    "greenhouse",
                    "allotment",
                    "agriculture",
                ],
            ),

        "min_wildland_distance_m":
            safe_min(
                osm_distance_dict,
                [
                    "forest",
                    "scrub",
                    "grassland",
                    "heath",
                    "natural_vegetation",
                ],
            ),
    }


# ============================================================
# MAIN OSM FEATURE FUNCTION
# ============================================================

def get_osm_features(
    lat: float,
    lon: float,
    radius_km: float = OSM_RADIUS_KM,
):
    """
    Run the complete Cell 2A-compatible OSM pipeline.

    Returns:

        osm_distance_dict
        osm_features

    osm_features contains:

        20 raw OSM distance features
        20 OSM log features
        4 minimum-distance features
    """

    # --------------------------------------------------------
    # BBOX
    # --------------------------------------------------------

    (
        west,
        south,
        east,
        north,
    ) = make_bbox(
        float(lat),
        float(lon),
        radius_km,
    )

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    overpass_query = (
        build_overpass_query(
            south=south,
            west=west,
            north=north,
            east=east,
        )
    )

    # --------------------------------------------------------
    # Overpass
    # --------------------------------------------------------

    osm_data = (
        query_overpass_with_failover(
            overpass_query,
            OVERPASS_URLS,
        )
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    osm_records = (
        extract_osm_records(
            osm_data
        )
    )

    print(
        f"OSM objects retrieved: "
        f"{len(osm_records)}"
    )

    # --------------------------------------------------------
    # Raw distance features
    # --------------------------------------------------------

    osm_distance_dict = (
        calculate_osm_distances(
            live_hotspot_lat=float(lat),
            live_hotspot_lon=float(lon),
            osm_records=osm_records,
        )
    )

    # --------------------------------------------------------
    # Log features
    # --------------------------------------------------------

    osm_features = (
        add_osm_log_features(
            osm_distance_dict
        )
    )

    # --------------------------------------------------------
    # Minimum features
    # --------------------------------------------------------

    osm_features.update(
        build_minimum_distance_features(
            osm_distance_dict
        )
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        "\nOSM nearest distances"
    )

    for feature_name in (
        OSM_FEATURE_TAGS
    ):

        key = (
            f"distance_to_nearest_"
            f"{feature_name}_m"
        )

        value = (
            osm_distance_dict[key]
        )

        print(
            f"{feature_name:<22}: "
            f"{value:.3f} m"
        )

    print(
        "\n✓ OSM features created"
    )

    print(
        f"✓ Raw distance features: "
        f"{len(osm_distance_dict)}"
    )

    print(
        "✓ Total OSM model features: "
        f"{len(osm_features)}"
    )

    return (
        osm_distance_dict,
        osm_features,
    )