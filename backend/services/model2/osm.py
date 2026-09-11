from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from .config import OSM_RADIUS_M, OVERPASS_SERVERS
from .events import haversine_m


OSM_CATEGORIES = [
    "flare",
    "oil_well",
    "mineshaft",
    "adit",
    "gasometer",
    "industrial",
    "quarry",
    "farmland",
    "farmyard",
    "orchard",
    "vineyard",
    "plant_nursery",
    "greenhouse",
    "allotment",
    "forest",
    "scrub",
    "grassland",
    "heath",
]


def build_osm_query(
    latitude: float,
    longitude: float,
) -> str:

    return f"""
[out:json][timeout:90];

(
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["man_made"="flare"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["man_made"="petroleum_well"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["man_made"="oil_well"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["man_made"="mineshaft"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["man_made"="adit"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["man_made"="gasometer"];

  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["landuse"="industrial"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["industrial"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["landuse"="quarry"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["landuse"="mine"];

  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["landuse"="farmland"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["landuse"="farmyard"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["landuse"="orchard"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["landuse"="vineyard"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["landuse"="plant_nursery"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["landuse"="greenhouse_horticulture"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["landuse"="allotments"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["building"="greenhouse"];

  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["natural"="wood"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["natural"="forest"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["natural"="scrub"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["natural"="grassland"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["natural"="heath"];
  nwr(around:{OSM_RADIUS_M},{latitude},{longitude})["landuse"="forest"];
);

out tags center qt;
"""


def query_overpass(
    query: str,
) -> list:

    for server in OVERPASS_SERVERS:

        try:
            response = requests.post(
                server,
                data={
                    "data": query,
                },
                headers={
                    "User-Agent":
                        "fire-source-classification-research",
                },
                timeout=100,
            )

            if response.status_code == 200:
                data = response.json()

                return data.get(
                    "elements",
                    [],
                )

        except Exception:
            continue

    return []


def element_coordinate(
    element: dict,
):
    if element.get("type") == "node":

        if (
            "lat" in element
            and
            "lon" in element
        ):
            return (
                float(element["lat"]),
                float(element["lon"]),
            )

        return None

    center = element.get(
        "center"
    )

    if center:

        if (
            "lat" in center
            and
            "lon" in center
        ):
            return (
                float(center["lat"]),
                float(center["lon"]),
            )

    geometry = element.get(
        "geometry",
        [],
    )

    if geometry:

        lats = [
            float(point["lat"])
            for point in geometry
            if "lat" in point
        ]

        lons = [
            float(point["lon"])
            for point in geometry
            if "lon" in point
        ]

        if lats and lons:
            return (
                float(np.mean(lats)),
                float(np.mean(lons)),
            )

    return None


def classify_osm(
    tags: dict,
):
    man_made = tags.get(
        "man_made"
    )

    landuse = tags.get(
        "landuse"
    )

    natural = tags.get(
        "natural"
    )

    building = tags.get(
        "building"
    )

    if man_made == "flare":
        return "flare"

    if man_made in [
        "petroleum_well",
        "oil_well",
    ]:
        return "oil_well"

    if man_made == "mineshaft":
        return "mineshaft"

    if man_made == "adit":
        return "adit"

    if man_made == "gasometer":
        return "gasometer"

    if (
        landuse == "industrial"
        or
        "industrial" in tags
    ):
        return "industrial"

    if landuse == "quarry":
        return "quarry"

    if landuse == "mine":
        return "quarry"

    if landuse == "farmland":
        return "farmland"

    if landuse == "farmyard":
        return "farmyard"

    if landuse == "orchard":
        return "orchard"

    if landuse == "vineyard":
        return "vineyard"

    if landuse == "plant_nursery":
        return "plant_nursery"

    if (
        landuse == "greenhouse_horticulture"
        or
        building == "greenhouse"
    ):
        return "greenhouse"

    if landuse == "allotments":
        return "allotment"

    if (
        natural in [
            "wood",
            "forest",
        ]
        or
        landuse == "forest"
    ):
        return "forest"

    if natural == "scrub":
        return "scrub"

    if natural == "grassland":
        return "grassland"

    if natural == "heath":
        return "heath"

    return None


def build_osm_features(
    latitude: float,
    longitude: float,
) -> tuple[dict, float]:

    query = build_osm_query(
        latitude,
        longitude,
    )

    elements = query_overpass(
        query
    )

    osm_nearest = {
        category: np.nan
        for category in OSM_CATEGORIES
    }

    osm_counts = {
        category: 0
        for category in OSM_CATEGORIES
    }

    for element in elements:

        tags = element.get(
            "tags",
            {},
        )

        category = classify_osm(
            tags
        )

        if category is None:
            continue

        coordinate = element_coordinate(
            element
        )

        if coordinate is None:
            continue

        xlat, xlon = coordinate

        distance = float(
            haversine_m(
                latitude,
                longitude,
                xlat,
                xlon,
            )
        )

        osm_counts[
            category
        ] += 1

        current_nearest = osm_nearest[
            category
        ]

        if (
            pd.isna(current_nearest)
            or
            distance < current_nearest
        ):
            osm_nearest[
                category
            ] = distance

    agriculture_osm_values = [
        osm_nearest[category]
        for category in [
            "farmland",
            "farmyard",
            "orchard",
            "vineyard",
            "plant_nursery",
            "greenhouse",
            "allotment",
        ]
        if pd.notna(
            osm_nearest[category]
        )
    ]

    agriculture_osm_distance = (
        float(
            min(
                agriculture_osm_values
            )
        )
        if agriculture_osm_values
        else np.nan
    )

    return (
        {
            "osm_nearest": osm_nearest,
            "osm_counts": osm_counts,
            "elements_count": len(elements),
        },
        agriculture_osm_distance,
    )
