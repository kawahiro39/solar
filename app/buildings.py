"""Utilities for fetching building polygons from OSM."""

from __future__ import annotations

import json
from typing import Iterable, Optional

import requests
from shapely.geometry import Point, Polygon

OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"


def _parse_osm_elements(elements: Iterable[dict]) -> Iterable[Polygon]:
    nodes = {
        element["id"]: (element["lon"], element["lat"])
        for element in elements
        if element.get("type") == "node"
        and "lat" in element
        and "lon" in element
    }

    polygons = []
    for element in elements:
        if element.get("type") != "way":
            continue
        node_ids = element.get("nodes") or []
        if len(node_ids) < 3:
            continue
        coords = []
        missing_node = False
        for node_id in node_ids:
            node = nodes.get(node_id)
            if node is None:
                missing_node = True
                break
            coords.append(node)
        if missing_node or len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        polygon = Polygon(coords)
        if polygon.is_empty:
            continue
        polygon = polygon.buffer(0)
        if polygon.is_valid and polygon.area > 0:
            polygons.append(polygon)
    return polygons


def fetch_osm_building_polygon(lat: float, lng: float, radius_m: float = 40.0) -> Optional[Polygon]:
    """Fetch the OSM building polygon nearest to the provided coordinate."""

    query = f"""
    [out:json][timeout:25];
    (
      way(around:{radius_m},{lat},{lng})["building"];
      relation(around:{radius_m},{lat},{lng})["building"];
    );
    (._;>;);
    out body;
    """

    try:
        response = requests.post(OVERPASS_ENDPOINT, data=query.encode("utf-8"), timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return None

    try:
        payload = response.json()
    except json.JSONDecodeError:
        return None

    elements = payload.get("elements") or []
    polygons = list(_parse_osm_elements(elements))
    if not polygons:
        return None

    point = Point(lng, lat)
    containing = [polygon for polygon in polygons if polygon.contains(point)]
    if containing:
        polygon = max(containing, key=lambda poly: poly.area)
        return polygon

    polygon = min(polygons, key=lambda poly: poly.distance(point))
    return polygon

