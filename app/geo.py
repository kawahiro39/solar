"""Geospatial utilities for working with Google Static Maps imagery."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple


EARTH_RADIUS_M = 6_378_137.0
TILE_SIZE = 256


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _mercator_lat_to_y(lat: float) -> float:
    siny = _clamp(math.sin(math.radians(lat)), -0.9999, 0.9999)
    return 0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)


def _mercator_y_to_lat(y: float) -> float:
    mercator = (0.5 - y) * 2 * math.pi
    return math.degrees(math.atan(math.sinh(mercator)))


def latlng_to_world(lat: float, lng: float, zoom: int) -> Tuple[float, float]:
    scale = TILE_SIZE * (2 ** zoom)
    x = (lng + 180.0) / 360.0 * scale
    y = _mercator_lat_to_y(lat) * scale
    return x, y


def world_to_latlng(x: float, y: float, zoom: int) -> Tuple[float, float]:
    scale = TILE_SIZE * (2 ** zoom)
    lng = x / scale * 360.0 - 180.0
    lat = _mercator_y_to_lat(y / scale)
    return lat, lng


def pixels_to_latlng(
    lat: float,
    lng: float,
    zoom: int,
    scale: int,
    width: int,
    height: int,
    pixels: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Convert image pixel coordinates to latitude/longitude pairs."""

    center_world_x, center_world_y = latlng_to_world(lat, lng, zoom)
    half_width = width / 2.0
    half_height = height / 2.0

    latlng_points: List[Tuple[float, float]] = []
    for px, py in pixels:
        world_x = center_world_x + (px - half_width) / scale
        world_y = center_world_y + (py - half_height) / scale
        lat_val, lng_val = world_to_latlng(world_x, world_y, zoom)
        latlng_points.append((lat_val, lng_val))
    return latlng_points


def polygon_pixels_to_latlng(
    lat: float,
    lng: float,
    zoom: int,
    scale: int,
    width: int,
    height: int,
    polygon: Iterable[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    return pixels_to_latlng(lat, lng, zoom, scale, width, height, list(polygon))


def meters_per_pixel(lat: float, zoom: int, scale: int) -> float:
    return (
        math.cos(math.radians(lat))
        * 2.0
        * math.pi
        * EARTH_RADIUS_M
        / (TILE_SIZE * (2 ** zoom))
        / scale
    )


def geodesic_area_m2(points: Sequence[Tuple[float, float]] ) -> float:
    if len(points) < 3:
        return 0.0

    ref_lat = sum(pt[0] for pt in points) / len(points)
    ref_lng = sum(pt[1] for pt in points) / len(points)

    coords_m = [
        (
            (lng - ref_lng)
            * math.cos(math.radians(ref_lat))
            * (math.pi / 180.0)
            * EARTH_RADIUS_M,
            (lat - ref_lat) * (math.pi / 180.0) * EARTH_RADIUS_M,
        )
        for lat, lng in points
    ]

    area = 0.0
    for (x1, y1), (x2, y2) in zip(coords_m, coords_m[1:] + coords_m[:1]):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def latlng_to_local_xy(
    lat: float,
    lng: float,
    ref_lat: float,
    ref_lng: float,
) -> Tuple[float, float]:
    dx = (lng - ref_lng) * math.cos(math.radians(ref_lat)) * (math.pi / 180.0) * EARTH_RADIUS_M
    dy = (lat - ref_lat) * (math.pi / 180.0) * EARTH_RADIUS_M
    return dx, dy


def polygon_latlng_to_local(
    polygon: Sequence[Tuple[float, float]],
    ref_lat: float,
    ref_lng: float,
) -> List[Tuple[float, float]]:
    return [latlng_to_local_xy(lat, lng, ref_lat, ref_lng) for lat, lng in polygon]
