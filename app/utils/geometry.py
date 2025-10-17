from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from shapely.geometry import Polygon


def meters_per_pixel(lat: float, zoom: int, scale: int = 1) -> float:
    scale = max(scale, 1)
    latitude_radians = math.radians(lat)
    meters_per_pixel_at_equator = 156543.03392
    return (meters_per_pixel_at_equator * math.cos(latitude_radians)) / (2 ** zoom * scale)


def quantize_points(points: Sequence[Sequence[float]], step: float) -> List[Tuple[float, float]]:
    if step <= 0:
        raise ValueError("Quantization step must be positive")
    quantized: List[Tuple[float, float]] = []
    for x, y in points:
        qx = round(x / step) * step
        qy = round(y / step) * step
        quantized.append((float(qx), float(qy)))
    return quantized


def polygon_area(polygon: Polygon) -> float:
    return float(polygon.area)
