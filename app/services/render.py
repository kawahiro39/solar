from __future__ import annotations

from typing import Sequence, Tuple

from PIL import Image, ImageDraw
from shapely.geometry import Polygon

BACKGROUND_COLOR = (15, 17, 26)
ROOF_FILL = (60, 66, 82)
ROOF_OUTLINE = (90, 96, 112)
PANEL_FILL = (255, 214, 0)
PANEL_OUTLINE = (190, 150, 0)
IMAGE_SIZE = 900
PADDING = 32


def render_layout_image(roof_polygons: Sequence[Polygon], panel_polygons: Sequence[Polygon]) -> Image.Image:
    if not roof_polygons and not panel_polygons:
        return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), BACKGROUND_COLOR)

    bounds = _aggregate_bounds(list(roof_polygons) + list(panel_polygons))
    minx, miny, maxx, maxy = bounds
    span_x = max(maxx - minx, 1.0)
    span_y = max(maxy - miny, 1.0)
    drawable_width = IMAGE_SIZE - 2 * PADDING
    drawable_height = IMAGE_SIZE - 2 * PADDING
    scale = min(drawable_width / span_x, drawable_height / span_y)

    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)

    for polygon in roof_polygons:
        _draw_polygon(draw, polygon, minx, miny, scale, ROOF_FILL, ROOF_OUTLINE)

    for polygon in panel_polygons:
        _draw_polygon(draw, polygon, minx, miny, scale, PANEL_FILL, PANEL_OUTLINE)

    return image


def _aggregate_bounds(polygons: Sequence[Polygon]) -> Tuple[float, float, float, float]:
    minx = min(poly.bounds[0] for poly in polygons)
    miny = min(poly.bounds[1] for poly in polygons)
    maxx = max(poly.bounds[2] for poly in polygons)
    maxy = max(poly.bounds[3] for poly in polygons)
    if minx == maxx:
        maxx += 1.0
    if miny == maxy:
        maxy += 1.0
    return minx, miny, maxx, maxy


def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    polygon: Polygon,
    minx: float,
    miny: float,
    scale: float,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
) -> None:
    if polygon.is_empty:
        return

    points = [
        _project_point(x, y, minx, miny, scale)
        for x, y in list(polygon.exterior.coords)
    ]
    draw.polygon(points, fill=fill, outline=outline)


def _project_point(x: float, y: float, minx: float, miny: float, scale: float) -> Tuple[float, float]:
    px = (x - minx) * scale + PADDING
    py = IMAGE_SIZE - ((y - miny) * scale + PADDING)
    return px, py
