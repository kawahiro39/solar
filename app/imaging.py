from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from .panel_layout import PanelPlacement, PanelSpec


@dataclass
class RenderGeometry:
    polygon: Polygon
    fill: Tuple[int, int, int]
    outline: Tuple[int, int, int]
    width: int = 1


def render_layout(
    roof_polygons: Sequence[Polygon],
    placements: Iterable[PanelPlacement],
    specs: Sequence[PanelSpec],
    image_size: int = 900,
    background: Optional[Image.Image] = None,
    background_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> str:
    placement_polygons: List[Polygon] = [placement.polygon for placement in placements]
    all_geometries: List[Polygon] = list(roof_polygons) + placement_polygons
    if not all_geometries:
        raise ValueError("No geometries to render")

    minx = min(poly.bounds[0] for poly in all_geometries)
    miny = min(poly.bounds[1] for poly in all_geometries)
    maxx = max(poly.bounds[2] for poly in all_geometries)
    maxy = max(poly.bounds[3] for poly in all_geometries)
    if background is not None and background_bounds is not None:
        bminx, bminy, bmaxx, bmaxy = background_bounds
        minx = min(minx, bminx)
        miny = min(miny, bminy)
        maxx = max(maxx, bmaxx)
        maxy = max(maxy, bmaxy)

    width_m = max(maxx - minx, 1e-6)
    height_m = max(maxy - miny, 1e-6)

    aspect = height_m / width_m
    base_width = image_size
    base_height = int(image_size * aspect)
    margin_px = int(image_size * 0.05)
    img_width = base_width + margin_px * 2
    img_height = base_height + margin_px * 2

    image = Image.new("RGB", (img_width, img_height), (242, 244, 248))

    if background is not None and background_bounds is not None:
        bminx, bminy, bmaxx, bmaxy = background_bounds
        top_left = project((bminx, bmaxy))
        bottom_right = project((bmaxx, bminy))
        left_px = min(top_left[0], bottom_right[0])
        right_px = max(top_left[0], bottom_right[0])
        top_px = min(top_left[1], bottom_right[1])
        bottom_px = max(top_left[1], bottom_right[1])
        width_px = max(int(math.ceil(right_px - left_px)), 1)
        height_px = max(int(math.ceil(bottom_px - top_px)), 1)
        resized = background.resize((width_px, height_px))
        image.paste(resized, (int(math.floor(left_px)), int(math.floor(top_px))))

    draw = ImageDraw.Draw(image, "RGBA")

    draw_width = base_width
    draw_height = base_height if base_height > 0 else 1

    def project(point: Tuple[float, float]) -> Tuple[float, float]:
        x, y = point
        px = ((x - minx) / width_m) * draw_width + margin_px
        py = margin_px + draw_height - ((y - miny) / height_m) * draw_height
        return px, py

    roof_color = (178, 205, 251, 200)
    roof_outline = (64, 112, 214, 255)
    panel_fill = (32, 69, 139, 220)
    panel_outline = (18, 37, 74, 255)

    for poly in roof_polygons:
        coords = [project((x, y)) for x, y in poly.exterior.coords]
        draw.polygon(coords, fill=roof_color, outline=roof_outline)

    for poly in placement_polygons:
        coords = [project((x, y)) for x, y in poly.exterior.coords]
        draw.polygon(coords, fill=panel_fill, outline=panel_outline)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
