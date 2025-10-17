from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Sequence

from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from ..models import PanelSpec, RoofPolygon
from ..utils.geometry import quantize_points
from ..utils.metrics import MetricsCollector

SIMPLIFY_TOLERANCE = 0.08
ROTATION_PRECISION = 2


@dataclass
class PanelLayoutResult:
    roof_polygons: Sequence[Polygon]
    panel_polygons: Sequence[Polygon]
    panel_count: int
    total_kw: float
    orientation_mode: str
    metrics: Dict[str, int]


def layout_panels_for_spec(
    roofs: Sequence[RoofPolygon],
    panel_spec: PanelSpec,
    orientation_mode: str,
    min_walkway_m: float,
    grid_step_mm: int,
) -> PanelLayoutResult:
    collector = MetricsCollector()
    grid_step_m = max(grid_step_mm, 1) / 1000.0
    gap_m = max(panel_spec.gap_mm, 0) / 1000.0
    walkway = max(min_walkway_m, 0.0)

    with collector.track("validate_ms"):
        roof_polygons = _prepare_roof_polygons(roofs, grid_step_m)
        if not roof_polygons:
            raise ValueError("No valid roof polygons provided")

    with collector.track("simplify_ms"):
        simplified_polygons = [
            poly.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
            for poly in roof_polygons
            if poly.area > 0
        ]
        simplified_polygons = [poly for poly in simplified_polygons if not poly.is_empty]
        if not simplified_polygons:
            raise ValueError("Simplified roof polygons are empty")

    with collector.track("rotation_ms"):
        rotation_candidates = _determine_rotation_candidates(
            simplified_polygons, orientation_mode
        )

    pack_start = time.perf_counter()
    best_layout = _evaluate_rotations(
        polygons=simplified_polygons,
        panel_spec=panel_spec,
        rotation_candidates=rotation_candidates,
        walkway=walkway,
        gap_m=gap_m,
        grid_step_m=grid_step_m,
        request_mode=orientation_mode,
    )
    collector.add_duration("pack_ms", (time.perf_counter() - pack_start) * 1000.0)

    metrics = collector.as_ints()
    return PanelLayoutResult(
        roof_polygons=list(simplified_polygons),
        panel_polygons=list(best_layout.panel_polygons),
        panel_count=best_layout.panel_count,
        total_kw=best_layout.total_kw,
        orientation_mode=best_layout.orientation_mode,
        metrics=metrics,
    )


@dataclass
class _RotationLayout:
    panel_polygons: Sequence[Polygon]
    panel_count: int
    total_kw: float
    orientation_mode: str
    rotation_deg: float


def _prepare_roof_polygons(roofs: Sequence[RoofPolygon], grid_step: float):
    polygons = []
    for roof in roofs:
        quantized = quantize_points(roof.polygon, grid_step)
        polygon = Polygon(quantized)
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
            continue
        polygons.append(polygon)
    return polygons


def _determine_rotation_candidates(polygons: Sequence[Polygon], mode: str):
    mode_lower = mode.lower()
    if mode_lower in {"north", "south"}:
        return [0.0]
    if mode_lower in {"east", "west"}:
        return [90.0]

    candidates = [0.0, 90.0]
    dominant = _dominant_orientation(polygons)
    if dominant is not None:
        dominant = round(dominant, ROTATION_PRECISION)
        if all(abs(dominant - candidate) > 1e-2 for candidate in candidates):
            candidates.append(dominant)
    return candidates[:3]


def _dominant_orientation(polygons: Sequence[Polygon]) -> float | None:
    try:
        merged = unary_union(list(polygons))
        rectangle = merged.minimum_rotated_rectangle
        coords = list(rectangle.exterior.coords)
        if len(coords) < 2:
            return None
        edge = (coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
        angle = math.degrees(math.atan2(edge[1], edge[0]))
        return angle % 180.0
    except Exception:  # pragma: no cover - shapely edge cases
        return None


def _evaluate_rotations(
    polygons: Sequence[Polygon],
    panel_spec: PanelSpec,
    rotation_candidates: Sequence[float],
    walkway: float,
    gap_m: float,
    grid_step_m: float,
    request_mode: str,
) -> _RotationLayout:
    panel_width = panel_spec.w_mm / 1000.0
    panel_height = panel_spec.h_mm / 1000.0
    best_layout = _RotationLayout([], 0, 0.0, request_mode, 0.0)

    for rotation in rotation_candidates:
        rotated = [affinity.rotate(poly, -rotation, origin=(0.0, 0.0)) for poly in polygons]
        placements = _pack_for_rotation(rotated, panel_width, panel_height, walkway, gap_m, grid_step_m)
        count = len(placements)
        total_kw = count * panel_spec.watt / 1000.0
        if count > best_layout.panel_count or (
            count == best_layout.panel_count and total_kw > best_layout.total_kw
        ):
            back_rotated = [affinity.rotate(panel, rotation, origin=(0.0, 0.0)) for panel in placements]
            best_layout = _RotationLayout(
                panel_polygons=back_rotated,
                panel_count=count,
                total_kw=round(total_kw, 3),
                orientation_mode=_orientation_label(request_mode, rotation),
                rotation_deg=rotation,
            )

    if not best_layout.panel_polygons:
        best_layout.orientation_mode = _orientation_label(request_mode, 0.0)
        best_layout.total_kw = 0.0
    return best_layout


def _orientation_label(mode: str, rotation: float) -> str:
    mode_lower = mode.lower()
    if mode_lower != "auto":
        return mode_lower
    normalized = round(rotation % 180.0, ROTATION_PRECISION)
    return f"auto@{normalized}"


def _pack_for_rotation(
    polygons: Sequence[Polygon],
    panel_width: float,
    panel_height: float,
    walkway: float,
    gap_m: float,
    grid_step_m: float,
):
    placements = []
    spacing_x = panel_width + gap_m
    spacing_y = panel_height + gap_m
    shrink_amount = walkway + gap_m / 2.0

    for polygon in polygons:
        candidate = polygon
        if shrink_amount > 0:
            candidate = candidate.buffer(-shrink_amount, join_style=2)
        if candidate.is_empty:
            continue
        geometries = list(candidate.geoms) if hasattr(candidate, "geoms") else [candidate]
        for geom in geometries:
            placements.extend(
                _pack_single_polygon(geom, panel_width, panel_height, spacing_x, spacing_y, grid_step_m)
            )
    return placements


def _pack_single_polygon(
    polygon: Polygon,
    panel_width: float,
    panel_height: float,
    spacing_x: float,
    spacing_y: float,
    grid_step_m: float,
):
    minx, miny, maxx, maxy = polygon.bounds
    if maxx - minx < panel_width or maxy - miny < panel_height:
        return []

    placements = []
    start_x = math.floor(minx / grid_step_m) * grid_step_m
    start_y = math.floor(miny / grid_step_m) * grid_step_m

    x = start_x
    while x + panel_width <= maxx + 1e-9:
        y = start_y
        while y + panel_height <= maxy + 1e-9:
            qx = round(x / grid_step_m) * grid_step_m
            qy = round(y / grid_step_m) * grid_step_m
            rect = box(qx, qy, qx + panel_width, qy + panel_height)
            if polygon.contains(rect):
                placements.append(rect)
            y += spacing_y
        x += spacing_x
    return placements
