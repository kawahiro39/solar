from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


@dataclass
class PanelSpec:
    index: int
    width_m: float
    height_m: float
    gap_m: float
    watt: float
    original: Dict[str, float]

    @property
    def area_m2(self) -> float:
        return self.width_m * self.height_m

    @property
    def efficiency(self) -> float:
        if self.area_m2 == 0:
            return 0.0
        return self.watt / self.area_m2


@dataclass
class PanelPlacement:
    spec_index: int
    polygon: Polygon


@dataclass
class SegmentLayout:
    segment_id: int
    placements: List[PanelPlacement]
    limit: Optional[int]


@dataclass
class SegmentInput:
    segment_id: int
    polygon: Polygon
    azimuth_deg: Optional[float]
    pitch_deg: Optional[float]
    inferred_azimuth_deg: Optional[float] = None


@dataclass
class DensityProfile:
    edge_margin_m: float
    gap_extra_m: float


class LayoutResult:
    def __init__(self, orientation: str, segments: List[SegmentLayout]):
        self.orientation = orientation
        self.segments = segments

    @property
    def placements(self) -> Iterable[PanelPlacement]:
        for segment in self.segments:
            for placement in segment.placements:
                yield placement


DENSITY_PROFILES: Dict[str, DensityProfile] = {
    "控えめ": DensityProfile(edge_margin_m=0.6, gap_extra_m=0.10),
    "標準": DensityProfile(edge_margin_m=0.4, gap_extra_m=0.05),
    "多め": DensityProfile(edge_margin_m=0.25, gap_extra_m=0.02),
}

MIN_GRID_STEP_M = 0.1


class LayoutEngine:
    def __init__(
        self,
        segments: Sequence[SegmentInput],
        panel_specs: Sequence[PanelSpec],
        density: DensityProfile,
        min_walkway: float,
        max_total: Optional[int],
        max_per_face: Dict[int, Optional[int]],
        rotation_override: Optional[float] = None,
    ) -> None:
        self.segments = list(segments)
        self.panel_specs = sorted(
            panel_specs,
            key=lambda spec: (spec.efficiency, spec.area_m2),
            reverse=True,
        )
        self.density = density
        self.min_walkway = min_walkway
        self.max_total = max_total
        self.max_per_face = max_per_face
        self.rotation_override = rotation_override

    def generate_layout(self, orientation: str) -> LayoutResult:
        assert orientation in {"portrait", "landscape"}
        remaining_total = self.max_total if self.max_total is not None else math.inf
        segment_layouts: List[SegmentLayout] = []

        # Sort segments by polygon area descending so larger areas get priority
        for segment in sorted(self.segments, key=lambda s: s.polygon.area, reverse=True):
            limit = self.max_per_face.get(segment.segment_id)
            if limit is None:
                limit = self.max_per_face.get(-1)  # default entry when request provided scalar
            remaining_face = limit if limit is not None else math.inf
            if remaining_total == 0:
                placements = []
            else:
                placements = self._fill_segment(segment, orientation, remaining_face, remaining_total)
                count_in_segment = len(placements)
                if not math.isinf(remaining_total):
                    remaining_total = max(0, remaining_total - count_in_segment)
            segment_layouts.append(
                SegmentLayout(segment.segment_id, placements, None if limit is math.inf else limit)
            )

        return LayoutResult(orientation, segment_layouts)

    def _fill_segment(
        self,
        segment: SegmentInput,
        orientation: str,
        remaining_face: float,
        remaining_total: float,
    ) -> List[PanelPlacement]:
        if remaining_face <= 0 or remaining_total <= 0:
            return []

        azimuth = segment.azimuth_deg
        if azimuth is None:
            if self.rotation_override is not None:
                azimuth = self.rotation_override
            elif segment.inferred_azimuth_deg is not None:
                azimuth = segment.inferred_azimuth_deg
            else:
                azimuth = 0.0
        rotation_angle = self._rotation_for_azimuth(azimuth)
        rotated_polygon = affinity.rotate(segment.polygon, rotation_angle, origin=(0.0, 0.0))
        margin = max(self.density.edge_margin_m, self.min_walkway)
        interior_polygon = rotated_polygon.buffer(-margin)
        if interior_polygon.is_empty:
            return []

        available_polygon = _ensure_polygon(interior_polygon)
        placements_rotated: List[Tuple[PanelSpec, Polygon]] = []
        inflated_shapes: List[BaseGeometry] = []

        for spec in self.panel_specs:
            if remaining_face <= 0 or remaining_total <= 0:
                break
            dimensions = self._panel_dimensions_for_orientation(spec, orientation)
            gap_total = spec.gap_m + self.density.gap_extra_m
            clearance = max(gap_total, 0.0) / 2.0
            spec_step_x = max(dimensions[0] + gap_total, MIN_GRID_STEP_M)
            spec_step_y = max(dimensions[1] + gap_total, MIN_GRID_STEP_M)

            candidate_sets = []
            minx, miny, maxx, maxy = available_polygon.bounds
            offset_options_x = [0.0, spec_step_x / 2.0]
            offset_options_y = [0.0, spec_step_y / 2.0]
            for offset_x in offset_options_x:
                for offset_y in offset_options_y:
                    candidate_sets.append((offset_x, offset_y))

            best_candidate: Tuple[int, List[Tuple[Polygon, BaseGeometry]]] = (0, [])
            for offset_x, offset_y in candidate_sets:
                candidates: List[Tuple[Polygon, BaseGeometry]] = []
                placed_buffers = list(inflated_shapes)
                count = 0
                y = miny + dimensions[1] / 2.0 + offset_y
                while y + dimensions[1] / 2.0 <= maxy + 1e-9:
                    x = minx + dimensions[0] / 2.0 + offset_x
                    while x + dimensions[0] / 2.0 <= maxx + 1e-9:
                        if count >= remaining_face or count >= remaining_total:
                            break
                        rect = box(
                            x - dimensions[0] / 2.0,
                            y - dimensions[1] / 2.0,
                            x + dimensions[0] / 2.0,
                            y + dimensions[1] / 2.0,
                        )
                        if not rect.within(available_polygon):
                            x += spec_step_x
                            continue
                        inflated = rect.buffer(clearance, join_style=2)
                        if not all(inflated.disjoint(existing) for existing in placed_buffers):
                            x += spec_step_x
                            continue
                        candidates.append((rect, inflated))
                        placed_buffers.append(inflated)
                        count += 1
                        x += spec_step_x
                    y += spec_step_y
                if count > best_candidate[0]:
                    best_candidate = (count, candidates)

            selected = best_candidate[1]
            for rect, inflated in selected:
                placements_rotated.append((spec, rect))
                inflated_shapes.append(inflated)
                remaining_face -= 1
                remaining_total -= 1
                if remaining_face <= 0 or remaining_total <= 0:
                    break

        placements: List[PanelPlacement] = []
        for spec, rect in placements_rotated:
            original_polygon = affinity.rotate(rect, -rotation_angle, origin=(0.0, 0.0))
            placements.append(PanelPlacement(spec.index, Polygon(original_polygon.exterior.coords)))
        return placements

    @staticmethod
    def _panel_dimensions_for_orientation(spec: PanelSpec, orientation: str) -> Tuple[float, float]:
        if orientation == "portrait":
            return spec.width_m, spec.height_m
        if orientation == "landscape":
            return spec.height_m, spec.width_m
        raise ValueError(f"Unsupported orientation: {orientation}")

    @staticmethod
    def _rotation_for_azimuth(azimuth: float) -> float:
        rad = math.radians(azimuth)
        dx = math.sin(rad)
        dy = math.cos(rad)
        theta = math.degrees(math.atan2(dx, dy))
        return theta


def _ensure_polygon(geometry: BaseGeometry) -> Polygon:
    if geometry.is_empty:
        raise ValueError("Geometry is empty")
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon):
        largest = max(geometry.geoms, key=lambda g: g.area)
        return Polygon(largest.exterior.coords)
    raise TypeError(f"Unsupported geometry type: {type(geometry)!r}")


def aggregate_mix(
    specs: Sequence[PanelSpec],
    placements: Iterable[PanelPlacement],
) -> Dict[int, int]:
    counts: Dict[int, int] = {spec.index: 0 for spec in specs}
    for placement in placements:
        counts[placement.spec_index] += 1
    return {spec_index: count for spec_index, count in counts.items() if count > 0}


def normalize_rotation(angle: float) -> float:
    normalized = (angle + 180.0) % 180.0
    if normalized > 90.0:
        normalized -= 180.0
    return normalized


def infer_segment_orientation(polygon: Polygon) -> Optional[float]:
    if polygon.is_empty:
        return None
    try:
        rotated = polygon.minimum_rotated_rectangle
    except Exception:
        return None
    coords = list(rotated.exterior.coords)
    if len(coords) < 4:
        return None
    edges: List[Tuple[float, float]] = []
    for idx in range(len(coords) - 1):
        x1, y1 = coords[idx]
        x2, y2 = coords[idx + 1]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        edges.append((length, angle))
    if not edges:
        return None
    longest = max(edges, key=lambda edge: edge[0])
    rotation = -longest[1]
    return normalize_rotation(rotation)


def simplify_polygon_for_layout(polygon: Polygon, tolerance: float = 0.05) -> Polygon:
    if polygon.is_empty:
        return polygon
    simplified = polygon.simplify(tolerance, preserve_topology=True)
    if simplified.is_empty or not simplified.is_valid:
        return polygon
    return simplified


def determine_rotation_candidates(
    segments: Sequence[SegmentInput],
) -> List[Optional[float]]:
    seen: Set[float] = set()
    candidates: List[Optional[float]] = [None]

    def add(angle: Optional[float]) -> None:
        if angle is None:
            return
        normalized = normalize_rotation(angle)
        key = round(normalized, 4)
        if key in seen:
            return
        seen.add(key)
        candidates.append(normalized)

    add(0.0)
    add(90.0)

    dominant = _dominant_orientation(segments)
    add(dominant)

    return candidates


def _dominant_orientation(segments: Sequence[SegmentInput]) -> Optional[float]:
    weighted_cos = 0.0
    weighted_sin = 0.0
    total_weight = 0.0

    for segment in segments:
        angle = segment.azimuth_deg
        if angle is None:
            angle = segment.inferred_azimuth_deg
        if angle is None:
            continue
        normalized = normalize_rotation(angle)
        weight = max(segment.polygon.area, 1e-6)
        rad = math.radians(normalized * 2.0)
        weighted_cos += math.cos(rad) * weight
        weighted_sin += math.sin(rad) * weight
        total_weight += weight

    if total_weight == 0.0:
        polygons = [segment.polygon for segment in segments if not segment.polygon.is_empty]
        if not polygons:
            return None
        try:
            combined = unary_union(polygons)
        except Exception:
            combined = polygons[0]
        try:
            polygon = _ensure_polygon(combined)
        except (TypeError, ValueError):
            polygon = max(polygons, key=lambda poly: poly.area)
        return infer_segment_orientation(polygon)

    angle = math.degrees(0.5 * math.atan2(weighted_sin, weighted_cos))
    return normalize_rotation(angle)


def fill_segments(
    segments: Sequence[SegmentInput],
    specs: Sequence[PanelSpec],
    density: DensityProfile,
    min_walkway: float,
    max_total: Optional[int],
    max_per_face: Dict[int, Optional[int]],
    orientation: str,
    rotation_override: Optional[float] = None,
) -> LayoutResult:
    engine = LayoutEngine(
        segments=segments,
        panel_specs=specs,
        density=density,
        min_walkway=min_walkway,
        max_total=max_total,
        max_per_face=max_per_face,
        rotation_override=rotation_override,
    )
    return engine.generate_layout(orientation)
