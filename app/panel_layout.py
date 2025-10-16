from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry


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


class LayoutEngine:
    def __init__(
        self,
        segments: Sequence[SegmentInput],
        panel_specs: Sequence[PanelSpec],
        density: DensityProfile,
        min_walkway: float,
        max_total: Optional[int],
        max_per_face: Dict[int, Optional[int]],
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

        azimuth = segment.azimuth_deg if segment.azimuth_deg is not None else 0.0
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
            spec_step_x = dimensions[0] + gap_total
            spec_step_y = dimensions[1] + gap_total

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


def fill_segments(
    segments: Sequence[SegmentInput],
    specs: Sequence[PanelSpec],
    density: DensityProfile,
    min_walkway: float,
    max_total: Optional[int],
    max_per_face: Dict[int, Optional[int]],
    orientation: str,
) -> LayoutResult:
    engine = LayoutEngine(
        segments=segments,
        panel_specs=specs,
        density=density,
        min_walkway=min_walkway,
        max_total=max_total,
        max_per_face=max_per_face,
    )
    return engine.generate_layout(orientation)
