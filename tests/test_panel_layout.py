import pathlib
import sys

import pytest

from shapely.geometry import Polygon

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.panel_layout import (
    DensityProfile,
    LayoutEngine,
    PanelSpec,
    SegmentInput,
    determine_rotation_candidates,
)


def _make_segment(index: int, angle: float) -> SegmentInput:
    polygon = Polygon([(0, 0), (index + 1, 0), (index + 1, 1), (0, 1)])
    return SegmentInput(
        segment_id=index,
        polygon=polygon,
        azimuth_deg=angle,
        pitch_deg=None,
        inferred_azimuth_deg=None,
    )


def test_rotation_candidates_capped_size():
    segments = [_make_segment(idx, angle=float(idx * 13 % 180)) for idx in range(10)]

    candidates = determine_rotation_candidates(segments)

    assert candidates[0] is None
    assert 0.0 in candidates
    assert 90.0 in candidates
    assert len(candidates) <= 4


def test_fill_segment_prepared_polygon_stable():
    segment = SegmentInput(
        segment_id=1,
        polygon=Polygon([(0, 0), (10, 0), (10, 5), (0, 5)]),
        azimuth_deg=0.0,
        pitch_deg=None,
        inferred_azimuth_deg=None,
    )

    spec = PanelSpec(
        index=1,
        width_m=1.0,
        height_m=2.0,
        gap_m=0.2,
        watt=300,
        original={},
    )

    engine = LayoutEngine(
        segments=[segment],
        panel_specs=[spec],
        density=DensityProfile(edge_margin_m=0.5, gap_extra_m=0.1),
        min_walkway=0.3,
        max_total=None,
        max_per_face={},
        rotation_override=None,
    )

    result = engine.generate_layout("portrait")
    placements = list(result.placements)

    assert len(placements) == 4
    assert sum(placement.polygon.area for placement in placements) == pytest.approx(
        4 * spec.area_m2
    )
    expected_bounds = [
        (0.5, 0.5, 1.5, 2.5),
        (3.1, 0.5, 4.1, 2.5),
        (5.7, 0.5, 6.7, 2.5),
        (8.3, 0.5, 9.3, 2.5),
    ]

    for placement, bounds in zip(placements, expected_bounds):
        minx, miny, maxx, maxy = placement.polygon.bounds
        assert placement.spec_index == spec.index
        for actual, expected in zip((minx, miny, maxx, maxy), bounds):
            assert actual == pytest.approx(expected)
