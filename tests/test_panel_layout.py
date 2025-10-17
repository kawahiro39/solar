import pathlib
import sys

from shapely.geometry import Polygon

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.panel_layout import SegmentInput, determine_rotation_candidates


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
