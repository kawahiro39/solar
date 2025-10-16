from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from shapely.geometry import Polygon

from .imaging import render_layout
from .panel_layout import (
    DENSITY_PROFILES,
    PanelPlacement,
    PanelSpec,
    SegmentInput,
    aggregate_mix,
    fill_segments,
)
from .schemas import SolarDesignRequest


class SolarApiError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class SolarDesignEngine:
    def __init__(self, request: SolarDesignRequest) -> None:
        self.request = request
        self.api_key = _get_api_key()

    def resolve_coordinates(self) -> Tuple[float, float]:
        if self.request.lat is not None and self.request.lng is not None:
            return float(self.request.lat), float(self.request.lng)
        if not self.request.map_url:
            raise ValueError("Location could not be resolved from the request")
        lat_lng = _parse_map_url(self.request.map_url)
        if lat_lng is None:
            raise ValueError("map_url did not contain latitude/longitude information")
        return lat_lng

    def fetch_building_insights(self, lat: float, lng: float) -> Dict[str, object]:
        endpoint = (
            "https://solar.googleapis.com/v1/buildingInsights:findClosest"
            f"?location.latitude={lat}&location.longitude={lng}&requiredQuality=HIGH"
        )
        headers = {"Accept": "application/json"}
        try:
            response = requests.get(endpoint, headers=headers, timeout=20, params={"key": self.api_key})
        except requests.RequestException as exc:
            raise SolarApiError(502, "Solar APIとの通信に失敗しました。再度お試しください。") from exc
        if response.status_code == 404:
            raise SolarApiError(404, "建物が見つかりませんでした。建物上にピンを置き直してください。")
        if response.status_code >= 500:
            raise SolarApiError(502, "Solar APIの応答に失敗しました。しばらくしてから再度お試しください。")
        if response.status_code >= 400:
            raise SolarApiError(400, "指定した位置を解釈できませんでした。座標を再確認してください。")
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise SolarApiError(502, "Solar APIからの応答の解析に失敗しました。") from exc
        return payload

    def build_specs(self) -> List[PanelSpec]:
        specs: List[PanelSpec] = []
        for idx, panel in enumerate(self.request.panels):
            specs.append(
                PanelSpec(
                    index=idx,
                    width_m=panel.w_mm / 1000.0,
                    height_m=panel.h_mm / 1000.0,
                    gap_m=panel.gap_mm / 1000.0,
                    watt=panel.watt,
                    original=panel.dict(),
                )
            )
        return specs

    def build_segments(self, solar_data: Dict[str, object]) -> List[SegmentInput]:
        segments_data = solar_data.get("solarPotential", {}).get("roofSegments", [])
        segments: List[SegmentInput] = []
        for index, segment in enumerate(segments_data):
            polygon_vertices = segment.get("segmentPolygon", {}).get("vertices", [])
            if len(polygon_vertices) < 3:
                continue
            coords = [(float(v.get("x", 0.0)), float(v.get("y", 0.0))) for v in polygon_vertices]
            # Ensure the polygon closes correctly
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            polygon = Polygon(coords)
            segment_id = segment.get("segmentId", index)
            azimuth = segment.get("azimuthDegrees")
            pitch = segment.get("pitchDegrees")
            segments.append(SegmentInput(segment_id=segment_id, polygon=polygon, azimuth_deg=azimuth, pitch_deg=pitch))
        if not segments:
            raise SolarApiError(404, "屋根ポリゴン情報が取得できませんでした。別の建物でお試しください。")
        return segments

    def compute_layout(
        self,
        segments: Sequence[SegmentInput],
        specs: Sequence[PanelSpec],
    ) -> Tuple[str, List[PanelPlacement], Dict[int, int], Dict[int, Dict[int, int]], Dict[int, Optional[int]]]:
        density_profile = DENSITY_PROFILES.get(self.request.density, DENSITY_PROFILES["標準"])
        min_walkway = max(self.request.min_walkway_m, 0.0)
        max_total = self.request.max_total
        max_per_face = self._resolve_face_limits(segments)

        orientations: List[str] = (
            [self.request.orientation_mode]
            if self.request.orientation_mode in {"portrait", "landscape"}
            else ["portrait", "landscape"]
        )

        best_orientation = "portrait"
        best_mix: Dict[int, int] = {}
        best_placements: List[PanelPlacement] = []
        best_face_mix: Dict[int, Dict[int, int]] = {}
        best_score: Tuple[float, List[int]] = (0.0, [])

        best_limits: Dict[int, Optional[int]] = max_per_face

        for orientation in orientations:
            layout = fill_segments(
                segments=segments,
                specs=specs,
                density=density_profile,
                min_walkway=min_walkway,
                max_total=max_total,
                max_per_face=max_per_face,
                orientation=orientation,
            )
            mix = aggregate_mix(specs, layout.placements)
            dc_kw = sum(specs[idx].watt * count for idx, count in mix.items()) / 1000.0
            efficiency_vector = [mix.get(spec.index, 0) for spec in specs]
            score = (dc_kw, efficiency_vector)
            if dc_kw > best_score[0] + 1e-6 or (
                math.isclose(dc_kw, best_score[0], rel_tol=1e-6) and efficiency_vector > best_score[1]
            ):
                best_score = score
                best_orientation = orientation
                best_mix = mix
                best_placements = list(layout.placements)
                best_face_mix = self._segment_mix(layout.segments, specs)
                best_limits = max_per_face

        return best_orientation, best_placements, best_mix, best_face_mix, best_limits

    def _segment_mix(
        self, segments: Sequence, specs: Sequence[PanelSpec]
    ) -> Dict[int, Dict[int, int]]:
        mix: Dict[int, Dict[int, int]] = {}
        for segment in segments:
            counts: Dict[int, int] = {}
            for placement in segment.placements:
                counts[placement.spec_index] = counts.get(placement.spec_index, 0) + 1
            mix[segment.segment_id] = counts
        return mix

    def _resolve_face_limits(self, segments: Sequence[SegmentInput]) -> Dict[int, Optional[int]]:
        limits: Dict[int, Optional[int]] = {}
        if isinstance(self.request.max_per_face, int):
            limits[-1] = self.request.max_per_face
        elif isinstance(self.request.max_per_face, dict):
            for key, value in self.request.max_per_face.items():
                try:
                    limits[int(key)] = value
                except (TypeError, ValueError):
                    continue
        for segment in segments:
            if segment.segment_id not in limits:
                limits[segment.segment_id] = limits.get(-1)
        return limits


_FLOAT_PAIR_RE = re.compile(r"(-?\d+\.\d+)")


def _parse_map_url(url: str) -> Optional[Tuple[float, float]]:
    if "@" in url:
        match = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", url)
        if match:
            return float(match.group(1)), float(match.group(2))
    if "!3d" in url and "!4d" in url:
        match = re.search(r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)", url)
        if match:
            return float(match.group(1)), float(match.group(2))
    if "q=" in url:
        match = re.search(r"q=(-?\d+\.\d+),(-?\d+\.\d+)", url)
        if match:
            return float(match.group(1)), float(match.group(2))
    if "/place/" in url:
        match = re.search(r"/place/(-?\d+\.\d+),(-?\d+\.\d+)", url)
        if match:
            return float(match.group(1)), float(match.group(2))

    matches = [float(value) for value in _FLOAT_PAIR_RE.findall(url)]
    for i in range(len(matches) - 1):
        lat, lng = matches[i], matches[i + 1]
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            return lat, lng
    return None


def _get_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set. Cloud Run 環境変数または Secret Manager を設定してください。"
        )
    return api_key
