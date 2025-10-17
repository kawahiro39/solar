from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
import numpy as np
from PIL import Image
from rasterio.features import shapes
from rasterio.io import MemoryFile
from shapely.geometry import Polygon, shape

from .imaging import render_layout
from .panel_layout import (
    DENSITY_PROFILES,
    PanelPlacement,
    PanelSpec,
    SegmentInput,
    aggregate_mix,
    determine_rotation_candidates,
    fill_segments,
    infer_segment_orientation,
    simplify_polygon_for_layout,
)
from .schemas import SolarDesignRequest


EARTH_RADIUS_M = 6_378_137.0
COVERAGE_UNAVAILABLE_MESSAGE = "この地域は衛星画像による太陽光解析データが未対応です。別の場所をお試しください。"


@dataclass
class DataLayerRenderContext:
    background: Optional[Image.Image]
    bounds_m: Optional[Tuple[float, float, float, float]]


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

    def fetch_data_layers(self, lat: float, lng: float) -> Dict[str, object]:
        endpoint = "https://solar.googleapis.com/v1/dataLayers:get"
        params = {
            "location.latitude": lat,
            "location.longitude": lng,
            "radiusMeters": 50,
            "view": "IMAGERY_AND_ANNUAL_FLUX_LAYERS",
            "requiredQuality": "BASE",
            "exactQualityRequired": "false",
            "key": self.api_key,
        }
        headers = {"Accept": "application/json"}
        try:
            response = requests.get(endpoint, headers=headers, params=params, timeout=60)
        except requests.RequestException as exc:
            raise SolarApiError(502, "Solar APIとの通信に失敗しました。再度お試しください。") from exc
        if response.status_code == 404:
            raise SolarApiError(404, COVERAGE_UNAVAILABLE_MESSAGE)
        if response.status_code >= 500:
            raise SolarApiError(502, "Solar APIの応答に失敗しました。しばらくしてから再度お試しください。")
        if response.status_code >= 400:
            raise SolarApiError(400, "指定した位置を解釈できませんでした。座標を再確認してください。")
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise SolarApiError(502, "Solar APIからの応答の解析に失敗しました。") from exc
        layers = payload.get("layers")
        if not layers:
            raise SolarApiError(404, COVERAGE_UNAVAILABLE_MESSAGE)
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
        reference_lon_lat: Optional[Tuple[float, float]] = None
        for index, segment in enumerate(segments_data):
            polygon_vertices = segment.get("segmentPolygon", {}).get("vertices", [])
            if len(polygon_vertices) < 3:
                continue
            if reference_lon_lat is None and polygon_vertices:
                first_vertex = polygon_vertices[0]
                reference_lon_lat = (
                    float(first_vertex.get("x", 0.0)),
                    float(first_vertex.get("y", 0.0)),
                )
            coords = _convert_vertices_to_meters(polygon_vertices, reference_lon_lat)
            # Ensure the polygon closes correctly
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            polygon = Polygon(coords)
            segment_id = segment.get("segmentId", index)
            azimuth = segment.get("azimuthDegrees")
            pitch = segment.get("pitchDegrees")
            polygon = simplify_polygon_for_layout(polygon)
            inferred = infer_segment_orientation(polygon)
            segments.append(
                SegmentInput(
                    segment_id=segment_id,
                    polygon=polygon,
                    azimuth_deg=azimuth,
                    pitch_deg=pitch,
                    inferred_azimuth_deg=inferred,
                )
            )
        if not segments:
            raise SolarApiError(404, "屋根ポリゴン情報が取得できませんでした。別の建物でお試しください。")
        return segments

    def build_segments_from_data_layers(
        self, lat: float, lng: float, data_layers: Dict[str, object]
    ) -> Tuple[List[SegmentInput], DataLayerRenderContext]:
        layers = data_layers.get("layers", {})
        annual_flux_url = layers.get("annualFluxUrl")
        if not annual_flux_url:
            raise SolarApiError(404, "日射量レイヤーを取得できませんでした。別の建物でお試しください。")

        flux_bytes = self._download_layer_bytes(annual_flux_url)
        reference = (float(lng), float(lat))
        segments: List[SegmentInput] = []
        with MemoryFile(flux_bytes) as memfile:
            with memfile.open() as dataset:
                flux_data = dataset.read(1, masked=True)
                valid_mask = ~flux_data.mask
                if not np.any(valid_mask):
                    raise SolarApiError(404, "日射量データから屋根領域を抽出できませんでした。")
                flux_values = flux_data[valid_mask]
                if flux_values.size == 0:
                    raise SolarApiError(404, "日射量データから屋根領域を抽出できませんでした。")
                threshold = float(np.percentile(flux_values, 60))
                roof_mask = (flux_data.filled(0) >= threshold) & valid_mask
                if not np.any(roof_mask):
                    roof_mask = valid_mask
                mask_uint8 = roof_mask.astype("uint8")
                polygons_lonlat: List[Polygon] = []
                for geom, value in shapes(mask_uint8, mask=roof_mask, transform=dataset.transform):
                    if value != 1:
                        continue
                    geom_shape = shape(geom)
                    if geom_shape.is_empty:
                        continue
                    if geom_shape.geom_type == "Polygon":
                        geom_iter = [geom_shape]
                    else:
                        geom_iter = [g for g in geom_shape.geoms if not g.is_empty]
                    polygons_lonlat.extend(geom_iter)

        segment_id = 0
        for polygon_lonlat in polygons_lonlat:
            polygon_lonlat = polygon_lonlat.buffer(0)
            polygon_m = _polygon_lonlat_to_meters(polygon_lonlat, reference)
            if polygon_m is None:
                continue
            if polygon_m.area < 5.0:
                continue
            simplified = simplify_polygon_for_layout(polygon_m)
            inferred = infer_segment_orientation(simplified)
            segments.append(
                SegmentInput(
                    segment_id=segment_id,
                    polygon=simplified,
                    azimuth_deg=None,
                    pitch_deg=None,
                    inferred_azimuth_deg=inferred,
                )
            )
            segment_id += 1

        if not segments:
            raise SolarApiError(404, "屋根ポリゴン情報が取得できませんでした。別の建物でお試しください。")

        background_image: Optional[Image.Image] = None
        background_bounds: Optional[Tuple[float, float, float, float]] = None
        rgb_url = layers.get("rgbUrl")
        if rgb_url:
            try:
                rgb_bytes = self._download_layer_bytes(rgb_url)
                with MemoryFile(rgb_bytes) as memfile:
                    with memfile.open() as dataset:
                        band_count = min(3, dataset.count)
                        if band_count == 0:
                            raise ValueError("empty rgb dataset")
                        bands = dataset.read(list(range(1, band_count + 1)))
                        bands = np.clip(bands, 0, 255)
                        if band_count == 1:
                            bands = np.repeat(bands, 3, axis=0)
                        if band_count == 2:
                            bands = np.concatenate([bands, bands[:1]], axis=0)
                        array = np.moveaxis(bands.astype("uint8"), 0, -1)
                        background_image = Image.fromarray(array, mode="RGB")
                        bounds = dataset.bounds
                        minx_m, miny_m = _project_lonlat_to_meters(bounds.left, bounds.bottom, reference)
                        maxx_m, maxy_m = _project_lonlat_to_meters(bounds.right, bounds.top, reference)
                        background_bounds = (
                            min(minx_m, maxx_m),
                            min(miny_m, maxy_m),
                            max(minx_m, maxx_m),
                            max(miny_m, maxy_m),
                        )
            except (SolarApiError, ValueError):
                background_image = None
                background_bounds = None

        context = DataLayerRenderContext(background=background_image, bounds_m=background_bounds)
        return segments, context

    def _download_layer_bytes(self, url: str) -> bytes:
        try:
            response = requests.get(url, timeout=60)
        except requests.RequestException as exc:
            raise SolarApiError(502, "データレイヤーの取得中に通信エラーが発生しました。") from exc
        if response.status_code >= 500:
            raise SolarApiError(502, "データレイヤーのダウンロードに失敗しました。しばらくしてから再試しください。")
        if response.status_code >= 400:
            raise SolarApiError(404, "データレイヤーの画像を取得できませんでした。")
        return response.content

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

        rotation_candidates = determine_rotation_candidates(segments)

        best_orientation = orientations[0]
        best_mix: Dict[int, int] = {}
        best_placements: List[PanelPlacement] = []
        best_face_mix: Dict[int, Dict[int, int]] = {}
        best_score: Tuple[float, List[int]] = (0.0, [])

        best_limits: Dict[int, Optional[int]] = max_per_face

        for rotation in rotation_candidates:
            for orientation in orientations:
                layout = fill_segments(
                    segments=segments,
                    specs=specs,
                    density=density_profile,
                    min_walkway=min_walkway,
                    max_total=max_total,
                    max_per_face=max_per_face,
                    orientation=orientation,
                    rotation_override=rotation,
                )
                mix = aggregate_mix(specs, layout.placements)
                dc_kw = sum(specs[idx].watt * count for idx, count in mix.items()) / 1000.0
                efficiency_vector = [mix.get(spec.index, 0) for spec in specs]
                score = (dc_kw, efficiency_vector)
                if dc_kw > best_score[0] + 1e-6 or (
                    math.isclose(dc_kw, best_score[0], rel_tol=1e-6)
                    and efficiency_vector > best_score[1]
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


def _convert_vertices_to_meters(
    vertices: Sequence[Dict[str, object]],
    reference_lon_lat: Optional[Tuple[float, float]] = None,
) -> List[Tuple[float, float]]:
    if not vertices:
        return []
    coords_deg: List[Tuple[float, float]] = []
    for vertex in vertices:
        lon = float(vertex.get("x", 0.0))
        lat = float(vertex.get("y", 0.0))
        coords_deg.append((lon, lat))

    if reference_lon_lat is None:
        ref_lon_deg, ref_lat_deg = coords_deg[0]
    else:
        ref_lon_deg, ref_lat_deg = reference_lon_lat
    ref_lat_rad = math.radians(ref_lat_deg)
    ref_lon_rad = math.radians(ref_lon_deg)
    cos_lat = math.cos(ref_lat_rad)

    coords_m: List[Tuple[float, float]] = []
    for lon_deg, lat_deg in coords_deg:
        lon_rad = math.radians(lon_deg)
        lat_rad = math.radians(lat_deg)
        x = (lon_rad - ref_lon_rad) * cos_lat * EARTH_RADIUS_M
        y = (lat_rad - ref_lat_rad) * EARTH_RADIUS_M
        coords_m.append((x, y))
    return coords_m


def _project_lonlat_to_meters(
    lon_deg: float, lat_deg: float, reference_lon_lat: Tuple[float, float]
) -> Tuple[float, float]:
    ref_lon_deg, ref_lat_deg = reference_lon_lat
    ref_lat_rad = math.radians(ref_lat_deg)
    ref_lon_rad = math.radians(ref_lon_deg)
    lon_rad = math.radians(lon_deg)
    lat_rad = math.radians(lat_deg)
    x = (lon_rad - ref_lon_rad) * math.cos(ref_lat_rad) * EARTH_RADIUS_M
    y = (lat_rad - ref_lat_rad) * EARTH_RADIUS_M
    return x, y


def _polygon_lonlat_to_meters(
    polygon: Polygon, reference_lon_lat: Tuple[float, float]
) -> Optional[Polygon]:
    if polygon.is_empty:
        return None
    exterior_coords = [
        _project_lonlat_to_meters(lon, lat, reference_lon_lat) for lon, lat in polygon.exterior.coords
    ]
    if len(exterior_coords) < 4:
        return None
    interiors_m: List[List[Tuple[float, float]]] = []
    for interior in polygon.interiors:
        coords = [_project_lonlat_to_meters(lon, lat, reference_lon_lat) for lon, lat in interior.coords]
        if len(coords) >= 4:
            interiors_m.append(coords)
    polygon_m = Polygon(exterior_coords, interiors_m)
    if not polygon_m.is_valid:
        polygon_m = polygon_m.buffer(0)
    if polygon_m.geom_type == "MultiPolygon":
        largest = max((geom for geom in polygon_m.geoms if not geom.is_empty), key=lambda g: g.area, default=None)
        if largest is None:
            return None
        polygon_m = largest
    if polygon_m.is_empty:
        return None
    return polygon_m


def _get_api_key() -> str:
    for env_name in ("GOOGLE_MAPS_API_KEY", "GOOGLE_API_KEY"):
        api_key = os.getenv(env_name)
        if api_key:
            return api_key
    raise RuntimeError(
        "GOOGLE_MAPS_API_KEY environment variable is not set. Cloud Run 環境変数または Secret Manager を設定してください。"
    )

