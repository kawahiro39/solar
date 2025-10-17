from __future__ import annotations

import base64
import logging
import re
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse

import httpx

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from starlette import status

from .fallback_roof import (
    analyze_roof,
    encode_image,
    panel_specs_from_inputs,
    run_fallback_detection,
    STATIC_MAP_SCALE,
    STATIC_MAP_ZOOM,
)
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
from .schemas import (
    ErrorResponse,
    DesignPipelineResponse,
    LayoutOptimizeRequest,
    LayoutOptimizeResponse,
    LayoutPanelsRequest,
    LayoutPanelsResponse,
    LayoutPanelsSummary,
    LayoutPanelsSummaryOption,
    SquareImageCenter,
    OrthoImageRequest,
    OrthoImageResponse,
    PanelMixEntry,
    PanelSpecInput,
    SquareImageRequest,
    SquareImageResponse,
    FallbackPanelResult,
    RoofDetectionResponse,
    RoofFaceInput,
    RoofSegmentRequest,
    RoofSegmentResponse,
    RoofFaceOutput,
    SegmentPlacementEntry,
    SolarDesignRequest,
    SolarDesignResponse,
)
from .solar_service import (
    COVERAGE_UNAVAILABLE_MESSAGE,
    DataLayerRenderContext,
    SolarApiError,
    SolarDesignEngine,
    _get_api_key,
)
from .geo import meters_per_pixel as compute_mpp, polygon_latlng_to_local, polygon_pixels_to_latlng

ALLOWED_CORS_ORIGINS = [
    "https://solar-nova.online",
    "https://www.solar-nova.online",
]

STATIC_MAP_ENDPOINT = "https://maps.googleapis.com/maps/api/staticmap"
GMAPS_AT_PATTERN = re.compile(
    r"@(?P<lat>-?\d+(?:\.\d+)?),(?P<lng>-?\d+(?:\.\d+)?),(?P<zoom>\d+(?:\.\d+)?)z"
)
GMAPS_34_PATTERN = re.compile(
    r"!3d(?P<lat>-?\d+(?:\.\d+)?)!4d(?P<lng>-?\d+(?:\.\d+)?)"
)
MAX_ERROR_DETAIL_LENGTH = 200

app = FastAPI(title="Solar Design Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_CORS_ORIGINS,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)

logger = logging.getLogger(__name__)

DEFAULT_ATTRIBUTION = ["© OpenStreetMap contributors", "Imagery © Google"]


def _summarize_validation_errors(errors: List[Dict[str, object]]) -> str:
    messages = []
    for error in errors:
        location = [str(part) for part in error.get("loc", []) if part not in {"body"}]
        field_path = ".".join(location)
        message = error.get("msg", "Invalid value")
        if field_path:
            messages.append(f"{field_path}: {message}")
        else:
            messages.append(str(message))
    summary = "; ".join(messages) if messages else "リクエスト内容が不正です。"
    return _truncate_detail(summary)


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    detail = _summarize_validation_errors(exc.errors())
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": detail})


def _truncate_detail(message: str) -> str:
    trimmed = message.strip()
    if len(trimmed) <= MAX_ERROR_DETAIL_LENGTH:
        return trimmed
    return trimmed[:MAX_ERROR_DETAIL_LENGTH]


def _parse_coordinate_string(value: str) -> Optional[Tuple[float, float]]:
    matches = re.findall(r"-?\d+(?:\.\d+)?", value)
    if len(matches) < 2:
        return None
    try:
        return float(matches[0]), float(matches[1])
    except ValueError:
        return None


def _extract_zoom_from_query(query: Dict[str, List[str]]) -> Optional[float]:
    for key in ("zoom", "z", "level"):
        if key in query and query[key]:
            candidate = query[key][0]
            try:
                return float(candidate)
            except ValueError:
                continue
    return None


def _parse_gmaps_url(url: str) -> Tuple[float, float, float]:
    normalized = url.strip()
    match = GMAPS_AT_PATTERN.search(normalized)
    if match:
        lat = float(match.group("lat"))
        lng = float(match.group("lng"))
        zoom = float(match.group("zoom"))
        if not 0 <= zoom <= 21:
            raise ValueError("Zoom level must be between 0 and 21.")
        return lat, lng, zoom

    parsed = urlparse(normalized)
    query = parse_qs(parsed.query)
    zoom = _extract_zoom_from_query(query)

    coordinate = None
    if query:
        coordinate = _parse_coordinate_string(
            next((value for key in ("ll", "center", "q") if (value := query.get(key))), [""])[0]
        )
        if coordinate is None:
            lat_candidate = query.get("lat")
            lng_candidate = query.get("lng")
            if lat_candidate and lng_candidate:
                try:
                    coordinate = float(lat_candidate[0]), float(lng_candidate[0])
                except ValueError:
                    coordinate = None

    if coordinate is None:
        alt_match = GMAPS_34_PATTERN.search(normalized)
        if alt_match:
            coordinate = float(alt_match.group("lat")), float(alt_match.group("lng"))

    if coordinate is None:
        raise ValueError("Google Maps URL から位置情報を抽出できませんでした。")

    if zoom is None:
        zoom = 20.0
    if not 0 <= zoom <= 21:
        raise ValueError("Zoom level must be between 0 and 21.")

    return coordinate[0], coordinate[1], zoom


async def _fetch_static_map_image(
    api_key: str,
    lat: float,
    lng: float,
    zoom: int,
    size_px: int,
    scale: int,
    maptype: str,
) -> bytes:
    params = {
        "center": f"{lat},{lng}",
        "zoom": str(zoom),
        "size": f"{size_px}x{size_px}",
        "scale": str(scale),
        "maptype": maptype,
        "format": "png",
        "key": api_key,
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(STATIC_MAP_ENDPOINT, params=params)
    except httpx.HTTPError as exc:
        logger.warning("Static map request failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=_truncate_detail("衛星画像の取得に失敗しました。時間をおいて再実行してください。"),
        ) from exc

    if response.status_code == 200 and response.content:
        return response.content

    if 400 <= response.status_code < 500:
        logger.info(
            "Static map request returned client error %s: %s",
            response.status_code,
            response.text,
        )
        raise HTTPException(
            status_code=400,
            detail=_truncate_detail("Google Maps Static API リクエストが無効です。パラメータを確認してください。"),
        )

    logger.error(
        "Static map provider returned status %s: %s",
        response.status_code,
        response.text,
    )
    raise HTTPException(
        status_code=502,
        detail=_truncate_detail("衛星画像プロバイダからエラーが返されました。再度お試しください。"),
    )


def _build_data_uri(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _pil_from_bgr(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image[:, :, ::-1])


def _scale_polygon(points: List[Tuple[float, float]], scale_x: float, scale_y: float) -> List[List[float]]:
    return [[float(x * scale_x), float(y * scale_y)] for x, y in points]


def _prepare_segments(
    faces: List[RoofFaceInput],
) -> Tuple[
    List[SegmentInput],
    List[Polygon],
    float,
    Tuple[float, float],
    List[Optional[float]],
]:
    if not faces:
        raise ValueError("No faces provided")

    first_polygon = faces[0].polygon
    if not first_polygon:
        raise ValueError("Face polygon is empty")

    ref_lng, ref_lat = first_polygon[0]
    segments: List[SegmentInput] = []
    roof_polygons: List[Polygon] = []
    for idx, face in enumerate(faces):
        if len(face.polygon) < 3:
            continue
        latlng_points = [(point[1], point[0]) for point in face.polygon]
        local_points = polygon_latlng_to_local(latlng_points, ref_lat, ref_lng)
        polygon = Polygon(local_points)
        if polygon.is_empty or polygon.area <= 0:
            continue
        polygon = simplify_polygon_for_layout(polygon)
        inferred_orientation = infer_segment_orientation(polygon)
        segments.append(
            SegmentInput(
                segment_id=idx,
                polygon=polygon,
                azimuth_deg=face.azimuth_deg,
                pitch_deg=None,
                inferred_azimuth_deg=inferred_orientation,
            )
        )
        roof_polygons.append(polygon)

    total_area = sum(poly.area for poly in roof_polygons)
    rotation_candidates = determine_rotation_candidates(segments)
    return segments, roof_polygons, total_area, (ref_lat, ref_lng), rotation_candidates


def _optimize_layout(
    faces: List[RoofFaceInput],
    panel_inputs: List[PanelSpecInput],
    orientation_mode: str,
    max_total: Optional[int],
    max_per_face: Optional[int],
    min_walkway: float,
) -> Tuple[
    List[PanelSpec],
    List[PanelPlacement],
    Dict[int, int],
    str,
    List[Polygon],
    float,
    float,
]:
    specs = panel_specs_from_inputs(panel_inputs)
    if not specs:
        raise ValueError("No panel specifications provided")

    segments, roof_polygons, roof_area, _, rotation_candidates = _prepare_segments(faces)
    if not segments:
        raise ValueError("有効な屋根ポリゴンが見つかりませんでした。")

    density = DENSITY_PROFILES.get("標準")
    min_walkway = max(min_walkway, 0.0)
    orientation_options = (
        [orientation_mode]
        if orientation_mode in {"portrait", "landscape"}
        else ["portrait", "landscape"]
    )

    def build_limits() -> Dict[int, Optional[int]]:
        if max_per_face is None:
            return {}
        return {segment.segment_id: max_per_face for segment in segments}

    best_orientation = orientation_options[0]
    best_layout = None
    best_mix: Dict[int, int] = {}
    best_dc_kw = -1.0
    best_rotation: Optional[float] = None

    for rotation in rotation_candidates:
        for orientation in orientation_options:
            layout = fill_segments(
                segments=segments,
                specs=specs,
                density=density,
                min_walkway=min_walkway,
                max_total=max_total,
                max_per_face=build_limits(),
                orientation=orientation,
                rotation_override=rotation,
            )
            mix = aggregate_mix(specs, layout.placements)
            dc_kw = sum(specs[idx].watt * count for idx, count in mix.items()) / 1000.0
            if dc_kw > best_dc_kw:
                best_dc_kw = dc_kw
                best_orientation = orientation
                best_layout = layout
                best_mix = mix
                best_rotation = rotation

    assert best_layout is not None
    placements = list(best_layout.placements)
    rotation_value = None if best_rotation is None else round(float(best_rotation), 2)

    return (
        specs,
        placements,
        best_mix,
        best_orientation,
        roof_polygons,
        roof_area,
        best_dc_kw,
        rotation_value,
    )


def _resolve_max_limit(max_per_face: Optional[Union[int, Dict[str, int]]]) -> Optional[int]:
    if isinstance(max_per_face, int):
        return max_per_face
    if isinstance(max_per_face, dict):
        numeric = [value for value in max_per_face.values() if isinstance(value, int)]
        if numeric:
            return min(numeric)
    return None


@app.post(
    "/square_image",
    response_model=SquareImageResponse,
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
async def generate_square_image(request: SquareImageRequest) -> SquareImageResponse:
    try:
        api_key = _get_api_key()
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=_truncate_detail(str(exc))) from exc

    try:
        lat, lng, zoom = _parse_gmaps_url(request.gmaps_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=_truncate_detail(str(exc))) from exc

    zoom_int = int(round(zoom))
    zoom_int = max(0, min(21, zoom_int))

    image_bytes = await _fetch_static_map_image(
        api_key=api_key,
        lat=lat,
        lng=lng,
        zoom=zoom_int,
        size_px=request.square_size_px,
        scale=request.scale,
        maptype=request.maptype,
    )

    data_uri = _build_data_uri(image_bytes)
    meters_per_px = compute_mpp(lat, zoom_int, request.scale)

    return SquareImageResponse(
        image_data_uri=data_uri,
        center=SquareImageCenter(lat=lat, lng=lng, zoom=float(zoom_int)),
        square_size_px=request.square_size_px,
        meters_per_pixel=round(meters_per_px, 6),
    )


@app.post(
    "/layout_panels",
    response_model=LayoutPanelsResponse,
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
def layout_panels(request: LayoutPanelsRequest) -> LayoutPanelsResponse:
    try:
        (
            specs,
            placements,
            mix,
            _orientation,
            roof_polygons,
            _roof_area,
            dc_kw,
            rotation_used,
        ) = _optimize_layout(
            request.roofs,
            request.panels,
            request.orientation_mode,
            request.max_total,
            request.max_per_face,
            request.min_walkway_m,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=_truncate_detail(str(exc))) from exc

    summary_options: List[LayoutPanelsSummaryOption] = []
    total_panels = 0
    for spec in specs:
        count = mix.get(spec.index, 0)
        if count <= 0:
            continue
        total_panels += count
        summary_options.append(
            LayoutPanelsSummaryOption(
                panel=PanelSpecInput(**spec.original),
                count=count,
                dc_kw=round(spec.watt * count / 1000.0, 3),
            )
        )

    total_kw = round(dc_kw, 3)

    try:
        image_data_uri = render_layout(roof_polygons, placements, specs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=_truncate_detail(str(exc))) from exc

    summary = LayoutPanelsSummary(
        total_panels=total_panels,
        total_kw=total_kw,
        by_option=summary_options,
        rotation_deg_used=rotation_used,
    )

    return LayoutPanelsResponse(layout_image_data_uri=image_data_uri, summary=summary)


@app.post(
    "/image/ortho",
    response_model=OrthoImageResponse,
    responses={502: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def generate_orthophoto(request: OrthoImageRequest) -> OrthoImageResponse:
    try:
        api_key = _get_api_key()
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    analysis = analyze_roof(api_key, request.lat, request.lng)
    if analysis is None:
        raise HTTPException(status_code=404, detail=COVERAGE_UNAVAILABLE_MESSAGE)

    rotated_image = _pil_from_bgr(analysis.rotated_image)
    original_width, original_height = rotated_image.size
    target_size = request.square_px
    if target_size != original_width:
        resized_image = rotated_image.resize((target_size, target_size))
        scale_x = target_size / original_width
        scale_y = target_size / original_height
    else:
        resized_image = rotated_image
        scale_x = scale_y = 1.0

    polygon_rotated = list(analysis.rotated_polygon.exterior.coords)[:-1]
    scaled_polygon = _scale_polygon(polygon_rotated, scale_x, scale_y)

    width = analysis.original_image.shape[1]
    height = analysis.original_image.shape[0]
    polygon_latlng = [
        [lng, lat]
        for lat, lng in polygon_pixels_to_latlng(
            analysis.lat,
            analysis.lng,
            STATIC_MAP_ZOOM,
            STATIC_MAP_SCALE,
            width,
            height,
            list(analysis.original_polygon.exterior.coords)[:-1],
        )
    ]

    image_b64 = encode_image(resized_image.convert("RGB"))
    normalized_orientation = (analysis.orientation + 180.0) % 180.0

    return OrthoImageResponse(
        image_png_base64=image_b64,
        m_per_px=round(analysis.mpp, 6),
        orientation_deg=round(float(normalized_orientation), 2),
        roof_polygon=scaled_polygon,
        roof_polygon_latlng=polygon_latlng,
        confidence=round(analysis.confidence, 3),
        attribution=list(DEFAULT_ATTRIBUTION),
    )


@app.post(
    "/roof/segment",
    response_model=RoofSegmentResponse,
    responses={502: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def segment_roof_faces(request: RoofSegmentRequest) -> RoofSegmentResponse:
    try:
        api_key = _get_api_key()
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    analysis = analyze_roof(api_key, request.lat, request.lng)
    if analysis is None:
        raise HTTPException(status_code=404, detail=COVERAGE_UNAVAILABLE_MESSAGE)

    mask_image = Image.fromarray(analysis.rotated_mask)
    mask_b64 = encode_image(mask_image.convert("L"))

    polygon_rotated = [[float(x), float(y)] for x, y in list(analysis.rotated_polygon.exterior.coords)[:-1]]
    width = analysis.original_image.shape[1]
    height = analysis.original_image.shape[0]
    polygon_latlng = [
        [lng, lat]
        for lat, lng in polygon_pixels_to_latlng(
            analysis.lat,
            analysis.lng,
            STATIC_MAP_ZOOM,
            STATIC_MAP_SCALE,
            width,
            height,
            list(analysis.original_polygon.exterior.coords)[:-1],
        )
    ]

    face_area = float(analysis.rotated_polygon.area) * (analysis.mpp ** 2)
    face = RoofFaceOutput(
        mask_png_base64=mask_b64,
        polygon=polygon_rotated,
        polygon_latlng=polygon_latlng,
        azimuth_deg=round((analysis.orientation + 360.0) % 360.0, 2),
        tilt_rel=0.0,
        area_m2=round(face_area, 2),
    )

    return RoofSegmentResponse(
        faces=[face],
        confidence=round(analysis.confidence, 3),
        attribution=list(DEFAULT_ATTRIBUTION),
    )


@app.post(
    "/layout/optimize",
    response_model=LayoutOptimizeResponse,
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
def optimize_layout(request: LayoutOptimizeRequest) -> LayoutOptimizeResponse:
    if not request.faces:
        raise HTTPException(status_code=400, detail="屋根面が指定されていません。")
    if not request.panels:
        raise HTTPException(status_code=400, detail="パネル仕様が指定されていません。")

    try:
        (
            specs,
            placements,
            mix,
            orientation,
            roof_polygons,
            roof_area,
            dc_kw,
            rotation_used,
        ) = _optimize_layout(
            request.faces,
            request.panels,
            request.orientation_mode,
            request.max_total,
            request.max_per_face,
            request.min_walkway_m,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mix_entries = [
        PanelMixEntry(spec=PanelSpecInput(**spec.original), count=mix.get(spec.index, 0))
        for spec in specs
        if mix.get(spec.index, 0) > 0
    ]

    result = FallbackPanelResult(
        orientation_used=orientation,
        dc_kw=round(dc_kw, 3),
        mix=mix_entries,
    )

    image_b64 = render_layout(roof_polygons, placements, specs)
    confidence = 1.0 if placements else 0.5

    return LayoutOptimizeResponse(
        result=result,
        image_png_base64=image_b64,
        confidence=confidence,
        roof_area_m2=round(roof_area, 2),
        dc_kw=round(dc_kw, 3),
        attribution=list(DEFAULT_ATTRIBUTION),
    )


@app.post(
    "/design",
    response_model=DesignPipelineResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
def integrated_design(request: SolarDesignRequest) -> DesignPipelineResponse:
    if request.lat is None or request.lng is None:
        raise HTTPException(status_code=400, detail="緯度経度が必要です。")
    if not request.panels:
        raise HTTPException(status_code=400, detail="パネル仕様が指定されていません。")

    try:
        api_key = _get_api_key()
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    analysis = analyze_roof(api_key, request.lat, request.lng)
    if analysis is None:
        raise HTTPException(status_code=404, detail=COVERAGE_UNAVAILABLE_MESSAGE)

    polygon_latlng = [
        [lng, lat]
        for lat, lng in polygon_pixels_to_latlng(
            analysis.lat,
            analysis.lng,
            STATIC_MAP_ZOOM,
            STATIC_MAP_SCALE,
            analysis.original_image.shape[1],
            analysis.original_image.shape[0],
            list(analysis.original_polygon.exterior.coords)[:-1],
        )
    ]

    mask_image = Image.fromarray(analysis.rotated_mask)
    mask_b64 = encode_image(mask_image.convert("L"))
    face_area = float(analysis.rotated_polygon.area) * (analysis.mpp ** 2)
    face_output = RoofFaceOutput(
        mask_png_base64=mask_b64,
        polygon=[[float(x), float(y)] for x, y in list(analysis.rotated_polygon.exterior.coords)[:-1]],
        polygon_latlng=polygon_latlng,
        azimuth_deg=round((analysis.orientation + 360.0) % 360.0, 2),
        tilt_rel=0.0,
        area_m2=round(face_area, 2),
    )

    face_inputs = [
        RoofFaceInput(
            polygon=polygon_latlng,
            azimuth_deg=face_output.azimuth_deg,
            tilt_rel=face_output.tilt_rel,
            area_m2=face_output.area_m2,
        )
    ]

    max_limit = _resolve_max_limit(request.max_per_face)

    try:
        (
            specs,
            placements,
            mix,
            orientation,
            roof_polygons,
            roof_area,
            dc_kw,
            rotation_used,
        ) = _optimize_layout(
            face_inputs,
            request.panels,
            request.orientation_mode,
            request.max_total,
            max_limit,
            request.min_walkway_m,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mix_entries = [
        PanelMixEntry(spec=PanelSpecInput(**spec.original), count=mix.get(spec.index, 0))
        for spec in specs
        if mix.get(spec.index, 0) > 0
    ]

    result = FallbackPanelResult(
        orientation_used=orientation,
        dc_kw=round(dc_kw, 3),
        mix=mix_entries,
    )

    image_b64 = render_layout(roof_polygons, placements, specs)
    layout_confidence = 1.0 if placements else 0.5
    combined_confidence = round(min(1.0, (analysis.confidence + layout_confidence) / 2.0), 3)

    return DesignPipelineResponse(
        faces=[face_output],
        result=result,
        image_png_base64=image_b64,
        confidence=combined_confidence,
        dc_kw=round(dc_kw, 3),
        attribution=list(DEFAULT_ATTRIBUTION),
    )


@app.post(
    "/solar/design",
    response_model=Union[SolarDesignResponse, RoofDetectionResponse],
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
def design_solar_system(
    request: SolarDesignRequest,
) -> Union[SolarDesignResponse, RoofDetectionResponse]:
    try:
        engine = SolarDesignEngine(request)
        lat, lng = engine.resolve_coordinates()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    specs = engine.build_specs()
    use_data_layers = request.force_data_layers
    solar_data: Dict[str, object] = {}
    data_layers_payload: Optional[Dict[str, object]] = None
    render_context: Optional[DataLayerRenderContext] = None

    segments = []
    building_insights_error: Optional[SolarApiError] = None
    if not use_data_layers:
        try:
            solar_data = engine.fetch_building_insights(lat, lng)
            segments = engine.build_segments(solar_data)
        except SolarApiError as exc:
            if exc.status_code == 404:
                use_data_layers = True
                solar_data = {}
                building_insights_error = exc
            else:
                raise HTTPException(status_code=exc.status_code, detail=exc.message)

    data_layers_error: Optional[SolarApiError] = None
    if use_data_layers:
        try:
            data_layers_payload = engine.fetch_data_layers(lat, lng)
            segments, render_context = engine.build_segments_from_data_layers(lat, lng, data_layers_payload)
        except SolarApiError as exc:
            if exc.status_code == 404:
                data_layers_error = exc
                logger.info(
                    "Solar API returned 404 for data layers at lat=%s, lng=%s. Attempting fallbacks.",
                    lat,
                    lng,
                )
                if building_insights_error and building_insights_error.status_code == 404:
                    try:
                        return fetch_roof_from_static_map(lat, lng, specs, request)
                    except HTTPException:
                        raise
                    except Exception as fallback_error:
                        logger.exception("Roof detection fallback failed: %s", fallback_error)
                        raise HTTPException(status_code=404, detail=COVERAGE_UNAVAILABLE_MESSAGE)
                try:
                    solar_data = engine.fetch_building_insights(lat, lng)
                    segments = engine.build_segments(solar_data)
                except SolarApiError as fallback_exc:
                    if fallback_exc.status_code == 404:
                        try:
                            return fetch_roof_from_static_map(lat, lng, specs, request)
                        except HTTPException:
                            raise
                        except Exception as roof_exc:
                            logger.exception("Roof detection fallback failed: %s", roof_exc)
                            raise HTTPException(status_code=404, detail=COVERAGE_UNAVAILABLE_MESSAGE)
                    raise HTTPException(status_code=fallback_exc.status_code, detail=fallback_exc.message)
            else:
                raise HTTPException(status_code=exc.status_code, detail=exc.message)

    if not segments:
        if data_layers_error and data_layers_error.status_code == 404:
            logger.info("Falling back to roof detection for lat=%s, lng=%s", lat, lng)
            return fetch_roof_from_static_map(lat, lng, specs, request)
        detail_message = (
            COVERAGE_UNAVAILABLE_MESSAGE
            if data_layers_error
            else "屋根ポリゴン情報が取得できませんでした。別の建物でお試しください。"
        )
        raise HTTPException(status_code=404, detail=detail_message)

    try:
        orientation, placements, mix, face_mix, face_limits = engine.compute_layout(segments, specs)
    except SolarApiError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    total_watts = sum(specs[idx].watt * count for idx, count in mix.items())
    dc_kw = round(total_watts / 1000.0, 3)

    roof_polygons: List[Polygon] = [segment.polygon for segment in segments]
    background_image = render_context.background if render_context else None
    background_bounds = render_context.bounds_m if render_context else None
    image_b64 = render_layout(
        roof_polygons,
        placements,
        specs,
        background=background_image,
        background_bounds=background_bounds,
    )

    mix_entries = [
        PanelMixEntry(spec=PanelSpecInput(**spec.original), count=mix[spec.index])
        for spec in specs
        if mix.get(spec.index, 0) > 0
    ]

    face_entries: List[SegmentPlacementEntry] = []
    for segment in segments:
        placements_for_segment = face_mix.get(segment.segment_id, {})
        placed_entries = [
            PanelMixEntry(spec=PanelSpecInput(**specs[idx].original), count=count)
            for idx, count in placements_for_segment.items()
            if count > 0
        ]
        limit_value = face_limits.get(segment.segment_id)
        face_entries.append(
            SegmentPlacementEntry(
                segment_id=segment.segment_id,
                limit=limit_value,
                placed=placed_entries,
            )
        )

    total_roof_area = sum(polygon.area for polygon in roof_polygons)
    total_panel_area = sum(specs[idx].width_m * specs[idx].height_m * count for idx, count in mix.items())
    fill_rate = 0.0 if total_roof_area == 0 else round(total_panel_area / total_roof_area, 3)

    solar_potential = solar_data.get("solarPotential") if solar_data else None
    max_orientation = solar_potential.get("maxSunshineOrientation", {}) if solar_potential else {}
    site_info = {
        "lat": lat,
        "lng": lng,
        "azimuth_deg": max_orientation.get("azimuthDegrees"),
        "tilt_deg": max_orientation.get("tiltDegrees"),
    }

    response_payload: Dict[str, object] = {
        "site": site_info,
        "result": {
            "orientation_used": orientation,
            "dc_kw": dc_kw,
            "mix": mix_entries,
            "by_face": face_entries,
            "utilization_metrics": {
                "roof_area_m2": round(total_roof_area, 3),
                "fill_rate": fill_rate,
            },
            "image_png_base64": image_b64,
        },
    }
    if solar_potential is not None:
        response_payload["solar_potential"] = solar_potential
    if data_layers_payload is not None:
        response_payload["data_layers"] = data_layers_payload.get("layers", {})

    return SolarDesignResponse(**response_payload)


@app.get("/healthz")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


def fetch_roof_from_static_map(
    lat: float,
    lng: float,
    specs: List[PanelSpec],
    request: SolarDesignRequest,
) -> RoofDetectionResponse:
    try:
        api_key = _get_api_key()
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    max_total = request.max_total
    max_face = None
    if isinstance(request.max_per_face, int):
        max_face = request.max_per_face
    elif isinstance(request.max_per_face, dict):
        numeric_limits = [value for value in request.max_per_face.values() if isinstance(value, int)]
        if numeric_limits:
            max_face = min(numeric_limits)

    try:
        result = run_fallback_detection(
            api_key=api_key,
            lat=lat,
            lng=lng,
            specs=specs,
            orientation_mode=request.orientation_mode,
            max_total=max_total,
            max_face=max_face,
            debug=request.debug,
        )
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        message = (
            "衛星画像の取得に失敗しました。リクエストを確認してください。"
            if 400 <= status < 500
            else "衛星画像の取得に失敗しました。しばらくしてから再度お試しください。"
        )
        raise HTTPException(status_code=502, detail=message) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="衛星画像の取得に失敗しました。再度お試しください。") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return RoofDetectionResponse(**result)
@app.post(
    "/image/ortho",
    response_model=OrthoImageResponse,
    responses={502: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def generate_orthophoto(request: OrthoImageRequest) -> OrthoImageResponse:
    try:
        api_key = _get_api_key()
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    analysis = analyze_roof(api_key, request.lat, request.lng)
    if analysis is None:
        raise HTTPException(status_code=404, detail=COVERAGE_UNAVAILABLE_MESSAGE)

    rotated_image = _pil_from_bgr(analysis.rotated_image)
    original_width, original_height = rotated_image.size
    target_size = request.square_px
    if target_size != original_width:
        resized_image = rotated_image.resize((target_size, target_size))
        scale_x = target_size / original_width
        scale_y = target_size / original_height
    else:
        resized_image = rotated_image
        scale_x = scale_y = 1.0

    polygon_rotated = list(analysis.rotated_polygon.exterior.coords)[:-1]
    scaled_polygon = _scale_polygon(polygon_rotated, scale_x, scale_y)

    width = analysis.original_image.shape[1]
    height = analysis.original_image.shape[0]
    polygon_latlng = [
        [lng, lat]
        for lat, lng in polygon_pixels_to_latlng(
            analysis.lat,
            analysis.lng,
            STATIC_MAP_ZOOM,
            STATIC_MAP_SCALE,
            width,
            height,
            list(analysis.original_polygon.exterior.coords)[:-1],
        )
    ]

    image_b64 = encode_image(resized_image.convert("RGB"))
    normalized_orientation = (analysis.orientation + 180.0) % 180.0

    return OrthoImageResponse(
        image_png_base64=image_b64,
        m_per_px=round(analysis.mpp, 6),
        orientation_deg=round(float(normalized_orientation), 2),
        roof_polygon=scaled_polygon,
        roof_polygon_latlng=polygon_latlng,
        confidence=round(analysis.confidence, 3),
        attribution=list(DEFAULT_ATTRIBUTION),
    )

