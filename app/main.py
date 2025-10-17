from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from shapely.geometry import Polygon

from .imaging import render_layout
from .schemas import (
    ErrorResponse,
    PanelMixEntry,
    PanelSpecInput,
    SegmentPlacementEntry,
    SolarDesignRequest,
    SolarDesignResponse,
)
from .solar_service import (
    COVERAGE_UNAVAILABLE_MESSAGE,
    DataLayerRenderContext,
    SolarApiError,
    SolarDesignEngine,
)

app = FastAPI(title="Solar Design Service", version="1.0.0")


@app.post("/solar/design", response_model=SolarDesignResponse, responses={
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
})
def design_solar_system(request: SolarDesignRequest) -> SolarDesignResponse:
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
                if building_insights_error and building_insights_error.status_code == 404:
                    raise HTTPException(status_code=404, detail=COVERAGE_UNAVAILABLE_MESSAGE)
                try:
                    solar_data = engine.fetch_building_insights(lat, lng)
                    segments = engine.build_segments(solar_data)
                except SolarApiError as fallback_exc:
                    if fallback_exc.status_code == 404:
                        raise HTTPException(status_code=404, detail=COVERAGE_UNAVAILABLE_MESSAGE)
                    raise HTTPException(status_code=fallback_exc.status_code, detail=fallback_exc.message)
            else:
                raise HTTPException(status_code=exc.status_code, detail=exc.message)

    if not segments:
        detail_message = COVERAGE_UNAVAILABLE_MESSAGE if data_layers_error else "屋根ポリゴン情報が取得できませんでした。別の建物でお試しください。"
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
