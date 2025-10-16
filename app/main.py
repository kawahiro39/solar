from __future__ import annotations

from typing import Dict, List

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
from .solar_service import SolarApiError, SolarDesignEngine

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

    try:
        solar_data = engine.fetch_building_insights(lat, lng)
        specs = engine.build_specs()
        segments = engine.build_segments(solar_data)
        orientation, placements, mix, face_mix, face_limits = engine.compute_layout(segments, specs)
    except SolarApiError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    total_watts = sum(specs[idx].watt * count for idx, count in mix.items())
    dc_kw = round(total_watts / 1000.0, 3)

    roof_polygons: List[Polygon] = [segment.polygon for segment in segments]
    image_b64 = render_layout(roof_polygons, placements, specs)

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

    solar_potential = solar_data.get("solarPotential", {})
    max_orientation = solar_potential.get("maxSunshineOrientation", {})
    site_info = {
        "lat": lat,
        "lng": lng,
        "azimuth_deg": max_orientation.get("azimuthDegrees"),
        "tilt_deg": max_orientation.get("tiltDegrees"),
    }

    return SolarDesignResponse(
        site=site_info,
        solar_potential=solar_potential,
        result={
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
    )


@app.get("/healthz")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}
