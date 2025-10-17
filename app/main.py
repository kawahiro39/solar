from __future__ import annotations

import base64
import io
import os
import time
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .models import (
    ErrorResponse,
    LayoutPanelsRequest,
    LayoutPanelsResponse,
    MetricsResponse,
    PanelSpec,
    RoofPolygon,
    SolarDesignRequest,
    SolarDesignResponse,
    SquareImageRequest,
    SquareImageResponse,
    LatLng,
)
from .services.imagery import ImageryError, fetch_square_image
from .services.packing import PanelLayoutResult, layout_panels_for_spec
from .services.render import render_layout_image

ALLOWED_CORS_ORIGINS = [
    "https://solar-nova.online",
    "https://www.solar-nova.online",
]

ALLOWED_METHODS = ["GET", "POST", "OPTIONS"]
ALLOWED_HEADERS = ["Content-Type", "X-Requested-With", "Accept"]

app = FastAPI(title="Solar Nova Layout API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_CORS_ORIGINS,
    allow_methods=ALLOWED_METHODS,
    allow_headers=ALLOWED_HEADERS,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = []
    for error in exc.errors():
        location = [str(part) for part in error.get("loc", []) if part != "body"]
        field = ".".join(location)
        message = error.get("msg", "Invalid value")
        errors.append(f"{field}: {message}" if field else message)
    detail = "; ".join(errors) if errors else "Invalid request body"
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": detail})


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else "Unexpected error"
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.options("/layout_panels")
async def options_layout_panels() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.options("/solar/design")
async def options_solar_design() -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _encode_image_to_data_uri(image_bytes: bytes, mime_type: str = "image/png") -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


async def _create_layout_response(
    roofs: List[RoofPolygon],
    panel_spec: PanelSpec,
    orientation_mode: str,
    min_walkway_m: float,
    grid_step_mm: int,
) -> PanelLayoutResult:
    return layout_panels_for_spec(
        roofs=roofs,
        panel_spec=panel_spec,
        orientation_mode=orientation_mode,
        min_walkway_m=min_walkway_m,
        grid_step_mm=grid_step_mm,
    )


@app.post(
    "/square_image",
    response_model=SquareImageResponse,
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
async def square_image(request: SquareImageRequest) -> SquareImageResponse:
    api_key = request.api_key or os.getenv("GOOGLE_MAPS_API_KEY")
    try:
        image, meters_per_pixel = await fetch_square_image(
            lat=request.lat,
            lng=request.lng,
            square_m=request.square_m,
            zoom=request.zoom,
            api_key=api_key,
        )
    except ImageryError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data_uri = _encode_image_to_data_uri(buffer.getvalue())

    return SquareImageResponse(
        center=LatLng(lat=request.lat, lng=request.lng),
        meters_per_pixel=meters_per_pixel,
        image_base64=data_uri,
    )


@app.post(
    "/layout_panels",
    response_model=LayoutPanelsResponse,
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
async def layout_panels(request: LayoutPanelsRequest) -> LayoutPanelsResponse:
    if not request.roofs:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No roof polygons provided")
    if not request.panels:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No panel specifications provided")

    start_time = time.perf_counter()
    grid_step_mm = request.grid_step_mm or 100

    try:
        layout_result = await _create_layout_response(
            roofs=request.roofs,
            panel_spec=request.panels[0],
            orientation_mode=request.orientation_mode,
            min_walkway_m=request.min_walkway_m,
            grid_step_mm=grid_step_mm,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    render_start = time.perf_counter()
    image = render_layout_image(layout_result.roof_polygons, layout_result.panel_polygons)
    render_duration = (time.perf_counter() - render_start) * 1000.0

    encode_start = time.perf_counter()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    layout_image = _encode_image_to_data_uri(buffer.getvalue())
    encode_duration = (time.perf_counter() - encode_start) * 1000.0

    metrics = dict(layout_result.metrics)
    metrics["render_ms"] = int(round(render_duration))
    metrics["encode_ms"] = int(round(encode_duration))
    metrics["total_ms"] = int(round((time.perf_counter() - start_time) * 1000.0))

    return LayoutPanelsResponse(
        layout_image=layout_image,
        panel_count=layout_result.panel_count,
        total_kw=layout_result.total_kw,
        orientation_mode=layout_result.orientation_mode,
        metrics=MetricsResponse(**metrics),
    )


@app.post(
    "/solar/design",
    response_model=SolarDesignResponse,
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
async def solar_design(request: SolarDesignRequest) -> SolarDesignResponse:
    if not request.roofs:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No roof polygons provided")
    if not request.panels:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No panel specifications provided")

    start_time = time.perf_counter()
    grid_step_mm = request.grid_step_mm or 100

    alternatives: List[Dict[str, float]] = []
    best_index = None
    best_result: PanelLayoutResult | None = None
    aggregated_metrics: Dict[str, int] = {}

    for index, panel_spec in enumerate(request.panels):
        try:
            result = await _create_layout_response(
                roofs=request.roofs,
                panel_spec=panel_spec,
                orientation_mode=request.orientation_mode,
                min_walkway_m=request.min_walkway_m,
                grid_step_mm=grid_step_mm,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        for key, value in result.metrics.items():
            aggregated_metrics[key] = aggregated_metrics.get(key, 0) + int(round(value))

        alternatives.append(
            {
                "panel_type_index": index,
                "panel_count": result.panel_count,
                "total_kw": result.total_kw,
            }
        )

        if best_result is None or result.total_kw > best_result.total_kw:
            best_result = result
            best_index = index

    if best_result is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unable to compute layout")

    render_start = time.perf_counter()
    image = render_layout_image(best_result.roof_polygons, best_result.panel_polygons)
    render_duration = (time.perf_counter() - render_start) * 1000.0

    encode_start = time.perf_counter()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    best_image = _encode_image_to_data_uri(buffer.getvalue())
    encode_duration = (time.perf_counter() - encode_start) * 1000.0

    aggregated_metrics["render_ms"] = aggregated_metrics.get("render_ms", 0) + int(round(render_duration))
    aggregated_metrics["encode_ms"] = aggregated_metrics.get("encode_ms", 0) + int(round(encode_duration))
    aggregated_metrics["total_ms"] = int(round((time.perf_counter() - start_time) * 1000.0))

    best_layout = {
        "image_base64": best_image,
        "panel_type_index": best_index if best_index is not None else 0,
        "panel_count": best_result.panel_count,
        "total_kw": best_result.total_kw,
    }

    return SolarDesignResponse(
        best_layout=best_layout,
        alternatives=alternatives,
        metrics=MetricsResponse(**aggregated_metrics),
    )
