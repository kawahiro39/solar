from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, root_validator, validator


class ErrorResponse(BaseModel):
    detail: str


class LatLng(BaseModel):
    lat: float
    lng: float


class SquareImageRequest(BaseModel):
    lat: float
    lng: float
    square_m: int = Field(..., gt=0)
    zoom: int = Field(..., ge=0, le=22)
    api_key: Optional[str] = None


class SquareImageResponse(BaseModel):
    center: LatLng
    meters_per_pixel: float
    image_base64: str


class PanelSpec(BaseModel):
    w_mm: int = Field(..., gt=0)
    h_mm: int = Field(..., gt=0)
    gap_mm: int = Field(0, ge=0)
    watt: int = Field(..., gt=0)


class RoofPolygon(BaseModel):
    polygon: List[List[float]]

    @validator("polygon")
    def validate_polygon(cls, value: List[List[float]]) -> List[List[float]]:
        if len(value) < 3:
            raise ValueError("Polygon must contain at least three points")
        return value


OrientationMode = Literal["north", "south", "east", "west", "auto"]


class LayoutPanelsRequest(BaseModel):
    roofs: List[RoofPolygon]
    panels: List[PanelSpec]
    orientation_mode: OrientationMode
    min_walkway_m: float = Field(0.0, ge=0.0)
    grid_step_mm: Optional[int] = Field(None, gt=0)

    @root_validator
    def ensure_panels(cls, values):
        panels = values.get("panels", [])
        if not panels:
            raise ValueError("At least one panel specification is required")
        return values


class MetricsResponse(BaseModel):
    validate_ms: int = 0
    simplify_ms: int = 0
    rotation_ms: int = 0
    pack_ms: int = 0
    render_ms: int = 0
    encode_ms: int = 0
    total_ms: int = 0


class LayoutPanelsResponse(BaseModel):
    layout_image: str
    panel_count: int
    total_kw: float
    orientation_mode: str
    metrics: MetricsResponse


class DesignAlternative(BaseModel):
    panel_type_index: int
    panel_count: int
    total_kw: float


class BestLayout(BaseModel):
    image_base64: str
    panel_type_index: int
    panel_count: int
    total_kw: float


class SolarDesignRequest(LayoutPanelsRequest):
    pass


class SolarDesignResponse(BaseModel):
    best_layout: BestLayout
    alternatives: List[DesignAlternative]
    metrics: MetricsResponse
