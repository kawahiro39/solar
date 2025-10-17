from __future__ import annotations

import math
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, root_validator, validator


class PanelSpecInput(BaseModel):
    w_mm: float = Field(..., gt=0, description="Panel width in millimetres")
    h_mm: float = Field(..., gt=0, description="Panel height in millimetres")
    gap_mm: float = Field(..., ge=0, description="Gap between panels in millimetres")
    watt: float = Field(..., gt=0, description="Panel DC watt rating")

    @root_validator
    def validate_dimensions(cls, values: Dict[str, float]) -> Dict[str, float]:
        width = values.get("w_mm")
        height = values.get("h_mm")
        if width is not None and height is not None:
            if width < 100 or height < 100:
                raise ValueError("パネル寸法はミリメートル単位で指定してください (例: 1722)。")
        gap = values.get("gap_mm")
        if gap is not None and 0 < gap < 1:
            raise ValueError("gap_mm はミリメートル単位 (例: 20) で指定してください。")
        return values


class SquareImageCenter(BaseModel):
    lat: float
    lng: float
    zoom: float


class SquareImageRequest(BaseModel):
    gmaps_url: str = Field(..., min_length=1)
    square_size_px: int = Field(640, ge=64, le=1024)
    scale: int = Field(2, ge=1, le=4)
    maptype: str = Field("satellite")

    @validator("maptype")
    def validate_maptype(cls, value: str) -> str:
        allowed = {"satellite"}
        if value not in allowed:
            raise ValueError("maptype must be one of: satellite")
        return value


class SquareImageResponse(BaseModel):
    image_data_uri: str
    center: SquareImageCenter
    square_size_px: int
    meters_per_pixel: float


class SolarDesignRequest(BaseModel):
    map_url: Optional[str] = None
    lat: Optional[float] = Field(None, description="Latitude in decimal degrees")
    lng: Optional[float] = Field(None, description="Longitude in decimal degrees")
    density: str = Field("標準", regex="^(控えめ|標準|多め)$")
    orientation_mode: str = Field(
        "auto", regex="^(portrait|landscape|auto)$", description="Orientation mode"
    )
    max_per_face: Optional[Union[int, Dict[str, int]]] = None
    max_total: Optional[int] = Field(None, gt=0)
    min_walkway_m: float = Field(0.4, ge=0)
    panels: List[PanelSpecInput]
    force_data_layers: bool = Field(
        False,
        description="When true, always use Solar API dataLayers instead of buildingInsights",
    )
    debug: bool = Field(False, description="When true, include debug imagery in fallback responses")

    @validator("panels")
    def validate_panels(cls, value: List[PanelSpecInput]) -> List[PanelSpecInput]:
        if not value:
            raise ValueError("At least one panel specification must be provided")
        if len(value) > 5:
            raise ValueError("No more than 5 panel specifications are allowed")
        return value

    @root_validator
    def validate_location(cls, values: Dict[str, object]) -> Dict[str, object]:
        lat = values.get("lat")
        lng = values.get("lng")
        map_url = values.get("map_url")
        if lat is None or lng is None:
            if not map_url:
                raise ValueError("Either lat/lng or map_url must be provided")
        return values


class PanelMixEntry(BaseModel):
    spec: PanelSpecInput
    count: int


class SegmentPlacementEntry(BaseModel):
    segment_id: int
    limit: Optional[int]
    placed: List[PanelMixEntry]


class PanelPlacementSummary(BaseModel):
    orientation_used: str
    dc_kw: float
    mix: List[PanelMixEntry]
    by_face: List[SegmentPlacementEntry]
    utilization_metrics: Dict[str, float]
    image_png_base64: str


class SolarDesignResponse(BaseModel):
    site: Dict[str, float]
    solar_potential: Optional[Dict[str, object]] = None
    data_layers: Optional[Dict[str, object]] = None
    result: PanelPlacementSummary


class ErrorResponse(BaseModel):
    detail: str


class PanelPlacementGeometry(BaseModel):
    spec: PanelSpecInput
    polygon_px: List[List[float]] = Field(default_factory=list)
    polygon_m: List[List[float]] = Field(default_factory=list)


class LayoutPanelsSummaryOption(BaseModel):
    panel: PanelSpecInput
    count: int
    dc_kw: float


class LayoutPanelsSummary(BaseModel):
    total_panels: int
    total_kw: float
    by_option: List[LayoutPanelsSummaryOption] = Field(default_factory=list)
    rotation_deg_used: Optional[float] = Field(
        None,
        description="Rotation angle (degrees) applied when optimizing placement, if any.",
    )


class LayoutPanelsResponse(BaseModel):
    layout_image_data_uri: str
    summary: LayoutPanelsSummary


class FallbackPanelResult(BaseModel):
    orientation_used: Optional[str] = None
    dc_kw: Optional[float] = None
    mix: List[PanelMixEntry] = Field(default_factory=list)
    count: int = 0
    portrait_count: int = 0
    landscape_count: int = 0
    auto_count: int = 0
    panels: List[PanelPlacementGeometry] = Field(default_factory=list)
    confidence: Optional[float] = None


class RoofDetectionResponse(BaseModel):
    roof_detected: bool
    confidence: Optional[float] = None
    orientation_deg: Optional[float] = None
    roof_area_m2: Optional[float] = None
    panel_counts: Optional[int] = None
    dc_kw: Optional[float] = None
    roof_polygon: List[List[float]] = Field(default_factory=list)
    roof_polygon_latlng: Optional[List[List[float]]] = None
    result: Optional[FallbackPanelResult] = None
    image_png_base64: Optional[str] = None
    debug_images: Optional[Dict[str, str]] = None
    fallback_reason: Optional[str] = None
    message: Optional[str] = None
    attribution: Optional[List[str]] = None


class OrthoImageRequest(BaseModel):
    lat: float
    lng: float
    zoom: int = Field(21, ge=0, le=21)
    square_px: int = Field(1024, gt=0, le=2048)


class OrthoImageResponse(BaseModel):
    image_png_base64: str
    m_per_px: float
    orientation_deg: float
    roof_polygon: List[List[float]] = Field(default_factory=list)
    roof_polygon_latlng: List[List[float]] = Field(default_factory=list)
    confidence: float
    attribution: List[str] = Field(default_factory=list)


class RoofFaceOutput(BaseModel):
    mask_png_base64: Optional[str] = None
    polygon: List[List[float]] = Field(default_factory=list)
    polygon_latlng: List[List[float]] = Field(default_factory=list)
    azimuth_deg: float
    tilt_rel: float
    area_m2: float


class RoofSegmentRequest(BaseModel):
    image_png_base64: str
    lat: float
    lng: float
    use_osm_mask: bool = True


class RoofSegmentResponse(BaseModel):
    faces: List[RoofFaceOutput]
    confidence: float
    attribution: List[str] = Field(default_factory=list)


class RoofFaceInput(BaseModel):
    polygon: List[List[float]]
    azimuth_deg: Optional[float] = None
    tilt_rel: Optional[float] = None
    area_m2: Optional[float] = None

    @validator("polygon")
    def validate_polygon(cls, value: List[List[float]]) -> List[List[float]]:
        if len(value) < 3:
            raise ValueError("各屋根ポリゴンには3点以上の座標 (m 単位) が必要です。")
        cleaned: List[List[float]] = []
        for point in value:
            if len(point) != 2:
                raise ValueError("屋根ポリゴン座標は [x, y] の2要素配列で指定してください。")
            x, y = point
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError("屋根ポリゴン座標に NaN/Inf が含まれています。")
            cleaned.append([float(x), float(y)])
        return cleaned


class LayoutOptimizeRequest(BaseModel):
    faces: List[RoofFaceInput]
    panels: List[PanelSpecInput]
    orientation_mode: str = Field("auto", regex="^(portrait|landscape|auto)$")
    max_per_face: Optional[int] = Field(None, ge=1)
    max_total: Optional[int] = Field(None, ge=1)
    min_walkway_m: float = Field(0.4, ge=0)


class LayoutPanelsRequest(BaseModel):
    roofs: List[RoofFaceInput]
    panels: List[PanelSpecInput]
    orientation_mode: str = Field("auto", regex="^(portrait|landscape|auto)$")
    max_total: Optional[int] = Field(None, ge=1)
    max_per_face: Optional[int] = Field(None, ge=1)
    min_walkway_m: float = Field(0.4, ge=0)

    @validator("roofs")
    def validate_roofs(cls, value: List[RoofFaceInput]) -> List[RoofFaceInput]:
        if not value:
            raise ValueError("At least one roof face must be provided")
        if len(value) > 10:
            raise ValueError("No more than 10 roof faces are allowed")
        return value

    @validator("panels")
    def validate_panels(cls, value: List[PanelSpecInput]) -> List[PanelSpecInput]:
        if not value:
            raise ValueError("At least one panel specification must be provided")
        if len(value) > 5:
            raise ValueError("No more than 5 panel specifications are allowed")
        return value


class LayoutOptimizeResponse(BaseModel):
    result: FallbackPanelResult
    image_png_base64: str
    confidence: float
    roof_area_m2: float
    dc_kw: float
    attribution: List[str] = Field(default_factory=list)


class DesignPipelineResponse(BaseModel):
    faces: List[RoofFaceOutput]
    result: FallbackPanelResult
    image_png_base64: str
    confidence: float
    dc_kw: float
    attribution: List[str] = Field(default_factory=list)
