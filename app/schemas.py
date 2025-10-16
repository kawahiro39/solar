from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, root_validator, validator


class PanelSpecInput(BaseModel):
    w_mm: float = Field(..., gt=0, description="Panel width in millimetres")
    h_mm: float = Field(..., gt=0, description="Panel height in millimetres")
    gap_mm: float = Field(..., ge=0, description="Gap between panels in millimetres")
    watt: float = Field(..., gt=0, description="Panel DC watt rating")


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
