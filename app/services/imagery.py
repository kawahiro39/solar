from __future__ import annotations

import io
import math
import os
from typing import Optional, Tuple

import httpx
from PIL import Image

from ..utils.geometry import meters_per_pixel

STATIC_MAP_ENDPOINT = "https://maps.googleapis.com/maps/api/staticmap"
MAX_SIZE = 640
DEFAULT_SCALE = 2
MAP_TYPE = "satellite"


class ImageryError(RuntimeError):
    """Raised when imagery retrieval fails."""


async def fetch_square_image(
    lat: float,
    lng: float,
    square_m: int,
    zoom: int,
    api_key: Optional[str] = None,
) -> Tuple[Image.Image, float]:
    key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
    if not key:
        raise ImageryError("Google Maps API key is required to fetch imagery")

    scale = DEFAULT_SCALE
    mpp = meters_per_pixel(lat, zoom, scale)
    final_pixels = max(1, int(math.ceil(square_m / mpp)))
    base_size = max(1, min(MAX_SIZE, int(math.ceil(final_pixels / scale))))

    params = {
        "center": f"{lat},{lng}",
        "zoom": str(zoom),
        "size": f"{base_size}x{base_size}",
        "scale": str(scale),
        "maptype": MAP_TYPE,
        "format": "png",
        "key": key,
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(STATIC_MAP_ENDPOINT, params=params)
    except httpx.HTTPError as exc:
        raise ImageryError("Failed to fetch satellite imagery") from exc

    if response.status_code != 200:
        raise ImageryError(
            f"Satellite imagery request failed with status {response.status_code}"
        )

    try:
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive
        raise ImageryError("Unable to decode satellite imagery response") from exc

    return image, mpp
