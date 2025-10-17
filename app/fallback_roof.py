from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from shapely import affinity
from shapely.geometry import Polygon, box

from .panel_layout import PanelSpec
from .schemas import (
    FallbackPanelResult,
    PanelMixEntry,
    PanelPlacementGeometry,
    PanelSpecInput,
)
from .geo import (
    meters_per_pixel as compute_mpp,
    polygon_pixels_to_latlng,
    latlng_to_pixels,
)
from .buildings import fetch_osm_building_polygon


STATIC_MAP_ENDPOINT = "https://maps.googleapis.com/maps/api/staticmap"
STATIC_MAP_SIZE = (640, 640)
STATIC_MAP_ZOOM = 21
STATIC_MAP_SCALE = 2


@dataclass
class FallbackPanelPlacement:
    spec: PanelSpec
    polygon_px: Polygon
    polygon_m: Polygon

    def translated(self, dx_px: float, dy_px: float, mpp: float) -> "FallbackPanelPlacement":
        return FallbackPanelPlacement(
            spec=self.spec,
            polygon_px=affinity.translate(self.polygon_px, xoff=dx_px, yoff=dy_px),
            polygon_m=affinity.translate(
                self.polygon_m,
                xoff=dx_px * mpp,
                yoff=dy_px * mpp,
            ),
        )


@dataclass
class OrientationSummary:
    mode: str
    orientation_label: str
    placements: List[FallbackPanelPlacement]
    total_watts: float
    panel_area_m2: float

    @property
    def count(self) -> int:
        return len(self.placements)


def panel_specs_from_inputs(inputs: Sequence[PanelSpecInput]) -> List[PanelSpec]:
    specs: List[PanelSpec] = []
    for idx, panel in enumerate(inputs):
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


def download_static_map(api_key: str, lat: float, lng: float) -> np.ndarray:
    params = {
        "center": f"{lat},{lng}",
        "zoom": STATIC_MAP_ZOOM,
        "size": f"{STATIC_MAP_SIZE[0]}x{STATIC_MAP_SIZE[1]}",
        "scale": STATIC_MAP_SCALE,
        "maptype": "satellite",
        "key": api_key,
    }
    response = requests.get(STATIC_MAP_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()
    array = np.frombuffer(response.content, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not decode satellite image")
    return image


def unsharp_mask(image: np.ndarray, amount: float = 1.5, sigma: float = 1.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def build_roof_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask_blue = cv2.inRange(hsv, (90, 40, 60), (140, 255, 255))
    mask_gray = cv2.inRange(hsv, (0, 0, 40), (180, 35, 95))
    mask_rust = cv2.inRange(hsv, (5, 25, 40), (25, 255, 255))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5
    )

    combined = cv2.bitwise_or(mask_blue, mask_gray)
    combined = cv2.bitwise_or(combined, mask_rust)
    combined = cv2.bitwise_or(combined, adaptive)

    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    return closed


def _initial_angle_candidates(angles: Sequence[float], max_candidates: int = 3) -> List[float]:
    histogram, bin_edges = np.histogram(angles, bins=36, range=(-90, 90))
    order = np.argsort(histogram)[::-1]
    candidates: List[float] = []
    for idx in order:
        count = histogram[idx]
        if count <= 0:
            continue
        center = (bin_edges[idx] + bin_edges[idx + 1]) / 2.0
        if any(abs(((center - existing) + 90) % 180 - 90) < 5.0 for existing in candidates):
            continue
        candidates.append(center)
        if len(candidates) >= max_candidates:
            break
    if not candidates:
        candidates.append(0.0)
    return candidates


def estimate_orientation(mask: np.ndarray) -> float:
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=40, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return 0.0

    angles: List[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        angles.append(angle)

    if not angles:
        return 0.0

    normalized = [((angle + 90) % 180) - 90 for angle in angles]
    candidates = _initial_angle_candidates(normalized)

    if len(normalized) >= 2 and len(candidates) >= 1:
        data = np.array(
            [[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in normalized],
            dtype=np.float32,
        )
        centers0 = np.array(
            [[math.cos(math.radians(c)), math.sin(math.radians(c))] for c in candidates],
            dtype=np.float32,
        )
        if centers0.ndim == 2 and centers0.shape[0] >= 1:
            distances = np.linalg.norm(data[:, None, :] - centers0[None, :, :], axis=2)
            labels0 = distances.argmin(axis=1).astype(np.int32).reshape(-1, 1)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
            try:
                _, labels, centers = cv2.kmeans(
                    data,
                    centers0.shape[0],
                    labels0,
                    criteria,
                    10,
                    cv2.KMEANS_USE_INITIAL_LABELS,
                )
                counts = np.bincount(labels.flatten(), minlength=centers.shape[0])
                dominant_center = centers[int(np.argmax(counts))]
                dominant_angle = math.degrees(math.atan2(dominant_center[1], dominant_center[0]))
                return dominant_angle
            except cv2.error:
                pass

    histogram, bin_edges = np.histogram(normalized, bins=36, range=(-90, 90))
    dominant_index = int(np.argmax(histogram))
    dominant_angle = (bin_edges[dominant_index] + bin_edges[dominant_index + 1]) / 2.0
    return dominant_angle


def rotate_image(
    image: np.ndarray,
    angle_deg: float,
    flags: int = cv2.INTER_LINEAR,
    border_value: int | Tuple[int, int, int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=flags,
        borderMode=cv2.BORDER_REFLECT if flags != cv2.INTER_NEAREST else cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return rotated, rotation_matrix


def rotate_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    ones = np.ones((points.shape[0], 1))
    hom = np.hstack([points, ones])
    rotated = hom @ matrix.T
    return rotated


def invert_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(matrix)


def merge_small_clusters(mask: np.ndarray, min_ratio: float = 0.05) -> np.ndarray:
    total = cv2.countNonZero(mask)
    if total == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    threshold = max(1, int(total * min_ratio))
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask
    largest_label = 1 + int(np.argmax(areas))

    merged = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= threshold:
            merged[labels == label] = 255

    merged[labels == largest_label] = 255

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < threshold:
            merged[labels == label] = 255
    return merged


def select_main_component(mask: np.ndarray) -> Optional[np.ndarray]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    height, width = mask.shape
    center = np.array([width / 2.0, height / 2.0])
    best_index = -1
    best_score = -float("inf")
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 1000:
            continue
        centroid = centroids[label]
        distance = np.linalg.norm(centroid - center)
        score = area - distance * 50.0
        if score > best_score:
            best_score = score
            best_index = label

    if best_index == -1:
        return None

    component_mask = np.zeros_like(mask)
    component_mask[labels == best_index] = 255
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest


def orthogonalize_contour(contour: np.ndarray) -> np.ndarray:
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(perimeter * 0.015, 2.0)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = approx.reshape(-1, 2).astype(float)
    if len(points) < 3:
        return approx.reshape(-1, 1, 2)

    def classify(edge: np.ndarray) -> str:
        dx = edge[1, 0] - edge[0, 0]
        dy = edge[1, 1] - edge[0, 1]
        if abs(dx) >= abs(dy):
            return "horizontal"
        return "vertical"

    n = len(points)
    new_points: List[Tuple[float, float]] = []
    for i in range(n):
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        prev_edge = classify(np.array([points[prev_idx], points[i]]))
        next_edge = classify(np.array([points[i], points[next_idx]]))
        x_vals: List[float] = []
        y_vals: List[float] = []
        if prev_edge == "horizontal":
            y_vals.extend([points[prev_idx][1], points[i][1]])
        if next_edge == "horizontal":
            y_vals.extend([points[i][1], points[next_idx][1]])
        if prev_edge == "vertical":
            x_vals.extend([points[prev_idx][0], points[i][0]])
        if next_edge == "vertical":
            x_vals.extend([points[i][0], points[next_idx][0]])
        new_x = points[i][0]
        new_y = points[i][1]
        if x_vals:
            new_x = sum(x_vals) / len(x_vals)
        if y_vals:
            new_y = sum(y_vals) / len(y_vals)
        new_points.append((new_x, new_y))

    new_array = np.array(new_points, dtype=np.float32)
    polygon = Polygon(new_array)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return approx.reshape(-1, 1, 2)
    cleaned = np.array(polygon.exterior.coords[:-1], dtype=np.float32)
    return cleaned.reshape(-1, 1, 2)


@dataclass
class RoofAnalysis:
    lat: float
    lng: float
    original_image: np.ndarray
    sharpened_image: np.ndarray
    crop_bounds: Tuple[int, int, int, int]
    crop_offset: Tuple[int, int]
    roof_mask: np.ndarray
    rotated_mask: np.ndarray
    rotated_image: np.ndarray
    rotation_matrix: np.ndarray
    inverse_rotation_matrix: np.ndarray
    component_contour: np.ndarray
    rotated_polygon: Polygon
    original_polygon: Polygon
    orientation: float
    mpp: float
    confidence: float


def polygon_from_contour(contour: np.ndarray) -> Polygon:
    points = contour.reshape(-1, 2)
    polygon = Polygon(points)
    polygon = polygon.buffer(0)
    return polygon


def principal_axis_angle(polygon: Polygon) -> float:
    coords = np.array(polygon.exterior.coords[:-1], dtype=np.float64)
    if coords.size == 0:
        return 0.0

    centroid = coords.mean(axis=0)
    centered = coords - centroid
    if centered.ndim != 2 or centered.shape[0] < 2:
        return 0.0

    covariance = np.cov(centered, rowvar=False)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return 0.0

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    index = int(np.argmax(eigenvalues))
    direction = eigenvectors[:, index]
    angle = math.degrees(math.atan2(direction[1], direction[0]))
    angle = (angle + 180.0) % 180.0
    if angle > 90.0:
        angle -= 180.0
    return angle


def analyze_roof(api_key: str, lat: float, lng: float) -> Optional[RoofAnalysis]:
    original_image = download_static_map(api_key, lat, lng)
    mpp = compute_mpp(lat, STATIC_MAP_ZOOM, STATIC_MAP_SCALE)
    sharpened_full = unsharp_mask(original_image)

    height, width = original_image.shape[:2]
    crop_bounds = (0, 0, width, height)
    crop_offset = (0, 0)

    building_polygon = fetch_osm_building_polygon(lat, lng)
    building_mask: Optional[np.ndarray] = None

    if building_polygon is not None and not building_polygon.is_empty:
        polygon_coords_latlng = [(coord[1], coord[0]) for coord in building_polygon.exterior.coords]
        pixel_points = latlng_to_pixels(
            lat,
            lng,
            STATIC_MAP_ZOOM,
            STATIC_MAP_SCALE,
            width,
            height,
            polygon_coords_latlng,
        )
        polygon_array = np.array(pixel_points, dtype=np.float32)
        if len(polygon_array) >= 3:
            building_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(building_mask, [np.round(polygon_array).astype(np.int32)], 255)
            pad_px = max(1, int(round(5.0 / max(mpp, 1e-6))))
            xs = polygon_array[:, 0]
            ys = polygon_array[:, 1]
            minx = max(0, int(np.floor(xs.min()) - pad_px))
            maxx = min(width, int(np.ceil(xs.max()) + pad_px))
            miny = max(0, int(np.floor(ys.min()) - pad_px))
            maxy = min(height, int(np.ceil(ys.max()) + pad_px))
            if maxx - minx > 10 and maxy - miny > 10:
                crop_bounds = (minx, miny, maxx, maxy)
                crop_offset = (minx, miny)

    minx, miny, maxx, maxy = crop_bounds
    cropped_sharpened = sharpened_full[miny:maxy, minx:maxx].copy()
    roof_mask = build_roof_mask(cropped_sharpened)
    if building_mask is not None:
        building_crop = building_mask[miny:maxy, minx:maxx]
        roof_mask = cv2.bitwise_and(roof_mask, building_crop)

    if cv2.countNonZero(roof_mask) == 0:
        if building_mask is not None:
            roof_mask = building_mask[miny:maxy, minx:maxx]
        else:
            return None

    roof_mask = merge_small_clusters(roof_mask)
    orientation = estimate_orientation(roof_mask)

    rotated_mask, rotation_matrix = rotate_image(roof_mask, -orientation, flags=cv2.INTER_NEAREST)
    rotated_image, _ = rotate_image(cropped_sharpened, -orientation)
    component = select_main_component(rotated_mask)
    if component is None:
        return None

    ortho_contour = orthogonalize_contour(component)
    rotated_polygon = polygon_from_contour(ortho_contour)
    if rotated_polygon.is_empty:
        return None

    inverse_matrix = invert_rotation_matrix(rotation_matrix)
    original_points = rotate_points(ortho_contour.reshape(-1, 2), inverse_matrix)
    if original_points.size == 0:
        return None
    original_points[:, 0] += crop_offset[0]
    original_points[:, 1] += crop_offset[1]
    original_polygon = Polygon(original_points).buffer(0)
    if original_polygon.is_empty:
        return None

    confidence = compute_confidence(component, rotated_polygon)

    return RoofAnalysis(
        lat=lat,
        lng=lng,
        original_image=original_image,
        sharpened_image=cropped_sharpened,
        crop_bounds=crop_bounds,
        crop_offset=crop_offset,
        roof_mask=roof_mask,
        rotated_mask=rotated_mask,
        rotated_image=rotated_image,
        rotation_matrix=rotation_matrix,
        inverse_rotation_matrix=inverse_matrix,
        component_contour=component,
        rotated_polygon=rotated_polygon,
        original_polygon=original_polygon,
        orientation=orientation,
        mpp=mpp,
        confidence=confidence,
    )


def _rectangles_overlap(rect: Polygon, others: Sequence[Polygon]) -> bool:
    for other in others:
        if rect.intersection(other).area > 1e-6:
            return True
    return False


def _layout_for_orientation(
    polygon_px: Polygon,
    polygon_m: Polygon,
    axis_angle: float,
    specs: Sequence[PanelSpec],
    max_count: Optional[int],
    orientation_label: str,
    mpp: float,
) -> OrientationSummary:
    centroid = polygon_m.centroid
    rotated_polygon = affinity.rotate(
        polygon_m,
        -axis_angle,
        origin=(centroid.x, centroid.y),
    )

    placements_aligned: List[Polygon] = []
    placement_records: List[Tuple[PanelSpec, Polygon]] = []
    max_slots = None if max_count is None else max(0, int(max_count))

    minx, miny, maxx, maxy = rotated_polygon.bounds
    if not all(math.isfinite(val) for val in (minx, miny, maxx, maxy)):
        return OrientationSummary(orientation_label, orientation_label, [], 0.0, 0.0)

    for spec in sorted(
        specs,
        key=lambda s: (s.efficiency, s.area_m2),
        reverse=True,
    ):
        if max_slots is not None and len(placement_records) >= max_slots:
            break

        width = spec.width_m if orientation_label == "portrait" else spec.height_m
        height = spec.height_m if orientation_label == "portrait" else spec.width_m
        gap = max(spec.gap_m, 0.0)
        step_x = width + gap
        step_y = height + gap
        if step_x <= 0 or step_y <= 0:
            continue

        half_w = width / 2.0
        half_h = height / 2.0
        offsets_x = [0.0, step_x / 2.0]
        offsets_y = [0.0, step_y / 2.0]

        best_local: List[Polygon] = []
        limit = None if max_slots is None else max_slots - len(placement_records)

        for ox in offsets_x:
            for oy in offsets_y:
                current: List[Polygon] = []
                y = miny + half_h + oy
                while y + half_h <= maxy + 1e-9:
                    x = minx + half_w + ox
                    while x + half_w <= maxx + 1e-9:
                        if limit is not None and len(current) >= limit:
                            break
                        rect = box(x - half_w, y - half_h, x + half_w, y + half_h)
                        if not rect.within(rotated_polygon):
                            x += step_x
                            continue
                        if _rectangles_overlap(rect, placements_aligned) or _rectangles_overlap(
                            rect, current
                        ):
                            x += step_x
                            continue
                        current.append(rect)
                        x += step_x
                    if limit is not None and len(current) >= limit:
                        break
                    y += step_y
                if len(current) > len(best_local):
                    best_local = current

        for rect in best_local:
            placements_aligned.append(rect)
            placement_records.append((spec, rect))
            if max_slots is not None and len(placement_records) >= max_slots:
                break
        if max_slots is not None and len(placement_records) >= max_slots:
            break

    placements: List[FallbackPanelPlacement] = []
    for spec, rect_aligned in placement_records:
        rect_metric = affinity.rotate(
            rect_aligned,
            axis_angle,
            origin=(centroid.x, centroid.y),
        )
        rect_px = affinity.scale(rect_metric, 1.0 / mpp, 1.0 / mpp, origin=(0.0, 0.0))
        placements.append(
            FallbackPanelPlacement(
                spec=spec,
                polygon_px=rect_px,
                polygon_m=rect_metric,
            )
        )

    total_watts = sum(p.spec.watt for p in placements)
    panel_area_m2 = sum(p.spec.area_m2 for p in placements)
    return OrientationSummary(orientation_label, orientation_label, placements, total_watts, panel_area_m2)


def place_panels(
    polygon: Polygon,
    specs: Sequence[PanelSpec],
    mpp: float,
    orientation_mode: str,
    max_count: Optional[int],
) -> Dict[str, OrientationSummary]:
    polygon_m = affinity.scale(polygon, xfact=mpp, yfact=mpp, origin=(0.0, 0.0))
    axis_angle = principal_axis_angle(polygon_m)

    portrait = _layout_for_orientation(
        polygon,
        polygon_m,
        axis_angle,
        specs,
        max_count,
        "portrait",
        mpp,
    )
    landscape = _layout_for_orientation(
        polygon,
        polygon_m,
        axis_angle,
        specs,
        max_count,
        "landscape",
        mpp,
    )

    auto_base = portrait if portrait.total_watts >= landscape.total_watts else landscape
    auto_summary = OrientationSummary(
        mode="auto",
        orientation_label=auto_base.orientation_label,
        placements=list(auto_base.placements),
        total_watts=auto_base.total_watts,
        panel_area_m2=auto_base.panel_area_m2,
    )

    return {"portrait": portrait, "landscape": landscape, "auto": auto_summary}


def encode_image(image: Image.Image) -> str:
    buffer = base64.b64encode(_image_to_png_bytes(image)).decode("ascii")
    return f"data:image/png;base64,{buffer}"


def _image_to_png_bytes(image: Image.Image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_layout_overlay(
    image_size: Tuple[int, int],
    roof_polygon: Polygon,
    placements: Sequence[FallbackPanelPlacement],
) -> Image.Image:
    width, height = image_size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    roof_coords = [(float(x), float(y)) for x, y in roof_polygon.exterior.coords]
    if roof_coords:
        fill_color = (0, 255, 255, int(0.25 * 255))
        outline_color = (0, 255, 255, 255)
        draw.polygon(roof_coords, fill=fill_color)
        draw.line(roof_coords + [roof_coords[0]], fill=outline_color, width=2)

    panel_fill = (255, 255, 255, 255)
    panel_outline = (255, 0, 0, 255)
    for placement in placements:
        coords = [(float(x), float(y)) for x, y in placement.polygon_px.exterior.coords]
        draw.polygon(coords, fill=panel_fill)
        draw.line(coords + [coords[0]], fill=panel_outline, width=1)

    return overlay


def draw_result(
    image_size: Tuple[int, int],
    roof_polygon: Polygon,
    placements: Sequence[FallbackPanelPlacement],
) -> str:
    overlay = render_layout_overlay(image_size, roof_polygon, placements)
    return encode_image(overlay)


def compute_confidence(contour: np.ndarray, polygon: Polygon) -> float:
    if contour is None or polygon.is_empty:
        return 0.0

    area = cv2.contourArea(contour)
    if area <= 0:
        return 0.0
    perimeter = cv2.arcLength(contour, True)
    circularity = 0.0 if perimeter == 0 else min(1.0, 4 * math.pi * area / (perimeter ** 2))

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    convex_ratio = 0.0 if hull_area == 0 else min(1.0, area / hull_area)

    area_score = min(1.0, area / 5000.0)

    coords = polygon.exterior.coords
    angles = []
    for idx in range(len(coords) - 1):
        x1, y1 = coords[idx]
        x2, y2 = coords[idx + 1]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx)))
        angle = min(angle % 180.0, 180.0 - (angle % 180.0))
        deviation = min(abs(angle), abs(90.0 - angle))
        angles.append(deviation)
    orthogonality = 1.0 if not angles else max(0.0, 1.0 - (sum(angles) / (len(angles) * 30.0)))

    confidence = 0.25 * (circularity + convex_ratio + area_score + orthogonality)
    return max(0.0, min(confidence, 1.0))


def pack_mix(placements: Sequence[FallbackPanelPlacement]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for placement in placements:
        idx = placement.spec.index
        counts[idx] = counts.get(idx, 0) + 1
    return counts


def placements_to_mix_entries(placements: Sequence[FallbackPanelPlacement]) -> List[PanelMixEntry]:
    mix_counts = pack_mix(placements)
    entries: List[PanelMixEntry] = []
    for spec_index, count in mix_counts.items():
        spec = next((placement.spec for placement in placements if placement.spec.index == spec_index), None)
        if spec is None:
            continue
        entries.append(
            PanelMixEntry(
                spec=PanelSpecInput(**spec.original),
                count=count,
            )
        )
    return entries


def build_debug_images(
    mask: np.ndarray,
    edges: np.ndarray,
    rotated_polygon: Polygon,
    layout_overlays: Optional[Dict[str, Image.Image]] = None,
) -> Dict[str, str]:
    debug_images: Dict[str, str] = {}

    def to_base64(img: np.ndarray) -> str:
        pil = Image.fromarray(img)
        return encode_image(pil.convert("RGB"))

    debug_images["mask"] = to_base64(mask)
    debug_images["edges"] = to_base64(edges)

    canvas = Image.new("RGB", (mask.shape[1], mask.shape[0]), (0, 0, 0))
    overlay = ImageDraw.Draw(canvas)
    coords = [(float(x), float(y)) for x, y in rotated_polygon.exterior.coords]
    overlay.line(coords + [coords[0]], fill=(0, 255, 0), width=2)
    debug_images["rotated"] = encode_image(canvas)

    if layout_overlays:
        modes = ["portrait", "landscape", "auto"]
        first_overlay = next((layout_overlays.get(mode) for mode in modes if layout_overlays.get(mode)), None)
        if first_overlay is not None:
            width = first_overlay.width
            height = first_overlay.height
            label_height = 24
            combined = Image.new(
                "RGBA",
                (width * len(modes), height + label_height),
                (0, 0, 0, 0),
            )
            font = ImageFont.load_default()
            draw_combined = ImageDraw.Draw(combined)
            for idx, mode in enumerate(modes):
                image = layout_overlays.get(mode)
                if image is None:
                    continue
                x_offset = idx * width
                combined.paste(image, (x_offset, 0), image)
                text = mode
                text_width, text_height = draw_combined.textsize(text, font=font)
                text_x = x_offset + (width - text_width) // 2
                text_y = height + (label_height - text_height) // 2
                draw_combined.text(
                    (text_x, text_y),
                    text,
                    fill=(255, 255, 255, 255),
                    font=font,
                )
            debug_images["layouts"] = encode_image(combined)

    return debug_images


def run_fallback_detection(
    api_key: str,
    lat: float,
    lng: float,
    specs: Sequence[PanelSpec],
    orientation_mode: str,
    max_total: Optional[int],
    max_face: Optional[int],
    debug: bool = False,
) -> Dict[str, object]:
    analysis = analyze_roof(api_key, lat, lng)
    if analysis is None:
        return {
            "roof_detected": False,
            "message": "屋根を特定できませんでした。",
        }

    original_polygon = analysis.original_polygon
    mpp = analysis.mpp
    orientation = analysis.orientation
    sharpened = analysis.sharpened_image

    limit = max_total
    if max_face is not None:
        limit = min(max_total, max_face) if max_total is not None else max_face

    orientation_mode = (orientation_mode or "auto").lower()
    placement_summaries = place_panels(original_polygon, specs, mpp, orientation_mode, limit)

    selected_key = orientation_mode if orientation_mode in placement_summaries else "auto"
    final_summary = placement_summaries[selected_key]
    orientation_used = final_summary.orientation_label
    placements = final_summary.placements

    mix_entries = placements_to_mix_entries(placements)
    total_panels = len(placements)
    total_watts = final_summary.total_watts
    dc_kw = total_watts / 1000.0

    roof_area_px = float(original_polygon.area)
    roof_area_m2 = roof_area_px * (mpp ** 2)
    coverage_ratio = 0.0
    if roof_area_m2 > 0:
        coverage_ratio = max(0.0, min(final_summary.panel_area_m2 / roof_area_m2, 1.0))

    roof_polygon_points = [
        [float(x), float(y)]
        for x, y in list(original_polygon.exterior.coords)[:-1]
    ]

    width = analysis.original_image.shape[1]
    height = analysis.original_image.shape[0]
    roof_polygon_latlng = [
        [lng_val, lat_val]
        for lat_val, lng_val in polygon_pixels_to_latlng(
            analysis.lat,
            analysis.lng,
            STATIC_MAP_ZOOM,
            STATIC_MAP_SCALE,
            width,
            height,
            [(x, y) for x, y in list(original_polygon.exterior.coords)[:-1]],
        )
    ]

    debug_images = None
    if debug:
        edges = cv2.Canny(analysis.rotated_mask, 50, 150)
        layout_overlays: Dict[str, Image.Image] = {}
        display_size = (sharpened.shape[1], sharpened.shape[0])
        offset_x, offset_y = analysis.crop_offset
        translated_polygon = affinity.translate(
            original_polygon, xoff=-offset_x, yoff=-offset_y
        )
        for key, summary in placement_summaries.items():
            placements_for_mode = summary.placements
            if offset_x != 0 or offset_y != 0:
                placements_for_mode = [
                    placement.translated(-offset_x, -offset_y, mpp)
                    for placement in placements_for_mode
                ]
            layout_overlays[key] = render_layout_overlay(
                display_size,
                translated_polygon,
                placements_for_mode,
            )
        debug_images = build_debug_images(
            analysis.rotated_mask,
            edges,
            analysis.rotated_polygon,
            layout_overlays=layout_overlays,
        )

    normalized_orientation = (orientation + 180.0) % 180.0

    offset_x, offset_y = analysis.crop_offset
    crop_minx, crop_miny, crop_maxx, crop_maxy = analysis.crop_bounds
    display_polygon = original_polygon
    display_placements: List[FallbackPanelPlacement]
    if (offset_x, offset_y) != (0, 0) or (crop_minx, crop_miny) != (0, 0) or (
        crop_maxx != analysis.original_image.shape[1]
        or crop_maxy != analysis.original_image.shape[0]
    ):
        display_polygon = affinity.translate(original_polygon, xoff=-offset_x, yoff=-offset_y)
        display_placements = [
            placement.translated(-offset_x, -offset_y, mpp) for placement in placements
        ]
    else:
        display_polygon = original_polygon
        display_placements = list(placements)

    image_size = (sharpened.shape[1], sharpened.shape[0])
    image_b64 = draw_result(image_size, display_polygon, display_placements)

    portrait_count = placement_summaries["portrait"].count
    landscape_count = placement_summaries["landscape"].count
    auto_count = placement_summaries["auto"].count

    panel_details = [
        PanelPlacementGeometry(
            spec=PanelSpecInput(**placement.spec.original),
            polygon_px=[
                [float(x), float(y)]
                for x, y in list(placement.polygon_px.exterior.coords)[:-1]
            ],
            polygon_m=[
                [float(x), float(y)]
                for x, y in list(placement.polygon_m.exterior.coords)[:-1]
            ],
        )
        for placement in placements
    ]

    result_model = FallbackPanelResult(
        orientation_used=orientation_used,
        dc_kw=round(dc_kw, 3),
        mix=mix_entries,
        count=total_panels,
        portrait_count=portrait_count,
        landscape_count=landscape_count,
        auto_count=auto_count,
        panels=panel_details,
        confidence=round(coverage_ratio, 3),
    )

    response: Dict[str, object] = {
        "roof_detected": True,
        "confidence": round(coverage_ratio, 3),
        "orientation_deg": round(float(normalized_orientation), 2),
        "roof_area_m2": round(roof_area_m2, 2),
        "panel_counts": total_panels,
        "dc_kw": round(dc_kw, 3),
        "roof_polygon": roof_polygon_points,
        "roof_polygon_latlng": roof_polygon_latlng,
        "result": result_model,
        "image_png_base64": image_b64,
    }

    if debug_images is not None:
        response["debug_images"] = debug_images

    return response

