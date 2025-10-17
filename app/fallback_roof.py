from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw
from shapely import affinity
from shapely.geometry import Polygon, box

from .panel_layout import PanelSpec
from .schemas import FallbackPanelResult, PanelMixEntry, PanelSpecInput


STATIC_MAP_ENDPOINT = "https://maps.googleapis.com/maps/api/staticmap"
STATIC_MAP_SIZE = (640, 640)
STATIC_MAP_ZOOM = 21
STATIC_MAP_SCALE = 2
EARTH_RADIUS_M = 6_378_137.0


@dataclass
class FallbackPanelPlacement:
    spec: PanelSpec
    rectangle: Polygon


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


def meters_per_pixel(lat: float) -> float:
    return (
        math.cos(math.radians(lat))
        * 2.0
        * math.pi
        * EARTH_RADIUS_M
        / (256.0 * (2 ** STATIC_MAP_ZOOM))
        / STATIC_MAP_SCALE
    )


def polygon_from_contour(contour: np.ndarray) -> Polygon:
    points = contour.reshape(-1, 2)
    polygon = Polygon(points)
    polygon = polygon.buffer(0)
    return polygon


def place_panels(
    polygon: Polygon,
    specs: Sequence[PanelSpec],
    mpp: float,
    orientation_mode: str,
    max_count: Optional[int],
) -> Tuple[str, List[FallbackPanelPlacement]]:
    orientations: Iterable[str]
    if orientation_mode == "auto":
        orientations = ("portrait", "landscape")
    else:
        orientations = (orientation_mode,)

    best_orientation = "portrait"
    best_placements: List[FallbackPanelPlacement] = []
    best_watts = -1.0

    for orientation in orientations:
        placements: List[FallbackPanelPlacement] = []
        remaining = max_count if max_count is not None else math.inf
        bounds = polygon.bounds
        if not math.isfinite(bounds[0]):
            continue

        for spec in sorted(specs, key=lambda s: (s.watt / max(s.area_m2, 1e-6), s.area_m2), reverse=True):
            panel_w_m, panel_h_m = (
                (spec.width_m, spec.height_m)
                if orientation == "portrait"
                else (spec.height_m, spec.width_m)
            )
            panel_gap = spec.gap_m
            panel_w_px = panel_w_m / mpp
            panel_h_px = panel_h_m / mpp
            gap_px = panel_gap / mpp
            step_x = panel_w_px + gap_px
            step_y = panel_h_px + gap_px
            half_w = panel_w_px / 2.0
            half_h = panel_h_px / 2.0
            if step_x <= 0 or step_y <= 0:
                continue

            minx, miny, maxx, maxy = polygon.bounds
            offset_x_values = [0.0, step_x / 2.0]
            offset_y_values = [0.0, step_y / 2.0]

            best_local: List[Polygon] = []
            for ox in offset_x_values:
                for oy in offset_y_values:
                    current: List[Polygon] = []
                    y = miny + half_h + oy
                    while y + half_h <= maxy + 1e-6:
                        x = minx + half_w + ox
                        while x + half_w <= maxx + 1e-6:
                            rect = box(x - half_w, y - half_h, x + half_w, y + half_h)
                            if not rect.within(polygon):
                                x += step_x
                                continue
                            if any(rect.intersects(existing) for existing in current):
                                x += step_x
                                continue
                            current.append(rect)
                            if len(current) >= remaining:
                                break
                            x += step_x
                        if len(current) >= remaining:
                            break
                        y += step_y
                    if len(current) > len(best_local):
                        best_local = current
            for rect in best_local:
                placements.append(FallbackPanelPlacement(spec=spec, rectangle=rect))
                remaining -= 1
                if remaining <= 0:
                    break
            if remaining <= 0:
                break

        total_watts = sum(p.spec.watt for p in placements)
        if total_watts > best_watts:
            best_watts = total_watts
            best_orientation = orientation
            best_placements = placements

    return best_orientation, best_placements


def encode_image(image: Image.Image) -> str:
    buffer = base64.b64encode(_image_to_png_bytes(image))
    return buffer.decode("ascii")


def _image_to_png_bytes(image: Image.Image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def draw_result(
    image: np.ndarray,
    roof_polygon: Polygon,
    placements: Sequence[FallbackPanelPlacement],
    fill_roof: bool = True,
) -> str:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    roof_coords = [(float(x), float(y)) for x, y in roof_polygon.exterior.coords]
    roof_fill = (102, 205, 255, 120)
    roof_outline = (64, 112, 214, 220)
    draw.polygon(roof_coords, fill=roof_fill if fill_roof else None, outline=roof_outline)
    if not fill_roof:
        draw.line(roof_coords + [roof_coords[0]], fill=roof_outline, width=2)

    panel_fill = (255, 255, 255, 180)
    panel_outline = (220, 32, 32, 255)
    for placement in placements:
        coords = [(float(x), float(y)) for x, y in placement.rectangle.exterior.coords]
        draw.polygon(coords, fill=panel_fill, outline=panel_outline)

    combined = Image.alpha_composite(pil_image, overlay)
    return encode_image(combined.convert("RGB"))


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


def build_debug_images(mask: np.ndarray, edges: np.ndarray, rotated_polygon: Polygon) -> Dict[str, str]:
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
    return debug_images


def cv_matrix_to_shapely(matrix: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    return (a, b, c, d, tx, ty)


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
    original_image = download_static_map(api_key, lat, lng)
    sharpened = unsharp_mask(original_image)
    roof_mask = build_roof_mask(sharpened)
    orientation = estimate_orientation(roof_mask)

    rotated_mask, rotation_matrix = rotate_image(roof_mask, -orientation, flags=cv2.INTER_NEAREST)
    rotated_image, _ = rotate_image(sharpened, -orientation)
    component = select_main_component(rotated_mask)
    if component is None:
        return {
            "roof_detected": False,
            "message": "屋根を特定できませんでした。",
        }

    ortho_contour = orthogonalize_contour(component)
    rotated_polygon = polygon_from_contour(ortho_contour)
    if rotated_polygon.is_empty:
        return {
            "roof_detected": False,
            "message": "屋根を特定できませんでした。",
        }

    inverse_matrix = invert_rotation_matrix(rotation_matrix)
    original_points = rotate_points(ortho_contour.reshape(-1, 2), inverse_matrix)
    original_polygon = Polygon(original_points).buffer(0)
    if original_polygon.is_empty:
        return {
            "roof_detected": False,
            "message": "屋根を特定できませんでした。",
        }

    mpp = meters_per_pixel(lat)
    limit = max_total
    if max_face is not None:
        limit = min(max_total, max_face) if max_total is not None else max_face

    orientation_mode = orientation_mode or "auto"
    orientation_used, placements_rotated = place_panels(rotated_polygon, specs, mpp, orientation_mode, limit)

    shapely_matrix = cv_matrix_to_shapely(inverse_matrix)
    placements: List[FallbackPanelPlacement] = []
    for placement in placements_rotated:
        transformed = affinity.affine_transform(placement.rectangle, shapely_matrix)
        placements.append(FallbackPanelPlacement(spec=placement.spec, rectangle=transformed))

    contour_area_px = float(cv2.contourArea(component))
    roof_area_m2 = contour_area_px * (mpp ** 2)
    confidence = compute_confidence(component, rotated_polygon)

    mix_entries = placements_to_mix_entries(placements)
    total_panels = sum(entry.count for entry in mix_entries)
    total_watts = sum(entry.spec.watt * entry.count for entry in mix_entries)
    dc_kw = total_watts / 1000.0

    roof_polygon_points = [
        [float(x), float(y)]
        for x, y in list(original_polygon.exterior.coords)[:-1]
    ]

    debug_images = None
    if debug:
        edges = cv2.Canny(rotated_mask, 50, 150)
        debug_images = build_debug_images(rotated_mask, edges, rotated_polygon)

    normalized_orientation = (orientation + 180.0) % 180.0

    fill_roof = confidence >= 0.6
    image_b64 = draw_result(sharpened, original_polygon, placements, fill_roof=fill_roof)

    result_model = FallbackPanelResult(
        orientation_used=orientation_used,
        dc_kw=round(dc_kw, 3),
        mix=mix_entries,
    )

    response: Dict[str, object] = {
        "roof_detected": True,
        "confidence": round(confidence, 3),
        "orientation_deg": round(float(normalized_orientation), 2),
        "roof_area_m2": round(roof_area_m2, 2),
        "panel_counts": total_panels,
        "dc_kw": round(dc_kw, 3),
        "roof_polygon": roof_polygon_points,
        "result": result_model,
        "image_png_base64": image_b64,
    }

    if debug_images is not None:
        response["debug_images"] = debug_images

    if confidence < 0.6:
        response["fallback_reason"] = "low_confidence"

    return response

