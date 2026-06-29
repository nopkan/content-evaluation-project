from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


METRIC_INFO = {
    "brightness_mean": {
        "label": "Brightness",
        "meaning": "Checks whether the image is bright enough to see the product and message clearly without looking washed out.",
        "criteria": "Best range is 95-165 on a 0-255 brightness scale. Lower values look dark or hidden; higher values can lose detail in bright areas.",
    },
    "contrast_rms": {
        "label": "Contrast",
        "meaning": "Measures separation between light and dark areas so the product, text, and offer are easier to scan.",
        "criteria": "Best range is 35-80. Too little contrast looks flat; too much contrast can make details harsh or harder to read.",
    },
    "sharpness_laplacian": {
        "label": "Sharpness",
        "meaning": "Detects whether the image appears crisp or blurry, especially around product edges and important text.",
        "criteria": "Higher is better. Scores are strongest above 300; values near 50 or below indicate likely blur or soft detail.",
    },
    "edge_density": {
        "label": "Visual Detail",
        "meaning": "Estimates how much visual structure is in the image, which affects scanability and perceived complexity.",
        "criteria": "Best range is 0.025-0.12. Very low values may feel empty; very high values can indicate clutter or busy detail.",
    },
    "saturation_mean": {
        "label": "Color Saturation",
        "meaning": "Measures color intensity and whether the image feels lively enough without becoming unnatural.",
        "criteria": "Best range is 55-140. Low saturation can feel dull; very high saturation can look artificial or distracting.",
    },
    "colorfulness_score": {
        "label": "Color Richness",
        "meaning": "Measures overall color variety and balance, which affects visual appeal and stopping power.",
        "criteria": "Best range is 20-90. Low values may look plain; very high values can make the creative feel noisy or off-brand.",
    },
    "underexposed_ratio": {
        "label": "Dark Area Risk",
        "meaning": "Shows how much of the image is extremely dark and may hide product or message details.",
        "criteria": "Lower is better. Strong images keep this below 3%; values near 20% or above are a serious visibility risk.",
    },
    "overexposed_ratio": {
        "label": "Bright Area Risk",
        "meaning": "Shows how much of the image is extremely bright and may lose product detail or make text harder to read.",
        "criteria": "Lower is better. Strong images keep this below 3%; values near 20% or above may look blown out.",
    },
    "noise_score": {
        "label": "Image Noise",
        "meaning": "Detects grain, compression artifacts, or dirty texture that can make the creative feel lower quality.",
        "criteria": "Lower is better. Values below 4 are strong; values near 20 or above suggest visible noise or compression issues.",
    },
    "aspect_ratio": {
        "label": "Format Fit",
        "meaning": "Checks how closely the image shape matches the automatically inferred target format.",
        "criteria": "Best score is within 0.05 of the target ratio. Larger differences can cause cropping, empty space, or poor platform fit.",
    },
}


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def score_good_range(
    value: float,
    good_min: float,
    good_max: float,
    bad_min: float,
    bad_max: float,
) -> float:
    if good_min <= value <= good_max:
        return 10.0

    if value < good_min:
        score = 1 + 9 * ((value - bad_min) / (good_min - bad_min))
        return clamp(score, 1, 10)

    score = 1 + 9 * ((bad_max - value) / (bad_max - good_max))
    return clamp(score, 1, 10)


def score_lower_is_better(value: float, good_max: float, bad_max: float) -> float:
    if value <= good_max:
        return 10.0

    score = 1 + 9 * ((bad_max - value) / (bad_max - good_max))
    return clamp(score, 1, 10)


def score_higher_is_better(value: float, good_min: float, bad_min: float) -> float:
    if value >= good_min:
        return 10.0

    score = 1 + 9 * ((value - bad_min) / (good_min - bad_min))
    return clamp(score, 1, 10)


def score_aspect_ratio(
    value: float,
    target_ratio: float = 1.0,
    tolerance: float = 0.05,
    bad_difference: float = 0.50,
) -> float:
    difference = abs(value - target_ratio)

    if difference <= tolerance:
        return 10.0

    score = 1 + 9 * ((bad_difference - difference) / (bad_difference - tolerance))
    return clamp(score, 1, 10)


def interpret_score(score: float) -> str:
    if score >= 9:
        return "Excellent"
    if score >= 8:
        return "Good"
    if score >= 6:
        return "Okay"
    if score >= 4:
        return "Weak"
    return "Poor"


def measure_picture_metrics(image_path: str | Path) -> dict[str, Any]:
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"Image not found or cannot be read: {image_path}")

    with Image.open(image_path) as pil_img:
        width, height = pil_img.size

    return _measure_picture_metrics_from_cv_image(img, width, height)


def measure_picture_metrics_from_bytes(image_bytes: bytes) -> dict[str, Any]:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Image cannot be read. Please upload a valid image file.")

    with Image.open(BytesIO(image_bytes)) as pil_img:
        width, height = pil_img.size

    return _measure_picture_metrics_from_cv_image(img, width, height)


def _measure_picture_metrics_from_cv_image(
    img: np.ndarray,
    width: int,
    height: int,
) -> dict[str, Any]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    brightness_mean = float(np.mean(gray))
    contrast_rms = float(np.std(gray))
    sharpness_laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.sum(edges > 0) / edges.size)

    saturation_mean = float(np.mean(hsv[:, :, 1]))

    b, g, r = cv2.split(img.astype("float"))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)

    colorfulness_score = float(
        np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )

    underexposed_ratio = float(np.sum(gray < 30) / gray.size)
    overexposed_ratio = float(np.sum(gray > 225) / gray.size)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_score = float(
        np.mean(np.abs(gray.astype("float") - blurred.astype("float")))
    )

    return {
        "width": width,
        "height": height,
        "aspect_ratio": float(width / height),
        "brightness_mean": brightness_mean,
        "contrast_rms": contrast_rms,
        "sharpness_laplacian": sharpness_laplacian,
        "edge_density": edge_density,
        "saturation_mean": saturation_mean,
        "colorfulness_score": colorfulness_score,
        "underexposed_ratio": underexposed_ratio,
        "overexposed_ratio": overexposed_ratio,
        "noise_score": noise_score,
    }


def score_picture_metrics(
    metrics: dict[str, Any],
    target_ratio: float = 1.0,
) -> dict[str, float]:
    scores = {
        "brightness_mean": score_good_range(metrics["brightness_mean"], 95, 165, 40, 220),
        "contrast_rms": score_good_range(metrics["contrast_rms"], 35, 80, 10, 120),
        "sharpness_laplacian": score_higher_is_better(
            metrics["sharpness_laplacian"],
            good_min=300,
            bad_min=50,
        ),
        "edge_density": score_good_range(metrics["edge_density"], 0.025, 0.12, 0.005, 0.25),
        "saturation_mean": score_good_range(metrics["saturation_mean"], 55, 140, 15, 220),
        "colorfulness_score": score_good_range(metrics["colorfulness_score"], 20, 90, 5, 160),
        "underexposed_ratio": score_lower_is_better(metrics["underexposed_ratio"], 0.03, 0.20),
        "overexposed_ratio": score_lower_is_better(metrics["overexposed_ratio"], 0.03, 0.20),
        "noise_score": score_lower_is_better(metrics["noise_score"], 4, 20),
        "aspect_ratio": score_aspect_ratio(metrics["aspect_ratio"], target_ratio=target_ratio),
    }

    return {key: round(value, 2) for key, value in scores.items()}


def explain_metric(metric_name: str, value: Any, score: float) -> dict[str, Any]:
    metric_info = METRIC_INFO.get(
        metric_name,
        {
            "label": metric_name.replace("_", " ").title(),
            "meaning": "No explanation available.",
            "criteria": "No scoring criteria available.",
        },
    )

    return {
        "metric": metric_info["label"],
        "metric_key": metric_name,
        "value": _format_metric_value(metric_name, value),
        "score": score,
        "status": interpret_score(score),
        "meaning": metric_info["meaning"],
        "criteria": metric_info["criteria"],
    }


def build_basic_metric_rows(
    metrics: dict[str, Any],
    scores: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []

    for metric_name, score in scores.items():
        explanation = explain_metric(metric_name, metrics[metric_name], score)
        rows.append(
            {
                "Metric": explanation["metric"],
                "Value": explanation["value"],
                "Score": f"{score}/10",
                "Status": explanation["status"],
                "Meaning": explanation["meaning"],
                "Criteria": explanation["criteria"],
            }
        )

    return rows


def average_score(scores: dict[str, float]) -> float:
    if not scores:
        return 0.0

    return round(sum(scores.values()) / len(scores), 2)


def _format_metric_value(metric_name: str, value: Any) -> str:
    if not isinstance(value, float):
        return str(value)

    if metric_name in {"underexposed_ratio", "overexposed_ratio", "edge_density"}:
        return f"{value * 100:.2f}%"

    if metric_name == "aspect_ratio":
        return f"{value:.3f}"

    return f"{value:.2f}"