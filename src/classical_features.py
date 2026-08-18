"""
src/classical_features.py

Classical image processing, segmentation, and quantitative morphometric feature
extraction utilities for the biomedical image-analysis assignment.

This module provides:
- Otsu automated intensity thresholding
- Morphological cleaning (noise filtering, hole filling, opening/closing)
- Connected component labeling
- Per-object quantitative feature extraction with skimage.measure.regionprops_table
- Deterministic numeric summarization for numbers-first LLM reasoning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_holes,
    remove_small_objects,
)


def apply_otsu_threshold(
    image: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Compute Otsu threshold on a normalized grayscale image [0, 1]
    and return the binary mask and threshold value.
    """
    image_float = np.asarray(image, dtype=np.float32)
    if image_float.max() > 1.0:
        image_float = image_float / 255.0

    # Guard against constant or near-constant image
    if image_float.max() - image_float.min() < 1e-6:
        thresh = 0.5
        binary = np.zeros_like(image_float, dtype=bool)
    else:
        thresh = float(threshold_otsu(image_float))
        binary = image_float > thresh

    return binary.astype(np.uint8), thresh


def morphological_cleanup(
    binary_mask: np.ndarray,
    min_size: int = 15,
    hole_area_threshold: int = 30,
    closing_radius: int = 1,
    opening_radius: int = 1,
) -> np.ndarray:
    """
    Clean up a binary mask using morphological operations:
    1. Binary closing to connect broken edges
    2. Hole filling
    3. Small object removal (speckle noise)
    4. Optional light opening to separate faint bridges
    """
    mask_bool = binary_mask.astype(bool)

    if closing_radius > 0:
        mask_bool = binary_closing(mask_bool, footprint=disk(closing_radius))

    if hole_area_threshold > 0:
        mask_bool = remove_small_holes(mask_bool, area_threshold=hole_area_threshold)

    if min_size > 0:
        mask_bool = remove_small_objects(mask_bool, min_size=min_size)

    if opening_radius > 0:
        mask_bool = binary_opening(mask_bool, footprint=disk(opening_radius))

    return mask_bool.astype(np.uint8)


def segment_classical(
    image: np.ndarray,
    min_size: int = 15,
    hole_area_threshold: int = 30,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    End-to-end classical segmentation:
    Otsu threshold -> morphological cleanup -> connected component labeling.

    Returns:
        (cleaned_binary_mask, labeled_mask, threshold_value)
    """
    raw_mask, thresh = apply_otsu_threshold(image)
    cleaned_mask = morphological_cleanup(
        raw_mask, min_size=min_size, hole_area_threshold=hole_area_threshold
    )
    labeled_mask = label(cleaned_mask)
    return cleaned_mask, labeled_mask, thresh


def extract_features(
    labeled_mask: np.ndarray,
    intensity_image: Optional[np.ndarray] = None,
    image_id: str = "sample",
) -> pd.DataFrame:
    """
    Compute per-object morphometric and photometric features using skimage.measure.regionprops_table.

    Features computed:
    - label: component identifier
    - area: area in pixels
    - perimeter: boundary perimeter in pixels
    - eccentricity: 0 for circles, approaching 1 for elongated ellipses
    - solidity: ratio of area to convex hull area
    - equivalent_diameter: diameter of circle with same area
    - mean_intensity (if intensity_image provided)
    - max_intensity (if intensity_image provided)
    - min_intensity (if intensity_image provided)
    """
    if labeled_mask.max() == 0:
        # Return empty DataFrame with defined columns
        cols = [
            "image_id",
            "label",
            "area",
            "perimeter",
            "eccentricity",
            "solidity",
            "equivalent_diameter",
            "mean_intensity",
        ]
        return pd.DataFrame(columns=cols)

    properties = [
        "label",
        "area",
        "perimeter",
        "eccentricity",
        "solidity",
        "equivalent_diameter_area",
    ]

    if intensity_image is not None:
        intensity_float = np.asarray(intensity_image, dtype=np.float32)
        if intensity_float.max() > 1.0:
            intensity_float = intensity_float / 255.0
        props_table = regionprops_table(
            labeled_mask,
            intensity_image=intensity_float,
            properties=properties + ["intensity_mean", "intensity_max", "intensity_min"],
        )
    else:
        props_table = regionprops_table(
            labeled_mask,
            properties=properties,
        )

    df = pd.DataFrame(props_table)
    df["image_id"] = image_id

    # Rename columns for clarity and consistency
    rename_map = {
        "equivalent_diameter_area": "equivalent_diameter",
        "intensity_mean": "mean_intensity",
        "intensity_max": "max_intensity",
        "intensity_min": "min_intensity",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Order columns
    leading_cols = ["image_id", "label", "area", "eccentricity", "solidity"]
    existing_leading = [c for c in leading_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_leading]
    df = df[existing_leading + other_cols]

    return df


def classify_density(n_objects: int) -> str:
    """Classify nucleus count into one of 4 standardized density regimes."""
    if n_objects <= 12:
        return "sparse"
    elif n_objects <= 40:
        return "normal"
    elif n_objects <= 65:
        return "clustered"
    else:
        return "dense"


def summarize_features(
    features_df: pd.DataFrame,
    image_id: str = "sample",
    image_shape: Tuple[int, int] = (256, 256),
) -> Dict[str, Any]:
    """
    Convert a per-object feature table into a structured numeric summary dict
    and a concise natural-language factual paragraph.
    """
    n_objects = len(features_df)
    total_pixels = image_shape[0] * image_shape[1]

    if n_objects == 0:
        return {
            "image_id": image_id,
            "n_objects": 0,
            "density_class": "sparse",
            "total_area": 0,
            "area_fraction": 0.0,
            "mean_area": 0.0,
            "std_area": 0.0,
            "min_area": 0.0,
            "max_area": 0.0,
            "mean_eccentricity": 0.0,
            "mean_solidity": 0.0,
            "mean_intensity": 0.0,
            "shape_regularity": "uncertain",
            "numeric_text": (
                f"Image {image_id}: Quantitative segmentation identified 0 objects. "
                "Total segmented area is 0 pixels (area fraction 0.00%). "
                "Density regime is classified as sparse with no detectable nuclei."
            ),
        }

    total_area = float(features_df["area"].sum())
    area_fraction = total_area / total_pixels
    mean_area = float(features_df["area"].mean())
    std_area = float(features_df["area"].std()) if n_objects > 1 else 0.0
    min_area = float(features_df["area"].min())
    max_area = float(features_df["area"].max())
    mean_ecc = float(features_df["eccentricity"].mean())
    mean_sol = float(features_df["solidity"].mean())
    mean_int = (
        float(features_df["mean_intensity"].mean())
        if "mean_intensity" in features_df.columns
        else 0.0
    )

    density_class = classify_density(n_objects)

    # Assess shape regularity based on eccentricity & solidity
    if mean_ecc < 0.65 and mean_sol > 0.90:
        shape_reg = "regular"
    elif mean_ecc > 0.82 or mean_sol < 0.80:
        shape_reg = "irregular"
    else:
        shape_reg = "moderate"

    # Format factual numbers-only text summary
    numeric_text = (
        f"Image {image_id}: Quantitative segmentation identified {n_objects} distinct cellular nuclei "
        f"occupying a total area of {total_area:.0f} pixels ({area_fraction*100:.2f}% field coverage). "
        f"Individual nucleus area averages {mean_area:.1f} ± {std_area:.1f} pixels (range {min_area:.0f}–{max_area:.0f}). "
        f"Mean eccentricity is {mean_ecc:.3f} and mean solidity is {mean_sol:.3f}, indicating predominantly {shape_reg} elliptical shapes. "
        f"Mean foreground pixel intensity is {mean_int:.3f}. Density is classified as {density_class}."
    )

    return {
        "image_id": image_id,
        "n_objects": n_objects,
        "density_class": density_class,
        "total_area": total_area,
        "area_fraction": area_fraction,
        "mean_area": mean_area,
        "std_area": std_area,
        "min_area": min_area,
        "max_area": max_area,
        "mean_eccentricity": mean_ecc,
        "mean_solidity": mean_sol,
        "mean_intensity": mean_int,
        "shape_regularity": shape_reg,
        "numeric_text": numeric_text,
    }
