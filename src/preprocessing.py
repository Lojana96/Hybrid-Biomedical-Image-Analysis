"""
Preprocessing utilities for the biomedical image-analysis assignment.

This module provides reusable functions for loading microscopy images,
converting them to grayscale, resizing them, loading binary masks, and
validating dataset consistency.

The functions are designed to support the classical image-processing,
U-Net segmentation, and hybrid pipeline stages of the assignment.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import resize


DEFAULT_IMAGE_SIZE = (256, 256)


def load_image(image_path: Path) -> np.ndarray:
    """
    Load an image from disk as an RGB NumPy array.

    Args:
        image_path: Path to the input image.

    Returns:
        RGB image as a NumPy array.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    return np.asarray(image)


def preprocess_image(
    image_path: Path,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """
    Load and preprocess a microscopy image.

    The input image is converted to RGB, transformed to grayscale,
    resized to the requested dimensions, and returned as a float32
    array with values constrained to the range [0, 1].

    Args:
        image_path: Path to the input microscopy image.
        target_size: Desired output size as (height, width).

    Returns:
        Preprocessed grayscale image with shape defined by target_size.
    """
    image = load_image(image_path)

    gray_image = rgb2gray(image)

    resized_image = resize(
        gray_image,
        target_size,
        anti_aliasing=True,
        preserve_range=True,
    )

    resized_image = np.clip(
        resized_image,
        0.0,
        1.0,
    )

    return resized_image.astype(np.float32)


def load_mask(
    mask_path: Path,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """
    Load and preprocess a binary segmentation mask.

    The mask is converted to grayscale, resized using nearest-neighbour
    interpolation, and converted to a binary array containing 0 and 1.

    Args:
        mask_path: Path to the binary mask image.
        target_size: Desired mask size as (height, width).

    Returns:
        Binary mask as a uint8 NumPy array.

    Raises:
        FileNotFoundError: If the mask file does not exist.
    """
    mask_path = Path(mask_path)

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask file not found: {mask_path}"
        )

    mask = Image.open(mask_path).convert("L")
    mask = np.asarray(mask)

    resized_mask = resize(
        mask,
        target_size,
        order=0,
        anti_aliasing=False,
        preserve_range=True,
    )

    binary_mask = resized_mask > 0

    return binary_mask.astype(np.uint8)


def validate_dataset_images(
    image_directory: Path,
    expected_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate image dimensions and colour mode within a directory.

    Args:
        image_directory: Directory containing PNG images.
        expected_size: Expected image size as (height, width).

    Returns:
        A tuple containing:
            - DataFrame describing all images.
            - DataFrame containing images with unexpected dimensions.

    Raises:
        FileNotFoundError: If the image directory does not exist.
    """
    image_directory = Path(image_directory)

    if not image_directory.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_directory}"
        )

    records = []

    for image_path in sorted(
        image_directory.glob("*.png")
    ):
        with Image.open(image_path) as image:
            records.append(
                {
                    "image_id": image_path.stem,
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                }
            )

    validation_df = pd.DataFrame(records)

    if validation_df.empty:
        return validation_df, validation_df.copy()

    invalid_size = validation_df[
        (
            validation_df["width"]
            != expected_size[1]
        )
        | (
            validation_df["height"]
            != expected_size[0]
        )
    ]

    return validation_df, invalid_size


def validate_image_mask_pairs(
    image_directory: Path,
    mask_directory: Path,
) -> pd.DataFrame:
    """
    Check whether each image has a corresponding segmentation mask.

    Args:
        image_directory: Directory containing input PNG images.
        mask_directory: Directory containing binary PNG masks.

    Returns:
        DataFrame showing image and mask availability for each image ID.

    Raises:
        FileNotFoundError: If either directory does not exist.
    """
    image_directory = Path(image_directory)
    mask_directory = Path(mask_directory)

    if not image_directory.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_directory}"
        )

    if not mask_directory.exists():
        raise FileNotFoundError(
            f"Mask directory not found: {mask_directory}"
        )

    image_ids = {
        path.stem
        for path in image_directory.glob("*.png")
    }

    mask_ids = {
        path.stem
        for path in mask_directory.glob("*.png")
    }

    all_ids = sorted(image_ids | mask_ids)

    records = []

    for image_id in all_ids:
        records.append(
            {
                "image_id": image_id,
                "image_exists": image_id in image_ids,
                "mask_exists": image_id in mask_ids,
                "pair_valid": (
                    image_id in image_ids
                    and image_id in mask_ids
                ),
            }
        )

    return pd.DataFrame(records)