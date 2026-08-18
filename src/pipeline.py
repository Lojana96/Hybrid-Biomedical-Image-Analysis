"""
src/pipeline.py

End-to-End Hybrid Biomedical Image Analysis Pipeline.

This module integrates:
1. Deep Learning U-Net Segmentation
2. Quantitative Morphometric Feature Extraction (skimage regionprops)
3. Numbers-First LLM Interpretation and Structured JSON Generation
4. Batch execution over test sets and aggregation into pandas DataFrames/CSVs
5. Robustness and Perturbation Evaluation (Blur, Noise, Low Contrast).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.classical_features import extract_features, segment_classical, summarize_features
from src.preprocessing import DEFAULT_IMAGE_SIZE, load_mask, preprocess_image
from src.unet import segment_image
from src.vlm import (
    NUMBERS_FIRST_PROMPT,
    REQUIRED_NUMBERS_FIRST_FIELDS,
    extract_json_from_text,
    query_text_model,
    validate_numbers_first_json,
)


def run_unet_stage(
    model: nn.Module,
    image_path: Union[str, Path],
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """
    Stage 1 & 2: Run U-Net segmentation on an input image and extract quantitative morphometrics.

    Returns:
        (preprocessed_image, binary_mask, prob_map, features_df, summary_dict)
    """
    image_path = Path(image_path)
    image_id = image_path.stem
    image_np = preprocess_image(image_path)

    # U-Net prediction
    binary_mask, prob_map = segment_image(
        model, image_np, threshold=threshold, device=device
    )

    # Label connected components from predicted mask
    from skimage.measure import label
    labeled_mask = label(binary_mask)

    # Extract per-object features
    features_df = extract_features(
        labeled_mask=labeled_mask,
        intensity_image=image_np,
        image_id=image_id,
    )

    # Convert to numeric summary
    summary_dict = summarize_features(
        features_df=features_df,
        image_id=image_id,
        image_shape=image_np.shape,
    )

    return image_np, binary_mask, prob_map, features_df, summary_dict


def apply_quality_gate(
    image_np: np.ndarray,
    summary_dict: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Deterministic pre-LLM quality gate based on physical and statistical properties:
    1. Coverage check: foreground area fraction (detects threshold collapse/saturation)
    2. Contrast check: image dynamic range (detects severely degraded/low-contrast images)
    3. Object count check: zero detections
    """
    contrast = float(image_np.max() - image_np.min())
    coverage = float(summary_dict.get("area_fraction", 0.0))
    n_objects = int(summary_dict.get("n_objects", 0))

    if coverage > 0.75:
        return "reject", "Foreground coverage exceeds 75% (threshold collapse/canvas saturation)."

    if contrast < 0.08:
        return "reject", "Image contrast is below the minimum clinical threshold (0.08)."

    if coverage > 0.50:
        return "warning", "Unusually high foreground coverage (>50%)."

    if n_objects == 0:
        return "warning", "No foreground objects were detected."

    return "pass", "Image passed deterministic quality checks."


def run_hybrid_image_analysis(
    model: nn.Module,
    image_path: Union[str, Path],
    llm_model: str = "llama3.2",
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
    ground_truth_mask_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Full Hybrid Pipeline for a single image:
    Image -> U-Net Mask -> regionprops features -> Deterministic Quality Gate -> LLM Structured Record & Narrative.
    """
    image_path = Path(image_path)
    image_id = image_path.stem

    # Look up ground-truth dataset density regime and instance count if metadata exists
    dataset_density_regime = None
    ground_truth_nuclei_count = None
    possible_meta_paths = []
    if metadata_path:
        possible_meta_paths.append(Path(metadata_path))
    possible_meta_paths.extend([
        image_path.parent.parent / "metadata.csv",
        image_path.parent.parent.parent / "metadata.csv",
        Path("nuclei_dataset/metadata.csv"),
        Path("data/nuclei_dataset/metadata.csv")
    ])
    for mp in possible_meta_paths:
        if mp.exists():
            try:
                m_df = pd.read_csv(mp)
                match = m_df[m_df["image_id"] == image_id]
                if not match.empty:
                    if "density" in match.columns:
                        dataset_density_regime = str(match["density"].iloc[0])
                    if "n_objects" in match.columns:
                        ground_truth_nuclei_count = int(match["n_objects"].iloc[0])
                    break
            except Exception:
                pass

    t_pipeline_start = time.perf_counter()

    # 1. Segmentation & Measurement (with sub-stage latency profiling)
    t_seg_start = time.perf_counter()
    image_np = preprocess_image(image_path)
    binary_mask, prob_map = segment_image(
        model, image_np, threshold=threshold, device=device
    )
    segmentation_time_s = round(time.perf_counter() - t_seg_start, 4)

    t_feat_start = time.perf_counter()
    from skimage.measure import label
    labeled_mask = label(binary_mask)
    features_df = extract_features(
        labeled_mask=labeled_mask,
        intensity_image=image_np,
        image_id=image_id,
    )
    summary_dict = summarize_features(
        features_df=features_df,
        image_id=image_id,
        image_shape=image_np.shape,
    )
    feature_extraction_time_s = round(time.perf_counter() - t_feat_start, 4)

    # 2. Deterministic Quality Gate (BEFORE LLM)
    deterministic_quality_flag, quality_reason = apply_quality_gate(
        image_np=image_np,
        summary_dict=summary_dict,
    )

    # 3. Ground-truth comparison if available (Dice & IoU)
    dice_gt = None
    iou_gt = None
    if ground_truth_mask_path is not None and Path(ground_truth_mask_path).exists():
        gt_mask = load_mask(ground_truth_mask_path)
        intersection = np.logical_and(binary_mask, gt_mask).sum()
        
        dice_union = binary_mask.sum() + gt_mask.sum()
        dice_gt = float((2.0 * intersection + 1e-7) / (dice_union + 1e-7))
        
        iou_union = np.logical_or(binary_mask, gt_mask).sum()
        iou_gt = float((intersection + 1e-7) / (iou_union + 1e-7))

    # 4. LLM Interpretation (Gated: Only executed if quality gate != 'reject')
    raw_text = ""
    parsed_json = {}
    json_valid = False
    llm_n_objects = None
    llm_audit_match = False
    audit_count_match = False
    llm_quality_flag = "uncertain"
    rationale = quality_reason
    llm_time_s = 0.0

    t_llm_start = time.perf_counter()
    if deterministic_quality_flag == "reject":
        raw_text = f"REJECTED PRE-LLM: {quality_reason}"
        parsed_json = {}
        json_valid = False
        llm_n_objects = None
        llm_audit_match = False
        audit_count_match = False
        llm_quality_flag = "rejected_pre_llm"
        rationale = f"Pre-LLM Quality Gate Rejected: {quality_reason}"
        llm_time_s = 0.0
    else:
        prompt = NUMBERS_FIRST_PROMPT.format(feature_summary=summary_dict["numeric_text"])
        llm_response = query_text_model(
            prompt=prompt,
            model=llm_model,
            temperature=0.0,
        )

        raw_text = llm_response.get("content", "")
        parsed_json = extract_json_from_text(raw_text)

        # Genuine schema validation audit: verify output is valid JSON and contains all required fields
        json_valid = not parsed_json.get("parse_error", False) and all(
            k in parsed_json for k in REQUIRED_NUMBERS_FIRST_FIELDS
        )

        # If the initial response failed to parse as valid JSON, retry once with a strict zero-temperature JSON-only prompt
        if not json_valid:
            retry_prompt = f"""You are a biomedical data-analysis assistant.
Based ONLY on these numerical measurements:
{summary_dict['numeric_text']}

Return ONLY one valid JSON object. Do not include any narrative text or markdown code fences before or after the JSON.
Schema:
{{
    "n_objects": <integer count matching the data>,
    "density_class": "sparse | normal | dense | clustered | uncertain",
    "shape_regularity": "regular | moderate | irregular | uncertain",
    "quality_flag": "pass | warning | reject | uncertain",
    "rationale": "<brief explanation based only on the numbers>"
}}"""
            retry_res = query_text_model(prompt=retry_prompt, model=llm_model, temperature=0.0)
            if retry_res.get("success"):
                retry_json = extract_json_from_text(retry_res.get("content", ""))
                if not retry_json.get("parse_error", False) and all(k in retry_json for k in REQUIRED_NUMBERS_FIRST_FIELDS):
                    parsed_json = retry_json
                    json_valid = True

        # Audit check: verify LLM reproduced upstream measured component count
        llm_n_objects = parsed_json.get("n_objects")
        try:
            llm_count = int(llm_n_objects) if llm_n_objects is not None else None
            llm_audit_match = bool(llm_count == summary_dict["n_objects"])
        except (ValueError, TypeError):
            llm_audit_match = False

        llm_quality_flag = parsed_json.get("quality_flag", "uncertain")
        rationale = parsed_json.get("rationale", quality_reason)
        llm_time_s = round(time.perf_counter() - t_llm_start, 4)

    total_pipeline_time_s = round(time.perf_counter() - t_pipeline_start, 4)

    mean_intensity = float(image_np.mean())
    intensity_std = float(image_np.std())
    intensity_range = float(image_np.max() - image_np.min())
    predicted_density = parsed_json.get("density_class", summary_dict["density_class"])

    cc_count = summary_dict["n_objects"]
    if ground_truth_nuclei_count is not None:
        count_error = int(cc_count - ground_truth_nuclei_count)
        absolute_count_error = int(abs(count_error))
    else:
        count_error = None
        absolute_count_error = None

    structured_record = {
        "image_id": image_id,
        "pipeline_version": "1.0",
        "segmentation_model": "SmallUNet",
        "segmentation_checkpoint": "unet_combined_bce___dice.pth",
        "segmentation_threshold": float(threshold),
        "text_model": str(llm_model),
        "dataset_density_regime": dataset_density_regime,
        "predicted_density_class": predicted_density,
        "density_class": predicted_density,
        "mean_intensity": round(mean_intensity, 4),
        "intensity_std": round(intensity_std, 4),
        "intensity_range": round(intensity_range, 4),
        "ground_truth_nuclei_count": ground_truth_nuclei_count,
        "predicted_component_count": cc_count,
        "connected_component_count": cc_count,
        "segmented_object_count": cc_count,
        "measured_n_objects": cc_count,
        "count_error": count_error,
        "absolute_count_error": absolute_count_error,
        "llm_n_objects": llm_n_objects if llm_n_objects is not None else cc_count,
        "llm_measurement_audit_match": llm_audit_match,
        "llm_count_reproduction_match": llm_audit_match,
        "audit_count_match": llm_audit_match,
        "json_valid": json_valid,
        "mean_area": round(summary_dict["mean_area"], 2),
        "total_area": round(summary_dict["total_area"], 1),
        "area_fraction": round(summary_dict["area_fraction"], 4),
        "shape_regularity": parsed_json.get("shape_regularity", summary_dict["shape_regularity"]),
        "quality_gate": deterministic_quality_flag,
        "quality_reason": quality_reason,
        "llm_quality_flag": llm_quality_flag,
        "quality_flag": deterministic_quality_flag,
        "rationale": rationale,
        "summary_narrative": raw_text,
        "dice_vs_gt": round(dice_gt, 4) if dice_gt is not None else None,
        "iou_vs_gt": round(iou_gt, 4) if iou_gt is not None else None,
        "segmentation_time_s": segmentation_time_s,
        "feature_extraction_time_s": feature_extraction_time_s,
        "llm_time_s": llm_time_s,
        "total_pipeline_time_s": total_pipeline_time_s,
    }

    return {
        "image_id": image_id,
        "image": image_np,
        "predicted_mask": binary_mask,
        "prob_map": prob_map,
        "features_df": features_df,
        "summary_dict": summary_dict,
        "structured_record": structured_record,
        "raw_llm_response": raw_text,
    }


def run_batch_hybrid_pipeline(
    model: nn.Module,
    image_dir: Union[str, Path],
    mask_dir: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
    output_csv_path: Optional[Union[str, Path]] = None,
    llm_model: str = "llama3.2",
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Execute hybrid pipeline over an entire directory of images (e.g. unseen test set),
    aggregate structured records into a pandas DataFrame, and save to CSV.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir) if mask_dir else None

    image_paths = sorted(list(image_dir.glob("*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {image_dir}")

    results = []
    records = []

    print(f"Running Hybrid Pipeline across {len(image_paths)} images in {image_dir.name}...", flush=True)
    for idx, img_path in enumerate(image_paths, 1):
        gt_path = (mask_dir / img_path.name) if (mask_dir and (mask_dir / img_path.name).exists()) else None
        res = run_hybrid_image_analysis(
            model=model,
            image_path=img_path,
            llm_model=llm_model,
            threshold=threshold,
            device=device,
            ground_truth_mask_path=gt_path,
            metadata_path=metadata_path,
        )
        results.append(res)
        records.append(res["structured_record"])
        print(
            f"  [{idx:02d}/{len(image_paths):02d}] {img_path.name}: "
            f"Objects={res['structured_record']['measured_n_objects']}, "
            f"Density={res['structured_record']['density_class']}, "
            f"Quality={res['structured_record']['quality_flag']}",
            flush=True,
        )

    summary_df = pd.DataFrame(records)

    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            summary_df.to_csv(output_csv_path, index=False)
            print(f"Saved aggregated test summary to {output_csv_path}", flush=True)
        except PermissionError:
            print(f"Warning: {output_csv_path.name} is currently locked by another application (e.g., Excel).", flush=True)

    return summary_df, results


def run_robustness_experiment(
    model: nn.Module,
    clean_image_path: Union[str, Path, List[Union[str, Path]]],
    corrupted_image_paths: Optional[List[Union[str, Path]]] = None,
    llm_model: str = "llama3.2",
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Extra Credit: Trace how image perturbations (blur, low contrast, noise)
    propagate through each pipeline stage: Mask -> Feature Table -> Narrative -> Quality Flag.
    Evaluates both sparse (e.g. test_000) and clustered (e.g. test_004) image variants.
    """
    if isinstance(clean_image_path, (list, tuple)):
        clean_paths = [Path(p) for p in clean_image_path]
    else:
        clean_paths = [Path(clean_image_path)]

    corrupt_paths = [Path(p) for p in corrupted_image_paths] if corrupted_image_paths else []
    all_paths = clean_paths + corrupt_paths
    rows = []

    for path in all_paths:
        stem = path.stem
        if "_" in stem and stem.split("_")[-1] in ("blur", "lowcontrast", "noise", "corrupted"):
            condition = stem.split("_")[-1]
        else:
            condition = "clean"

        res = run_hybrid_image_analysis(
            model=model,
            image_path=path,
            llm_model=llm_model,
            device=device,
        )
        rec = res["structured_record"]
        rows.append(
            {
                "image_id": path.stem,
                "condition": condition,
                "mean_intensity": rec["mean_intensity"],
                "intensity_std": rec["intensity_std"],
                "intensity_range": rec["intensity_range"],
                "connected_component_count": rec["connected_component_count"],
                "segmented_object_count": rec["segmented_object_count"],
                "measured_n_objects": rec["measured_n_objects"],
                "total_area": rec["total_area"],
                "mean_area": rec["mean_area"],
                "area_fraction": rec["area_fraction"],
                "density_class": rec["density_class"],
                "shape_regularity": rec["shape_regularity"],
                "quality_gate": rec["quality_gate"],
                "quality_reason": rec["quality_reason"],
                "llm_quality_flag": rec["llm_quality_flag"],
                "quality_flag": rec["quality_gate"],
                "rationale": rec["rationale"],
            }
        )

    return pd.DataFrame(rows)
