"""
run_all_experiments.py

Comprehensive experimental pipeline for the Biomedical Image Analysis Assignment.
Executes Tasks 1 to 4 and the Extra Credit extensions, generating all figures,
CSVs, JSON records, and model checkpoints.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project roots to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classical_features import (
    apply_otsu_threshold,
    extract_features,
    morphological_cleanup,
    segment_classical,
    summarize_features,
)
from src.pipeline import (
    run_batch_hybrid_pipeline,
    run_hybrid_image_analysis,
    run_robustness_experiment,
    run_unet_stage,
)
from src.preprocessing import DEFAULT_IMAGE_SIZE, load_image, load_mask, preprocess_image
from src.unet import (
    NucleiDataset,
    SmallUNet,
    bce_loss,
    combined_loss,
    dice_coefficient,
    evaluate,
    iou_score,
    segment_image,
    soft_dice_loss,
    train_unet,
)
from src.vlm import (
    NAIVE_PROMPT,
    NUMBERS_FIRST_PROMPT,
    STRUCTURED_VLM_PROMPT,
    evaluate_vlm_stochasticity,
    extract_json_from_text,
    query_llm_numbers_first,
    run_naive_vlm,
    run_structured_vlm,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dirs(base_dirs: List[Path]) -> None:
    for base in base_dirs:
        for sub in ["figures", "csv", "json", "models"]:
            (base / sub).mkdir(parents=True, exist_ok=True)


def copy_outputs_to_mirrors(src_dir: Path, dest_dir: Path) -> None:
    """Mirror generated files between project directories."""
    import shutil
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["figures", "csv", "json", "models"]:
        s_sub = src_dir / sub
        d_sub = dest_dir / sub
        if s_sub.exists():
            d_sub.mkdir(parents=True, exist_ok=True)
            for item in s_sub.glob("*.*"):
                shutil.copy2(item, d_sub / item.name)


# -----------------------------------------------------------------------------
# TASK 1: Data Preparation, EDA & Multimodal VLM Prompt Engineering
# -----------------------------------------------------------------------------

def run_task_1(data_dir: Path, output_dir: Path) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("TASK 1: Data Preparation, EDA & Multimodal VLM Exploration")
    print("=" * 70)

    train_img_dir = data_dir / "train" / "images"
    train_mask_dir = data_dir / "train" / "masks"
    metadata_path = data_dir / "metadata.csv"

    metadata_df = pd.read_csv(metadata_path)
    print(f"Loaded metadata.csv with {len(metadata_df)} total records across splits:")
    print(metadata_df["split"].value_counts().to_string())

    # --- 1.1 Multi-Regime Visual Samples ---
    density_classes = ["sparse", "normal", "dense", "clustered"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    for ax, density in zip(axes, density_classes):
        sample_meta = metadata_df[
            (metadata_df["split"] == "train") & (metadata_df["density"] == density)
        ].iloc[0]
        img_id = sample_meta["image_id"]
        img_path = train_img_dir / f"{img_id}.png"
        rgb_img = load_image(img_path)

        ax.imshow(rgb_img)
        ax.set_title(
            f"{density.capitalize()} Regime\n({sample_meta['n_objects']} nuclei, {sample_meta['area_fraction']*100:.1f}% area)",
            fontsize=11,
            fontweight="bold",
        )
        ax.axis("off")

    plt.tight_layout()
    fig_path = output_dir / "figures" / "eda_density_samples.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved EDA density sample grid to {fig_path}")

    # --- 1.2 Intensity & Count Distributions ---
    all_train_pixels = []
    for p in sorted(train_img_dir.glob("*.png")):
        all_train_pixels.append(preprocess_image(p).ravel())
    all_train_pixels = np.concatenate(all_train_pixels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # Pixel Intensity Histogram
    axes[0].hist(all_train_pixels, bins=60, color="#1f77b4", edgecolor="black", alpha=0.8, density=True)
    axes[0].set_title("Training Set Pixel Intensity Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Normalized Pixel Intensity [0, 1]")
    axes[0].set_ylabel("Probability Density")
    axes[0].grid(True, alpha=0.3)

    # Object Count Distribution by Density Regime
    for density in density_classes:
        sub = metadata_df[metadata_df["density"] == density]["n_objects"]
        axes[1].hist(sub, bins=12, alpha=0.6, label=density.capitalize(), edgecolor="black")
    axes[1].set_title("Nucleus Count Distribution Across Regimes", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Nuclei per Image")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    hist_fig_path = output_dir / "figures" / "eda_intensity_histogram.png"
    plt.savefig(hist_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved EDA intensity and count histograms to {hist_fig_path}")

    # --- 1.3 VLM Prompt Engineering Experiments (Task 1: llama3.2-vision) ---
    rep_image_path = train_img_dir / "train_000.png"
    vlm_model_name = "llama3.2-vision"
    print(f"\nEvaluating Vision-Language Model ({vlm_model_name}) on representative image: {rep_image_path.name}")

    # 1. Naive Prompt
    print(f"Running Naive VLM Prompt with {vlm_model_name} (temperature=0.7)...")
    naive_res = run_naive_vlm(rep_image_path, model=vlm_model_name, temperature=0.7)
    print("Naive Prompt Output:")
    print(naive_res.get("raw_response", ""))

    # 2. Structured Prompt
    print(f"\nRunning Structured VLM Prompt with {vlm_model_name} (temperature=0.0)...")
    structured_res = run_structured_vlm(rep_image_path, model=vlm_model_name, temperature=0.0)
    print("Structured VLM JSON Output:")
    print(json.dumps(structured_res.get("json", {}), indent=2))

    # 3. Repeated Runs (Stochasticity vs Determinism)
    print(f"\nEvaluating Stochasticity: 3 repeated runs with {vlm_model_name} at temperature=0.7...")
    stochastic_runs = evaluate_vlm_stochasticity(rep_image_path, prompt=NAIVE_PROMPT, model=vlm_model_name, n_runs=3, temperature=0.7)

    print(f"Evaluating Determinism: 2 repeated runs with {vlm_model_name} at temperature=0.0...")
    deterministic_runs = evaluate_vlm_stochasticity(rep_image_path, prompt=STRUCTURED_VLM_PROMPT, model=vlm_model_name, n_runs=2, temperature=0.0)

    actual_model = naive_res.get("model", "moondream")
    fallback_used = naive_res.get("fallback_used", False)
    fallback_reason = naive_res.get("fallback_reason", "None")

    task1_data = {
        "model": actual_model,
        "requested_model": vlm_model_name,
        "actual_model": actual_model,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "json_valid": structured_res.get("json_valid", True),
        "schema_complete": structured_res.get("schema_complete", True),
        "task": "Task 1: Multimodal LLM Description & Prompt Optimization",
        "naive_prompt": NAIVE_PROMPT,
        "naive_response": naive_res.get("raw_response"),
        "structured_prompt": STRUCTURED_VLM_PROMPT,
        "structured_response": structured_res.get("json"),
        "stochastic_runs": [r["response"] for r in stochastic_runs],
        "deterministic_runs": [r["response"] for r in deterministic_runs],
    }

    json_path = output_dir / "json" / "task1_vlm_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(task1_data, f, indent=2)
    print(f"Saved Task 1 VLM results (Executed: {actual_model} | Requested: {vlm_model_name}) to {json_path}")

    return task1_data


# -----------------------------------------------------------------------------
# TASK 2: Classical Image Processing & Numbers-First LLM Interpretation
# -----------------------------------------------------------------------------

def run_task_2(data_dir: Path, output_dir: Path) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("TASK 2: Classical Features & Numbers-First LLM Interpretation")
    print("=" * 70)

    sample_img_path = data_dir / "train" / "images" / "train_000.png"
    sample_gt_mask_path = data_dir / "train" / "masks" / "train_000.png"

    img_np = preprocess_image(sample_img_path)
    gt_mask = load_mask(sample_gt_mask_path)

    # Classical Otsu + Morphology
    raw_otsu, thresh = apply_otsu_threshold(img_np)
    cleaned_mask, labeled_mask, _ = segment_classical(img_np)

    # Feature extraction via regionprops_table
    features_df = extract_features(labeled_mask, intensity_image=img_np, image_id="train_000")
    csv_path = output_dir / "csv" / "classical_features_sample.csv"
    features_df.to_csv(csv_path, index=False)
    print(f"Extracted {len(features_df)} objects with regionprops. Saved table to {csv_path}")

    # Visual Multi-Panel Comparison
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Grayscale Input (256x256)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(raw_otsu, cmap="Blues")
    axes[1].set_title(f"Raw Otsu Threshold\n(T = {thresh:.3f})", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(cleaned_mask, cmap="Blues")
    axes[2].set_title("Morphological Cleanup\n(Hole filling + Speck removal)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # Labeled instances with color map
    axes[3].imshow(labeled_mask, cmap="nipy_spectral")
    axes[3].set_title(f"Connected Components\n({labeled_mask.max()} objects identified)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    fig_path = output_dir / "figures" / "classical_segmentation_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved classical segmentation panel to {fig_path}")

    # Generate Numeric Summary
    summary_dict = summarize_features(features_df, image_id="train_000")
    print(f"\nGenerated Numeric Summary string:\n{summary_dict['numeric_text']}")

    # Query local LLM (Numbers-First)
    print("\nQuerying text-only LLM (llama3.2) with numbers-first prompt...")
    llm_res = query_llm_numbers_first(summary_dict["numeric_text"], model="llama3.2")
    print("Numbers-First LLM Output:")
    print(llm_res.get("raw_response", ""))

    task2_data = {
        "numeric_summary": summary_dict,
        "llm_response": llm_res.get("raw_response"),
        "parsed_json": llm_res.get("json"),
    }

    json_path = output_dir / "json" / "task2_numbers_first_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(task2_data, f, indent=2)
    print(f"Saved Task 2 results to {json_path}")

    return task2_data


# -----------------------------------------------------------------------------
# TASK 3: PyTorch U-Net Segmentation & Loss Ablations
# -----------------------------------------------------------------------------

def run_task_3(data_dir: Path, output_dir: Path, n_epochs: int = 15) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("TASK 3: PyTorch U-Net Segmentation & Loss Function Ablation")
    print("=" * 70)

    train_img_dir = data_dir / "train" / "images"
    train_mask_dir = data_dir / "train" / "masks"
    val_img_dir = data_dir / "val" / "images"
    val_mask_dir = data_dir / "val" / "masks"

    train_ds = NucleiDataset(train_img_dir, train_mask_dir, augment=True, seed=42)
    val_ds = NucleiDataset(val_img_dir, val_mask_dir, augment=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Loss Ablation setup: 3 models trained with pinned seed
    loss_experiments = {
        "Combined (BCE + Dice)": combined_loss,
        "BCE Only": bce_loss,
        "Soft Dice Only": soft_dice_loss,
    }

    histories = {}
    best_models = {}

    for exp_name, loss_fn in loss_experiments.items():
        print(f"\n--- Training U-Net with Loss: {exp_name} ({n_epochs} epochs) ---")
        set_seed(42)  # Pin seed for exact ablation comparability

        model = SmallUNet(in_channels=1, out_channels=1, features=(16, 32, 64, 128)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        ckpt_name = f"unet_{exp_name.lower().replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')}.pth"
        ckpt_path = output_dir / "models" / ckpt_name

        history = train_unet(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            n_epochs=n_epochs,
            device=device,
            checkpoint_path=ckpt_path,
            verbose=True,
        )

        histories[exp_name] = history
        best_models[exp_name] = model

    # Save training histories table
    ablation_rows = []
    for epoch_idx in range(n_epochs):
        row = {"epoch": epoch_idx + 1}
        for exp_name, hist in histories.items():
            slug = exp_name.lower().replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')
            row[f"{slug}_train_loss"] = hist["train_loss"][epoch_idx]
            row[f"{slug}_val_loss"] = hist["val_loss"][epoch_idx]
            row[f"{slug}_val_dice"] = hist["val_dice"][epoch_idx]
            row[f"{slug}_val_iou"] = hist["val_iou"][epoch_idx]
        ablation_rows.append(row)

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_csv_path = output_dir / "csv" / "unet_loss_ablation_history.csv"
    ablation_df.to_csv(ablation_csv_path, index=False)
    print(f"\nSaved full loss ablation history to {ablation_csv_path}")

    # Generate loss ablation summary table (Final & Peak Metrics)
    summary_rows = []
    for exp_name, hist in histories.items():
        val_dice_list = hist["val_dice"]
        peak_idx = int(np.argmax(val_dice_list))
        summary_rows.append({
            "loss_function": exp_name,
            "final_train_loss": round(float(hist["train_loss"][-1]), 4),
            "final_val_loss": round(float(hist["val_loss"][-1]), 4),
            "final_val_dice": round(float(hist["val_dice"][-1]), 4),
            "final_val_iou": round(float(hist["val_iou"][-1]), 4),
            "peak_val_dice": round(float(hist["val_dice"][peak_idx]), 4),
            "peak_val_iou": round(float(hist["val_iou"][peak_idx]), 4),
            "peak_epoch": peak_idx + 1,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "csv" / "unet_loss_ablation_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved loss ablation summary to {summary_csv_path}")

    # Primary Model Learning Curves Plot (Combined Loss)
    comb_hist = histories["Combined (BCE + Dice)"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    epochs_range = range(1, n_epochs + 1)
    axes[0].plot(epochs_range, comb_hist["train_loss"], label="Train Loss", marker="o", color="#1f77b4")
    axes[0].plot(epochs_range, comb_hist["val_loss"], label="Val Loss", marker="s", color="#ff7f0e")
    axes[0].set_title("U-Net Loss Curves (BCE + Dice)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, comb_hist["val_dice"], label="Validation Dice", marker="o", color="#2ca02c")
    axes[1].plot(epochs_range, comb_hist["val_iou"], label="Validation IoU", marker="^", color="#9467bd")
    axes[1].set_title("Validation Segmentation Metrics", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score [0, 1]")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = output_dir / "figures" / "unet_learning_curves.png"
    plt.savefig(curves_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved primary learning curves to {curves_path}")

    # Loss Ablation Comparison Plot (Validation Dice across losses)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Combined (BCE + Dice)": "#2ca02c", "BCE Only": "#1f77b4", "Soft Dice Only": "#d62728"}
    markers = {"Combined (BCE + Dice)": "o", "BCE Only": "s", "Soft Dice Only": "^"}

    for exp_name, hist in histories.items():
        ax.plot(
            epochs_range,
            hist["val_dice"],
            label=f"{exp_name} (Peak: {max(hist['val_dice']):.4f})",
            color=colors[exp_name],
            marker=markers[exp_name],
            linewidth=2,
        )

    ax.set_title("Validation Dice Trajectory Across Loss Functions", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Dice Score")
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ablation_curves_path = output_dir / "figures" / "unet_loss_ablation_curves.png"
    plt.savefig(ablation_curves_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved loss ablation curves to {ablation_curves_path}")

    # --- 3.3 Visual Multi-Panel Validation Triplets ---
    # Pick 4 validation images spanning different regimes
    val_metadata = pd.read_csv(data_dir / "metadata.csv")
    val_meta = val_metadata[val_metadata["split"] == "val"]

    selected_val_ids = []
    for density in ["sparse", "normal", "dense", "clustered"]:
        match = val_meta[val_meta["density"] == density]
        if not match.empty:
            selected_val_ids.append(match.iloc[0]["image_id"])

    best_model = best_models["Combined (BCE + Dice)"]
    best_model.eval()

    fig, axes = plt.subplots(len(selected_val_ids), 4, figsize=(16, 4 * len(selected_val_ids)))

    val_metrics_table = []

    for row_idx, val_id in enumerate(selected_val_ids):
        img_p = val_img_dir / f"{val_id}.png"
        mask_p = val_mask_dir / f"{val_id}.png"

        img_np = preprocess_image(img_p)
        gt_mask = load_mask(mask_p)

        pred_mask, prob_map = segment_image(best_model, img_np, device=device)

        # Compute metrics
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum()
        dice = float((2.0 * intersection + 1e-7) / (union + 1e-7))
        iou = float((intersection + 1e-7) / (union - intersection + 1e-7))

        val_metrics_table.append({
            "image_id": val_id,
            "dice": dice,
            "iou": iou,
        })

        # Columns: [Original Image, Ground Truth, Predicted Probability, Binary Mask Prediction]
        axes[row_idx, 0].imshow(img_np, cmap="gray")
        axes[row_idx, 0].set_title(f"Input: {val_id}", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(gt_mask, cmap="Blues")
        axes[row_idx, 1].set_title(f"Ground Truth Mask\n({gt_mask.sum()} px)", fontsize=11, fontweight="bold")
        axes[row_idx, 1].axis("off")

        im_prob = axes[row_idx, 2].imshow(prob_map, cmap="magma", vmin=0.0, vmax=1.0)
        axes[row_idx, 2].set_title("Probability Heatmap", fontsize=11, fontweight="bold")
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(pred_mask, cmap="Blues")
        axes[row_idx, 3].set_title(f"U-Net Prediction\n(Dice={dice:.4f}, IoU={iou:.4f})", fontsize=11, fontweight="bold")
        axes[row_idx, 3].axis("off")

    plt.tight_layout()
    triplet_fig_path = output_dir / "figures" / "unet_val_predictions.png"
    plt.savefig(triplet_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved validation prediction panels to {triplet_fig_path}")

    return {
        "histories": histories,
        "best_model": best_model,
        "device": device,
    }


# -----------------------------------------------------------------------------
# TASK 4: End-to-End Hybrid Pipeline on Unseen Test Images
# -----------------------------------------------------------------------------

def run_task_4(
    model: nn.Module,
    data_dir: Path,
    output_dir: Path,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("TASK 4: End-to-End Hybrid Pipeline on Unseen Test Images")
    print("=" * 70)

    test_img_dir = data_dir / "test" / "images"
    test_mask_dir = data_dir / "test" / "masks"

    out_csv_path = output_dir / "csv" / "hybrid_pipeline_test_summary.csv"
    summary_df, results = run_batch_hybrid_pipeline(
        model=model,
        image_dir=test_img_dir,
        mask_dir=test_mask_dir,
        output_csv_path=out_csv_path,
        llm_model="llama3.2",
        device=device,
    )

    # Save detailed JSON records
    records_clean = []
    for r in results:
        records_clean.append({
            "image_id": r["image_id"],
            "structured_record": r["structured_record"],
            "numeric_summary": r["summary_dict"],
            "raw_narrative": r["raw_llm_response"],
        })

    json_path = output_dir / "json" / "hybrid_pipeline_test_records.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records_clean, f, indent=2)
    print(f"Saved full hybrid pipeline JSON records to {json_path}")

    # Generate Otsu vs U-Net Visual Comparison Panel (Sparse vs Clustered)
    cases = [
        ("Sparse Isolated Example (test_000)", data_dir / "test" / "images" / "test_000.png", data_dir / "test" / "masks" / "test_000.png"),
        ("Dense Clustered Example (test_004)", data_dir / "test" / "images" / "test_004.png", data_dir / "test" / "masks" / "test_004.png"),
    ]
    fig, axes = plt.subplots(len(cases), 4, figsize=(15, 7.5))
    for row_idx, (title, img_p, mask_p) in enumerate(cases):
        img_np = preprocess_image(img_p)
        gt_mask = load_mask(mask_p)

        # Classical Otsu
        otsu_mask, otsu_labeled, thresh = segment_classical(img_np)
        otsu_inter = np.logical_and(otsu_mask, gt_mask).sum()
        otsu_union = otsu_mask.sum() + gt_mask.sum()
        otsu_dice = (2.0 * otsu_inter + 1e-7) / (otsu_union + 1e-7)

        # Deep Learning U-Net
        _, pred_mask, _, _, summary_dict = run_unet_stage(model, img_p, device=device)
        unet_inter = np.logical_and(pred_mask, gt_mask).sum()
        unet_union = pred_mask.sum() + gt_mask.sum()
        unet_dice = (2.0 * unet_inter + 1e-7) / (unet_union + 1e-7)

        axes[row_idx, 0].imshow(img_np, cmap="gray")
        axes[row_idx, 0].set_title(f"{title}\nGrayscale Input (256x256)", fontsize=10.5, fontweight="bold", pad=6)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(gt_mask, cmap="Blues")
        axes[row_idx, 1].set_title(f"Ground Truth Mask\n({gt_mask.sum()} fg pixels)", fontsize=10.5, fontweight="bold", pad=6)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(otsu_mask, cmap="Blues")
        axes[row_idx, 2].set_title(f"Classical Otsu (T = {thresh:.3f})\nDice: {otsu_dice:.4f} | Detected: {otsu_labeled.max()}", fontsize=10.5, fontweight="bold", pad=6)
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(pred_mask, cmap="Blues")
        axes[row_idx, 3].set_title(f"Deep Learning U-Net\nDice: {unet_dice:.4f} | Detected: {summary_dict['n_objects']}", fontsize=10.5, fontweight="bold", pad=6)
        axes[row_idx, 3].axis("off")

    plt.subplots_adjust(wspace=0.08, hspace=0.28, left=0.02, right=0.98, top=0.92, bottom=0.04)
    comp_fig_p = output_dir / "figures" / "otsu_vs_unet_examples.png"
    plt.savefig(comp_fig_p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Otsu vs U-Net visual comparison panel to {comp_fig_p}")

    # Generate full Otsu vs U-Net test set quantitative comparison table
    comp_rows = []
    for r in results:
        img_id = r["image_id"]
        gt_count = r["structured_record"]["ground_truth_nuclei_count"]
        unet_dice = r["structured_record"]["dice_vs_gt"]
        unet_iou = r["structured_record"]["iou_vs_gt"]
        unet_count = r["structured_record"]["predicted_component_count"]
        density = r["structured_record"]["dataset_density_regime"]

        img_p = test_images[0].parent / f"{img_id}.png"
        mask_p = test_masks[0].parent / f"{img_id}.png"
        img_np = preprocess_image(img_p)
        gt_mask = load_mask(mask_p)
        otsu_mask, otsu_labeled, _ = segment_classical(img_np)
        otsu_inter = np.logical_and(otsu_mask, gt_mask).sum()
        otsu_dice = float((2.0 * otsu_inter + 1e-7) / (otsu_mask.sum() + gt_mask.sum() + 1e-7))
        otsu_iou = float((otsu_inter + 1e-7) / (np.logical_or(otsu_mask, gt_mask).sum() + 1e-7))
        otsu_count = int(otsu_labeled.max())

        comp_rows.append({
            "image_id": img_id,
            "density_regime": density,
            "ground_truth_count": gt_count,
            "otsu_dice": round(otsu_dice, 4),
            "otsu_iou": round(otsu_iou, 4),
            "otsu_count": otsu_count,
            "otsu_count_error": otsu_count - gt_count if gt_count is not None else None,
            "unet_dice": unet_dice,
            "unet_iou": unet_iou,
            "unet_count": unet_count,
            "unet_count_error": unet_count - gt_count if gt_count is not None else None,
            "dice_improvement": round(unet_dice - otsu_dice, 4) if unet_dice is not None else None,
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_csv_p = output_dir / "csv" / "otsu_vs_unet_test_comparison.csv"
    comp_df.to_csv(comp_csv_p, index=False)
    print(f"Saved Otsu vs U-Net full test comparison CSV to {comp_csv_p}")

    return summary_df


# -----------------------------------------------------------------------------
# Robustness & Error Propagation Analysis
# -----------------------------------------------------------------------------

def run_task_extra_credit(
    model: nn.Module,
    data_dir: Path,
    output_dir: Path,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("EXTRA CREDIT EXTENSION: Robustness & Corruption Propagation")
    print("=" * 70)

    clean_img_p = data_dir / "test" / "images" / "test_000.png"
    blur_img_p = data_dir / "test_corrupted" / "images" / "test_000_blur.png"
    lowcontrast_img_p = data_dir / "test_corrupted" / "images" / "test_000_lowcontrast.png"

    # Also evaluate test_004 if available
    clean_004_p = data_dir / "test" / "images" / "test_004.png"
    blur_004_p = data_dir / "test_corrupted" / "images" / "test_004_blur.png"
    lowcontrast_004_p = data_dir / "test_corrupted" / "images" / "test_004_lowcontrast.png"

    cases = [
        ("Clean Image (test_000)", clean_img_p),
        ("Heavy Blur (test_000_blur)", blur_img_p),
        ("Low Contrast (test_000_lowcontrast)", lowcontrast_img_p),
        ("Clean Image (test_004)", clean_004_p),
        ("Heavy Blur (test_004_blur)", blur_004_p),
        ("Low Contrast (test_004_lowcontrast)", lowcontrast_004_p),
    ]

    robustness_records = []
    fig, axes = plt.subplots(len(cases), 3, figsize=(14, 3.8 * len(cases)))

    for idx, (label, path) in enumerate(cases):
        if not path.exists():
            continue

        res = run_hybrid_image_analysis(
            model=model,
            image_path=path,
            llm_model="llama3.2",
            device=device,
        )

        rec = res["structured_record"]
        robustness_records.append(rec)

        # Plot Input, U-Net Mask, and Heatmap
        axes[idx, 0].imshow(res["image"], cmap="gray")
        axes[idx, 0].set_title(f"Input: {label}", fontsize=11, fontweight="bold")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(res["predicted_mask"], cmap="Blues")
        axes[idx, 1].set_title(f"U-Net Mask ({rec['measured_n_objects']} objects)", fontsize=11, fontweight="bold")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(res["prob_map"], cmap="magma", vmin=0, vmax=1)
        axes[idx, 2].set_title(f"Quality Flag: {rec['quality_flag'].upper()}\nDensity: {rec['density_class']}", fontsize=11, fontweight="bold")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    robust_fig_p = output_dir / "figures" / "robustness_analysis.png"
    plt.savefig(robust_fig_p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved robustness visual analysis to {robust_fig_p}")

    robust_df = pd.DataFrame(robustness_records)
    robust_csv_p = output_dir / "csv" / "robustness_experiment_results.csv"
    robust_df.to_csv(robust_csv_p, index=False)
    print(f"Saved robustness quantitative table to {robust_csv_p}")

    return robust_df


# -----------------------------------------------------------------------------
# Main Execution Entrypoint
# -----------------------------------------------------------------------------

def main():
    set_seed(42)

    # Resolve paths
    data_dir = PROJECT_ROOT / "nuclei_dataset"
    if not data_dir.exists():
        data_dir = PROJECT_ROOT.parent / "Biomedical_Image_Analysis_Assignment" / "data" / "nuclei_dataset"

    output_dir = PROJECT_ROOT / "outputs"
    mirror_output_dir = PROJECT_ROOT.parent / "Biomedical_Image_Analysis_Assignment" / "outputs"

    ensure_output_dirs([output_dir, mirror_output_dir])

    print(f"Dataset root : {data_dir}")
    print(f"Output root  : {output_dir}")

    # Task 1: EDA & Multimodal VLM
    t1_res = run_task_1(data_dir=data_dir, output_dir=output_dir)

    # Task 2: Classical Features & Numbers-First LLM
    t2_res = run_task_2(data_dir=data_dir, output_dir=output_dir)

    # Task 3: U-Net Training & Loss Ablations
    t3_res = run_task_3(data_dir=data_dir, output_dir=output_dir, n_epochs=15)
    best_model = t3_res["best_model"]
    device = t3_res["device"]

    # Task 4: Hybrid Pipeline on Test Set
    t4_res = run_task_4(model=best_model, data_dir=data_dir, output_dir=output_dir, device=device)

    # Extra Credit: Robustness Experiment
    ec_res = run_task_extra_credit(model=best_model, data_dir=data_dir, output_dir=output_dir, device=device)

    # Mirror outputs to Biomedical_Image_Analysis_Assignment folder
    copy_outputs_to_mirrors(output_dir, mirror_output_dir)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
