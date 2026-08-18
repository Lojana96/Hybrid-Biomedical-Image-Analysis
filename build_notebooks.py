"""
build_notebooks.py

Generates publication-quality, crash-resistant Jupyter notebooks with OpenMP protection:
1. 01_data_preparation_eda.ipynb
2. 02_multimodal_vlm.ipynb (and multimidel_vlm.ipynb)
3. 03_classical_features.ipynb (and classical_features.ipynb)
4. 04_unet_training.ipynb (and unet_training.ipynb)
5. 05_hybrid_pipeline.ipynb (and hybrid_pipeline.ipynb)
6. 06_robustness_experiment.ipynb (and robustness_experiment.ipynb)
"""

import json
from pathlib import Path

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def build_all(nb_dir: Path):
    nb_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 01_data_preparation_eda.ipynb
    # -------------------------------------------------------------
    eda_cells = [
        md_cell("# Stage 01: Data Preparation and Exploratory Data Analysis (EDA)\n\n## Objective\nThis notebook loads, preprocesses, and explores the synthetic stained-nuclei fluorescence microscopy dataset.\nIt validates file integrity across all splits (`train`, `val`, `test`), visualizes representative samples from each density regime (`sparse`, `normal`, `dense`, `clustered`), and computes pixel intensity and object count distributions."),
        code_cell("""# 1. Imports and Environment Setup
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image"""),
        code_cell("""# 2. Dynamic Project Path Resolution
CURRENT_DIR = Path.cwd()
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name.lower() == "notebooks" else CURRENT_DIR

# Dataset path resolution
DATA_DIR = PROJECT_ROOT / "data" / "nuclei_dataset"
if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / "nuclei_dataset"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

TRAIN_IMAGES_DIR = TRAIN_DIR / "images"
TRAIN_MASKS_DIR = TRAIN_DIR / "masks"
VAL_IMAGES_DIR = VAL_DIR / "images"
VAL_MASKS_DIR = VAL_DIR / "masks"
TEST_IMAGES_DIR = TEST_DIR / "images"
TEST_MASKS_DIR = TEST_DIR / "masks"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("Project root :", PROJECT_ROOT)
print("Dataset path :", DATA_DIR)
print("Dataset exists:", DATA_DIR.exists())"""),
        code_cell("""# 3. Add project root to Python path and import preprocessing module
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    load_image,
    preprocess_image,
    load_mask,
    validate_dataset_images,
    validate_image_mask_pairs,
)

print("Preprocessing module imported successfully.")"""),
        code_cell("""# 4. Validate Directory Structure and Dataset Counts
required_directories = {
    "Train images": TRAIN_IMAGES_DIR,
    "Train masks": TRAIN_MASKS_DIR,
    "Validation images": VAL_IMAGES_DIR,
    "Validation masks": VAL_MASKS_DIR,
    "Test images": TEST_IMAGES_DIR,
    "Test masks": TEST_MASKS_DIR,
}

for name, path in required_directories.items():
    print(f"{name:<20}: {'FOUND' if path.exists() else 'MISSING'} ({len(list(path.glob('*.png')))} images)")"""),
        code_cell("""# 5. Metadata Inspection
metadata_path = DATA_DIR / "metadata.csv"
if not metadata_path.exists():
    raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

metadata = pd.read_csv(metadata_path)
print("Metadata shape:", metadata.shape)
display(metadata.head(10))"""),
        code_cell("""# 6. Dataset Summary by Data Split
split_summary = (
    metadata.groupby("split")
    .agg(
        n_images=("image_id", "count"),
        min_objects=("n_objects", "min"),
        mean_objects=("n_objects", "mean"),
        max_objects=("n_objects", "max"),
        mean_intensity=("mean_intensity", "mean"),
        mean_area_fraction=("area_fraction", "mean"),
    )
    .reset_index()
)
print("=== Dataset Summary by Data Split ===")
display(split_summary)

# 7. Dataset Summary by Density Regime
rows = []
for d in ["sparse", "normal", "dense", "clustered"]:
    sub = metadata[metadata["density"] == d]
    c_m, c_s = sub["n_objects"].mean(), sub["n_objects"].std()
    a_m, a_s = sub["area_fraction"].mean() * 100, sub["area_fraction"].std() * 100
    rows.append({
        "Density": d.capitalize(),
        "Images": len(sub),
        "Mean nuclei count": f"{c_m:.2f} ± {c_s:.2f}",
        "Mean area coverage": f"{a_m:.2f}% ± {a_s:.2f}%"
    })

total_c_m, total_c_s = metadata["n_objects"].mean(), metadata["n_objects"].std()
total_a_m, total_a_s = metadata["area_fraction"].mean() * 100, metadata["area_fraction"].std() * 100
rows.append({
    "Density": "Overall",
    "Images": len(metadata),
    "Mean nuclei count": f"{total_c_m:.2f} ± {total_c_s:.2f}",
    "Mean area coverage": f"{total_a_m:.2f}% ± {total_a_s:.2f}%"
})

density_summary_df = pd.DataFrame(rows)
print()
print("=== Dataset Summary by Density Regime ===")
display(density_summary_df)

# 8. Split vs. Density Regime Cross-Tabulation
cross_tab = pd.crosstab(metadata["split"], metadata["density"], margins=True)
ordered_cols = [c for c in ["sparse", "normal", "dense", "clustered", "All"] if c in cross_tab.columns]
ordered_rows = [r for r in ["train", "val", "test", "All"] if r in cross_tab.index]
print()
print("=== Split vs. Density Regime Image Distribution ===")
display(cross_tab.loc[ordered_rows, ordered_cols])"""),
        code_cell("""# 9. Image Preprocessing Verification
sample_image_path = TRAIN_IMAGES_DIR / "train_000.png"
sample_mask_path = TRAIN_MASKS_DIR / "train_000.png"

processed_image = preprocess_image(sample_image_path)
processed_mask = load_mask(sample_mask_path)

print("Processed image shape:", processed_image.shape, "| dtype:", processed_image.dtype, "| Range:", f"[{processed_image.min():.3f}, {processed_image.max():.3f}]")
print("Processed mask shape :", processed_mask.shape, "| dtype:", processed_mask.dtype, "| Values:", np.unique(processed_mask))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(load_image(sample_image_path))
axes[0].set_title("Original RGB Image (256x256)")
axes[0].axis("off")

axes[1].imshow(processed_image, cmap="gray")
axes[1].set_title("Preprocessed Grayscale Normalized")
axes[1].axis("off")
plt.tight_layout()
plt.show()"""),
        code_cell("""# 8. Visual Exploration Across Density Regimes
density_classes = ["sparse", "normal", "dense", "clustered"]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
for ax, density in zip(axes, density_classes):
    sample_meta = metadata[(metadata["split"] == "train") & (metadata["density"] == density)].iloc[0]
    img_id = sample_meta["image_id"]
    img_path = TRAIN_IMAGES_DIR / f"{img_id}.png"
    rgb_img = load_image(img_path)

    ax.imshow(rgb_img)
    ax.set_title(f"{density.capitalize()} Regime\\n({sample_meta['n_objects']} nuclei, {sample_meta['area_fraction']*100:.1f}% area)", fontsize=11, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
plt.savefig(FIGURE_DIR / "eda_density_samples.png", dpi=300, bbox_inches="tight")
plt.show()"""),
        code_cell("""# 9. Pixel Intensity & Nucleus Count Distributions
all_train_pixels = []
for image_path in sorted(TRAIN_IMAGES_DIR.glob("*.png")):
    all_train_pixels.append(preprocess_image(image_path).ravel())
all_train_pixels = np.concatenate(all_train_pixels)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

axes[0].hist(all_train_pixels, bins=60, color="#1f77b4", edgecolor="black", alpha=0.8, density=True)
axes[0].set_title("Training Set Pixel Intensity Distribution", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Normalized Pixel Intensity [0, 1]")
axes[0].set_ylabel("Probability Density")
axes[0].grid(True, alpha=0.3)

for density in density_classes:
    sub = metadata[metadata["density"] == density]["n_objects"]
    axes[1].hist(sub, bins=12, alpha=0.6, label=density.capitalize(), edgecolor="black")
axes[1].set_title("Nucleus Count Distribution Across Regimes", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Number of Nuclei per Image")
axes[1].set_ylabel("Frequency")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "eda_intensity_histogram.png", dpi=300, bbox_inches="tight")
plt.show()"""),
        md_cell("## 10. Summary\nThe synthetic stained-nuclei dataset is validated and standardized to 256x256 grayscale float32 tensors $[0, 1]$. Bimodal intensity characteristics and variable density distributions have been quantified and saved to `outputs/figures/`.")
    ]

    with open(nb_dir / "01_data_preparation_eda.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(eda_cells), f, indent=2)

    # -------------------------------------------------------------
    # 02_multimodal_vlm.ipynb
    # -------------------------------------------------------------
    vlm_cells = [
        md_cell("# Stage 02: Multimodal VLM (llama3.2-vision) Prompt Engineering & Stochasticity\n\n## Overview\nThis notebook evaluates the **`llama3.2-vision`** model on biomedical microscopy images in accordance with **Task 1**.\nWe evaluate an unconstrained naive prompt against an optimized structured descriptive prompt with a strict JSON schema, uncertainty tokens, and temperature stochasticity analysis ($T=0.7$ vs $T=0.0$).\n\n> **Note on Model Execution & Runtime Fallback:**\n> The primary specified model is `llama3.2-vision`. If the local Ollama backend encounters a runtime architecture limitation with `llama3.2-vision` (such as the known Windows Ollama `mllama` backend issue), the pipeline automatically and gracefully reroutes inference to `llava:7b` while preserving strict, transparent data provenance (`model`, `requested_model`, `fallback_used`)."),
        code_cell("""import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
import json

CURRENT_DIR = Path.cwd()
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name.lower() == "notebooks" else CURRENT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vlm import (
    run_naive_vlm,
    run_structured_vlm,
    evaluate_vlm_stochasticity,
    NAIVE_PROMPT,
    STRUCTURED_VLM_PROMPT
)"""),
        md_cell("## 1. Naive Prompting with llama3.2-vision (Baseline)\nIn this test, llama3.2-vision is queried with an open-ended prompt without domain constraints."),
        code_cell("""data_dir = PROJECT_ROOT / "data" / "nuclei_dataset"
if not data_dir.exists():
    data_dir = PROJECT_ROOT / "nuclei_dataset"

sample_img = data_dir / "train" / "images" / "train_000.png"
naive_res = run_naive_vlm(sample_img, model="llama3.2-vision", temperature=0.7)

print("VLM Execution Provenance:")
print(f"  Requested Model : {naive_res.get('requested_model', 'llama3.2-vision')}")
print(f"  Actual Model    : {naive_res.get('actual_model', naive_res.get('model', 'llama3.2-vision'))}")
print(f"  Fallback Used   : {naive_res.get('fallback_used', False)}")
if naive_res.get('fallback_used'):
    print(f"  Fallback Reason : {naive_res.get('fallback_reason')}")

print()
print("Naive Prompt Output:")
print(naive_res.get("raw_response", ""))"""),
        md_cell("## 2. Structured Descriptive Prompting with llama3.2-vision\nHere, llama3.2-vision is anchored as purely descriptive (prohibiting ungrounded diagnosis) and forced into a strict JSON schema permitting 'uncertain'."),
        code_cell("""structured_res = run_structured_vlm(sample_img, model="llama3.2-vision", temperature=0.0)
print(f"Structured VLM Execution Provenance (Actual Model: {structured_res.get('actual_model', structured_res.get('model'))} | Fallback Used: {structured_res.get('fallback_used', False)})")
print(f"Schema Validation Status:")
print(f"  json_valid      : {structured_res.get('json_valid', True)}")
print(f"  schema_complete : {structured_res.get('schema_complete', True)}")
print()
print("Structured VLM JSON Output:")
print(json.dumps(structured_res.get("json", {}), indent=2))"""),
        md_cell("## 3. Stochasticity vs Determinism Analysis\nWe evaluate repeated runs at temperature=0.7 (showing output variability) vs temperature=0.0 (showing strict determinism)."),
        code_cell("""stoch_runs = evaluate_vlm_stochasticity(sample_img, prompt=NAIVE_PROMPT, model="llama3.2-vision", n_runs=3, temperature=0.7)
for r in stoch_runs:
    print(f"--- Run {r['run_index']} (Temp 0.7 | Model: {r.get('model')}) ---")
    print(r['response'][:200], "...\\n")

det_runs = evaluate_vlm_stochasticity(sample_img, prompt=STRUCTURED_VLM_PROMPT, model="llama3.2-vision", n_runs=2, temperature=0.0)
for r in det_runs:
    print(f"--- Run {r['run_index']} (Temp 0.0 | Model: {r.get('model')}) ---")
    print(r['response'][:200], "...\\n")""")
    ]
    with open(nb_dir / "02_multimodal_vlm.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(vlm_cells), f, indent=2)

    # -------------------------------------------------------------
    # 03_classical_features.ipynb
    # -------------------------------------------------------------
    classic_cells = [
        md_cell("# Stage 03: Classical Features & Numbers-First LLM Interpretation\n\n## Overview\nThis notebook demonstrates classical Otsu thresholding, morphological cleanup, connected-component analysis, and regionprops feature extraction, followed by numbers-first LLM reasoning."),
        code_cell("""import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import json

CURRENT_DIR = Path.cwd()
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name.lower() == "notebooks" else CURRENT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_image, load_mask
from src.classical_features import segment_classical, extract_features, summarize_features
from src.vlm import query_llm_numbers_first"""),
        md_cell("## 1. Classical Segmentation Pipeline"),
        code_cell("""data_dir = PROJECT_ROOT / "data" / "nuclei_dataset"
if not data_dir.exists():
    data_dir = PROJECT_ROOT / "nuclei_dataset"

img_p = data_dir / "train" / "images" / "train_000.png"
img_np = preprocess_image(img_p)
cleaned_mask, labeled_mask, thresh = segment_classical(img_np)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(img_np, cmap="gray")
axes[0].set_title("Grayscale Input")
axes[1].imshow(cleaned_mask, cmap="Blues")
axes[1].set_title(f"Cleaned Mask (Otsu T={thresh:.3f})")
axes[2].imshow(labeled_mask, cmap="nipy_spectral")
axes[2].set_title(f"Connected Components ({labeled_mask.max()} objects)")
plt.show()"""),
        md_cell("## 2. Quantitative Morphometrics & Numeric Summarization"),
        code_cell("""features_df = extract_features(labeled_mask, intensity_image=img_np, image_id="train_000")
print("Per-object feature table preview:")
display(features_df.head())

summary_dict = summarize_features(features_df, image_id="train_000")
print()
print("Generated Numeric Summary string:")
print(summary_dict["numeric_text"])"""),
        md_cell("## 3. Numbers-First LLM Reasoning\nPassing quantitative measurements to text-only LLM without images."),
        code_cell("""llm_res = query_llm_numbers_first(summary_dict["numeric_text"], model="llama3.2")
print("LLM Interpretation:")
print(llm_res.get("raw_response", ""))""")
    ]
    with open(nb_dir / "03_classical_features.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(classic_cells), f, indent=2)

    # -------------------------------------------------------------
    # 04_unet_training.ipynb
    # -------------------------------------------------------------
    unet_cells = [
        md_cell("# Stage 04: PyTorch U-Net Segmentation & Loss Function Ablation\n\n## Overview\nThis notebook trains a 4-level U-Net model on the mini-dataset and evaluates performance across loss functions (BCE vs Dice vs Combined BCE+Dice)."),
        code_cell("""import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = Path.cwd()
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name.lower() == "notebooks" else CURRENT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.unet import SmallUNet, NucleiDataset, combined_loss, bce_loss, soft_dice_loss, train_unet"""),
        md_cell("## 1. Dataset & DataLoader Setup"),
        code_cell("""data_dir = PROJECT_ROOT / "data" / "nuclei_dataset"
if not data_dir.exists():
    data_dir = PROJECT_ROOT / "nuclei_dataset"

train_ds = NucleiDataset(data_dir / "train" / "images", data_dir / "train" / "masks", augment=True, seed=42)
val_ds = NucleiDataset(data_dir / "val" / "images", data_dir / "val" / "masks", augment=False)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False)
print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")"""),
        md_cell("## 2. Model Training & Loss Ablations"),
        code_cell("""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallUNet(features=(16, 32, 64, 128)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

history = train_unet(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=combined_loss,
    optimizer=optimizer,
    n_epochs=15,
    device=device,
    verbose=True
)"""),
        md_cell("## 3. Evaluation Curves & Results"),
        code_cell("""plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Loss Trajectory")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["val_dice"], label="Val Dice", color="green")
plt.plot(history["val_iou"], label="Val IoU", color="purple")
plt.title("Validation Segmentation Metrics")
plt.legend()
plt.show()"""),
        md_cell("## 4. Loss Function Ablation Summary Table\nAutomatically computed from training histories without manual hardcoding."),
        code_cell("""ablation_history_path = PROJECT_ROOT / "outputs" / "csv" / "unet_loss_ablation_history.csv"
if ablation_history_path.exists():
    history_df = pd.read_csv(ablation_history_path)
    configs = [
        ("Combined (BCE + Dice)", "combined_bce___dice"),
        ("BCE Only", "bce_only"),
        ("Soft Dice Only", "soft_dice_only")
    ]
    summary_rows = []
    for name, slug in configs:
        sub = pd.DataFrame({
            "epoch": history_df["epoch"],
            "train_loss": history_df[f"{slug}_train_loss"],
            "val_loss": history_df[f"{slug}_val_loss"],
            "val_dice": history_df[f"{slug}_val_dice"],
            "val_iou": history_df[f"{slug}_val_iou"]
        })
        final_row = sub.iloc[-1]
        peak_idx = sub["val_dice"].idxmax()
        peak_row = sub.loc[peak_idx]
        summary_rows.append({
            "loss_function": name,
            "final_train_loss": round(final_row["train_loss"], 4),
            "final_val_loss": round(final_row["val_loss"], 4),
            "final_val_dice": round(final_row["val_dice"], 4),
            "final_val_iou": round(final_row["val_iou"], 4),
            "peak_val_dice": round(peak_row["val_dice"], 4),
            "peak_val_iou": round(peak_row["val_iou"], 4),
            "peak_epoch": int(peak_row["epoch"])
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = PROJECT_ROOT / "outputs" / "csv" / "unet_loss_ablation_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print("Loss Ablation Summary Table (Computed from CSV):")
    display(summary_df)"""),
        md_cell("## 5. Multi-Loss Trajectory Comparison Plot\nVisualizes the epoch-by-epoch validation Dice convergence across loss formulations."),
        code_cell("""# Plot Validation Dice Trajectory Across Loss Functions
plt.figure(figsize=(9.5, 5.5), dpi=120)

epochs = history_df["epoch"]
comb_dice = history_df["combined_bce___dice_val_dice"]
bce_dice = history_df["bce_only_val_dice"]
dice_only = history_df["soft_dice_only_val_dice"]

peak_comb = comb_dice.max()
peak_bce = bce_dice.max()
peak_dice = dice_only.max()

plt.plot(epochs, comb_dice, marker="o", color="#2ca02c", linewidth=2, markersize=6, label=f"Combined (BCE + Dice) (Peak: {peak_comb:.4f})")
plt.plot(epochs, bce_dice, marker="s", color="#1f77b4", linewidth=2, markersize=6, label=f"BCE Only (Peak: {peak_bce:.4f})")
plt.plot(epochs, dice_only, marker="^", color="#d62728", linewidth=2, markersize=6, label=f"Soft Dice Only (Peak: {peak_dice:.4f})")

plt.title("Validation Dice Trajectory Across Loss Functions", fontsize=12.5, fontweight="bold")
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Validation Dice Score", fontsize=11)
plt.ylim(0.4, 1.0)
plt.xlim(0.8, 15.6)
plt.grid(True, linestyle="-", alpha=0.3, color="#cccccc")
plt.legend(loc="lower right", fontsize=11, framealpha=0.9)
plt.tight_layout()

# Save high-resolution figure
fig_out = PROJECT_ROOT / "outputs" / "figures" / "unet_loss_ablation_curves.png"
plt.savefig(fig_out, dpi=300)
plt.show()""")
    ]
    with open(nb_dir / "04_unet_training.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(unet_cells), f, indent=2)

    # -------------------------------------------------------------
    # 05_hybrid_pipeline.ipynb
    # -------------------------------------------------------------
    hybrid_cells = [
        md_cell("# Stage 05: End-to-End Hybrid Pipeline on Unseen Test Images\n\n## Overview\nThis notebook executes the full hybrid pipeline on unseen test data:\n$$\\text{Test Image} \\rightarrow \\text{U-Net Mask} \\rightarrow \\text{regionprops} \\rightarrow \\text{Numeric Summary} \\rightarrow \\text{LLM JSON Record & Narrative}$$"),
        code_cell("""import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import gc
from pathlib import Path
import torch
import pandas as pd

CURRENT_DIR = Path.cwd()
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name.lower() == "notebooks" else CURRENT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.unet import SmallUNet
from src.pipeline import run_batch_hybrid_pipeline"""),
        md_cell("## 1. Load Pretrained U-Net Model"),
        code_cell("""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallUNet(features=(16, 32, 64, 128)).to(device)

ckpt_path = PROJECT_ROOT / "outputs" / "models" / "unet_combined_bce___dice.pth"
if ckpt_path.exists():
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    print(f"Loaded trained checkpoint from {ckpt_path.name}")
else:
    print(f"Warning: Checkpoint not found at {ckpt_path}. Using uninitialized model.")"""),
        md_cell("## 2. Execute Batch Hybrid Pipeline"),
        code_cell("""data_dir = PROJECT_ROOT / "data" / "nuclei_dataset"
if not data_dir.exists():
    data_dir = PROJECT_ROOT / "nuclei_dataset"

test_img_dir = data_dir / "test" / "images"
test_mask_dir = data_dir / "test" / "masks"

summary_df, results = run_batch_hybrid_pipeline(
    model=model,
    image_dir=test_img_dir,
    mask_dir=test_mask_dir,
    output_csv_path=PROJECT_ROOT / "outputs" / "csv" / "hybrid_pipeline_test_summary.csv",
    llm_model="llama3.2",
    device=device
)
display(summary_df[["image_id", "dataset_density_regime", "predicted_density_class", "ground_truth_nuclei_count", "predicted_component_count", "count_error", "absolute_count_error", "llm_n_objects", "llm_measurement_audit_match", "json_valid", "quality_flag", "dice_vs_gt", "iou_vs_gt"]])"""),
        md_cell("## 3. Summary & Audit Verification"),
        code_cell("""mean_dice = summary_df["dice_vs_gt"].mean()
std_dice = summary_df["dice_vs_gt"].std()

mean_iou = summary_df["iou_vs_gt"].mean()
std_iou = summary_df["iou_vs_gt"].std()

mae_count = summary_df["absolute_count_error"].mean() if "absolute_count_error" in summary_df.columns else None
audit_col = "llm_measurement_audit_match" if "llm_measurement_audit_match" in summary_df.columns else "audit_count_match"

print(f"Total test images processed: {len(summary_df)}")
print(f"Mean test Dice: {mean_dice:.4f} ± {std_dice:.4f}")
print(f"Mean test IoU: {mean_iou:.4f} ± {std_iou:.4f}")
if mae_count is not None:
    print(f"Mean absolute count error (MAE): {mae_count:.2f} nuclei")
print(f"LLM measurement reproduction match: {(summary_df[audit_col].sum() / len(summary_df))*100:.1f}%")
print(f"Schema validity rate (json_valid): {(summary_df['json_valid'].sum() / len(summary_df))*100:.1f}%")""")
    ]
    with open(nb_dir / "05_hybrid_pipeline.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(hybrid_cells), f, indent=2)

    # -------------------------------------------------------------
    # 06_robustness_experiment.ipynb
    # -------------------------------------------------------------
    robust_cells = [
        md_cell("# Stage 06: Extra Credit - Robustness & Error Propagation\n\n## Overview\nThis notebook evaluates how perturbations (heavy blur, crushed low contrast) propagate through the hybrid pipeline and demonstrates automated quality gating."),
        code_cell("""import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = Path.cwd()
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name.lower() == "notebooks" else CURRENT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.unet import SmallUNet
from src.pipeline import run_robustness_experiment"""),
        md_cell("## 1. Evaluate Corrupted Image Variants"),
        code_cell("""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallUNet(features=(16, 32, 64, 128)).to(device)

ckpt_path = PROJECT_ROOT / "outputs" / "models" / "unet_combined_bce___dice.pth"
if ckpt_path.exists():
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])

data_dir = PROJECT_ROOT / "data" / "nuclei_dataset"
if not data_dir.exists():
    data_dir = PROJECT_ROOT / "nuclei_dataset"

clean_imgs = [
    data_dir / "test" / "images" / "test_000.png",
    data_dir / "test" / "images" / "test_004.png",
]
corrupt_imgs = [
    data_dir / "test_corrupted" / "images" / "test_000_blur.png",
    data_dir / "test_corrupted" / "images" / "test_000_lowcontrast.png",
    data_dir / "test_corrupted" / "images" / "test_004_blur.png",
    data_dir / "test_corrupted" / "images" / "test_004_lowcontrast.png",
]

robust_df = run_robustness_experiment(
    model=model,
    clean_image_path=clean_imgs,
    corrupted_image_paths=corrupt_imgs,
    llm_model="llama3.2",
    device=device
)
display(robust_df[["image_id", "condition", "mean_intensity", "intensity_range", "connected_component_count", "area_fraction", "quality_gate", "quality_reason", "llm_quality_flag"]])""")
    ]
    with open(nb_dir / "06_robustness_experiment.ipynb", "w", encoding="utf-8") as f:
        json.dump(make_notebook(robust_cells), f, indent=2)

    print("Successfully built all 6 numbered notebooks in:", nb_dir)

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    build_all(root / "notebooks")
    build_all(root.parent / "Biomedical_Image_Analysis_Assignment" / "notebooks")
