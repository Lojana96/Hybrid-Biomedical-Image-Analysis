"""
src/unet.py

Small U-Net implementation in PyTorch for biomedical nuclei segmentation.

This module provides:
- Reusable DoubleConv block and 4-level U-Net architecture
- PyTorch Dataset and DataLoader wrappers for nuclei images and binary masks
- Loss functions: BCEWithLogitsLoss, Soft Dice Loss, and Combined BCE+Dice Loss
- Evaluation metrics: Hard Dice Coefficient, Intersection-over-Union (IoU/Jaccard), Precision, Recall
- Comprehensive training, validation, and inference routines with checkpointing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.preprocessing import DEFAULT_IMAGE_SIZE, load_mask, preprocess_image


# -----------------------------------------------------------------------------
# U-Net Architecture
# -----------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """
    Two consecutive Convolution -> BatchNorm -> ReLU blocks.
    Preserves spatial dimensions via padding=1.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SmallUNet(nn.Module):
    """
    Lightweight 4-level U-Net for microscopy nucleus segmentation.
    Default feature depths: [16, 32, 64, 128] with bottleneck at 256.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Tuple[int, ...] = (16, 32, 64, 128),
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder path
        curr_in = in_channels
        for feat in features:
            self.downs.append(DoubleConv(curr_in, feat))
            curr_in = feat

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder path
        for feat in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feat * 2, feat, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feat * 2, feat))

        # Final 1x1 classifier returning raw logits
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Downward encoder pass
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upward decoder pass with skip connections
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Handle potential spatial mismatch via padding if dimensions differ
            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True
                )

            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return self.final_conv(x)


# -----------------------------------------------------------------------------
# Dataset & DataLoaders
# -----------------------------------------------------------------------------

class NucleiDataset(Dataset):
    """
    PyTorch Dataset for paired nuclei images and binary masks.
    """

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Optional[Path] = None,
        target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        augment: bool = False,
        seed: Optional[int] = None,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.target_size = target_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        self.image_paths = sorted(list(self.image_dir.glob("*.png")))
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, str]]:
        img_path = self.image_paths[idx]
        image_np = preprocess_image(img_path, target_size=self.target_size)

        if self.mask_dir is not None:
            mask_path = self.mask_dir / img_path.name
            mask_np = load_mask(mask_path, target_size=self.target_size)

            # Simple spatial augmentations (flips / 90-degree rotations)
            if self.augment:
                if self.rng.random() > 0.5:
                    image_np = np.fliplr(image_np).copy()
                    mask_np = np.fliplr(mask_np).copy()
                if self.rng.random() > 0.5:
                    image_np = np.flipud(image_np).copy()
                    mask_np = np.flipud(mask_np).copy()
                k = self.rng.integers(0, 4)
                if k > 0:
                    image_np = np.rot90(image_np, k).copy()
                    mask_np = np.rot90(mask_np, k).copy()

            image_tensor = torch.from_numpy(image_np).unsqueeze(0).float()
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
            return image_tensor, mask_tensor
        else:
            image_tensor = torch.from_numpy(image_np).unsqueeze(0).float()
            return image_tensor, img_path.stem


# -----------------------------------------------------------------------------
# Losses & Metrics
# -----------------------------------------------------------------------------

def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Differentiable Soft Dice Loss computed from logits.
    Target must be binary [0, 1]. Range: [0, 1], lower is better.
    """
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard Binary Cross-Entropy with logits."""
    return F.binary_cross_entropy_with_logits(logits, target)


def combined_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    """
    Combined BCE and Soft Dice loss for balanced gradient propagation
    and boundary overlap optimization.
    """
    bce = bce_loss(logits, target)
    dice = soft_dice_loss(logits, target)
    return (bce_weight * bce) + (dice_weight * dice)


@torch.no_grad()
def dice_coefficient(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """Hard Dice score on thresholded predictions. Range: [0, 1], higher is better."""
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * target).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return float(dice.mean().item())


@torch.no_grad()
def iou_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """Intersection-over-Union (Jaccard index). Range: [0, 1], higher is better."""
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * target).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return float(iou.mean().item())


@torch.no_grad()
def precision_recall_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Tuple[float, float]:
    """Calculate pixel-level Precision and Recall."""
    preds = (torch.sigmoid(logits) > threshold).float()
    tp = (preds * target).sum(dim=(1, 2, 3))
    fp = (preds * (1.0 - target)).sum(dim=(1, 2, 3))
    fn = ((1.0 - preds) * target).sum(dim=(1, 2, 3))

    precision = ((tp + eps) / (tp + fp + eps)).mean().item()
    recall = ((tp + eps) / (tp + fn + eps)).mean().item()
    return float(precision), float(recall)


# -----------------------------------------------------------------------------
# Training & Validation
# -----------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        bs = images.size(0)

        logits = model(images)
        loss = loss_fn(logits, masks)

        total_loss += loss.item() * bs
        total_dice += dice_coefficient(logits, masks) * bs
        total_iou += iou_score(logits, masks) * bs
        n += bs

    return {
        "val_loss": total_loss / n,
        "val_dice": total_dice / n,
        "val_iou": total_iou / n,
    }


def train_unet(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    n_epochs: int = 15,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Train U-Net model and record epoch history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
    }

    best_val_dice = -1.0

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_dice"].append(val_metrics["val_dice"])
        history["val_iou"].append(val_metrics["val_iou"])

        if val_metrics["val_dice"] > best_val_dice:
            best_val_dice = val_metrics["val_dice"]
            if checkpoint_path is not None:
                checkpoint_path = Path(checkpoint_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_dice": best_val_dice,
                        "val_iou": val_metrics["val_iou"],
                    },
                    checkpoint_path,
                )

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == n_epochs):
            print(
                f"Epoch {epoch:02d}/{n_epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Dice: {val_metrics['val_dice']:.4f} | "
                f"Val IoU: {val_metrics['val_iou']:.4f}"
            )

    return history


@torch.no_grad()
def segment_image(
    model: nn.Module,
    image: np.ndarray,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a single 2D grayscale image array.

    Returns:
        (binary_mask_uint8, probability_map_float32)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    img_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
    logits = model(img_tensor)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    binary_mask = (probs > threshold).astype(np.uint8)

    return binary_mask, probs
