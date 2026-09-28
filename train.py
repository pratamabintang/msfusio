# coding:utf-8
import argparse
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from model import MFNet
from util.landslide_dataset import LandslideDataset

logger = logging.getLogger("MS2Fusion")

def setup_logger(log_file: Optional[str] = None):
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


class LandslideLoss(nn.Module):
    """
    Combined Cross-Entropy and Dice loss for landslide segmentation.
    Addresses extreme class imbalance between background and landslide areas.
    """

    def __init__(self, num_classes: int = 2, weight: Optional[torch.Tensor] = None, use_dice: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.use_dice = use_dice

    def dice_loss(self, logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        if self.num_classes == 2:
            fg_prob = probs[:, 1]
            fg_target = (targets == 1).float()
            intersection = (fg_prob * fg_target).sum(dim=(1, 2))
            union = fg_prob.sum(dim=(1, 2)) + fg_target.sum(dim=(1, 2))
            dice = (2.0 * intersection + smooth) / (union + smooth)
            return (1.0 - dice).mean()
        else:
            targets_one_hot = F.one_hot(
                targets.clamp(0, self.num_classes - 1), num_classes=self.num_classes
            ).permute(0, 3, 1, 2).float()
            intersection = (probs * targets_one_hot).sum(dim=(2, 3))
            union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
            dice = (2.0 * intersection + smooth) / (union + smooth)
            return (1.0 - dice.mean(dim=1)).mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        if self.use_dice and self.num_classes == 2:
            d_loss = self.dice_loss(logits, targets)
            return ce_loss + d_loss
        return ce_loss


def calculate_metrics(cf: np.ndarray) -> Dict[str, float]:
    """Calculates accuracy, IoU per class, mean IoU, Precision, Recall, and F1/Dice."""
    n_class = cf.shape[0]
    total = cf.sum()
    overall_acc = float(np.diag(cf).sum() / max(total, 1))

    iou_list = []
    for c in range(n_class):
        intersection = cf[c, c]
        union = cf[c, :].sum() + cf[:, c].sum() - intersection
        iou = float(intersection / max(union, 1))
        iou_list.append(iou)

    metrics = {
        "accuracy": overall_acc,
        "iou_background": iou_list[0] if len(iou_list) > 0 else 0.0,
        "iou_landslide": iou_list[1] if len(iou_list) > 1 else 0.0,
        "mIoU": float(np.mean(iou_list)),
    }

    if n_class >= 2:
        tp = cf[1, 1]
        fp = cf[:, 1].sum() - tp
        fn = cf[1, :].sum() - tp
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float(2 * precision * recall / max(precision + recall, 1e-8))
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1_dice"] = f1

    return metrics


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    grad_clip: float,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{total_epochs:03d} [Train]", leave=False)
    for batch in pbar:
        if isinstance(batch, dict):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
        else:
            images, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1).long()
        else:
            labels = labels.long()

        optimizer.zero_grad(set_to_none=True)

        if amp_enabled and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(images)
                loss = criterion(logits, labels)

            loss_val = loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val):
                logger.warning(f"Epoch {epoch:03d} encountered NaN/Inf loss ({loss_val})! Skipping step.")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)

            loss_val = loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val):
                logger.warning(f"Epoch {epoch:03d} encountered NaN/Inf loss ({loss_val})! Skipping step.")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += loss_val

        preds = logits.argmax(dim=1)
        correct_pixels += (preds == labels).sum().item()
        total_pixels += labels.numel()

        pbar.set_postfix({"loss": f"{loss_val:.4f}", "acc": f"{correct_pixels / max(total_pixels, 1):.4f}"})

    n_batches = len(train_loader)
    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = correct_pixels / max(total_pixels, 1)
    return {"loss": avg_loss, "accuracy": avg_acc}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    epoch: int,
    total_epochs: int,
    amp_enabled: bool = True,
    amp_dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    cf = np.zeros((num_classes, num_classes), dtype=np.int64)

    pbar = tqdm(val_loader, desc=f"Epoch {epoch:03d}/{total_epochs:03d} [Val]", leave=False)
    for batch in pbar:
        if isinstance(batch, dict):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
        else:
            images, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1).long()
        else:
            labels = labels.long()

        if amp_enabled and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        # Vectorized confusion matrix on GPU/device without per-cell synchronization
        flat_mask = (labels * num_classes + preds).view(-1)
        batch_cf = torch.bincount(flat_mask, minlength=num_classes ** 2).view(num_classes, num_classes)
        cf += batch_cf.cpu().numpy()

    n_batches = len(val_loader)
    metrics = calculate_metrics(cf)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Train MS2Fusion / MFNet for Landslide Segmentation")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml", help="Path to config YAML file")
    parser.add_argument("--device", "-G", type=str, default=None, help="Device to use (e.g. cuda, cuda:0, cpu)")
    parser.add_argument("--batch_size", "-B", type=int, default=None, help="Batch size override")
    parser.add_argument("--epochs", "-E", type=int, default=None, help="Max epochs override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory override")
    parser.add_argument("--runs_dir", type=str, default=None, help="Runs directory override")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume checkpoint path")
    parser.add_argument("--model_name", "-M", type=str, default="MFNet", help="Model name")
    parser.add_argument("--num_workers", "-j", type=int, default=None, help="Num workers override")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        alt_path = os.path.join(os.path.dirname(__file__), config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Config file not found: {args.config}")

    cfg = load_config(config_path)

    # 1. Config sections with fallback defaults
    exp_cfg = cfg.get("experiment", {})
    ds_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    # CLI overrides
    exp_name = exp_cfg.get("name", "base")
    seed = int(exp_cfg.get("seed", 42))
    runs_dir = args.runs_dir or exp_cfg.get("runs_dir", "runs")

    data_dir = args.data_dir or ds_cfg.get("data_dir", "datasets/landslide")
    modalities = ds_cfg.get("modalities", ["IMAGE", "DTM"])
    blacklist_path = ds_cfg.get("blacklist_path", None)
    img_size = int(ds_cfg.get("size", 512))
    batch_size = args.batch_size or int(ds_cfg.get("batch_size", 4))
    num_workers = args.num_workers if args.num_workers is not None else int(ds_cfg.get("num_workers", 4))

    num_classes = int(model_cfg.get("num_classes", 2))
    model_name = args.model_name or model_cfg.get("name", "MFNet")

    epochs = args.epochs or int(train_cfg.get("epochs", 50))
    lr = args.lr or float(train_cfg.get("lr", 0.0001))
    weight_decay = float(train_cfg.get("weight_decay", 0.0005))
    min_lr = float(train_cfg.get("min_lr", 1.0e-7))
    save_interval = int(train_cfg.get("save_interval", 5))
    eval_interval = int(train_cfg.get("eval_interval", 1))
    amp_enabled = bool(train_cfg.get("amp", True))
    amp_dtype_str = str(train_cfg.get("amp_dtype", "auto")).lower()
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seed
    set_seed(seed)

    # 2. Output directory setup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_dir, f"{exp_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Setup logger
    log_file = os.path.join(run_dir, "log.txt")
    setup_logger(log_file)

    logger.info("=" * 60)
    logger.info("MS2Fusion / MFNet Landslide Segmentation Training")
    logger.info("=" * 60)
    logger.info(f"Config loaded from: {config_path}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Modalities: {modalities}")
    logger.info(f"Image size: {img_size}x{img_size}, Batch size: {batch_size}, Epochs: {epochs}")
    logger.info(f"Learning rate: {lr} (min: {min_lr}), Weight decay: {weight_decay}")

    # Save resolved config in run_dir for reproducibility and testing
    resolved_config = {
        "experiment": {
            "name": exp_name,
            "seed": seed,
            "runs_dir": runs_dir,
            "run_dir": run_dir,
        },
        "dataset": {
            "data_dir": data_dir,
            "modalities": modalities,
            "blacklist_path": blacklist_path,
            "size": img_size,
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
        "model": {
            "name": model_name,
            "num_classes": num_classes,
            **{k: v for k, v in model_cfg.items() if k not in ("name", "num_classes")},
        },
        "training": {
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "min_lr": min_lr,
            "save_interval": save_interval,
            "eval_interval": eval_interval,
            "amp": amp_enabled,
            "amp_dtype": amp_dtype_str,
            "grad_clip": grad_clip,
        },
    }
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, sort_keys=False)

    # 3. Datasets & DataLoaders
    train_dataset = LandslideDataset(
        data_dir=data_dir,
        split="train",
        size=img_size,
        modalities=modalities,
        blacklist_path=blacklist_path,
        mode="train",
    )
    val_dataset = LandslideDataset(
        data_dir=data_dir,
        split="val",
        size=img_size,
        modalities=modalities,
        blacklist_path=blacklist_path,
        mode="val",
    )

    pin_mem = device.type == "cuda"
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_mem,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True if len(train_dataset) > batch_size else False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 4. Model setup
    # Calculate auxiliary (non-RGB) channels from modalities
    non_rgb_modalities = [m for m in modalities if m != "IMAGE"]
    in_channels_inf = len(non_rgb_modalities)

    logger.info(f"Initializing {model_name} (in_channels_inf={in_channels_inf}, num_classes={num_classes})...")
    model = MFNet(n_class=num_classes, in_channels_inf=in_channels_inf)
    model.to(device)

    # Resume checkpoint if provided
    start_epoch = 1
    best_iou = 0.0
    if args.checkpoint and os.path.isfile(args.checkpoint):
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        # Handle DataParallel 'module.' prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if isinstance(ckpt, dict) and "best_iou" in ckpt:
            best_iou = float(ckpt["best_iou"])
        logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")

    # 5. Optimizer, Scheduler, Loss, AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    criterion = LandslideLoss(num_classes=num_classes, use_dice=True)

    if amp_dtype_str == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")

    # 6. Training Loop
    logger.info("Starting training loop...")
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_res = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            grad_clip=grad_clip,
            epoch=epoch,
            total_epochs=epochs,
        )
        scheduler.step()

        elapsed = time.time() - epoch_start_time
        logger.info(
            f"Epoch [{epoch:03d}/{epochs:03d}] lr: {current_lr:.6f} | "
            f"Train Loss: {train_res['loss']:.4f} Acc: {train_res['accuracy']:.4f} | Time: {elapsed:.1f}s"
        )

        # Validation
        if epoch % eval_interval == 0 or epoch == epochs:
            val_res = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
                epoch=epoch,
                total_epochs=epochs,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )

            val_iou = val_res.get("iou_landslide", val_res.get("mIoU", 0.0))
            is_best = val_iou > best_iou
            if is_best:
                best_iou = val_iou

            logger.info(
                f"   [Val Eval] Loss: {val_res['loss']:.4f} | "
                f"mIoU: {val_res['mIoU']:.4f} | Landslide IoU: {val_res['iou_landslide']:.4f} | "
                f"F1/Dice: {val_res.get('f1_dice', 0.0):.4f} | Acc: {val_res['accuracy']:.4f}"
                + (" (NEW BEST!)" if is_best else "")
            )

            # Save best checkpoint
            if is_best:
                best_ckpt_path = os.path.join(checkpoints_dir, "best.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_iou": best_iou,
                        "val_metrics": val_res,
                        "config": resolved_config,
                    },
                    best_ckpt_path,
                )
                logger.info(f"Saved best model checkpoint to: {best_ckpt_path}")

        # Save interval checkpoint
        if epoch % save_interval == 0 or epoch == epochs:
            interval_ckpt_path = os.path.join(checkpoints_dir, f"epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": resolved_config,
                },
                interval_ckpt_path,
            )

        # Always save last checkpoint
        last_ckpt_path = os.path.join(checkpoints_dir, "last.pth")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_iou": best_iou,
                "config": resolved_config,
            },
            last_ckpt_path,
        )

    logger.info("=" * 60)
    logger.info(f"Training completed! Best Validation IoU: {best_iou:.4f}")
    logger.info(f"Checkpoints directory: {checkpoints_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
