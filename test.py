# coding:utf-8
import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from model import MFNet
from util.landslide_dataset import LandslideDataset


logger = logging.getLogger("MS2Fusion-Test")


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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Test and Evaluate MS2Fusion / MFNet on Landslide Dataset")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml", help="Path to config YAML file")
    parser.add_argument("--checkpoint", "-ckpt", type=str, required=False, default=None, help="Path to checkpoint .pth file")
    parser.add_argument("--split", "-s", type=str, default="test", help="Dataset split to evaluate (test, val, train)")
    parser.add_argument("--save_dir", "-o", type=str, default="results/test", help="Directory to save predictions and metrics")
    parser.add_argument("--save_masks", type=str2bool, nargs="?", const=True, default=True, help="Save prediction masks (.png)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for landslide detection")
    parser.add_argument("--batch_size", "-B", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--device", "-G", type=str, default=None, help="Device to use (e.g. cuda, cuda:0, cpu)")
    parser.add_argument("--gpu", type=int, default=None, help="Legacy GPU index parameter")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset directory override")
    parser.add_argument("--num_workers", "-j", type=int, default=None, help="Number of dataloader workers")
    parser.add_argument("--model_name", "-M", type=str, default="MFNet", help="Model name")
    return parser.parse_args()


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

    ds_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})

    data_dir = args.data_dir or ds_cfg.get("data_dir", "datasets/landslide")
    modalities = ds_cfg.get("modalities", ["IMAGE", "DTM"])
    blacklist_path = ds_cfg.get("blacklist_path", None)
    img_size = int(ds_cfg.get("size", 512))
    batch_size = args.batch_size or int(ds_cfg.get("batch_size", 1))
    num_workers = args.num_workers if args.num_workers is not None else int(ds_cfg.get("num_workers", 4))

    num_classes = int(model_cfg.get("num_classes", 2))
    model_name = args.model_name or model_cfg.get("name", "MFNet")

    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif args.gpu is not None and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    setup_logger(os.path.join(args.save_dir, "test.log"))

    logger.info("=" * 60)
    logger.info("MS2Fusion / MFNet Landslide Segmentation Evaluation")
    logger.info("=" * 60)
    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Save directory: {args.save_dir}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Device: {device}")
    logger.info(f"Modalities: {modalities}")

    # 1. Dataset & DataLoader
    dataset = LandslideDataset(
        data_dir=data_dir,
        split=args.split,
        size=img_size,
        modalities=modalities,
        blacklist_path=blacklist_path,
        mode="test",
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    logger.info(f"Total test samples: {len(dataset)}")

    # 2. Model setup
    non_rgb_modalities = [m for m in modalities if m != "IMAGE"]
    in_channels_inf = len(non_rgb_modalities)

    logger.info(f"Initializing {model_name} (in_channels_inf={in_channels_inf}, num_classes={num_classes})...")
    model = MFNet(n_class=num_classes, in_channels_inf=in_channels_inf)

    # 3. Load Checkpoint
    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
        logger.info(f"Loading weights from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        # Remove 'module.' prefix if trained with DataParallel
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        logger.info("Weights loaded successfully.")
    else:
        logger.warning("No checkpoint specified! Running with randomly initialized weights.")

    model.to(device)
    model.eval()

    # 4. Evaluation Loop
    cf = np.zeros((num_classes, num_classes), dtype=np.int64)
    has_any_labels = False

    masks_dir = os.path.join(args.save_dir, "masks") if args.save_masks else None
    if masks_dir:
        os.makedirs(masks_dir, exist_ok=True)

    logger.info("Running inference...")
    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Testing [{args.split}]", leave=True):
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                names = batch["name"]
                has_label_flag = bool(batch.get("has_label", [True])[0])
            else:
                images, labels, names = batch[0].to(device), batch[1].to(device), batch[2]
                has_label_flag = True

            if labels.dim() == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1).long()
            else:
                labels = labels.long()

            logits = model(images)

            if num_classes == 2:
                probs = F.softmax(logits, dim=1)[:, 1]
                preds = (probs >= args.threshold).long()
            else:
                preds = logits.argmax(dim=1)

            # Confusion matrix accumulation if ground truth is present
            if has_label_flag and (labels > 0).any() or (labels == 0).any():
                has_any_labels = True
                for c1 in range(num_classes):
                    for c2 in range(num_classes):
                        cf[c1, c2] += int(((labels == c1) & (preds == c2)).sum().item())

            # Save prediction masks
            if args.save_masks:
                preds_np = preds.cpu().numpy().astype(np.uint8)
                for i, name in enumerate(names):
                    mask = preds_np[i] * 255  # 255 for landslide, 0 for background
                    img = Image.fromarray(mask)
                    # Save both to root save_dir and masks subfolder for compatibility
                    img.save(os.path.join(args.save_dir, f"{name}.png"))

    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.2f}s ({len(dataset) / max(inference_time, 0.001):.1f} samples/sec)")

    # 5. Report & Save Metrics
    if has_any_labels:
        metrics = calculate_metrics(cf)
        metrics["total_samples"] = len(dataset)
        metrics["inference_time_seconds"] = float(inference_time)
        metrics["threshold"] = float(args.threshold)

        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS:")
        logger.info("=" * 60)
        logger.info(f"Overall Accuracy:  {metrics['accuracy'] * 100:.2f}%")
        logger.info(f"Background IoU:    {metrics['iou_background'] * 100:.2f}%")
        logger.info(f"Landslide IoU:     {metrics['iou_landslide'] * 100:.2f}%")
        logger.info(f"Mean IoU (mIoU):   {metrics['mIoU'] * 100:.2f}%")
        if "precision" in metrics:
            logger.info(f"Precision:         {metrics['precision'] * 100:.2f}%")
            logger.info(f"Recall:            {metrics['recall'] * 100:.2f}%")
            logger.info(f"F1 / Dice Score:   {metrics['f1_dice'] * 100:.2f}%")
        logger.info("=" * 60)

        # Save metrics to JSON and text
        metrics_json_path = os.path.join(args.save_dir, "metrics.json")
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        metrics_txt_path = os.path.join(args.save_dir, "metrics.txt")
        with open(metrics_txt_path, "w", encoding="utf-8") as f:
            f.write("Landslide Segmentation Evaluation Results\n")
            f.write("=" * 50 + "\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

        logger.info(f"Metrics saved to: {metrics_json_path}")
    else:
        logger.info("No ground-truth labels detected; quantitative metrics skipped.")

    if args.save_masks:
        logger.info(f"Prediction masks saved to: {args.save_dir}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
