"""
Training script — DeepLabV3+ on Cityscapes with robustness augmentations.

Key features over baseline:
  • DeepLabV3+ with ResNet-101 backbone
  • Exponential Moving Average (EMA) of weights
  • Mixed-precision training (AMP)
  • Per-supercategory IoU / Dice logging
  • OneCycleLR scheduler
"""

import os
from argparse import ArgumentParser

import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from segmentation_models_pytorch.losses import DiceLoss

from model import Model, EMA
from dataset import CityscapeAlbumentations, train_transformation, validation_transformation
import config

# ── Cityscapes label helpers ──────────────────────────────────────────
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}

def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid.get(x, 255))

train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    b, _, h, w = prediction.shape
    color = torch.zeros((b, 3, h, w), dtype=torch.uint8)
    for tid, c in train_id_to_color.items():
        mask = prediction[:, 0] == tid
        for ch in range(3):
            color[:, ch][mask] = c[ch]
    return color


# ── Per-class / supercategory metrics ─────────────────────────────────
def compute_metrics(preds: torch.Tensor, labels: torch.Tensor,
                    n_classes: int = 19, ignore_index: int = 255):
    """
    Returns per-class IoU and Dice arrays (length n_classes).
    """
    iou = np.zeros(n_classes)
    dice = np.zeros(n_classes)
    valid = np.zeros(n_classes, dtype=bool)

    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    for c in range(n_classes):
        pred_c = preds_np == c
        true_c = labels_np == c
        mask = labels_np != ignore_index

        pred_c = pred_c & mask
        true_c = true_c & mask

        intersection = (pred_c & true_c).sum()
        union = (pred_c | true_c).sum()
        sum_both = pred_c.sum() + true_c.sum()

        if true_c.sum() == 0 and pred_c.sum() == 0:
            continue

        valid[c] = True
        iou[c] = intersection / (union + 1e-8)
        dice[c] = 2 * intersection / (sum_both + 1e-8)

    return iou, dice, valid


def supercategory_metrics(iou, dice, valid):
    """Aggregate per-class metrics into supercategory means."""
    results = {}
    for cat, class_ids in config.SUPERCATEGORY_MAP.items():
        mask = np.array([valid[c] for c in class_ids])
        if mask.any():
            results[f"IoU_{cat}"] = np.mean([iou[c] for c in class_ids if valid[c]])
            results[f"Dice_{cat}"] = np.mean([dice[c] for c in class_ids if valid[c]])
        else:
            results[f"IoU_{cat}"] = 0.0
            results[f"Dice_{cat}"] = 0.0
    results["MeanIoU"] = np.mean(iou[valid]) if valid.any() else 0.0
    results["MeanDice"] = np.mean(dice[valid]) if valid.any() else 0.0
    return results


# ── Argument parser ───────────────────────────────────────────────────
def get_args_parser():
    p = ArgumentParser("Training script for Cityscapes segmentation")
    p.add_argument("--data-dir", type=str, default=config.DATA_DIR)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--experiment-id", type=str, default=config.EXPERIMENT_ID)
    p.add_argument("--encoder-name", type=str, default=config.ENCODER_NAME)
    p.add_argument("--apply-fourier", type=lambda x: str(x).lower() == "true",
                   default=config.APPLY_FOURIER)
    p.add_argument("--apply-copypaste", type=lambda x: str(x).lower() == "true",
                   default=config.APPLY_COPYPASTE)
    p.add_argument("--apply-freq-band-dropout", type=lambda x: str(x).lower() == "true",
                   default=config.APPLY_FREQ_BAND_DROPOUT)
    p.add_argument("--apply-semantic-style-swap", type=lambda x: str(x).lower() == "true",
                   default=config.APPLY_SEMANTIC_STYLE_SWAP)
    return p


# ── Main ──────────────────────────────────────────────────────────────
def main(args):
    wandb.init(project=config.WANDB_PROJECT, name=args.experiment_id,
               config=vars(args))

    output_dir = os.path.join(config.CHECKPOINT_DIR, args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Datasets ──────────────────────────────────────────────────
    train_raw = Cityscapes(args.data_dir, split="train", mode="fine",
                           target_type="semantic")
    valid_raw = Cityscapes(args.data_dir, split="val", mode="fine",
                           target_type="semantic")

    train_dataset = CityscapeAlbumentations(
        train_raw, transform=train_transformation,
        apply_fourier=args.apply_fourier, apply_copypaste=args.apply_copypaste,
        apply_freq_band_dropout=args.apply_freq_band_dropout,
        apply_semantic_style_swap=args.apply_semantic_style_swap)
    valid_dataset = CityscapeAlbumentations(
        valid_raw, transform=validation_transformation,
        apply_fourier=False, apply_copypaste=False,
        apply_freq_band_dropout=False, apply_semantic_style_swap=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────
    model = Model(
        in_channels=config.IN_CHANNELS,
        n_classes=config.N_CLASSES,
        encoder_name=args.encoder_name,
    ).to(device)

    ema = EMA(model, decay=config.EMA_DECAY) if config.USE_EMA else None

    # ── Loss ──────────────────────────────────────────────────────
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    dice_loss_fn = DiceLoss(mode="multiclass", ignore_index=255)
    ce_w = config.LOSS_WEIGHTS["cross_entropy"]
    dice_w = config.LOSS_WEIGHTS["dice"]

    # ── Optimizer & Scheduler ─────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.05,
        anneal_strategy="cos",
    )

    scaler = GradScaler()  # AMP

    # ── Training loop ─────────────────────────────────────────────
    best_miou = 0.0
    best_model_path = None
    global_step = 0

    for epoch in range(args.epochs):
        # ── Train ─────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(images)
                loss = ce_w * ce_loss_fn(logits, labels) + dice_w * dice_loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if ema is not None:
                ema.update(model)

            epoch_loss += loss.item()
            wandb.log({"train_loss": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"]},
                       step=global_step)
            global_step += 1

        avg_train = epoch_loss / len(train_loader)

        # ── Validate (use EMA weights if available) ───────────
        eval_model = ema.module() if ema is not None else model
        eval_model.eval()

        val_losses = []
        all_iou = np.zeros(config.N_CLASSES)
        all_dice = np.zeros(config.N_CLASSES)
        all_valid = np.zeros(config.N_CLASSES, dtype=bool)
        n_batches = 0

        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_loader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)

                with autocast():
                    logits = eval_model(images)
                    loss = ce_w * ce_loss_fn(logits, labels) + dice_w * dice_loss_fn(logits, labels)
                val_losses.append(loss.item())

                preds = logits.softmax(1).argmax(1)

                iou, dice, valid = compute_metrics(preds, labels, config.N_CLASSES)
                all_iou += iou
                all_dice += dice
                all_valid |= valid
                n_batches += 1

                # Log visual predictions for first batch
                if i == 0:
                    pred_vis = convert_train_id_to_color(preds.unsqueeze(1))
                    lbl_vis = convert_train_id_to_color(labels.unsqueeze(1))
                    wandb.log({
                        "predictions": [wandb.Image(
                            make_grid(pred_vis.cpu(), nrow=4).permute(1, 2, 0).numpy())],
                        "labels": [wandb.Image(
                            make_grid(lbl_vis.cpu(), nrow=4).permute(1, 2, 0).numpy())],
                    }, step=global_step - 1)

        # Average metrics over batches
        avg_iou = all_iou / np.maximum(n_batches, 1)
        avg_dice = all_dice / np.maximum(n_batches, 1)
        metrics = supercategory_metrics(avg_iou, avg_dice, all_valid)
        avg_val_loss = sum(val_losses) / len(val_losses)

        log_dict = {"valid_loss": avg_val_loss, "epoch": epoch + 1}
        log_dict.update(metrics)
        wandb.log(log_dict, step=global_step - 1)

        print(f"Epoch {epoch+1:03}/{args.epochs}  "
              f"train_loss={avg_train:.4f}  val_loss={avg_val_loss:.4f}  "
              f"mIoU={metrics['MeanIoU']:.4f}  mDice={metrics['MeanDice']:.4f}")

        # ── Checkpoint best model ─────────────────────────────
        if metrics["MeanIoU"] > best_miou:
            best_miou = metrics["MeanIoU"]
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_model_path = os.path.join(
                output_dir,
                f"best_model-epoch={epoch:04}-mIoU={best_miou:.4f}.pt")
            sd = ema.state_dict() if ema is not None else model.state_dict()
            torch.save(sd, best_model_path)
            print(f"  ↳ saved best model (mIoU={best_miou:.4f})")

    # ── Save final model ──────────────────────────────────────────
    final_sd = ema.state_dict() if ema is not None else model.state_dict()
    torch.save(final_sd, os.path.join(
        output_dir, f"final_model-epoch={epoch:04}-mIoU={metrics['MeanIoU']:.4f}.pt"))

    print("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
