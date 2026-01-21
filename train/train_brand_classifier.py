# train/train_brand_classifier.py

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import argparse
import csv
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.seed import set_seed
from utils.data import build_datasets
from utils.metrics import accuracy_top1
from model.brand_classifier import BrandClassifier


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def train_one_epoch(model, loader, optimizer, device) -> tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_top1(logits.detach(), y)
        steps += 1

    return total_loss / max(1, steps), total_acc / max(1, steps)


@torch.no_grad()
def eval_one_epoch(model, loader, device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    for x, y in tqdm(loader, desc="Val", leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        total_acc += accuracy_top1(logits, y)
        steps += 1

    return total_loss / max(1, steps), total_acc / max(1, steps)


def main():
    parser = argparse.ArgumentParser("Train milk-tea brand classifier (ViT transfer learning)")
    parser.add_argument("--data_dir", type=str, default="data/images",
                        help="Path to data/images (ImageFolder style)")
    parser.add_argument("--out_dir", type=str, default="results/brand_classifier",
                        help="Where to save checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # Training strategy switches
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze ViT backbone and train only head (default recommended)")
    parser.add_argument("--unfreeze_after", type=int, default=-1,
                        help="If >=0, unfreeze backbone after this epoch (for fine-tuning)")

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

 
    # 1) Build datasets / loaders
    split = build_datasets(
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        image_size=224,
    )
    idx_to_class = split.idx_to_class
    num_classes = len(idx_to_class)

    train_loader = DataLoader(
        split.train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        split.val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

 
    # 2) Build model
    model = BrandClassifier(num_classes=num_classes, freeze_backbone=args.freeze_backbone)
    model.to(device)

    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)


    # 3) Prepare output dir / logs
    os.makedirs(args.out_dir, exist_ok=True)

    # Save label mapping for inference / report
    labels_path = os.path.join(args.out_dir, "labels.json")
    save_json(labels_path, idx_to_class)

    log_path = os.path.join(args.out_dir, "train_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_acc = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")
    last_path = os.path.join(args.out_dir, "last.pt")

    # 4) Training loop
    for epoch in range(1, args.epochs + 1):
        # Optional staged fine-tuning:
        # Unfreeze backbone after a few epochs to improve performance.
        if args.unfreeze_after >= 0 and epoch == args.unfreeze_after + 1:
            model.set_backbone_trainable(True)
            # Rebuild optimizer with newly trainable parameters
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, device)

        # Append log row
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        # Save last checkpoint each epoch
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "idx_to_class": idx_to_class,
            "args": vars(args),
        }, last_path)

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "idx_to_class": idx_to_class,
                "args": vars(args),
            }, best_path)

        print(f"[Epoch {epoch:02d}/{args.epochs}] "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} "
              f"(best_val_acc={best_val_acc:.4f})")

    print(f"\nDone. Best checkpoint: {best_path}")
    print(f"Logs saved to: {log_path}")
    print(f"Labels saved to: {labels_path}")


if __name__ == "__main__":
    main()
