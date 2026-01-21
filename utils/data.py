# utils/data.py
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


@dataclass
class DataSplit:
    """
    A container for train/val datasets and label mapping.
    """
    train_set: Dataset
    val_set: Dataset
    idx_to_class: Dict[int, str]


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build train/val transforms.

    Notes:
    - Train transform uses augmentation to reduce overfitting.
    - Val transform is deterministic for fair evaluation.
    """
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        # ImageNet normalization (matches pretrained ViT weights convention)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def stratified_split_indices(
    targets: List[int],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Stratified split by class.

    Why:
    - Your brand folders may be imbalanced.
    - Stratified split keeps class distribution similar in train/val.
    """
    g = torch.Generator().manual_seed(seed)

    targets_tensor = torch.tensor(targets, dtype=torch.long)
    classes = torch.unique(targets_tensor).tolist()

    train_indices: List[int] = []
    val_indices: List[int] = []

    for c in classes:
        idx = torch.where(targets_tensor == c)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]  # shuffle indices of this class

        n_val = max(1, int(len(idx) * val_ratio)) if len(idx) > 1 else 0
        val_idx = idx[:n_val].tolist()
        train_idx = idx[n_val:].tolist()

        val_indices.extend(val_idx)
        train_indices.extend(train_idx)

    return train_indices, val_indices


def build_datasets(
    data_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    image_size: int = 224,
) -> DataSplit:
    """
    Build ImageFolder dataset and create stratified train/val subsets.

    Expected folder structure:
      data_dir/
        BrandA/
          xxx.jpg
        BrandB/
          yyy.png
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    train_tf, val_tf = build_transforms(image_size=image_size)

    # Important trick:
    # - ImageFolder stores samples as (path, class_index)
    # - But transform is a property of dataset
    # We build two datasets with same file list but different transforms.
    base = ImageFolder(root=data_dir)  # no transform yet, just to read targets/class_to_idx
    targets = [t for _, t in base.samples]

    train_idx, val_idx = stratified_split_indices(targets, val_ratio=val_ratio, seed=seed)

    train_ds = ImageFolder(root=data_dir, transform=train_tf)
    val_ds = ImageFolder(root=data_dir, transform=val_tf)

    train_set = Subset(train_ds, train_idx)
    val_set = Subset(val_ds, val_idx)

    # Build idx_to_class mapping for saving to json
    idx_to_class = {v: k for k, v in base.class_to_idx.items()}

    return DataSplit(train_set=train_set, val_set=val_set, idx_to_class=idx_to_class)
