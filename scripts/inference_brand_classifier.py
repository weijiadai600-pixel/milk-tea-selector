# scripts/inference_brand_classifier.py
# ------------------------------------------------------------
# Use the trained brand classifier for inference:
# Input an image -> Output Top-K brand predictions + confidence scores
#
# Default files:
#   ckpt:   results/brand_classifier/best.pt
#   labels: results/brand_classifier/labels.json
# ------------------------------------------------------------

import os
import sys
import json
import argparse
import warnings
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Add project root to sys.path to make sure we can import model/*
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.brand_classifier import BrandClassifier  # noqa: E402


def build_transform():
    """Preprocessing consistent with training: 224 + ImageNet mean/std."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_labels(labels_path: str) -> List[str]:
    """
    Load labels.json and support three common formats, returning:
      class_names: List[str]  (index -> class_name)
    """
    with open(labels_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # 1) Direct list format
    if isinstance(obj, list):
        return obj

    # 2) dict: {"0":"A", "1":"B", ...}
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        if all(k.isdigit() for k in obj.keys()):
            items = sorted(((int(k), v) for k, v in obj.items()), key=lambda x: x[0])
            return [v for _, v in items]

        # 3) dict: {"A":0, "B":1, ...}
        if all(isinstance(v, int) for v in obj.values()):
            inv = {v: k for k, v in obj.items()}
            return [inv[i] for i in sorted(inv.keys())]

    raise ValueError(f"Cannot parse labels.json format: {type(obj)}")


def _torch_load_safely(path: str, device: torch.device):
    """
    Newer torch versions warn that weights_only default may change.
    We provide a compatibility wrapper:
    - If weights_only=True is supported, use it.
    - Otherwise fall back to normal torch.load.
    """
    try:
        # Newer torch supports weights_only
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Older torch does not support weights_only
        return torch.load(path, map_location=device)


def load_checkpoint(ckpt_path: str, device: torch.device) -> dict:
    """
    best.pt can be stored as:
      A) torch.save(model.state_dict())
      B) torch.save({"model_state": state_dict, ...})
      C) torch.save({"model": state_dict, ...}) / {"state_dict": state_dict, ...}
    This function extracts and returns the state_dict.
    """
    ckpt = _torch_load_safely(ckpt_path, device)

    # Case A: already a state_dict (keys like "backbone.xxx" / "head.xxx")
    if isinstance(ckpt, dict):
        # Case B: common packed checkpoint format (your current best.pt uses this)
        if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            return ckpt["model_state"]

        # Case C: other naming conventions
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]

        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]

        # Heuristic: if dict values contain tensors, treat it as state_dict
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt

    raise ValueError("Unsupported checkpoint format (best.pt)")


@torch.no_grad()
def predict_one(
    image_path: str,
    model: torch.nn.Module,
    class_names: List[str],
    device: torch.device,
    topk: int = 3,
) -> List[Tuple[str, float]]:
    """Predict one image and return Top-K results [(name, prob), ...]."""
    tfm = build_transform()
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)  # [1,3,224,224]

    logits = model(x)                   # [1, num_classes]
    probs = F.softmax(logits, dim=1)[0] # [num_classes]

    k = min(topk, probs.numel())
    vals, idxs = torch.topk(probs, k=k)

    results = []
    for p, i in zip(vals.tolist(), idxs.tolist()):
        results.append((class_names[i], float(p)))
    return results


def main():
    # Suppress torch FutureWarning about default weights_only change (does not affect results)
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the image to classify")
    parser.add_argument("--ckpt", type=str, default="results/brand_classifier/best.pt", help="Model checkpoint best.pt")
    parser.add_argument("--labels", type=str, default="results/brand_classifier/labels.json", help="labels.json path")
    parser.add_argument("--topk", type=int, default=3, help="Output Top-K predictions")
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda (default: auto)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"labels.json not found: {args.labels}")

    class_names = load_labels(args.labels)
    num_classes = len(class_names)

    # Architecture must match training
    model = BrandClassifier(num_classes=num_classes).to(device).eval()

    state_dict = load_checkpoint(args.ckpt, device)
    model.load_state_dict(state_dict, strict=True)

    results = predict_one(args.image, model, class_names, device, topk=args.topk)

    print(f"\nImage: {args.image}")
    print("Top-K predictions:")
    for rank, (name, prob) in enumerate(results, start=1):
        print(f"{rank:>2d}) {name:<24s}  prob={prob:.4f}")


if __name__ == "__main__":
    main()
