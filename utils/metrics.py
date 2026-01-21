# utils/metrics.py
import torch


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute top-1 accuracy.

    logits: [B, C]
    targets: [B]
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / max(1, targets.numel())
