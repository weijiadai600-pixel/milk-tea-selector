# model/brand_classifier.py
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class BrandClassifier(nn.Module):
    """
    ViT-based brand classifier.

    What we train:
    - By default, we freeze ViT backbone and train only the classification head.
      This is fast and stable for small datasets.

    You can also fine-tune:
    - Unfreeze backbone after a few epochs to improve accuracy (optional).
    """

    def __init__(self, num_classes: int, freeze_backbone: bool = True):
        super().__init__()

        # Use official pretrained weights (avoid deprecated 'pretrained=True' warning)
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.backbone = vit_b_16(weights=weights)

        in_dim = self.backbone.heads.head.in_features  # usually 768

        # Replace classification head
        self.backbone.heads = nn.Identity()

        self.head = nn.Linear(in_dim, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, 224, 224]
        return logits: [B, num_classes]
        """
        feat = self.backbone(x)      # [B, 768]
        logits = self.head(feat)     # [B, C]
        return logits

    def set_backbone_trainable(self, trainable: bool) -> None:
        """
        Enable/disable training for backbone parameters.
        Useful for staged fine-tuning.
        """
        for p in self.backbone.parameters():
            p.requires_grad = trainable
