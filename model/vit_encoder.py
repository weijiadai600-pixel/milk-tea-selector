import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Use the ViT model provided by HuggingFace
from transformers import ViTModel

class ViTEncoder(nn.Module):
    """
    Use a pretrained Vision Transformer to extract image features
    Output is the image embedding (no classification)
    """
    def __init__(self):
        super(ViTEncoder, self).__init__()

        # Load pretrained ViT model (without classification head)
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )

        # Freeze ViT parameters (no training, feature extraction only for stability)
        for param in self.vit.parameters():
            param.requires_grad = False

        # ViT requires fixed input preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),   # ViT fixed input size
            transforms.ToTensor(),            # Convert to Tensor
            transforms.Normalize(             # ImageNet normalization
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def forward(self, image: Image.Image):
        """
        Input: PIL Image
        Output: 768-dimensional image feature vector
        """

        # Image preprocessing
        x = self.transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]

        # ViT forward pass
        outputs = self.vit(pixel_values=x)

        # Use the [CLS] token as the global image representation
        embedding = outputs.last_hidden_state[:, 0, :]  # [1, 768]

        return embedding.squeeze(0)  # [768]
