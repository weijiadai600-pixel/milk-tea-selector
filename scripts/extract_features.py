import os
import torch
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
from tqdm import tqdm

# ===============================
# 1. Basic configuration (use absolute paths for robustness)
# ===============================

# __file__ is the path of this extract_features.py
# os.path.dirname(__file__) -> scripts/ directory
# then join("..") -> back to final_project/ root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Image root directory: final_project/data/images
IMAGE_ROOT = os.path.join(PROJECT_ROOT, "data", "images")

# Feature save path: final_project/data/embeddings/milktea_vit_embeddings.pt
SAVE_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings", "milktea_vit_embeddings.pt")

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (Optional) Print paths at startup to verify correctness
print("PROJECT_ROOT =", PROJECT_ROOT)
print("IMAGE_ROOT   =", IMAGE_ROOT)
print("SAVE_PATH    =", SAVE_PATH)


# ===============================
# 2. Image preprocessing (ViT standard)
# ===============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT requires a fixed input size
    transforms.ToTensor(),          # PIL Image -> Tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # ImageNet mean
        std=[0.229, 0.224, 0.225]
    )
])

# ===============================
# 3. Load pretrained ViT model
# ===============================

# Use ImageNet pretrained weights
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights)

# ⚠️ Key: remove the classification head, keep only the feature extractor
model.heads = torch.nn.Identity()

model = model.to(DEVICE)
model.eval()  # inference mode (no training)

# ===============================
# 4. Traverse all images
# ===============================

image_paths = []   # path for each image
labels = []        # brand name (string)

for brand in sorted(os.listdir(IMAGE_ROOT)):
    brand_dir = os.path.join(IMAGE_ROOT, brand)
    if not os.path.isdir(brand_dir):
        continue

    for fname in os.listdir(brand_dir):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_paths.append(os.path.join(brand_dir, fname))
            labels.append(brand)

print(f"Found {len(image_paths)} milk tea images in total")

# ===============================
# 5. Extract features
# ===============================

features = []

with torch.no_grad():  # disable gradient computation (less memory + faster)
    for img_path in tqdm(image_paths):
        # Load image
        img = Image.open(img_path).convert("RGB")

        # Preprocess
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        # unsqueeze(0): [1, 3, 224, 224]

        # Forward pass to get feature vector
        feat = model(img_tensor)
        # feat shape: [1, 768]

        features.append(feat.cpu())

# Concatenate into one large tensor
features = torch.cat(features, dim=0)  # [N, 768]

# ===============================
# 6. Save results
# ===============================

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

torch.save({
    "features": features,   # Tensor [N, 768]
    "paths": image_paths,   # image paths
    "labels": labels        # brand names
}, SAVE_PATH)

print(f"Features saved to {SAVE_PATH}")
