# retrieval/search.py
# ------------------------------------------------------------
# Command-line version of milk tea image retrieval
#   - If the query image belongs to the database (same image),
#     exclude it to avoid top1 always being itself
#   - Reuse utility functions from similarity.py (do not duplicate logic)
# ------------------------------------------------------------

import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

# ✅[Update] More robust import: ensure similarity.py can be found
# regardless of the working directory (e.g., final_project/retrieval or final_project/scripts)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from retrieval.similarity import (
    load_embeddings,
    build_hash_index,
    find_self_index_by_hash_or_path,
    retrieve_top_k,
)


def load_vit_model(device: str = "cpu"):
    """
    Load the same ViT model as extract_features.py (feature extraction only)
    ✅[Update] Use ViT_B_16_Weights to avoid deprecated pretrained warnings (more standard)
    """
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)

    # Remove classification head, keep backbone feature vector output [768]
    model.heads = torch.nn.Identity()

    model.eval()
    model.to(device)
    return model


def extract_query_feature(image_path: str, model, device: str = "cpu") -> torch.Tensor:
    """
    Extract ViT features for the query image, return [D]
    ✅ Note: Normalize must match the one used during feature extraction
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ✅ recommended to match extract_features.py (ImageNet)
            std=[0.229, 0.224, 0.225],
        )
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)  # [1,3,224,224]

    with torch.no_grad():
        feat = model(img)  # [1,768]

    return feat.squeeze(0)  # [768]


def main():
    parser = argparse.ArgumentParser(description="Milk Tea Image Retrieval (CLI)")
    parser.add_argument("--query", type=str, required=True, help="Query image path")
    parser.add_argument("--embedding", type=str, default="../data/embeddings/milktea_vit_embeddings.pt",
                        help="Embedding file path")
    parser.add_argument("--topk", type=int, default=5, help="Return top K results")
    parser.add_argument("--no_exclude_self", action="store_true",
                        help="If set, do not exclude the query itself (excluded by default)")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load database embeddings
    embeddings, paths, labels = load_embeddings(args.embedding)

    # 2) Load ViT model
    model = load_vit_model(device)

    # 3) Extract query feature
    query_feat = extract_query_feature(args.query, model, device).detach().cpu()
    embeddings_cpu = embeddings.detach().cpu()

    # 4) ✅ Check if query belongs to the database; if yes, exclude it
    exclude_index = None
    if not args.no_exclude_self:
        # Build hash index (md5 -> index)
        # For CLI, building it once is fine (dataset size is not huge)
        hash2idx = build_hash_index(paths, PROJECT_ROOT)

        exclude_index = find_self_index_by_hash_or_path(
            query_path=args.query,
            paths=paths,
            project_root=PROJECT_ROOT,
            hash2idx=hash2idx,
        )

    # 5) TopK retrieval (internally sets exclude_index score to -1e9)
    results = retrieve_top_k(
        query_feat=query_feat,
        embeddings=embeddings_cpu,
        paths=paths,
        labels=labels,
        top_k=args.topk,
        exclude_index=exclude_index,
    )

    # 6) Print results
    print(f"\nQuery image: {args.query}")
    if exclude_index is not None:
        print(f"✅ Excluded query itself (database index={exclude_index})")
    else:
        print("ℹ️ Query is not in the database (or exclusion disabled), no need to exclude itself")
    print(f"\nTop {args.topk} most similar milk tea images:\n")

    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['path']}")
        print(f"    Brand: {r['label']}")
        print(f"    Similarity: {r['score']:.4f}\n")


if __name__ == "__main__":
    main()
