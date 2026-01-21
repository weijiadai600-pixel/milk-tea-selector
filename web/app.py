# web/app.py
# ------------------------------------------------------------
# Interactive Milk Tea Recommendation System (No Payment Version)
# Key ideas:
# 1) Use ViT to extract image features
# 2) Use cosine similarity for retrieval
# 3) User clicks "I prefer this one" -> update preference vector (Human-in-the-loop)
#
# ✅ Updates in this revision:
# - After uploading an image: automatically display the brand classifier prediction + confidence
# - Fix best.pt loading (support ckpt["model_state"])
# - Add the missing predict_brand_topk function (it was called before but not defined)
# - ✅ Key fix: resolve Streamlit session_state limitation (cannot modify after widget instantiation)
#   Approach: introduce brand_filter_pending and apply it on the next rerun (before widget creation)
# ------------------------------------------------------------

import sys
import os
import hashlib
import json
from typing import List

import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from pathlib import Path


# ============================================================
# 0) Path setup: ensure we can import final_project/*
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Reuse retrieval module
from retrieval.similarity import load_embeddings, build_hash_index, retrieve_top_k  # noqa: E402
from model.brand_classifier import BrandClassifier  # noqa: E402


# ============================================================
# 1) Page settings
# ============================================================
st.set_page_config(page_title="Milk Tea Selector", layout="wide")
st.title("🧋 模拟奶茶挑选系统")

st.write(
    "上传一张奶茶图片，系统会推荐视觉上相似的奶茶。\n\n"
    "进阶玩法：你可以在推荐结果里点 **「我更喜欢这杯」**，系统会根据你的选择继续推荐。\n\n"
    "你不需要真的购买，只是享受沉浸式挑选的过程~"
)
st.latex(r"\mathbf{q}_{t+1} = (1-\alpha)\mathbf{q}_{t} + \alpha \mathbf{f}_{\text{selected}}")


# ============================================================
# 2) Load ViT (only once)
# ============================================================
@st.cache_resource
def load_vit_model():
    """Load ViT-B/16 feature extractor (remove classification head)."""
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    model.heads = torch.nn.Identity()
    model.eval()
    return model


# ============================================================
# 3) Load database embeddings (only once)
# ============================================================
@st.cache_resource
def load_database():
    embeddings, paths, labels = load_embeddings("../data/embeddings/milktea_vit_embeddings.pt")
    return embeddings, paths, labels


# ============================================================
# 4) Database hash index (build once, with caching)
# ============================================================
@st.cache_resource
def load_db_hash_index(paths: List[str], project_root: str):
    """Build md5(bytes) -> index mapping (used to exclude "self")."""
    return build_hash_index(paths, project_root)


# ============================================================
# 5) ViT feature extraction: PIL -> [768]
# ============================================================
def extract_query_feature(image: Image.Image, model: torch.nn.Module) -> torch.Tensor:
    """
    Note: Use ImageNet mean/std for Normalize (keep consistent with extract_features.py)
    """
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    x = tfm(image).unsqueeze(0)  # [1,3,224,224]
    with torch.no_grad():
        feat = model(x)          # [1,768]
    return feat.squeeze(0)       # [768]


# ============================================================
# 6) "Fake price" (for UI)
# ============================================================
def fake_price_from_path(path: str) -> float:
    h = hashlib.md5(path.encode("utf-8")).hexdigest()
    x = int(h[:8], 16)
    price = 8 + (x % 2100) / 100.0
    return round(price, 2)


# ============================================================
# 7) Brand classifier: load (only once)
# ============================================================
@st.cache_resource
def load_brand_classifier(
    ckpt_path: str = "results/brand_classifier/best.pt",
    labels_path: str = "results/brand_classifier/labels.json",
):
    """
    Returns:
      - clf_model: BrandClassifier (eval)
      - class_names: List[str]  (index -> brand name)
    """
    labels_full = os.path.join(PROJECT_ROOT, labels_path)
    with open(labels_full, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # labels.json compatibility: list / {"0":"A"} / {"A":0}
    if isinstance(obj, list):
        class_names = obj
    elif isinstance(obj, dict) and all(k.isdigit() for k in obj.keys()):
        items = sorted(((int(k), v) for k, v in obj.items()), key=lambda x: x[0])
        class_names = [v for _, v in items]
    elif isinstance(obj, dict) and all(isinstance(v, int) for v in obj.values()):
        inv = {v: k for k, v in obj.items()}
        class_names = [inv[i] for i in sorted(inv.keys())]
    else:
        raise ValueError("Cannot parse labels.json format, please check saved format")

    num_classes = len(class_names)

    clf_model = BrandClassifier(num_classes=num_classes).eval()

    ckpt_full = os.path.join(PROJECT_ROOT, ckpt_path)

    # best.pt is often a packed checkpoint: {"model_state":..., "epoch":...}
    ckpt = torch.load(ckpt_full, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt  # Case where it is directly a state_dict

    clf_model.load_state_dict(state_dict, strict=True)
    return clf_model, class_names


# ============================================================
# 8) ✅ Brand classifier inference: return Top-K
# ============================================================
@torch.no_grad()
def predict_brand_topk(
    image: Image.Image,
    clf_model: torch.nn.Module,
    class_names: List[str],
    topk: int = 3,
):
    """
    Output format:
      [{"name": "GongCha", "prob": 0.87}, ...]
    """
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    x = tfm(image).unsqueeze(0)  # [1,3,224,224]
    logits = clf_model(x)        # [1,C]
    probs = F.softmax(logits, dim=1)[0]  # [C]

    k = min(topk, probs.numel())
    vals, idxs = torch.topk(probs, k=k)

    out = []
    for p, i in zip(vals.tolist(), idxs.tolist()):
        out.append({"name": class_names[i], "prob": float(p)})
    return out


# ============================================================
# 9) SessionState
# ============================================================
def init_state():
    if "base_image" not in st.session_state:
        st.session_state["base_image"] = None
    if "base_feat" not in st.session_state:
        st.session_state["base_feat"] = None
    if "pref_feat" not in st.session_state:
        st.session_state["pref_feat"] = None
    if "cart" not in st.session_state:
        st.session_state["cart"] = []
    if "last_query_source" not in st.session_state:
        st.session_state["last_query_source"] = "初始上传"
    if "upload_hash" not in st.session_state:
        st.session_state["upload_hash"] = None
    if "exclude_index" not in st.session_state:
        st.session_state["exclude_index"] = None
    if "pred_brand" not in st.session_state:
        st.session_state["pred_brand"] = None

    # ✅ Bind multiselect to session_state
    if "brand_filter" not in st.session_state:
        st.session_state["brand_filter"] = []

    # ✅ Whether the user has manually changed brand filter (if yes, do not auto-override)
    if "brand_filter_touched" not in st.session_state:
        st.session_state["brand_filter_touched"] = False

    # ✅ [Key fix] pending: write brand_filter only on next rerun (before widget creation)
    if "brand_filter_pending" not in st.session_state:
        st.session_state["brand_filter_pending"] = None

    if "brand_pred_topk" not in st.session_state:
        st.session_state["brand_pred_topk"] = None  # list[{"name","prob"}]
    if "brand_pred_top1" not in st.session_state:
        st.session_state["brand_pred_top1"] = None  # {"name","prob"}



init_state()


# ============================================================
# 10) Load models & data
# ============================================================
vit_model = load_vit_model()
embeddings, paths, labels = load_database()
embeddings_cpu = embeddings.detach().cpu()

db_hash2idx = load_db_hash_index(paths, PROJECT_ROOT)
all_brands = sorted(list(set(labels)))


# ============================================================
# ✅ Apply brand_filter_pending BEFORE creating sidebar widgets
# Streamlit rule: cannot modify session_state bound to a widget after instantiation
# ============================================================
pending = st.session_state.get("brand_filter_pending", None)
if pending is not None:
    # Only effective if "use predicted brand as default filter" is checked
    if st.session_state.get("auto_use_pred", True):
        # Only override if the user has NOT manually adjusted the filter
        if not st.session_state.get("brand_filter_touched", False):
            if isinstance(pending, str):
                pending_list = [pending]
            else:
                pending_list = list(pending)
            st.session_state["brand_filter"] = pending_list

    # ✅ Clear after applying
    st.session_state["brand_filter_pending"] = None


# ============================================================
# 11) Sidebar
# ============================================================
st.sidebar.header("📋 菜单栏")
topk = st.sidebar.slider("推荐数量 Top-K", 3, 12, 6)
alpha = st.sidebar.slider("偏好融合系数 α（越大越“相信你刚选的”）", 0.05, 0.95, 0.70, 0.05)


def _on_brand_filter_change():
    st.session_state["brand_filter_touched"] = True


auto_use_pred = st.sidebar.checkbox("用预测品牌作为默认筛选", value=True, key="auto_use_pred")

st.sidebar.multiselect(
    "只看这些品牌",
    options=all_brands,
    key="brand_filter",
    on_change=_on_brand_filter_change,
)

brand_filter = st.session_state["brand_filter"]

price_min, price_max = st.sidebar.slider("价格区间（假数据，单位：元）", 8.0, 29.0, (8.0, 29.0), 0.5)

st.sidebar.markdown("---")

if st.sidebar.button("不买了，我只是看看（清空购物车，回到初始）"):
    st.session_state["cart"] = []
    if st.session_state["base_feat"] is not None:
        st.session_state["pref_feat"] = st.session_state["base_feat"].clone()
        st.session_state["last_query_source"] = "初始上传"
    st.rerun()

if st.sidebar.button("全部重新开始（清空所有状态）"):
    for k in ["base_image", "base_feat", "pref_feat", "cart", "last_query_source",
              "upload_hash", "exclude_index", "pred_brand",
              "brand_filter", "brand_filter_touched", "brand_filter_pending"]:
        st.session_state[k] = None if k != "cart" else []
    st.session_state["last_query_source"] = "初始上传"
    st.rerun()


# ============================================================
# 12) Main layout: left recommendations + right cart
# ============================================================
main_col, cart_col = st.columns([2, 1], gap="large")

with main_col:
    uploaded_file = st.file_uploader("📤 请上传一张奶茶图片", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        current_hash = hashlib.md5(file_bytes).hexdigest()

        # Only reset preference when a new image is uploaded
        if st.session_state["upload_hash"] != current_hash:
            st.session_state["upload_hash"] = current_hash

            image = Image.open(uploaded_file).convert("RGB")
            st.session_state["base_image"] = image

            base_feat = extract_query_feature(image, vit_model)
            st.session_state["base_feat"] = base_feat
            st.session_state["pref_feat"] = base_feat.clone()
            st.session_state["last_query_source"] = "初始上传"

            # ========== Brand prediction ==========
            clf_model, class_names = load_brand_classifier()
            brand_preds = predict_brand_topk(image, clf_model, class_names, topk=3)

            top1 = brand_preds[0]
            # ✅ Keep predictions across reruns for stable display
            st.session_state["brand_pred_topk"] = brand_preds
            st.session_state["brand_pred_top1"] = top1

            st.success(f"🧾 预测品牌：**{top1['name']}**（置信度：{top1['prob']:.3f}）")
            st.session_state["pred_brand"] = top1["name"]

            with st.expander("查看 Top-3 品牌预测"):
                for i, r in enumerate(brand_preds, 1):
                    st.write(f"{i}. {r['name']} —— {r['prob']:.3f}")

            # ✅ New image: allow auto strategy to take over again
            # (otherwise touched=True would disable auto forever)
            st.session_state["brand_filter_touched"] = False

            # ✅ Key: do NOT write brand_filter directly (widget already created)
            # Write pending instead; apply it on next rerun BEFORE sidebar is created
            if st.session_state.get("auto_use_pred", True):
                if top1["name"] in all_brands:
                    st.session_state["brand_filter_pending"] = [top1["name"]]
                else:
                    st.session_state["brand_filter_pending"] = []
                st.rerun()

            # ========== Exclude self: md5 first, filename fallback ==========
            exclude_idx = db_hash2idx.get(current_hash, None)
            if exclude_idx is None:
                uploaded_name = uploaded_file.name
                cand = [i for i, p in enumerate(paths) if os.path.basename(p) == uploaded_name]
                if cand:
                    exclude_idx = cand[0]

            st.session_state["exclude_index"] = exclude_idx

    if st.session_state["base_image"] is None:
        st.info("先上传一张奶茶图片开始挑选～")
        st.stop()

    st.subheader(f"你选择的奶茶（当前参考：{st.session_state['last_query_source']}）")
    st.image(st.session_state["base_image"], width=320)
    # ✅ Stable display for predicted brand (won't be cleared by rerun)
    top1 = st.session_state.get("brand_pred_top1")
    topk_list = st.session_state.get("brand_pred_topk")

    if top1 is not None:
        st.info(f"🧾 预测品牌：**{top1['name']}**（置信度：{top1['prob']:.3f}）")

    if topk_list:
        with st.expander("查看 Top-3 品牌预测"):
            for i, r in enumerate(topk_list, 1):
                st.write(f"{i}. {r['name']} —— {r['prob']:.3f}")


    # ------- Filters -------
    allowed_indices: List[int] = []
    for i, (p, lab) in enumerate(zip(paths, labels)):
        if brand_filter and (lab not in brand_filter):
            continue
        pr = fake_price_from_path(p)
        if not (price_min <= pr <= price_max):
            continue
        allowed_indices.append(i)

    if not allowed_indices:
        st.warning("当前筛选条件下没有可推荐图片，放宽品牌/价格范围试试～")
        st.stop()

    # ------- Retrieval -------
    query_feat = st.session_state["pref_feat"].detach().cpu()

    exclude_index = st.session_state["exclude_index"]
    if exclude_index is not None and exclude_index not in allowed_indices:
        exclude_index = None

    sub_embeddings = embeddings_cpu[allowed_indices]
    sub_paths = [paths[i] for i in allowed_indices]
    sub_labels = [labels[i] for i in allowed_indices]

    sub_exclude_index = None
    if exclude_index is not None:
        sub_exclude_index = allowed_indices.index(exclude_index)

    results = retrieve_top_k(
        query_feat=query_feat,
        embeddings=sub_embeddings,
        paths=sub_paths,
        labels=sub_labels,
        top_k=topk,
        exclude_index=sub_exclude_index,
    )

    for r in results:
        r["price"] = fake_price_from_path(r["path"])

    st.subheader("你可能也会想挑这些奶茶（点一下接着推荐）")
    cols = st.columns(min(topk, len(results)))

    for col, r in zip(cols, results):
        with col:
            img_path = os.path.join(PROJECT_ROOT, r["path"])
            if not os.path.exists(img_path):
                st.warning(f"找不到图片：{r['path']}")
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                st.image(img, width='stretch')
            except Exception:
                st.warning(f"图片读取失败：{r['path']}")
                continue

            st.caption(
                f"品牌：{r['label']}\n\n"
                f"价格：￥{r['price']}\n\n"
                f"相似度：{r['score']:.3f}"
            )

            like_btn_key = f"like_{r['path']}"
            if st.button("❤️ 我更喜欢这杯", key=like_btn_key):
                picked_path = r["path"]
                picked_index_global = paths.index(picked_path)

                picked_feat = embeddings_cpu[picked_index_global]
                old_pref = st.session_state["pref_feat"].detach().cpu()
                new_pref = (1 - alpha) * old_pref + alpha * picked_feat

                st.session_state["pref_feat"] = new_pref
                st.session_state["last_query_source"] = f"你点的：{os.path.basename(picked_path)}"
                st.session_state["base_image"] = img
                st.session_state["exclude_index"] = picked_index_global  # Exclude the one you just clicked
                st.rerun()

            add_btn_key = f"cart_{r['path']}"
            if st.button("🛒 加入购物车", key=add_btn_key):
                existed = any(item["path"] == r["path"] for item in st.session_state["cart"])
                if not existed:
                    st.session_state["cart"].append({
                        "path": r["path"],
                        "label": r["label"],
                        "price": r["price"],
                    })
                st.rerun()


with cart_col:
    st.subheader("🛍️ 购物车")
    st.write("你可以把喜欢的奶茶先放这里，但不会跳转支付，只是单纯挑选解压。")

    cart = st.session_state["cart"]
    if not cart:
        st.info("购物车还是空的～去左边挑选几杯放进来吧。")
    else:
        total = sum(item["price"] for item in cart)
        st.metric("购物车总价（假）", f"￥{total:.2f}")

        for i, item in enumerate(cart, start=1):
            st.write(f"[{i}] {item['label']} | ￥{item['price']:.2f}")
            thumb_path = os.path.join(PROJECT_ROOT, item["path"])
            if os.path.exists(thumb_path):
                st.image(Image.open(thumb_path).convert("RGB"), width=180)

        st.markdown("---")
        if st.button("清空购物车"):
            st.session_state["cart"] = []
            st.rerun()
