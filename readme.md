# Milk Tea Selector：交互式奶茶检索推荐系统（不付款版）

 本项目为《深度学习实践》课程大作业，实现了一个完整的交互式系统：

- **上传奶茶图片 → 检索相似奶茶（Top-K）**
- **Human-in-the-loop**：点击“❤️ 我更喜欢这杯”后更新偏好向量，多轮迭代推荐
- 额外训练 **品牌分类器（Brand Classifier）**：在 Web 页面显示预测品牌与置信度，并可自动作为筛选条件

---

## 1. 课程要求对应实现

 (1) 深度学习 CV：ViT 特征提取
使用 torchvision 的 **ViT-B/16**（ImageNet 预训练）作为 backbone，移除分类头，仅作为 encoder 输出 **768 维特征**。

 (2) 推荐/检索：余弦相似度 Top-K
基于 ViT 特征向量做 **Cosine Similarity**，实现 Top-K 图像检索；支持品牌筛选、价格区间筛选（假价格）。

 (3) 交互式偏好更新
用户点击“❤️ 我更喜欢这杯”后按公式更新偏好向量，实现多轮交互推荐：

 (4) 训练模型：Brand Classifier
在 `data/images/` 数据集上训练品牌分类器，训练日志、曲线图与模型权重统一保存到 `results/brand_classifier/`。


## 2. 项目结构

final_project/
├── data/
│ ├── images/ # ImageFolder: data/images/<brand>/*.jpg
│ │ ├── ai_gen/
│ │ ├── AunteaJenny/
│ │ └── ...
│ └── embeddings/
│ └── milktea_vit_embeddings.pt # ViT 特征库（由 extract_features.py 生成）
│
├── retrieval/
│ ├── similarity.py # embeddings 加载 + TopK 检索函数
│ └── search.py # CLI 检索入口
│
├── model/
│ ├── vit_encoder.py
│ └── brand_classifier.py # BrandClassifier 模型定义
│
├── train/
│ └── train_brand_classifier.py # 训练品牌分类器（输出到 results/）
│
├── scripts/
│ ├── extract_features.py # 提取检索 embeddings
│ ├── plot_train_log.py # 训练曲线保存（loss/acc）-> results/
│ └── inference_brand_classifier.py # 单张图片推理品牌
│
├── results/
│ └── brand_classifier/
│ ├── best.pt # val 最优权重
│ ├── last.pt # 最后一轮权重
│ ├── labels.json # 类别映射（Web 推理也使用）
│ ├── train_log.csv # 每 epoch 的 loss/acc
│ ├── loss_curve.png # loss 曲线（脚本生成）
│ └── acc_curve.png # acc 曲线（脚本生成）
│
└── web/
└── app.py # Streamlit 主程序，推荐系统 UI


## 3. 环境配置（可复现运行）

推荐 conda（Python 3.9）：
conda create -n pracdl python=3.9 -y
conda activate pracdl

安装依赖（推荐 requirements.txt）：
cd ~/Practical_Deep_Learning/final_project
pip install -U pip
pip install -r requirements.txt

若没有 requirements.txt，可手动安装，不推荐，但也能跑：
pip install torch torchvision torchaudio
pip install streamlit pillow tqdm matplotlib pandas


## 4. Quick Start（一键复现核心结果）
复现流程：提取特征库 → 训练分类器 → 画训练曲线 → 启动 Web

在项目根目录运行：
conda activate pracdl
cd ~/Practical_Deep_Learning/final_project

 1) 提取检索数据库 ViT embeddings（生成 data/embeddings/milktea_vit_embeddings.pt）
python scripts/extract_features.py \
  --data_dir data/images \
  --out_path data/embeddings/milktea_vit_embeddings.pt

 2) 训练 Brand Classifier（输出到 results/brand_classifier）
python train/train_brand_classifier.py \
  --data_dir data/images \
  --out_dir results/brand_classifier \
  --epochs 10 \
  --batch_size 32 \
  --lr 3e-4

 冻结/解冻策略：只训练分类头
 python train/train_brand_classifier.py ... --freeze_backbone
 python train/train_brand_classifier.py ... --freeze_backbone --unfreeze_after 5

 3) 绘制训练曲线（保存到 results/brand_classifier/loss_curve.png, acc_curve.png）
python scripts/plot_train_log.py --log_csv results/brand_classifier/train_log.csv

 4) 启动 Web 应用(路径原因，必须到web文件夹下运行app.py)
cd web
streamlit run app.py


## 5. 核心功能说明
5.1 特征库生成（用于检索推荐）
python scripts/extract_features.py \
  --data_dir data/images \
  --out_path data/embeddings/milktea_vit_embeddings.pt

输出文件：
data/embeddings/milktea_vit_embeddings.pt

5.2 训练 Brand Classifier（训练模型）
python train/train_brand_classifier.py \
  --data_dir data/images \
  --out_dir results/brand_classifier \
  --epochs 10 \
  --batch_size 32 \
  --lr 3e-4

训练输出：
 results/brand_classifier/best.pt
 results/brand_classifier/last.pt
 results/brand_classifier/train_log.csv
 results/brand_classifier/labels.json

5.3 绘制训练曲线（loss/acc）
WSL/服务器无图形界面时，脚本会将图片直接保存到 results：
python scripts/plot_train_log.py --log_csv results/brand_classifier/train_log.csv

输出：
 results/brand_classifier/loss_curve.png
 results/brand_classifier/acc_curve.png

5.4 推理：预测单张图片品牌
python scripts/inference_brand_classifier.py \
  --image data/images/GongCha/GongCha_007.jpg \
  --ckpt results/brand_classifier/best.pt \
  --labels results/brand_classifier/labels.json \
  --topk 3
输出包括：
Top-1 品牌预测
置信度
Top-K 列表

5.5 Web 推荐系统（主程序）
cd web
streamlit run app.py

Web 功能：
上传奶茶图片
展示预测品牌 + 置信度（Brand Classifier，Top-1 + Top-3）
推荐相似奶茶 Top-K（余弦相似度检索）
用戶點擊“❤️ 我更喜欢这杯” ，則多轮推荐（偏好向量更新）
购物车功能（不付款，仅模拟点单体验）


## 6. Results关键结果保存
所有训练/评估输出统一保存到：
results/brand_classifier/

包含：
best.pt：最佳模型权重（按验证集指标保存）
last.pt：最后一轮权重
train_log.csv：训练过程记录（loss/acc）
labels.json：类别映射
loss_curve.png：loss 曲线图
acc_curve.png：acc 曲线图


## 7. 数据组织规范
数据必须按 ImageFolder 形式组织：
data/images/<brand_name>/*.jpg

例如：
data/images/GongCha/GongCha_001.jpg
data/images/MixueBingcheng/MixueBingcheng_001.jpg
data/images/ai_gen/ai_gen_001.jpg

新增类别后，要让 Web 的品牌筛选里出现该类，必须：
首先重新提取 embeddings（extract_features.py），然後重新训练 brand classifier（train_brand_classifier.py）
因为 Web 的筛选项来自 embeddings 文件里保存的 labels。


## 8. 常见问题 FAQ
Q1：FileNotFoundError（找不到 data/images 或 embeddings）
请确保在项目根目录运行，或传入正确路径：
cd ~/Practical_Deep_Learning/final_project
python scripts/extract_features.py --data_dir data/images --out_path data/embeddings/milktea_vit_embeddings.pt

Q2：torch.load weights_only=False 的 FutureWarning
PyTorch 安全提示，不影响运行。
若要消除 warning，可改为：
torch.load(path, map_location="cpu", weights_only=True)

Q3：Streamlit 提示 use_container_width 将弃用
可以把：
st.image(img, use_container_width=True)
替换为：
st.image(img, width="stretch")
warning 不影响运行。


## 9. 说明
 本项目仅为课程作业用途，包含检索推荐系统 + 品牌分类模型训练与推理，UI 为“模拟点单体验（不付款版）”。











