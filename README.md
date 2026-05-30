# 🔬 Polyp Segmentation: U-Net vs SegFormer on Kvasir-SEG

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning project comparing a **CNN-based U-Net** (ResNet50 encoder + scSE attention) against a **Vision Transformer-based SegFormer-B2** for binary polyp segmentation on the Kvasir-SEG colonoscopy dataset. Trained on Google Colab (T4 GPU), 40 epochs each, with full EDA, augmentation, quantitative evaluation, and failure analysis.

---

## 📋 Table of Contents

- [Results](#-results)
- [Dataset](#-dataset)
- [Models](#-models)
- [Methodology](#-methodology)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Key Findings](#-key-findings)
- [Technologies](#-technologies)

---

## 📊 Results

| Metric | U-Net (ResNet50) | SegFormer-B2 | Winner |
|--------|-----------------|--------------|--------|
| **IoU** | 0.7903 | **0.8448** | SegFormer +5.4% |
| **Dice** | 0.8828 | **0.9159** | SegFormer +3.3% |
| **Precision** | 0.9000 | **0.9305** | SegFormer +3.1% |
| **Recall** | 0.8664 | **0.9017** | SegFormer +3.5% |
| **F2-Score** | 0.8729 | **0.9073** | SegFormer +3.4% |
| **Specificity** | 0.9817 | **0.9872** | SegFormer +0.6% |
| Parameters | 33.8M | 27.3M | SegFormer (smaller!) |
| Training Time | ~20 min | ~30 min | U-Net |

> **SegFormer-B2 outperforms U-Net across all six metrics** while using ~19% fewer parameters, demonstrating the advantage of global self-attention for medical image segmentation.

---

## 🗂 Dataset

**Kvasir-SEG** — 1,000 colonoscopy images with pixel-level polyp masks from Simula Research Laboratory.

| Split | Images |
|-------|--------|
| Train | 700 (70%) |
| Validation | 150 (15%) |
| Test | 150 (15%) |

Key dataset statistics (from EDA):
- Image dimensions: 332–1920 px wide, 352–1072 px tall
- Polyp coverage: 0.5%–81.2%, mean 15.4%
- Size distribution: Tiny (<5%) · Small (5–15%) · Medium (15–30%) · Large (>30%)

The dataset downloads automatically (~168 MB) from Simula's servers when the notebook is run.

---

## 🧠 Models

### Model 1 — U-Net with ResNet50 + scSE Attention
- **Architecture:** Encoder-decoder with skip connections (Ronneberger et al., 2015)
- **Encoder:** ResNet50 pretrained on ImageNet-1k
- **Decoder:** Progressive upsampling with squeeze-and-channel excitation (scSE) attention
- **Parameters:** 33.8M (all trainable)
- **Library:** `segmentation-models-pytorch`

### Model 2 — SegFormer-B2
- **Architecture:** Hierarchical Vision Transformer with lightweight MLP decoder (Xie et al., 2021)
- **Encoder:** Mix Transformer B2 (MiT-B2) pretrained on ImageNet-1k
- **Decoder:** Linear MLP fusion across 4 stages
- **Parameters:** 27.3M (all trainable)
- **Source:** `nvidia/mit-b2` via HuggingFace Transformers

---

## ⚙️ Methodology

### Loss Function
**Combined Dice + BCE Loss** (α = 0.5):
- Dice loss handles class imbalance (polyps occupy ~15% of pixels on average)
- BCE provides stable pixel-level gradients

### Training Setup

| Setting | U-Net | SegFormer |
|---------|-------|-----------|
| Optimiser | AdamW | AdamW (layer-wise LR) |
| Learning Rate | 3e-4 | 6e-5 (encoder) / 6e-4 (decoder) |
| Scheduler | Cosine Annealing | Cosine Annealing |
| Mixed Precision | ✅ AMP (FP16) | ✅ AMP (FP16) |
| Gradient Clipping | max_norm=1.0 | max_norm=1.0 |
| Epochs | 40 | 40 |
| Batch Size | 8 | 8 |
| Image Size | 352×352 | 352×352 |

### Augmentation Pipeline (Training)
Horizontal flip · Vertical flip · RandomRotate90 · ShiftScaleRotate · Colour jitter (brightness/contrast/HSV/CLAHE) · Gaussian noise/blur/motion blur · CoarseDropout · ImageNet normalisation

### Evaluation Metrics
IoU (Jaccard) · Dice (F1) · Precision · Recall · F2-Score · Specificity

---

## 🚀 How to Run

**Environment:** Google Colab with T4 GPU
**Estimated runtime:** ~45–60 minutes

```
1. Open the notebook in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells (Runtime → Run all)
4. Dataset downloads automatically in Section 2
5. All figures saved to /content/outputs/
```

No manual downloads required. All dependencies install automatically in Section 1.

### Dependencies (auto-installed)
```
torch  torchvision  transformers  segmentation-models-pytorch
albumentations  torchmetrics==0.11.4  timm  scikit-learn
matplotlib  seaborn  Pillow  pandas  numpy
```

---

## 📁 Project Structure

```
polyp-segmentation/
├── 3531966_ITNPAI1_Assignment_PolySegmentation.ipynb   # Main notebook
├── README.md
└── outputs/                    # Generated after running
    ├── fig1_dataset_overview.png
    ├── fig2_dataset_statistics.png
    ├── fig3_augmentation_examples.png
    ├── fig4_training_curves.png
    ├── fig5_metric_comparison.png
    ├── fig6_qualitative_results.png
    ├── fig7_failure_analysis.png
    ├── fig8_summary_table.png
    ├── unet_best.pth
    └── segformer_best.pth
```

---

## 🔍 Key Findings

1. **SegFormer-B2 outperforms U-Net on all metrics** — global self-attention captures long-range context that CNN skip connections cannot, which is critical for irregularly shaped polyps.

2. **Both models struggle with tiny polyps** (coverage < 2%) — U-Net failure cases cluster at coverage ~1.5%, with Dice scores near 0. SegFormer shows more graceful degradation (min Dice 0.007 vs U-Net's 0.000).

3. **SegFormer trains faster in terms of loss convergence** — it achieves 0.83 val Dice at epoch 1 vs U-Net's 0.67, reflecting the benefit of transformer pretraining on large-scale data.

4. **SegFormer is surprisingly compact** — with 27.3M parameters vs U-Net's 33.8M, it is both better and smaller, making it more suitable for deployment.

5. **Combined Dice+BCE loss was effective** — the class-imbalanced nature of polyp segmentation (mean coverage ~15%) was handled well without additional weighting tricks.

---

## 🛠 Technologies

`Python` · `PyTorch` · `HuggingFace Transformers` · `segmentation-models-pytorch` · `Albumentations` · `torchmetrics` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Google Colab`

---

## 📚 References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.
- Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., & Luo, P. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. *NeurIPS*.
- Jha, D., et al. (2020). Kvasir-SEG: A Segmented Polyp Dataset. *MMM Conference*.
- Roy, A.G., et al. (2018). Recalibrating Fully Convolutional Networks with Spatial and Channel 'Squeeze & Excitation' Blocks. *IEEE TMI*.

---

*University of Stirling — ITNPAI1 Deep Learning Assignment 2*
