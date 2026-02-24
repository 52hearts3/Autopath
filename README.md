# AutoPath
![License: MIT](images/license.svg)
**AutoPath** is a groundbreaking cross-modal generative framework that enables noninvasive synthesis of histologically plausible H&E images directly from prostate multiparametric MRI (mpMRI). By bridging the gap between imaging and histopathology, it provides interpretable pathological insights without the need for invasive prostate biopsy, pioneering a *"virtual histopathology"* paradigm for prostate cancer diagnosis and grading.
## 📌 Core Overview

Prostate cancer diagnosis currently relies on invasive biopsy for definitive histopathological evaluation. **AutoPath** addresses this critical limitation by generating high-fidelity H&E images and pathological semantic representations from standard MRI sequences (T2WI, ADC, DWI). The framework integrates multimodal MRI encoding, autoregressive semantic generation, billion-scale cell database retrieval, and conditional diffusion reconstruction to achieve robust statistical mapping between MRI and pathology—bypassing long-standing pixel-wise registration challenges.
## 🔑 Key Features

- **Noninvasive H&E Synthesis**: First framework to directly generate analyzable H&E images from MRI, eliminating biopsy dependency.
- **High Pathological Fidelity**: SSIM 0.8678, PSNR 29.59 dB, and nuclear segmentation Dice 0.8066 (closely aligned with real histopathology).
- **Clinical-Grade Performance**: Quadrant-level diagnosis AUC 0.926, 88.5% csPCa localization hit rate, and GGG classification accuracy 0.8947.
- **Multilevel Interpretability**: Supports cellular segmentation, TME quantification, Gleason grading, and single-cell gene expression inference.

![technical route](images/technical_route.png)
## ⚙️ Workflow Overview

### 1. Input: Multiparametric MRI Data
- **Modalities**: T2-weighted imaging (T2WI), Apparent Diffusion Coefficient (ADC), Diffusion-Weighted Imaging (DWI) — standard clinical MRI sequences.
- **Preprocessing**: Clinical registration, prostate segmentation, quadrant division (upper/lower/left/right), and data augmentation (to capture pathological variability).

### 2. Stage 1: Autoregressive Pathological Semantic Generation
- **Feature Extraction**: Coordinate Attention-enhanced ResNet extracts modality-specific features from MRI, capturing spatial directional information of tumor regions.
- **Semantic Label Generation**: Transformer-based autoregressive model generates structured labels (e.g., AAAA00001) encoding Gleason patterns, tumor aggressiveness, and pathological semantics.
- **Multiple Sampling**: ~2,000 augmented MRI samples per quadrant to ensure statistical representativeness of pathological distributions.

### 3. Stage 2: Billion-Scale Cell Database Retrieval
- **Database Foundation**: Built from 2.64M H&E slices, containing spatial coordinates, morphological attributes, and category labels of 1B+ cells (6 categories: nolabe, neopla, inflam, connec, necros, no-neo).
- **Mask Reconstruction**: Autoregressive labels retrieve matching cell populations from the database, reorganizing them into cell-level spatial masks (structural priors for synthesis).

### 4. Stage 3: Conditional Diffusion H&E Synthesis
- **Model Architecture**: U-Net-based diffusion model with ControlNet branch, Adaptive Group Normalization (AdaGN), and cross-attention mechanisms.
- **Condition Fusion**: Fuses cell masks, MRI feature embeddings, and timestep information to constrain synthesis.
- **Output**: High-resolution (2048×2048) H&E patches with histologically plausible tissue architecture, cellular density, and staining characteristics.

### 5. Stage 4: Pathological Analysis & Clinical Report
- **Downstream Analyses**:
  - *Cellular*: Nuclear segmentation (HoverNet), cell-type composition quantification.
  - *Tissue*: Tumor microenvironment (TME) characterization, spatial heterogeneity assessment.
  - *Clinical*: Gleason grading, csPCa localization, cancer cell differentiation prediction.
  - *Molecular*: Single-cell gene expression inference (via GHIST framework).
- **Report Generation**: Integrates rapid inference from autoregressive labels and fine-grained analysis from synthetic H&E images to output diagnosis, grading, and clinical suggestions.



