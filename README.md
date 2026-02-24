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



