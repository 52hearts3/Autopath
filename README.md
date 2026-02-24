# AutoPath
![License: MIT](images/license.svg)
**AutoPath** is a groundbreaking cross-modal generative framework that enables noninvasive synthesis of histologically plausible H&E images directly from prostate multiparametric MRI (mpMRI). By bridging the gap between imaging and histopathology, it provides interpretable pathological insights without the need for invasive prostate biopsy, pioneering a *"virtual histopathology"* paradigm for prostate cancer diagnosis and grading.


![technical route](images/technical_route.png)
## 📌 Core Overview

Prostate cancer diagnosis currently relies on invasive biopsy for definitive histopathological evaluation. **AutoPath** addresses this critical limitation by generating high-fidelity H&E images and pathological semantic representations from standard MRI sequences (T2WI, ADC, DWI). The framework integrates multimodal MRI encoding, autoregressive semantic generation, billion-scale cell database retrieval, and conditional diffusion reconstruction to achieve robust statistical mapping between MRI and pathology—bypassing long-standing pixel-wise registration challenges.

