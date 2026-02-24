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
![technical route](images/diff.jpg)
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
## 📊 Core Validation Results

**AutoPath** is rigorously validated on **6.98M+ images** from 5 independent multicenter cohorts  
(*Renji, PI-CAI, TCIA, PROSTATE-MRI, CHIMERA*).

| Evaluation Dimension    | Key Metrics                                                                 |
|-------------------------|------------------------------------------------------------------------------|
| **Image Quality**       | SSIM = 0.8678, PSNR = 29.59 dB (82–88% samples meet clinical acceptability) |
| **Nuclear Segmentation**| Dice = 0.8066, Mean Centroid Distance = 9.57 pixels, Classification Accuracy = 0.9001 |
| **Clinical Diagnosis**  | AUC = 0.926, Recall = 0.990, Specificity = 0.880                            |
| **csPCa Localization**  | Quadrant-level hit rate = 88.5%                                              |
| **Cellular Composition**| Similarity = 91.15% (tumor/stromal/inflammatory cells well-matched)         |
| **Gleason Grading**     | GGG Accuracy = 0.8947, Composite Soft Score = 94.28                         |
| **Tumor Microenvironment** | Similarity = 87.86% (tumor purity, immune score, spatial interactions)   |


## 🚀 Quick Start

### 1. Data Processing
![technical route](images/data-process.jpg)
1. Run **Split into four positions.py**  
   Perform quadrant segmentation (upper/lower/left/right) on registered MRI.

2. Run **SVS Processing.py**  
   Extract multiple 2048-pixel H&E image patches.

3. Use **HoverNet**  
   Segment and classify nuclei in H&E images, generating detailed JSON files with cell information.

4. Run **Compress JSON.py**  
   Simplify JSON storage size.

5. Run **Store H5.py**  
   Store processed cell information in HDF5 format.

6. Run **Match Position.py**  
   Match positional information of H&E images, then run **1-csv_creat.py** to generate autoregressive labels (pathology IDs).

7. Run **2-csv_sort.py**  
   Sort pathology IDs, then run **3-csv_text.py** to group CSV files.

8. Run **4-Transfer Column.py** and **5-Rename Column.py**  
   For multiple batches, run **6-Merge Tables.py** to merge tables.

9. Run **Image Enhancement.py**  
   Augment MRI samples to match the number of H&E images.

### 2. Autoregressive Training
![technical route](images/Autoregressive.jpg)
1. **Prepare MRI–Pathology Pairs**  
   Use the preprocessed MRI images (T2, ADC, DWI) together with the corresponding pathology labels (CSV files).

2. **Run Multimodal Training**  
   If you have T2, ADC, and DWI modalities available, run:  
   **multimodal autoregressive model.py**  
   This trains the autoregressive model across multiple modalities, learning joint spatial and semantic features.

3. **Run Unimodal Training**  
   If only T2 images are available, run:  
   **unimodal autoregressive model.py**  
   This trains the autoregressive model using single-modality features to generate pathology semantic labels.

4. **Training Output**  
   - Produces autoregressive pathology label sequences (e.g., `AAAA00001`) encoding Gleason patterns and tumor semantics.  
   - Saves model weights and training logs for downstream database retrieval and conditional diffusion synthesis.

### 3. Conditional Diffusion Model Training

1. **Train the Diffusion Model**  
   Use the cell masks obtained from database retrieval as control conditions.  
   Run **ddpm.py** to train the conditional diffusion model.  
   This step leverages ControlNet and Adaptive Group Normalization (AdaGN) to synthesize histologically plausible H&E patches.

2. **Inference with DDIM**  
   For fast inference and image generation, run **ddim.py**.  
   This performs deterministic sampling to reconstruct high-resolution H&E images from MRI-derived conditions.



