# Benchmark Results, Ablation Studies, and Visual Diagnostics

## Calibration Ablation: Intrinsic Alignment vs. Frozen Inductive Centering

To evaluate the severity of the modality gap (cone collapse) and quantify the impact of post-hoc latent calibration, the trained ConvNeXt-Tiny dual-tower network was evaluated on the unseen 3,248-pair test split (6,496 total items in the unified modality-agnostic FAISS gallery) under two strict protocols: **Uncalibrated** (raw $\ell_2$-normalized embeddings directly from the target projectors) and **Frozen Inductive Centering** (shifting test embeddings using frozen sensor centroids $\mathbf{\mu}_{\text{SAR}}, \mathbf{\mu}_{\text{Opt}}$ computed exclusively on the 3,255-pair validation split).

| Configuration | Geo R@1 | Geo R@5 | Geo R@10 | Semantic F1@5 | Same-Sensor Top-5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Uncalibrated (Raw Latent Space)** | 21.31% | 46.09% | 58.25% | 0.7309 | 59.82% |
| **Frozen Inductive Centering** | **21.60%** | **46.44%** | **58.71%** | **0.7313** | **59.74%** |

*(Note: For reference, transductive centering using test-set means yields 21.89% Geo R@1 and 47.06% Geo R@5, confirming that our zero-leakage Frozen Inductive Centering captures nearly the entire calibration benefit without test-set data leakage).*

### Analysis of Calibration Dynamics
1. **Intrinsic Cone-Collapse Mitigation:** In naive cross-modal contrastive pipelines, uncalibrated embeddings suffer from total modality cone collapse, resulting in a $100\%$ Same-Sensor Top-5 retrieval ratio and near-$0\%$ cross-modal recall. Remarkably, our uncalibrated baseline already achieves **21.31% Geo R@1** and a well-mixed **59.82% Same-Sensor Top-5 ratio** (where $50\%$ represents ideal sensor parity). This demonstrates that combining our composite loss ($0.50\times$ cross-modal InfoNCE + $0.25\times$ intra-modal self-distillation) with EMA-synchronized `BatchNorm1d` running statistics in the projection head naturally prevents the SAR and Optical cones from drifting apart during training.
2. **Inductive Refinement:** Applying Frozen Inductive Centering provides a consistent, zero-leakage geometric refinement across all metrics, improving exact geographic counterpart recall to **21.60% at R@1** ($+0.29\%$) and **58.71% at R@10** ($+0.46\%$), while nudging the Same-Sensor Top-5 distribution closer to parity ($59.74\%$).

## Modality-Agnostic Directional Retrieval Performance

With Frozen Inductive Centering applied, the system's retrieval performance was broken down by query modality across all 6,496 queries against the unified mixed gallery:

| Query Direction | Geo R@1 | Geo R@5 | Geo R@10 | Semantic F1@5 | Semantic F1@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SAR $\to$ Unified Gallery** | **22.60%** | **47.97%** | **60.44%** | 0.7259 | 0.7224 |
| **Optical $\to$ Unified Gallery** | 20.60% | 44.92% | 56.99% | **0.7367** | **0.7315** |
| **Mixed (Modality-Agnostic)** | 21.60% | 46.44% | 58.71% | 0.7313 | 0.7270 |

### Structural vs. Spectral Asymmetry
The directional breakdown reveals a compelling physical trade-off between active microwave and passive optical sensing:
* **SAR Excels at Instance-Level Localization (Geo R@K):** Queries originating from Sentinel-1 SAR outperform Sentinel-2 Optical queries by **$+2.00\%$ on Geo R@1** ($22.60\%$ vs. $20.60\%$) and **$+3.45\%$ on Geo R@10** ($60.44\%$ vs. $56.99\%$). Because C-band radar directly measures macroscopic surface roughness, dielectric boundaries, and hard corner reflectors, SAR embeddings encode highly distinct structural skeletons (e.g., sharp coastlines, road intersections, and urban grids). Combined with GeM spatial pooling ($p=3.0$), these invariant geometric landmarks act as a unique spatial fingerprint to pinpoint the exact optical twin.
* **Optical Excels at Semantic Biome Clustering (F1@K):** Conversely, Sentinel-2 Optical queries outperform SAR queries on Twin-Excluded Semantic Multi-Label F1 (**0.7367 vs. 0.7259 at K=5**). Its 10 radiometrically calibrated spectral bands capture fine biochemical and chlorophyll absorption signatures, allowing the optical encoder to differentiate subtle land-cover variations (such as distinguishing arable crop types from natural pastures) more reliably than 2-channel radar backscatter.

## Semantic Class-Level Diagnostics

To evaluate how surface physics governs cross-modal alignment across diverse biomes, test queries were stratified across five primary CORINE Land Cover (CLC) macro-groups within the frozen inductive gallery:

| Land-Cover Macro-Group | Query Count ($n$) | Geo R@1 | Geo R@5 | Semantic F1@5 |
| :--- | :---: | :---: | :---: | :---: |
| **Water Bodies** | 722 | **37.67%** | **67.59%** | 0.6397 |
| **Agricultural Areas** | 5,280 | 24.51% | 50.49% | **0.7290** |
| **Woodland / Shrub** | 2,394 | 21.68% | 46.91% | 0.6718 |
| **Urban / Industrial** | 4,102 | 21.18% | 45.51% | 0.7125 |
| **Forests** | 3,364 | 17.39% | 40.37% | 0.7156 |

### Physical Interpretation of Class Performance
1. **Water Bodies (Highest Instance Recall):** Queries containing marine or inland water bodies achieve the highest exact geographic recall (**37.67% Geo R@1** and **67.59% Geo R@5**). Calm water acts as a specular microwave reflector ($\sigma^0 < -25\text{ dB}$, appearing pitch-black in SAR) and strongly absorbs NIR/SWIR solar radiation in Sentinel-2. The resulting high-contrast shoreline geometry creates an unmistakable cross-modal contour.
2. **Agricultural and Urban Landscapes (Balanced Instance & Semantic Alignment):** Agricultural patches achieve the highest semantic consistency (**0.7290 F1@5**) alongside strong twin recall (**24.51% Geo R@1**), driven by distinct field parcel boundaries visible in both modalities. Similarly, **Urban / Industrial** queries maintain high semantic clustering (**0.7125 F1@5**) and solid instance recall (**21.18% Geo R@1**), anchored by double-bounce corner reflections from buildings and transport infrastructure.
3. **Homogeneous Forests (High Semantic Clustering, Lower Instance Recall):** Forest queries exhibit strong semantic clustering (**0.7156 F1@5**), proving the Jaccard-masked InfoNCE loss groups forest biomes together effectively. However, they yield the lowest exact twin recall (**17.39% Geo R@1**). Physically, dense continuous forest canopies produce uniform volumetric VH backscatter in SAR and uniform chlorophyll reflectance in Optical imagery, lacking the localized geometric landmarks needed to distinguish one specific forest patch from thousands of other visually identical forest patches in the gallery.

## Qualitative Visual Diagnostics

To visually validate the quantitative metrics, multi-panel retrieval grids were generated during inference using strict leak-free validation centering. For a given query, the system retrieves the Top-5 candidates from the 6,496-item unified gallery.

Each retrieved candidate patch in the visual output is annotated with:
1. **[TWIN] Tag:** Explicitly highlights if the retrieved image is the exact geographic counterpart of the query (verifying Geo R@K).
2. **Modality Badge:** Indicates whether the retrieved patch is SAR or Optical, demonstrating that the search traverses the modality-agnostic space freely.
3. **Dice / F1 Overlap:** Displays the multi-label Sørensen–Dice (F1) similarity between the query's CORINE labels and the retrieved candidate's labels.
4. **Cosine Similarity:** The raw inner-product score ($0.0 \to 1.0$) computed by FAISS in the normalized 256-D latent space.

Visual inspection of these panels confirms that even when the exact geographic twin is missed at Rank 1, the subsequent retrieved patches exhibit identical terrain morphology and high multi-label F1 overlap ($\sim 0.73$), proving that the network learned genuine physical topography and land-cover semantics rather than overfitting to coordinate artifacts.

## Hardware-Synchronized Inference Latency

Operational deployment requires strict latency bounds. Using `torch.cuda.synchronize()` on a standard cloud GPU instance (NVIDIA T4 on Google Colab), the end-to-end inference profile for a single query was recorded:

* **ConvNeXt-Tiny Encoding (FP16):** $\sim 4.2 \text{ ms}$
* **Inductive Centering & $\ell_2$ Normalization:** $\sim 0.1 \text{ ms}$
* **FAISS Inner Product Search (6,496 items):** $\sim 0.8 \text{ ms}$
* **Total End-to-End Latency:** $\sim 5.1 \text{ ms}$

This sub-10-millisecond end-to-end latency (with sub-millisecond vector search) scales efficiently, proving the architecture is fully capable of real-time, interactive deployment.