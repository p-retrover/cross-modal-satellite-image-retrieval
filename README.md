# Cross-Modal Satellite Image Retrieval

## Project overview

This project investigates **cross-modal satellite image retrieval**
using paired Sentinel-1 Synthetic Aperture Radar (SAR) and Sentinel-2
optical imagery from the BigEarthNet family of datasets.

The central objective is to learn a shared feature-embedding space in
which images from different sensors can be compared. Given an image from
one modality, the system should retrieve its corresponding or otherwise
relevant images from the other modality using embedding similarity.

**Working research question**

> Can contrastive representation learning align heterogeneous satellite
> imagery into a shared embedding space to enable accurate and
> computationally efficient same-modal and cross-modal image retrieval?

The implementation and experimental choices below describe the current
plan. Items explicitly marked *to be confirmed* will be finalised only
after inspecting the downloaded TIFF data.

------------------------------------------------------------------------

## 1. Dataset currently in use

### Kaggle dataset

-   **Dataset:**
    [BigEarthNet-14K](https://www.kaggle.com/datasets/narendraaironi/bigearthnet-14k)
-   **Kaggle identifier:** `narendraaironi/bigearthnet-14k`
-   **Source type:** Third-party Kaggle dataset upload. It should not be
    described as an official BigEarthNet distribution.
-   **Local Colab extraction root:** `/content/datasets/BEN14K/BEN_14k/`
-   **Metadata file:** `metadata.parquet`

The dataset contains paired Sentinel-1 and Sentinel-2 TIFF files
arranged into predefined train, validation, and test directories.

### Verified image counts

The extracted directory structure has been checked against the expected
file counts. All six modality/split counts match.

  Modality            Train   Validation    Test    Total
  ----------------- ------- ------------ ------- --------
  Sentinel-1 (S1)     7,180        3,255   3,248   13,683
  Sentinel-2 (S2)     7,180        3,255   3,248   13,683

This gives **13,683 Sentinel-1 / Sentinel-2 pairs**, or **27,366 TIFF
images** in total, plus one Parquet metadata file.

The metadata file contains 480,038 rows because it describes a broader
collection than the image subset included in this Kaggle upload. Only
metadata records that match the downloaded image files should be used in
the project.

### Metadata fields

The Parquet file has the following columns:

-   `patch_id`: full Sentinel-2 patch identifier; use this to match S2
    TIFF filenames.
-   `s1_name`: Sentinel-1 image identifier; use this to match S1 TIFF
    filenames.
-   `s2v1_name`: shortened S2 identifier. It is **not** the correct
    exact filename key for the downloaded S2 TIFFs.
-   `labels`: multi-label land-cover annotations.
-   `split`: predefined train, validation, or test assignment.
-   `country`: country metadata.
-   `contains_seasonal_snow`: seasonal-snow quality/context flag.
-   `contains_cloud_or_shadow`: cloud/shadow quality flag.

The verified matching convention is:

-   S1 TIFF basename (without `.tif`) ↔ `s1_name`
-   S2 TIFF basename (without `.tif`) ↔ `patch_id`

An exact check found 13,683 `patch_id` matches among the S2 image
basenames, and the S1 matching check also found 13,683 matches using
`s1_name`.

**Important:** the final paired metadata table should be restricted to
records for which both corresponding image files exist. Duplicate IDs,
missing labels, split counts, and quality-flag distributions still need
a final diagnostic check.

------------------------------------------------------------------------

## 2. How the dataset was downloaded

The dataset was downloaded in Google Colab using the Kaggle CLI.

### Kaggle CLI download

The dataset manifest was first inspected through the Kaggle CLI. The
manifest contained:

-   27,366 `.tif` files
-   1 `.parquet` metadata file
-   27,367 files in total

The full dataset was then downloaded and extracted.

``` bash
!kaggle datasets download \
    -d narendraaironi/bigearthnet-14k \
    -p /content/datasets
```

Extraction:

``` python
import os
import zipfile

archive = "/content/datasets/bigearthnet-14k.zip"
DATA_ROOT = "/content/datasets/BEN14K"

os.makedirs(DATA_ROOT, exist_ok=True)

with zipfile.ZipFile(archive, "r") as z:
    z.extractall(DATA_ROOT)
```

The archive contains an additional `BEN_14k` directory, so the effective
data root is:

``` python
DATA_ROOT = "/content/datasets/BEN14K/BEN_14k"
```

The expected structure is:

``` text
/content/datasets/BEN14K/
└── BEN_14k/
    ├── metadata.parquet
    ├── BigEarthNet-S1/
    │   ├── train/
    │   ├── validation/
    │   └── test/
    └── BigEarthNet-S2/
        ├── train/
        ├── validation/
        └── test/
```

The metadata file was also downloaded separately during the initial
inspection. The copy inside the extracted dataset root is the one to use
consistently:

``` python
import pandas as pd

metadata_path = f"{DATA_ROOT}/metadata.parquet"
df = pd.read_parquet(metadata_path)
```

### Reproducibility and provenance notes

-   The Kaggle dataset is a third-party upload; its relationship to any
    particular official BigEarthNet release or curated research subset
    has not been independently established byte-for-byte.
-   The file counts and filename-to-metadata matches have been verified
    locally.
-   The dataset manifest and extracted split counts agree: 7,180 train,
    3,255 validation, and 3,248 test images per modality.
-   The Kaggle CLI previously emitted a version warning (installed
    version 2.0.2; the warning recommended 2.2.5). The download
    nevertheless completed successfully.
-   Colab's `/content` storage is temporary. Preserve experiment
    configurations, logs, checkpoints, and final results in persistent
    Google Drive storage. Re-download the dataset if the Colab runtime
    is reset.

------------------------------------------------------------------------

## 3. Planned implementation

The implementation will be built and validated incrementally. Model
training should not begin until the TIFF formats, pairing,
preprocessing, and data loaders have been checked.

### Stage 1 --- Dataset validation and pairing

1.  Inspect representative S1 and S2 TIFF files with a geospatial raster
    reader such as `rasterio`.
2.  Record each modality's image dimensions, number of bands, data type,
    CRS, nodata value, and representative pixel-value ranges.
3.  Build a paired index using `s1_name` for S1 and `patch_id` for S2.
4.  Verify unique IDs, missing files, duplicate records, label
    availability, and per-split pair counts.
5.  Retain the provided train/validation/test split assignments. Do not
    replace them with a random image-level split.
6.  Inspect the split construction and geographic identifiers for
    possible spatial leakage, especially neighbouring patches from the
    same geographic tile.

### Stage 2 --- Preprocessing and data loaders

Preprocessing will be chosen only after inspecting the actual TIFF
values and band layout.

-   **Sentinel-1:** determine the available polarisation bands and their
    value representation. Preserve meaningful backscatter information;
    do not apply per-image normalisation blindly.
-   **Sentinel-2:** determine the available spectral bands and
    reflectance scale. Handle nodata and invalid pixels explicitly; do
    not assume a scale factor before checking the data.
-   **Spatial size:** inspect the actual raster dimensions before
    choosing resizing or cropping.
-   **Quality flags:** retain cloud/shadow and seasonal-snow metadata.
    Decide whether to filter or stratify these samples after assessing
    their prevalence and relevance.
-   Ensure that paired S1 and S2 samples remain correctly aligned
    through all transforms.
-   Use deterministic validation and test preprocessing. Any stochastic
    augmentation should be confined to training.

A Sentinel-2 RGB view may be derived for visualisation or a separate
baseline, but it is not a third sensor or an independent modality.

### Stage 3 --- Shared representation learning

The planned approach is a dual-branch encoder:

-   one encoder branch for Sentinel-1 SAR;
-   one encoder branch for Sentinel-2 optical imagery;
-   projection heads that map both branches into a common embedding
    dimension;
-   a contrastive objective that brings corresponding cross-modal pairs
    together while separating non-matching samples.

A pretrained image backbone (the current candidate is ConvNeXt-Small
with ImageNet initialisation) is under consideration. The final
backbone, input-channel adaptation, embedding dimension, loss details,
and batch size remain subject to the TIFF inspection and available GPU
memory.

The current experimental direction is **semantic-aware contrastive
learning**. The exact loss formulation and any hard-negative strategy
will be selected and documented before training. A simple paired
contrastive baseline should be retained so that added complexity can be
evaluated fairly.

### Stage 4 --- Retrieval and indexing

1.  Encode the gallery images and store their embeddings.
2.  L2-normalise embeddings if cosine similarity is used.
3.  Retrieve the top-K candidates using cosine similarity.
4.  Use FAISS for efficient nearest-neighbour search if the gallery size
    and experiments justify it.
5.  Report embedding/index construction time separately from query-time
    retrieval latency.

------------------------------------------------------------------------

## 4. Evaluation protocol and metrics

Evaluation must distinguish two different notions of relevance. They
answer different research questions and should not be mixed into one
score.

### A. Geographic / paired cross-modal retrieval

**Question:** Given one sensor's image of a geographic patch, can the
system retrieve the corresponding image of the same patch from the other
sensor?

-   A matched S1/S2 pair is a positive pair.
-   Query and gallery modalities are specified explicitly (S1 → S2 and
    S2 → S1).
-   The paired counterpart is relevant by definition.
-   Report both retrieval directions where applicable.

This is the primary direct test of cross-sensor alignment.

### B. Semantic retrieval

**Question:** Can the system retrieve images with similar land-cover
content, even when they depict different geographic locations?

-   Use the multi-label `labels` metadata to define semantic relevance.
-   A candidate is relevant according to a clearly documented
    label-overlap rule (to be finalised before evaluation).
-   The exact same-location counterpart should be excluded from the
    semantic gallery or separately reported. Otherwise, the task can be
    dominated by the easy geographic match.
-   State the relevance rule, including how multi-label overlap is
    calculated and how samples with no shared labels are treated.

Geographic and semantic retrieval results must be reported separately.

### Metrics

The planned ranking metrics are:

  -----------------------------------------------------------------------
  Metric                              What it measures
  ----------------------------------- -----------------------------------
  **Precision@K**                     Fraction of the top K retrieved
                                      items that are relevant.

  **Recall@K**                        Fraction of all relevant gallery
                                      items recovered in the top K.

  **F1@K**                            Harmonic mean of Precision@K and
                                      Recall@K.

  **mAP**                             Mean Average Precision across
                                      queries; evaluates the ranking of
                                      relevant results across the ranked
                                      list.

  **nDCG@K**                          Normalised Discounted Cumulative
                                      Gain; evaluates ranking quality and
                                      can support graded relevance.

  **Retrieval latency**               Time required to retrieve
                                      candidates for a query, reported
                                      with the indexing/search setup.
  -----------------------------------------------------------------------

Use more than one K value (for example, K = 1, 5, and 10), selected and
kept consistent across experiments. Define metric implementations and
relevance rules once, then use them unchanged for model comparisons.

For semantic retrieval with multiple relevant candidates, calculate AP
over the ranked gallery and average it across valid queries. For paired
geographic retrieval, report the rank of the exact counterpart and
corresponding top-K success/recall in addition to any aggregate ranking
metrics.

### Split discipline and leakage prevention

-   Train only on the training split.
-   Use validation for model selection, hyperparameter tuning, and
    checkpoint selection.
-   Keep the test split untouched until the final evaluation.
-   Do not randomly repartition individual images if that breaks the
    supplied pairing or geographic split structure.
-   Investigate whether nearby patches or patches from the same tile
    occur across splits. If the provided split does not adequately
    prevent geographic leakage, document the limitation and consider a
    geographically grouped evaluation protocol.
-   Never use test retrieval results to tune the model.

------------------------------------------------------------------------

## 5. Current project status

  -----------------------------------------------------------------------
  Component                           Status
  ----------------------------------- -----------------------------------
  Kaggle dataset selected             Complete

  Kaggle file manifest inspected      Complete

  Full archive downloaded and         Complete
  extracted in Colab                  

  Correct extracted data root         Complete
  identified                          

  S1 train/validation/test counts     Complete
  verified                            

  S2 train/validation/test counts     Complete
  verified                            

  Metadata Parquet file present       Complete

  S1/S2 filename matching convention  Complete
  established                         

  Full paired-table integrity         Pending
  diagnostics                         

  TIFF band, shape, dtype, nodata,    **Next step**
  and value inspection                

  Preprocessing and data loaders      Pending

  Baseline model and contrastive      Pending
  training                            

  Retrieval index and evaluation      Pending
  implementation                      

  Final test evaluation and analysis  Pending
  -----------------------------------------------------------------------

### Immediate next step

Inspect a small representative sample of TIFF files from both modalities
(including their band counts, shapes, data types, nodata values, and
pixel-value distributions). Use those observations to design the
preprocessing pipeline before implementing the data loaders.

------------------------------------------------------------------------

## 6. Experimental discipline

For reproducibility, record for every experiment:

-   dataset identifier and data-root/version;
-   exact train/validation/test manifest or record counts;
-   preprocessing and band-selection configuration;
-   model architecture and pretrained-weight source;
-   loss function and all loss hyperparameters;
-   optimiser, learning rate, scheduler, batch size, epochs, and random
    seeds;
-   checkpoint-selection rule;
-   retrieval direction, gallery definition, relevance rule, and K
    values;
-   metric implementation and retrieval latency measurement method.

Keep a simple baseline and change one major experimental factor at a
time. Do not report expected or notebook-reported scores as results
unless they have been reproduced with this dataset and evaluation
protocol.
