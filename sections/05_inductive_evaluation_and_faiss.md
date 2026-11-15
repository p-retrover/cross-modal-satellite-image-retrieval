# Inductive Zero-Leakage Evaluation and Latency Benchmarks

## The Threat of Transductive Leakage and Modality Cone Collapse

Cross-modal retrieval systems frequently suffer from a geometric failure called "cone collapse" or the "modality gap". Even after contrastive alignment, neural networks map different physical modalities into narrow, separate sub-cones on the unit hypersphere. As a result, intra-modality similarity (SAR-to-SAR or Optical-to-Optical) exceeds cross-modality similarity (SAR-to-Optical).

Some researchers correct this shift with transductive zero-mean centering, which introduces a serious methodological flaw. In that method, they compute the mean of all SAR vectors and the mean of all Optical vectors across the *test set*, and then subtract those means before evaluation. This practice creates severe data leakage because the test-set distribution calibrates individual test queries. Operational deployments cannot use this method because queries arrive one at a time.

## Frozen Inductive Centering

To prevent test-set leakage and remove modality cone collapse, this pipeline uses Frozen Inductive Centering.

The pipeline computes the sensor-specific calibration centroids strictly over the held-out validation split ($N_{\text{val}} = 3,255$ pairs):

$$\boldsymbol{\mu}_{\text{SAR}} = \frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} f_{\text{SAR}}^{\text{target}}(\mathbf{x}_{\text{S1}, i}^{\text{val}}), \quad \boldsymbol{\mu}_{\text{Opt}} = \frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} f_{\text{Opt}}^{\text{target}}(\mathbf{x}_{\text{S2}, i}^{\text{val}})$$

The system then freezes these pre-computed centroids. During final test-set evaluation or live production inference, the pipeline shifts incoming single queries and gallery embeddings with these frozen validation centroids before $\ell_2$-normalization:

$$\hat{\mathbf{z}}_m = \frac{\mathbf{z}_m - \boldsymbol{\mu}_m}{\Vert{}\mathbf{z}_m - \boldsymbol{\mu}_m\Vert{}_2}, \quad m \in \{\text{SAR}, \text{Opt}\}$$

This procedure keeps the test set completely independent while it aligns the SAR and Optical representation cones.

## Unified Modality-Agnostic FAISS Indexing

Directional pipelines restrict searches strictly from one modality to another, such as querying SAR against a gallery of Optical images. In contrast, this architecture uses a unified, modality-agnostic gallery.

The pipeline stacks the embeddings for all 3,248 Sentinel-1 test patches and all 3,248 Sentinel-2 test patches into one 6,496-item database. Because the vectors are $\ell_2$-normalized, maximizing cosine similarity equals maximizing the inner product. The system uses the Facebook AI Similarity Search (FAISS) library[@johnson2019], specifically `faiss.IndexFlatIP(256)`, to run exact inner-product searches across the 256-dimensional latent space.

When a user queries an image, FAISS searches this mixed gallery. To prevent trivial self-retrieval, the system filters out exact identical twins ($\text{pid}_h = \text{pid}_q \land \text{mod}_h = \text{mod}_q$) from the top-$K$ results.

## Dual-Dimensional Evaluation Metrics

Because the system models both exact instance matching and broad semantic clustering, the evaluation framework separates performance into two independent metrics.

### Geographic Counterpart Recall (Geo R@K)

This metric evaluates instance-level cross-modal alignment. It measures whether the system retrieves the exact physical twin (same geographic patch ID, opposite sensor) within the top-$K$ candidate slots:

$$\text{Geo R}@K = \mathbb{I}(\exists r \le K : \text{pid}_{h_r} = \text{pid}_q \land \text{mod}_{h_r} \neq \text{mod}_q)$$

### Twin-Excluded Mean Label Dice@K

This metric measures multi-label semantic retrieval across different geographic locations. To prevent the exact physical twin from inflating the semantic score, the evaluation excludes the twin from this calculation ($\text{pid}_h \neq \text{pid}_q$).

For a query $q$ and its top-$K$ retrieved candidates $h_r$, the system calculates semantic overlap with the Sørensen–Dice coefficient over the multi-label sets $Y$:

$$\text{Dice}@K(q) = \frac{1}{K} \sum_{r=1}^{K} \frac{2 \vert{}Y_q \cap Y_{h_r}\vert{}}{\vert{}Y_q\vert{} + \vert{}Y_{h_r}\vert{}}$$

The system then computes the mean Dice score across all valid queries.

### Algorithmic Evaluation Protocol

The complete zero-leakage evaluation workflow—integrating frozen validation centroid calibration, unified modality-agnostic FAISS indexing, self-match filtering, and dual-dimensional metric computation—is formalized below:

**Algorithm 5.1: Frozen Validation-Centered Retrieval and Test Evaluation**

* **Require:** Best saved model checkpoint; paired validation set $\mathcal{D}_{\mathrm{val}}$ ($N_{\mathrm{val}} = 3{,}255$); paired test set $\mathcal{D}_{\mathrm{test}}$ ($N_{\mathrm{test}} = 3{,}248$) with multi-labels $Y$ and pair IDs ($\text{pid}$); evaluation cutoffs $K \in \{1, 5, 10\}$.
* **Ensure:** Unified retrieval rankings, $\text{Geo R}@K$, and twin-excluded mean label $\text{Dice}@K$.

1. Load the best model checkpoint and set both target encoders ($f_{\mathrm{SAR}}^{\mathrm{target}}, f_{\mathrm{Opt}}^{\mathrm{target}}$) to evaluation mode.
2. Initialize empty validation embedding sets $\mathcal{Z}^{\mathrm{val}}_{\mathrm{SAR}}$ and $\mathcal{Z}^{\mathrm{val}}_{\mathrm{Opt}}$.
3. **For** each paired mini-batch in $\mathcal{D}_{\mathrm{val}}$ **do:**
   a. Extract SAR and optical embeddings without gradients using their corresponding target encoders.
   b. Append the embeddings to $\mathcal{Z}^{\mathrm{val}}_{\mathrm{SAR}}$ and $\mathcal{Z}^{\mathrm{val}}_{\mathrm{Opt}}$.
4. Compute the modality-specific validation centroids:
   $$\boldsymbol{\mu}_{m} \gets \frac{1}{N_{\mathrm{val}}} \sum_{i=1}^{N_{\mathrm{val}}} \mathbf{z}^{\mathrm{val}}_{m,i}, \quad m \in \{\mathrm{SAR}, \mathrm{Opt}\}$$
5. Freeze both centroids ($\boldsymbol{\mu}_{\mathrm{SAR}}, \boldsymbol{\mu}_{\mathrm{Opt}}$); do not compute or update centroids using test embeddings.
6. Extract raw SAR and optical embeddings, pair IDs ($\text{pid}$), and multi-label sets ($Y$) for every pair in $\mathcal{D}_{\mathrm{test}}$.
7. **For** each test pair $i \in \{1, \ldots, N_{\mathrm{test}}\}$ and modality $m \in \{\mathrm{SAR}, \mathrm{Opt}\}$ **do:**
   a. Center and $\ell_2$-normalize the embedding:
      $$\widehat{\mathbf{z}}_{m,i} \gets \frac{\mathbf{z}_{m,i} - \boldsymbol{\mu}_{m}}{\left\|\mathbf{z}_{m,i} - \boldsymbol{\mu}_{m}\right\|_2}$$
8. Form the unified $2N_{\mathrm{test}}$-item gallery ($6{,}496$ items) by stacking the calibrated SAR and optical test embeddings, retaining each item's pair ID ($\text{pid}$), modality ($\text{mod}$), and label set ($Y$).
9. Build an exact FAISS inner-product index `faiss.IndexFlatIP(256)` and populate it with the unified gallery embeddings.
10. **For** each modality-specific test item $q \in \{1, \ldots, 2N_{\mathrm{test}}\}$ used as a query **do:**
    a. Search the unified FAISS index for the top $25$ candidates in descending inner-product order.
    b. Remove the exact self-match query item ($\text{pid}_h = \text{pid}_q \land \text{mod}_h = \text{mod}_q$) from the candidate list.
    c. Compute $\text{Geo R}@K(q) \gets 1$ if the opposite-modality counterpart with the same pair ID ($\text{pid}_{h_r} = \text{pid}_q \land \text{mod}_{h_r} \neq \text{mod}_q$) occurs among the first $K$ remaining candidates; otherwise set $\text{Geo R}@K(q) \gets 0$.
    d. Remove all candidates sharing the query's pair ID ($\text{pid}_h = \text{pid}_q$) to form the twin-excluded semantic ranking.
    e. Compute $\text{Dice}@K(q)$ over the first $K$ candidates in this twin-excluded semantic ranking.
11. Average $\text{Geo R}@K(q)$ and $\text{Dice}@K(q)$ over all $2N_{\mathrm{test}}$ modality-specific test queries (and separately by query modality for directional analysis).
12. **Return** unified retrieval rankings, $\text{Geo R}@K$, and twin-excluded mean label $\text{Dice}@K$.

## Hardware-Synchronized Latency Profiling

Operational deployment requires verified sub-millisecond retrieval latency. PyTorch executes CUDA kernels asynchronously. Therefore, standard Python `time.time()` calls around a forward pass return before the GPU finishes its computation, which underestimates latency.

To measure latency accurately, the benchmark enforces explicit stream barrier synchronization with `torch.cuda.synchronize()` before and after each timing block. This step isolates and measures two metrics:

1. **Encoding Latency:** The wall-clock time that the ConvNeXt-Tiny backbone needs to process one query image.

2. **Search Latency:** The time that FAISS needs to compute inner products and sort the top-$K$ candidates across the 6,496-item gallery.

## Experimental Discipline and Reproducibility Protocol

Every experiment and ablation follows four strict rules to ensure scientific reproducibility:

* **Split Integrity:** The training loop uses only the 7,180 training pairs, and the test split remains untouched during training and hyperparameter tuning.

* **Checkpoint Selection:** The pipeline selects model weights from composite validation performance ($\text{Geo R}@5 + \text{Mean Label Dice}@5$) and saves them to disk with OS-level buffer synchronization (`os.sync()`) to prevent file corruption.

* **Baseline Discipline:** The benchmark maintains a standard baseline (unmasked InfoNCE and an uncalibrated FAISS gallery) and changes only one major architectural or mathematical factor at a time.

* **Reporting:** The evaluation never uses transductive centering or test-set tuning, so all metrics reflect strict inductive, zero-leakage deployment.