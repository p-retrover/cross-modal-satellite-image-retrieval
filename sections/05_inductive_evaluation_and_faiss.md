# Inductive Zero-Leakage Evaluation and Latency Benchmarks

## The Threat of Transductive Leakage and Modality Cone Collapse

In cross-modal retrieval, a well-documented phenomenon known as "cone collapse" or the "modality gap" frequently occurs. Despite applying contrastive alignment, neural networks tend to map different physical modalities into distinct, narrow sub-cones of the hyperspherical latent space. Consequently, the intra-modality similarity (SAR-to-SAR or Optical-to-Optical) dominates the cross-modality similarity (SAR-to-Optical).

A common, yet methodologically flawed, approach in the literature to correct this shift is transductive zero-mean centering. In this flawed approach, researchers compute the centroid of all SAR vectors and the centroid of all Optical vectors across the *test set*, subtracting these means before evaluation. This constitutes a severe form of data leakage, as the distribution of the entire test set is used to calibrate individual test queries—a mechanism impossible in real-world, online deployments where queries arrive one at a time.

## Frozen Inductive Centering

To strictly prevent test-set leakage while neutralizing modality cone collapse, this pipeline implements Frozen Inductive Centering. 

The sensor-specific calibration centroids are computed strictly and exclusively over the held-out validation split ($N_{\text{val}} = 3,255$ pairs):
$$\boldsymbol{\mu}_{\text{SAR}} = \frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} f_{\text{SAR}}^{\text{target}}(\mathbf{x}_{\text{S1}, i}^{\text{val}}), \quad \boldsymbol{\mu}_{\text{Opt}} = \frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} f_{\text{Opt}}^{\text{target}}(\mathbf{x}_{\text{S2}, i}^{\text{val}})$$
These pre-computed scalars are frozen. During the final test-set evaluation, or in a production web application, incoming single queries and gallery embeddings are inductively shifted by these frozen validation centroids prior to $\ell_2$-normalization:
$$\hat{\mathbf{z}}_m = \frac{\mathbf{z}_m - \boldsymbol{\mu}_m}{\|\mathbf{z}_m - \boldsymbol{\mu}_m\|_2}, \quad m \in \{\text{SAR}, \text{Opt}\}$$
This completely preserves the strict independence of the test set while effectively overlapping the SAR and Optical representation cones.

## Unified Modality-Agnostic FAISS Indexing

Unlike rigid directional pipelines that constrain searches strictly from one modality to another (e.g., querying SAR against a database of only Optical images), this architecture features a unified, modality-agnostic gallery.

The embeddings for all 3,248 Sentinel-1 test patches and all 3,248 Sentinel-2 test patches are stacked into a single 6,496-item database. Because the vectors are $\ell_2$-normalized, maximizing cosine similarity is mathematically equivalent to maximizing the inner product. We utilize the Facebook AI Similarity Search (FAISS) library[@johnson2019], specifically `faiss.IndexFlatIP(256)`, for exact, exhaustive inner-product search across the 256-dimensional latent space. 

When an image is queried, it traverses this mixed gallery. To prevent trivial self-retrieval, exact identical twins (where both the patch ID and the modality exactly match the query: $\text{pid}_h = \text{pid}_q \land \text{mod}_h = \text{mod}_q$) are dynamically filtered out of the top-$K$ results.

## Dual-Dimensional Evaluation Metrics

Because the system models both exact instance matching and broad semantic clustering, evaluation is explicitly decoupled into two independent metrics.

### Geographic Counterpart Recall (Geo R@K)
This metric evaluates instance-level cross-modal alignment. It measures whether the exact physical twin (same geographic patch ID, but captured by the opposite sensor) is successfully retrieved in the top-$K$ candidate slots:
$$\text{Geo R}@K = \mathbb{I}(\exists r \le K : \text{pid}_{h_r} = \text{pid}_q \land \text{mod}_{h_r} \neq \text{mod}_q)$$

### Twin-Excluded Mean Label Dice@K
This metric measures pure multi-label semantic retrieval across geographically distinct locations. To prevent the exact physical twin from artificially inflating the semantic score, the twin is explicitly excluded from this calculation ($\text{pid}_h \neq \text{pid}_q$). 

For a query $q$ and its top-$K$ retrieved candidates $h_r$, the semantic overlap is calculated using the Sørensen–Dice coefficient over the multi-label sets $Y$:
$$\text{Dice}@K(q) = \frac{1}{K} \sum_{r=1}^{K} \frac{2 |Y_q \cap Y_{h_r}|}{|Y_q| + |Y_{h_r}|}$$
The system computes the mean of this Dice score across all valid queries.

## Hardware-Synchronized Latency Profiling

For operational deployment readiness, sub-millisecond retrieval latency must be empirically verified. PyTorch executes CUDA kernels asynchronously; standard Python `time.time()` calls around a neural network forward pass will return immediately before the GPU has finished computing, resulting in severely underestimated latencies.

To ensure rigorous profiling, explicit stream barrier synchronization is enforced using `torch.cuda.synchronize()` before and after the timing blocks. This isolates and accurately measures:
1.  **Encoding Latency:** The wall-clock time required for the ConvNeXt-Tiny backbone to process a single query image.
2.  **Search Latency:** The time required for FAISS to compute inner products and sort the top-$K$ candidates across the 6,496-item gallery.

## Experimental Discipline and Reproducibility Protocol

Every experiment and ablation strictly adheres to the following empirical parameters to ensure scientific reproducibility:
   **Split Integrity:** Models are trained exclusively on the 7,180 training pairs. The test split is entirely untouched during training and hyperparameter tuning.
   **Checkpoint Selection:** Model weights are selected based on composite validation performance ($\text{Geo R}@5 + \text{Mean Label Dice}@5$) and persisted directly to disk with OS-level buffer synchronization (`os.sync()`) to prevent corruption.
   **Baseline Discipline:** A simple baseline (standard InfoNCE without masking, uncalibrated FAISS gallery) is maintained, modifying only one major architectural or mathematical factor at a time.
   **Reporting:** Scores are never derived from transductive centering or tuned on the test set. All metrics reflect strictly inductive, zero-leakage deployments.