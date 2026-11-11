# Literature Survey and Theoretical Foundations

## Taxonomy of Remote Sensing Image Retrieval

The rapid proliferation of Earth observation data has driven the evolution of Remote Sensing Image Retrieval (RSIR) from early metadata-based querying to advanced Content-Based Remote Sensing Image Retrieval (CBRSIR). Recent comprehensive reviews of the field hierarchically categorize modern RSIR methodologies into two primary branches based on the relationship between the query and the database: uni-source retrieval and cross-source retrieval[@zhou2023].

![RSIR Hierarchical Taxonomy](./images/rsir_network.svg)

### Uni-Source Retrieval
In uni-source retrieval, the query and gallery images are acquired by the same sensor type[@zhou2023]. This branch is divided by the semantic complexity of the annotations:
   **Single-Label RSIR (SLRSIR):** The traditional retrieval paradigm where each image is associated with a single dominant land-cover class[@zhou2023]. Benchmark datasets such as UC Merced, WHU-RS19, and PatternNet have historically driven this domain[@zhou2023]. However, SLRSIR fails to capture the heterogeneous nature of real-world remote sensing scenes.
   **Multi-Label RSIR (MLRSIR):** A more advanced paradigm where images are annotated with multiple primitive classes (e.g., buildings, roads, and water simultaneously)[@zhou2023]. Algorithms in this space must model complex label co-occurrences. Datasets like DLRSD, WHDLD, and MLRSNet support this task[@zhou2023], yet defining robust similarity measures for multi-label ranking remains an active research challenge.

### Cross-Source Retrieval
Cross-source retrieval addresses scenarios where the query and database images originate from different physical domains, sensors, or perspectives[@zhou2023].
   **Retrieval Across Different Sensors (RASRSIR):** Involves images from the same modality but different sensors or resolutions, such as retrieving high-resolution panchromatic images using a multispectral query[@zhou2023].
   **Cross-View RSIR (CVRSIR):** Also known as geolocalization, this task matches images captured from drastically different viewpoints, such as querying a ground-level panorama against an overhead drone or satellite database[@zhou2023].
   **Cross-Modal RSIR (CMRSIR):** The focus of this project, CMRSIR bridges entirely different data representations[@zhou2023]. This includes sketch-to-image, text-to-image, audio-to-image, and Optical-to-SAR retrieval[@zhou2023]. Aligning active microwave (SAR) and passive optical imagery requires overcoming a severe physical modality gap, as the sensors measure fundamentally different properties of the Earth's surface.

## Evolution of Deep Metric Learning

To bridge the modality gap, modern RSIR relies heavily on Deep Metric Learning (DML). DML aims to learn a parameterized embedding space where semantically similar samples are mapped closely together, and dissimilar samples are pushed apart.

### From Pairwise to Triplet Formulations
Early metric learning relied on the Siamese architecture and the **Contrastive Loss** (Hadsell et al., 2006)[@hadsell2006], which minimized the Euclidean distance between positive pairs and penalized negative pairs that fell within a predefined margin. This evolved into the **Triplet Margin Loss** (Schroff et al., 2015)[@schroff2015], which optimizes relative distances: anchoring a query image, pulling a positive sample closer, and pushing a negative sample further away by a set margin. While effective, triplet loss suffers from cubic sampling complexity ($O(N^3)$) and heavily depends on computationally expensive hard-negative mining strategies.

### Information Noise Contrastive Estimation (InfoNCE)
The field experienced a paradigm shift with the introduction of **InfoNCE** (Oord et al., 2018)[@oord2018], rooted in maximizing the mutual information between representations. Instead of comparing one positive and one negative sample, InfoNCE treats the task as a multi-class classification problem over a batch, comparing the query against one positive and $K$ negative samples simultaneously. By operating in a hyperspherical latent space (via $\ell_2$-normalization) and scaling similarities with a temperature parameter $\tau$, InfoNCE creates highly uniform and linearly separable feature spaces.

### Asymmetric Momentum Self-Distillation
In cross-modal applications, maintaining consistent representation targets is difficult when the network updates rapidly. Architectures like **MoCo (Momentum Contrast)** (He et al., 2020)[@he2020] and **BYOL (Bootstrap Your Own Latent)** (Grill et al., 2020)[@grill2020] introduced the asymmetric dual-tower paradigm. Instead of updating both the online and target networks via backpropagation, the target network's weights are updated using an Exponential Moving Average (EMA) of the online network. This provides a stable, slowly-evolving target for the contrastive loss, which is critical when aligning highly heterogeneous domains like SAR and Optical imagery.

## The Modality Gap and Cone Collapse in Cross-Modal Retrieval

When applying contrastive learning to dual-modality inputs (e.g., passing SAR through one encoder and Optical through another), the system is prone to a geometric failure known as the **modality gap** or **cone collapse** (Liang et al., 2022)[@liang2022]. 

Even when the InfoNCE loss successfully aligns paired SAR and Optical patches, the underlying differences in the sensor data distributions cause the network to map all SAR embeddings into one narrow cone of the hyperspherical space, and all Optical embeddings into a completely separate cone. While paired vectors remain relatively close across the gap, querying a SAR image against a unified gallery containing both SAR and Optical images will inadvertently retrieve only SAR images, as the intra-modality similarity heavily outweighs the cross-modality similarity.

### Transductive vs. Inductive Calibration
Recent literature attempts to correct this gap by centering the embeddings—subtracting the mean of the SAR vectors and the mean of the Optical vectors. However, computing these centroids over the *test set* constitutes transductive learning, introducing severe data leakage that invalidates the evaluation. 

A rigorous, operationally sound system must employ **Frozen Inductive Centering**. Calibration centroids ($\boldsymbol{\mu}_{\text{SAR}}$ and $\boldsymbol{\mu}_{\text{Opt}}$) must be computed exclusively on a strictly held-out validation set. During inference, incoming test queries are shifted using these frozen, pre-computed scalars. This satisfies strict zero-leakage protocols while successfully neutralizing modality cone collapse, enabling true modality-agnostic unified retrieval.