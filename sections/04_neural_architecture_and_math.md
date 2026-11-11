# Neural Architecture and Mathematical Formulation

\begin{center}
  \includegraphics[width=\textwidth,keepaspectratio]{images/training_pipeline_final.pdf}
  \captionof{figure}{End-to-end asymmetric dual-tower training pipeline. Co-registered Sentinel-1 SAR and Sentinel-2 optical image pairs undergo synchronized $D_4$ dihedral augmentations before routing into online encoders ($\theta_q$) and momentum-updated target encoders ($\theta_k$, $m=0.996$). Latent projections on the 256-dimensional unit hypersphere are jointly optimized using a composite multi-label Jaccard-masked InfoNCE loss ($\tau_{\text{IoU}}=0.40$), executing both cross-modal alignment and intra-modal self-distillation while suppressing false negatives.}
  \label{fig:training-pipeline}
\end{center}

## Sensor-Specific Stem Adaptation

To leverage the robust inductive biases and hierarchical feature extraction capabilities of modern convolutional architectures, the system utilizes ConvNeXt-Tiny[@liu2022] as the foundational backbone for both the SAR and Optical branches. The standard ConvNeXt architecture begins with a "patchify" stem: a convolutional layer with a $4 \times 4$ kernel and stride of 4 (`Conv2d(3, 96, 4, 4)`), originally designed for 3-channel RGB inputs pre-trained on ImageNet-1K.

Because our remote sensing tensors possess non-standard channel depths ($C=2$ for SAR, $C=10$ for Optical), the first convolutional layer must be mathematically adapted to ingest these modalities while strictly preserving the pre-trained variance and energy of the ImageNet weights $\mathbf{W}_{\text{RGB}} \in \mathbb{R}^{96 \times 3 \times 4 \times 4}$.

### SAR 2-Channel Initialization
For the 2-channel Sentinel-1 input ($\text{VV}, \text{VH}$), the new stem weights $\mathbf{W}_{\text{S1}}$ are derived by computing the mean across the 3 RGB channels of the pre-trained weights, scaling it to conserve the original energy, and repeating it across the 2 SAR channels:
$$\mathbf{W}_{\text{S1}} = \text{repeat}_2\left(\frac{1}{3}\sum_{c=1}^{3} \mathbf{W}_{\text{RGB}}[:, c, :, :]\right) \in \mathbb{R}^{96 \times 2 \times 4 \times 4}$$

### Optical 10-Channel Initialization
For the 10-channel Sentinel-2 input, a tiling and scaling approach is employed. The original 3-channel weights are tiled to cover the 10 dimensions, and the output is scaled down by a factor of $10/3 \approx 3.33$ to ensure that the sum of the activations matches the expected ImageNet pre-trained magnitude:
$$\mathbf{W}_{\text{S2}} = \frac{3}{10} \left(\text{repeat}_4(\mathbf{W}_{\text{RGB}})\right)[:, :10, :, :] \in \mathbb{R}^{96 \times 10 \times 4 \times 4}$$

\begin{figure}[p]
  \centering
  \begin{subfigure}[b]{0.48\textwidth}
    \centering
    \includegraphics[height=0.76\textheight,keepaspectratio]{images/sar_encoder_compact.pdf}
    \caption{Sentinel-1 SAR Target Encoder (2-ch)}
    \label{fig:sar-encoder}
  \end{subfigure}\hfill
  \begin{subfigure}[b]{0.48\textwidth}
    \centering
    \includegraphics[height=0.76\textheight,keepaspectratio]{images/opt_encoder_compact.pdf}
    \caption{Sentinel-2 Optical Target Encoder (10-ch)}
    \label{fig:opt-encoder}
  \end{subfigure}
  \caption{Computational graph topologies for the target encoders generated via \texttt{torchview} ($\text{depth}=2$). Both backbones map multi-channel rasters to a shared 256-d normalized latent space.}
  \label{fig:target-encoders}
\end{figure}

## Generalized Mean (GeM) Spatial Pooling

Standard convolutional architectures typically conclude feature extraction with Global Average Pooling (GAP). However, for retrieval tasks where specific localized terrain features (e.g., a small bridge or isolated building) are crucial, GAP tends to dilute strong localized activations into the background noise.

To address this, we implement Generalized Mean (GeM) spatial pooling[@radenovic2018]. Given the final convolutional feature map $\mathbf{F} \in \mathbb{R}^{768 \times H' \times W'}$, the GeM descriptor $\mathbf{g} \in \mathbb{R}^{768}$ is computed as:
$$\mathbf{g} = \left( \frac{1}{H'W'} \sum_{u=1}^{H'} \sum_{v=1}^{W'} \max(\mathbf{F}_{:, u, v}, \epsilon)^p \right)^{\frac{1}{p}}$$
where $p$ is a learnable pooling parameter initialized at $p=3.0$, and $\epsilon=10^{-6}$ ensures numerical stability during backpropagation. The GeM formulation is a continuous generalization: as $p \to 1$, it reduces to GAP, and as $p \to \infty$, it approaches Global Max Pooling (GMP). By allowing the network to learn $p$, the system dynamically balances background context with salient object activations.

## Asymmetric Momentum Dual-Tower Dynamics

To align the heterogeneous SAR and Optical domains, the system employs an asymmetric online–target framework (`CrossModalSAROpticalNet`), inspired by self-supervised momentum architectures like MoCo[@he2020] and BYOL[@grill2020].

### Online and Target Encoders
The architecture maintains two pairs of networks:
   **Online Networks ($\theta_q$):** Process aggressively augmented views of the SAR and Optical batches $(\tilde{\mathbf{X}}_{\text{S1}}^{\text{aug}}, \tilde{\mathbf{X}}_{\text{S2}}^{\text{aug}})$. These networks are updated dynamically via standard gradient descent.
   **Target Networks ($\theta_k$):** Process the original, unaugmented views $(\tilde{\mathbf{X}}_{\text{S1}}^{\text{orig}}, \tilde{\mathbf{X}}_{\text{S2}}^{\text{orig}})$. These networks do not receive gradients. Instead, their parameters are updated via a strict Exponential Moving Average (EMA) of the online networks' parameters:
$$\theta_k \leftarrow m\theta_k + (1 - m)\theta_q$$
where $m = 0.996$ is the momentum coefficient.

### Projection Head and Buffer Synchronization

| Stage / Component | Operation | Input Shape | Output Shape | Spatial / Feature Details |
| :--- | :--- | :--- | :--- | :--- |
| **SAR Stem** | Conv2d ($4\times4, s=4$) | $(B, 2, 120, 120)$ | $(B, 96, 30, 30)$ | Patch projection (VV, VH) |
| **Optical Stem** | Conv2d ($4\times4, s=4$) | $(B, 10, 120, 120)$ | $(B, 96, 30, 30)$ | Patch projection (10 MSI bands) |
| **Stage 1** | ConvNeXt Block $\times 3$ | $(B, 96, 30, 30)$ | $(B, 96, 30, 30)$ | Depthwise $7\times 7$, GELU, LN |
| **Downsample 1** | LayerNorm + Conv2d ($2\times 2, s=2$) | $(B, 96, 30, 30)$ | $(B, 192, 15, 15)$ | Spatial reduction |
| **Stage 2** | ConvNeXt Block $\times 3$ | $(B, 192, 15, 15)$ | $(B, 192, 15, 15)$ | Mid-level spatial features |
| **Downsample 2** | LayerNorm + Conv2d ($2\times 2, s=2$) | $(B, 192, 15, 15)$ | $(B, 384, 7, 7)$ | Spatial reduction |
| **Stage 3** | ConvNeXt Block $\times 9$ | $(B, 384, 7, 7)$ | $(B, 384, 7, 7)$ | Core semantic feature extraction |
| **Downsample 3** | LayerNorm + Conv2d ($2\times 2, s=2$) | $(B, 384, 7, 7)$ | $(B, 768, 3, 3)$ | Deep semantic bottleneck |
| **Stage 4** | ConvNeXt Block $\times 3$ | $(B, 768, 3, 3)$ | $(B, 768, 3, 3)$ | Final deep spatial maps |
| **GeM Pooling** | Generalized Mean ($p=3.0$) | $(B, 768, 3, 3)$ | $(B, 768, 1, 1)$ | Learnable spatial focus |
| **Flatten** | Reshape | $(B, 768, 1, 1)$ | $(B, 768)$ | Feature vectorization |
| **Projector** | Linear $\rightarrow$ BN $\rightarrow$ GELU $\rightarrow$ Dropout $\rightarrow$ Linear | $(B, 768)$ | $(B, 256)$ | Dimensionality compression |
| **L2 Norm** | Normalization ($p=2$) | $(B, 256)$ | $(B, 256)$ | Unit hypersphere projection |

Following the GeM pooling layer, the features are projected into a normalized latent space using a two-layer Multi-Layer Perceptron (MLP): $\text{Linear}(768 \to 512) \to \text{BatchNorm1d}(512) \to \text{GELU} \to \text{Dropout}(0.1) \to \text{Linear}(512 \to 256)$. The final vector is $\ell_2$-normalized onto the hypersphere $\mathcal{S}^{255}$:
$$\mathbf{z} = \frac{\phi(\mathbf{g})}{\|\phi(\mathbf{g})\|_2} \in \mathbb{R}^{256}$$
Crucially, stable target representations over long training runs require tracking not just the convolutional weights, but also the floating-point running statistics of the `BatchNorm1d` layers. The EMA update is explicitly applied to the batch normalization running mean and variance buffers:
$$\mathbf{\mu}_{\text{target}} \leftarrow m\mathbf{\mu}_{\text{target}} + (1 - m)\mathbf{\mu}_{\text{online}}, \quad \mathbf{\sigma}^2_{\text{target}} \leftarrow m\mathbf{\sigma}^2_{\text{target}} + (1 - m)\mathbf{\sigma}^2_{\text{online}}$$
Failure to synchronize these buffers results in severe representation drift between the online and target spaces.

## Multi-Label Semantic InfoNCE Objective

Standard InfoNCE loss functions treat any non-matching sample in a mini-batch as a strict negative. In remote sensing, this leads to the "false-negative" problem: if two distinct geographic patches in the same batch both feature "Coniferous Forest" and "River", pushing them apart degrades the semantic manifold. 

### Pairwise Jaccard (IoU) Masking
To inject semantic awareness into the contrastive objective, we utilize the CORINE land-cover multi-labels. For any two samples $i$ and $j$ in a mini-batch of size $B$, with label sets $Y_i$ and $Y_j$, we compute the pairwise Jaccard index (Intersection over Union):
$$\text{IoU}_{i,j} = \frac{|Y_i \cap Y_j|}{|Y_i \cup Y_j| + 10^{-8}}$$
A binary mask $\mathcal{M}$ is constructed to suppress false negatives. If the semantic overlap exceeds a threshold $\tau_{\text{IoU}} = 0.40$, the pair is masked out from the contrastive denominator:
$$\mathcal{M}_{i,j} = \mathbb{I}(\text{IoU}_{i,j} > 0.40 \land i \neq j)$$

### Masked Symmetric InfoNCE
Let $\mathbf{u}_i$ be the online embedding of sample $i$ (e.g., SAR) and $\mathbf{v}_j$ be the target embedding of sample $j$ (e.g., Optical). The masked logits are defined as:
$$\tilde{\ell}_{i,j} = \begin{cases} -\infty \ (-10^9) & \text{if } \mathcal{M}_{i,j} = 1 \\ \frac{\mathbf{u}_i^\top \mathbf{v}_j}{\tau} & \text{otherwise} \end{cases}$$
where $\tau = 0.1$ is the temperature hyperparameter. The asymmetric masked InfoNCE loss for a batch is formulated as:
$$\mathcal{L}_{\text{InfoNCE}}(\mathbf{U}, \mathbf{V}) = \frac{1}{2B} \sum_{i=1}^{B} \left[ \left( -\frac{\mathbf{u}_i^\top \mathbf{v}_i}{\tau} + \log \sum_{j=1}^{B} \exp(\tilde{\ell}_{i,j}) \right) + \left( -\frac{\mathbf{v}_i^\top \mathbf{u}_i}{\tau} + \log \sum_{j=1}^{B} \exp(\tilde{\ell}_{j,i}) \right) \right]$$

### Composite Objective
To balance cross-modal alignment (sensor-to-sensor) with intra-modal semantic clustering (sensor-to-itself), the final loss is a weighted composite of four contrastive pairs:
$$\mathcal{L}_{\text{total}} = \frac{1}{2}\Big(\mathcal{L}_{\text{InfoNCE}}(\mathbf{Z}_{\text{S1}}, \mathbf{T}_{\text{S2}}) + \mathcal{L}_{\text{InfoNCE}}(\mathbf{Z}_{\text{S2}}, \mathbf{T}_{\text{S1}})\Big) + \frac{1}{4}\Big(\mathcal{L}_{\text{InfoNCE}}(\mathbf{Z}_{\text{S1}}, \mathbf{T}_{\text{S1}}) + \mathcal{L}_{\text{InfoNCE}}(\mathbf{Z}_{\text{S2}}, \mathbf{T}_{\text{S2}})\Big)$$
where $\mathbf{Z}$ represents the online predictions and $\mathbf{T}$ represents the target projections.