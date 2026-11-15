# Neural Architecture and Mathematical Formulation

\begin{center}
\includegraphics[width=\textwidth,keepaspectratio]{images/training_pipeline_final.pdf}
\captionof{figure}{End-to-end asymmetric dual-tower training pipeline. Co-registered Sentinel-1 SAR and Sentinel-2 optical image pairs undergo synchronized $D_4$ dihedral augmentations. The system routes these pairs into online encoders ($\theta_q$) and momentum-updated target encoders ($\theta_k$, $m=0.996$). A composite multi-label Jaccard-masked InfoNCE loss ($\tau_{\text{IoU}}=0.40$) optimizes the latent projections on the 256-dimensional unit hypersphere. This loss executes cross-modal alignment and intra-modal self-distillation while it suppresses false negatives.}
\label{fig:training-pipeline}
\end{center}

## Sensor-Specific Stem Adaptation

The system uses ConvNeXt-Tiny[@liu2022] as the backbone for both the SAR and Optical branches to extract hierarchical features. The standard ConvNeXt architecture starts with a "patchify" stem. This stem is a convolutional layer with a $4 \times 4$ kernel and a stride of 4 (`Conv2d(3, 96, 4, 4)`), which ImageNet-1K pre-trained for 3-channel RGB inputs.

Remote sensing tensors use different channel depths ($C=2$ for SAR and $C=10$ for Optical). Therefore, the system must adapt the first convolutional layer to accept these modalities while it preserves the pre-trained variance, energy, and bias ($\mathbf{b}_{\text{RGB}} \in \mathbb{R}^{96}$) of the ImageNet weights $\mathbf{W}_{\text{RGB}} \in \mathbb{R}^{96 \times 3 \times 4 \times 4}$.

### SAR 2-Channel Initialization

For the 2-channel Sentinel-1 input ($\text{VV}, \text{VH}$), the system computes the mean across the 3 RGB channels of the pre-trained weights. It scales this mean to conserve the original energy, repeats it across the 2 SAR channels to create the new stem weights $\mathbf{W}_{\text{S1}}$, and copies the pre-trained stem bias when present:

$$\mathbf{W}_{\text{S1}} = \text{repeat}_2\left(\frac{1}{3}\sum_{c=1}^{3} \mathbf{W}_{\text{RGB}}[:, c, :, :]\right) \in \mathbb{R}^{96 \times 2 \times 4 \times 4}, \qquad \mathbf{b}_{\text{S1}} = \mathbf{b}_{\text{RGB}}$$

### Optical 10-Channel Initialization

For the 10-channel Sentinel-2 input, the system tiles and scales the pre-trained weights. It tiles the original 3-channel weights to cover 10 dimensions, scales the output by $3/10$ to match the expected ImageNet pre-trained activation magnitude, and copies the original stem bias:

$$\mathbf{W}_{\text{S2}} = \frac{3}{10} \left(\text{repeat}_4(\mathbf{W}_{\text{RGB}})\right)[:, :10, :, :] \in \mathbb{R}^{96 \times 10 \times 4 \times 4}, \qquad \mathbf{b}_{\text{S2}} = \mathbf{b}_{\text{RGB}}$$

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

Standard convolutional architectures usually end feature extraction with Global Average Pooling (GAP). In image retrieval, localized terrain features such as a small bridge or an isolated building are critical. GAP dilutes these strong localized activations into the background noise.

To prevent this signal loss, the system uses Generalized Mean (GeM) spatial pooling[@radenovic2018]. From the final convolutional feature map $\mathbf{F} \in \mathbb{R}^{768 \times H' \times W'}$, the system computes the GeM descriptor $\mathbf{g} \in \mathbb{R}^{768}$:

$$\mathbf{g} = \left( \frac{1}{H'W'} \sum_{u=1}^{H'} \sum_{v=1}^{W'} \max(\mathbf{F}_{:, u, v}, \epsilon)^p \right)^{\frac{1}{p}}$$

Here, $p$ is a learnable pooling parameter initialized at $p=3.0$, and $\epsilon=10^{-6}$ maintains numerical stability during backpropagation. The GeM equation provides a continuous generalization. It reduces to GAP as $p \to 1$, and it approaches Global Max Pooling (GMP) as $p \to \infty$. Because the network learns $p$, the system dynamically balances background context with salient object activations.

## Asymmetric Momentum Dual-Tower Dynamics

To align the heterogeneous SAR and Optical domains, the system uses an asymmetric online–target framework (`CrossModalSAROpticalNet`). This design follows self-supervised momentum architectures such as MoCo[@he2020] and BYOL[@grill2020].

### Online and Target Encoders

The architecture maintains two pairs of networks:

* **Online Networks ($\theta_q$):** These networks operate in training mode and process heavily augmented views of the SAR and Optical batches $(\tilde{\mathbf{X}}_{\text{S1}}^{\text{aug}}, \tilde{\mathbf{X}}_{\text{S2}}^{\text{aug}})$. Standard gradient descent updates their parameters dynamically.

* **Target Networks ($\theta_k$):** These networks are locked in evaluation mode and process the original, unaugmented views $(\tilde{\mathbf{X}}_{\text{S1}}^{\text{orig}}, \tilde{\mathbf{X}}_{\text{S2}}^{\text{orig}})$. They do not receive gradients. Instead, an Exponential Moving Average (EMA) of the online network parameters updates the target parameters:

$$\theta_k \leftarrow m\theta_k + (1 - m)\theta_q$$

where $m = 0.996$ is the momentum coefficient.

### Projection Head and Buffer Synchronization

| Stage / Component | Operation | Input Shape | Output Shape | Spatial / Feature Details |
| --- | --- | --- | --- | --- |
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

After the GeM pooling layer, a two-layer Multi-Layer Perceptron (MLP) projects the features into a normalized latent space. This MLP follows the sequence $\text{Linear}(768 \to 512) \to \text{BatchNorm1d}(512) \to \text{GELU} \to \text{Dropout}(0.1) \to \text{Linear}(512 \to 256)$. The system then $\ell_2$-normalizes the final vector onto the hypersphere $\mathcal{S}^{255}$:

$$\mathbf{z} = \frac{\phi(\mathbf{g})}{\Vert{}\phi(\mathbf{g})\Vert{}_2} \in \mathbb{R}^{256}$$

To keep target representations stable over long training runs, the system must track both the convolutional weights and the floating-point running statistics of the `BatchNorm1d` layers. The system applies the EMA update directly to the batch normalization running mean and variance buffers:

$$\mathbf{\mu}_{\text{target}} \leftarrow m\mathbf{\mu}_{\text{target}} + (1 - m)\mathbf{\mu}_{\text{online}}, \quad \mathbf{\sigma}^2_{\text{target}} \leftarrow m\mathbf{\sigma}^2_{\text{target}} + (1 - m)\mathbf{\sigma}^2_{\text{online}}$$

If the system does not synchronize these buffers, severe representation drift occurs between the online and target spaces.

## Multi-Label Semantic InfoNCE Objective

Standard InfoNCE loss functions treat every non-matching sample in a mini-batch as a strict negative. In remote sensing, this rule creates a false-negative problem. For example, two distinct geographic patches in the same batch can both contain "Coniferous Forest" and "River". Pushing these semantically similar patches apart degrades the semantic manifold.

### Pairwise Jaccard (IoU) Masking

To add semantic awareness to the contrastive objective, the system uses CORINE land-cover multi-labels. For any two samples $i$ and $j$ in a mini-batch of size $B$ with label sets $Y_i$ and $Y_j$, the system calculates the pairwise Jaccard index (Intersection over Union):

$$\text{IoU}_{i,j} = \frac{\vert{}Y_i \cap Y_j\vert{}}{\vert{}Y_i \cup Y_j\vert{} + 10^{-8}}$$

The system builds a binary mask $\mathcal{M}$ to suppress false negatives. When the semantic overlap exceeds the threshold $\tau_{\text{IoU}} = 0.40$, the mask removes that pair from the contrastive denominator:

$$\mathcal{M}_{i,j} = \mathbb{I}(\text{IoU}_{i,j} > 0.40 \land i \neq j)$$

### Masked Symmetric InfoNCE

For two batches of normalized embeddings $\mathbf{U} = [\mathbf{u}_1, \ldots, \mathbf{u}_B]^\top$ (e.g., online SAR representations) and $\mathbf{V} = [\mathbf{v}_1, \ldots, \mathbf{v}_B]^\top$ (e.g., target Optical representations), define the temperature-scaled cosine similarity matrix $S_{i,j}$ and the false-negative masked logit matrix $\widetilde{S}_{i,j}$ as:

$$S_{i,j} = \frac{\mathbf{u}_i^\top \mathbf{v}_j}{\tau}, \qquad \widetilde{S}_{i,j} = \begin{cases} -10^9, & \mathcal{M}_{i,j} = 1, \\ S_{i,j}, & \mathcal{M}_{i,j} = 0, \end{cases}$$

where $\tau = 0.1$ is the temperature hyperparameter. The symmetric masked InfoNCE loss is formulated as:

$$\mathcal{L}_{\mathrm{sym}}(\mathbf{U}, \mathbf{V}; \mathcal{M}) = \frac{1}{2B} \sum_{i=1}^{B} \left[ -S_{i,i} + \log \sum_{j=1}^{B} \exp(\widetilde{S}_{i,j}) - S_{i,i} + \log \sum_{j=1}^{B} \exp(\widetilde{S}_{j,i}) \right]$$

Here, the diagonal pair $(i, i)$ serves as the true geographic positive pair. Off-diagonal pairs with high semantic overlap ($\mathcal{M}_{i,j} = 1$) are suppressed in the denominator via $-10^9$ rather than being added as extra positives, preserving instance-level spatial discrimination while preventing false-negative repulsion.

### Composite Objective

To balance cross-modal alignment (sensor-to-sensor) with intra-modal semantic clustering (sensor-to-itself), the system calculates the final loss as a weighted sum of four contrastive pairs:

$$\mathcal{L}_{\text{total}} = \frac{1}{2}\Big(\mathcal{L}_{\mathrm{sym}}(\mathbf{Z}_{\text{S1}}, \mathbf{T}_{\text{S2}}; \mathcal{M}) + \mathcal{L}_{\mathrm{sym}}(\mathbf{Z}_{\text{S2}}, \mathbf{T}_{\text{S1}}; \mathcal{M})\Big) + \frac{1}{4}\Big(\mathcal{L}_{\mathrm{sym}}(\mathbf{Z}_{\text{S1}}, \mathbf{T}_{\text{S1}}; \mathcal{M}) + \mathcal{L}_{\mathrm{sym}}(\mathbf{Z}_{\text{S2}}, \mathbf{T}_{\text{S2}}; \mathcal{M})\Big)$$

where $\mathbf{Z}$ denotes the online predictions and $\mathbf{T}$ denotes the target projections.

### Algorithmic Training Protocol

The complete end-to-end training procedure—combining sensor-specific stem initialization, synchronized $D_4$ augmentations, false-negative masked contrastive optimization, and momentum buffer updates—is formalized below:

**Algorithm 4.1: Asymmetric Momentum Cross-Modal Training**

* **Require:** Paired training set $\mathcal{D}_{\mathrm{train}}$ with multi-labels $Y$; validation set $\mathcal{D}_{\mathrm{val}}$; batch size $B=32$; epochs $E=30$; momentum $m=0.996$; temperature $\tau=0.1$; IoU threshold $\tau_{\mathrm{IoU}}=0.40$; initial learning rate $\eta_0=3\times10^{-4}$; weight decay $10^{-3}$; minimum learning rate $\eta_{\min}=10^{-6}$.
* **Ensure:** Best saved model checkpoint selected using validation retrieval metrics.

1. Normalize SAR inputs by clipping to $[-35, 0]$ dB and scaling to $[0, 1]$.
2. Normalize optical inputs by dividing by $10{,}000$ and clipping to $[0, 1]$.
3. Initialize SAR and optical online towers $\theta_q$ from pretrained ConvNeXt-Tiny.
4. Adapt the SAR stem by averaging RGB stem weights and repeating them over 2 channels; copy the original stem bias when present.
5. Adapt the optical stem by tiling RGB stem weights to 10 channels and scaling them by $3/10$; copy the original stem bias when present.
6. Add learnable GeM pooling initialized at $p=3.0$, and an MLP projector $768 \to 512 \to 256$ with `BatchNorm1d`, `GELU`, and dropout $0.1$.
7. Initialize target towers $\theta_k \gets \mathrm{copy}(\theta_q)$ and disable gradients for $\theta_k$.
8. Initialize AdamW on $\theta_q$ and a cosine-annealing scheduler with $T_{\max}=E$ and $\eta_{\min}=10^{-6}$.
9. **For** epoch $e \gets 1$ **to** $E$ **do:**
   a. Set online towers $\theta_q$ to training mode and target towers $\theta_k$ to evaluation mode.
   b. **For** each mini-batch $(\mathbf{X}_{\mathrm{S1}}, \mathbf{X}_{\mathrm{S2}}, Y)$ of size $B$ **do:**
      i. Zero optimizer gradients.
      ii. Sample a random horizontal flip, vertical flip, and $90^\circ k$ rotation ($D_4$) for each pair; apply the identical transformation to both modalities to obtain $(\widetilde{\mathbf{X}}_{\mathrm{S1}}, \widetilde{\mathbf{X}}_{\mathrm{S2}})$.
      iii. Compute online embeddings $\mathbf{Z}_{\mathrm{S1}} \gets f^{q}_{\mathrm{S1}}(\widetilde{\mathbf{X}}_{\mathrm{S1}})$ and $\mathbf{Z}_{\mathrm{S2}} \gets f^{q}_{\mathrm{S2}}(\widetilde{\mathbf{X}}_{\mathrm{S2}})$.
      iv. Compute target embeddings $\mathbf{T}_{\mathrm{S1}} \gets f^{k}_{\mathrm{S1}}(\mathbf{X}_{\mathrm{S1}})$ and $\mathbf{T}_{\mathrm{S2}} \gets f^{k}_{\mathrm{S2}}(\mathbf{X}_{\mathrm{S2}})$ without gradients.
      v. Compute pairwise label overlap $\mathrm{IoU}_{i,j} \gets \vert Y_i \cap Y_j \vert / (\vert Y_i \cup Y_j \vert + 10^{-8})$.
      vi. Set false-negative mask $\mathcal{M}_{i,j} \gets \mathbb{I}(\mathrm{IoU}_{i,j} > \tau_{\mathrm{IoU}} \land i \neq j)$.
      vii. Compute the composite loss:
           $$\mathcal{L}_{\text{total}} \gets \frac{1}{2}\Big[\mathcal{L}_{\mathrm{sym}}(\mathbf{Z}_{\mathrm{S1}}, \mathbf{T}_{\mathrm{S2}}; \mathcal{M}) + \mathcal{L}_{\mathrm{sym}}(\mathbf{Z}_{\mathrm{S2}}, \mathbf{T}_{\mathrm{S1}}; \mathcal{M})\Big] + \frac{1}{4}\Big[\mathcal{L}_{\mathrm{sym}}(\mathbf{Z}_{\mathrm{S1}}, \mathbf{T}_{\mathrm{S1}}; \mathcal{M}) + \mathcal{L}_{\mathrm{sym}}(\mathbf{Z}_{\mathrm{S2}}, \mathbf{T}_{\mathrm{S2}}; \mathcal{M})\Big]$$
      viii. **If** CUDA is available, use AMP FP16 and `GradScaler` to backpropagate $\mathcal{L}_{\text{total}}$ and update $\theta_q$; **else** backpropagate $\mathcal{L}_{\text{total}}$ and update $\theta_q$ in full precision.
      ix. Update target parameters via in-place EMA: $\theta_k \gets m\theta_k + (1 - m)\theta_q$.
      x. Apply the same EMA update to floating-point buffers, including `BatchNorm1d` running statistics ($\mathbf{\mu}_{\text{target}}, \mathbf{\sigma}^2_{\text{target}}$).
   c. Step the cosine-annealing learning-rate scheduler.
   d. **If** $(e \bmod 3 = 0)$ **or** $(e = E)$ **then:**
      i. Evaluate the target towers on $\mathcal{D}_{\mathrm{val}}$ without centering in the SAR-to-Optical direction.
      ii. Compute $\mathrm{Score}_{\mathrm{val}} = \text{Geo R@5} + \text{Semantic F1@5}$.
      iii. Save the full model checkpoint if $\mathrm{Score}_{\mathrm{val}}$ is the best observed so far.
10. **Return** the best saved model checkpoint.
