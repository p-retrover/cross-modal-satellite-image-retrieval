# Appendices

## Appendix A: CORINE Land Cover (CLC) Multi-Label Taxonomy

The BigEarthNet-14K dataset utilizes the CORINE Land Cover (CLC) database nomenclature, representing a hierarchical multi-label taxonomy of 43 distinct environmental and urban classes. The retrieval pipeline tracks semantic overlap (Jaccard Index / Dice Coefficient) across these classes to evaluate true geographic generalization.

### Level 1: Artificial Surfaces
*   **1.1 Urban fabric:** Continuous urban fabric, Discontinuous urban fabric.
*   **1.2 Industrial, commercial and transport units:** Industrial or commercial units, Road and rail networks and associated land, Port areas, Airports.
*   **1.3 Mine, dump and construction sites:** Mineral extraction sites, Dump sites, Construction sites.
*   **1.4 Artificial, non-agricultural vegetated areas:** Green urban areas, Sport and leisure facilities.

### Level 2: Agricultural Areas
*   **2.1 Arable land:** Non-irrigated arable land, Permanently irrigated land, Rice fields.
*   **2.2 Permanent crops:** Vineyards, Fruit trees and berry plantations, Olive groves.
*   **2.3 Pastures:** Pastures.
*   **2.4 Heterogeneous agricultural areas:** Annual crops associated with permanent crops, Complex cultivation patterns, Land principally occupied by agriculture with significant areas of natural vegetation, Agro-forestry areas.

### Level 3: Forest and Semi-Natural Areas
*   **3.1 Forests:** Broad-leaved forest, Coniferous forest, Mixed forest.
*   **3.2 Scrub and/or herbaceous vegetation associations:** Natural grasslands, Moors and heathland, Sclerophyllous vegetation, Transitional woodland-shrub.
*   **3.3 Open spaces with little or no vegetation:** Beaches/dunes/sands, Bare rocks, Sparsely vegetated areas, Burnt areas, Glaciers and perpetual snow.

### Level 4 & 5: Wetlands and Water Bodies
*   **4.1 Inland wetlands:** Inland marshes, Peat bogs.
*   **4.2 Maritime wetlands:** Salt marshes, Salines, Intertidal flats.
*   **5.1 Inland waters:** Water courses, Water bodies.
*   **5.2 Marine waters:** Coastal lagoons, Estuaries, Sea and ocean.

---

## Appendix B: Hyperparameter Grid and Hardware Configuration

The system was optimized for a single-GPU cloud runtime (e.g., NVIDIA T4 / L4 on Google Colab) utilizing Automatic Mixed Precision (AMP) to maximize batch throughput and minimize VRAM consumption.

| Hyperparameter / Configuration | Value / Setting |
| :--- | :--- |
| **Hardware / Engine** | 1x NVIDIA T4 (16GB VRAM), PyTorch 2.x |
| **Precision** | `torch.cuda.amp.GradScaler` (FP16 / FP32 Mixed) |
| **Epochs** | 30 |
| **Batch Size ($B$)** | 32 (yielding a $32 \times 32$ contrastive matrix) |
| **Optimizer** | AdamW |
| **Base Learning Rate ($\eta$)** | $3 \times 10^{-4}$ |
| **Weight Decay** | $1 \times 10^{-3}$ |
| **LR Schedule** | Cosine Annealing ($\eta_{\text{min}} = 10^{-6}$) |
| **EMA Target Momentum ($m$)** | $0.996$ |
| **GeM Initial Exponent ($p$)** | $3.0$ (Learnable) |
| **InfoNCE Temperature ($\tau$)** | $0.1$ |
| **IoU False-Negative Threshold ($\tau_{\text{IoU}}$)** | $0.40$ |

---

## Appendix C: Core PyTorch Implementation Listings

### C.1 ConvNeXt-Tiny Stem Channel Adaptation
This snippet demonstrates the physical energy-conserving channel adaptation for the SAR (2-channel) and Optical (10-channel) branches.

```python
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

def adapt_convnext_stem(model, in_channels, modality='SAR'):
    original_stem = model.features[0][0]
    original_weights = original_stem.weight.data # [96, 3, 4, 4]
    
    new_stem = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)
    
    if modality == 'SAR':
        # Mean across RGB, repeated for VV and VH
        mean_weight = original_weights.mean(dim=1, keepdim=True)
        new_stem.weight.data = mean_weight.repeat(1, in_channels, 1, 1)
    elif modality == 'Optical':
        # Tile across 10 channels, scale by 3/10 to conserve energy
        tiled_weights = original_weights.repeat(1, 4, 1, 1)[:, :in_channels, :, :]
        new_stem.weight.data = tiled_weights * (3.0 / in_channels)
        
    new_stem.bias.data = original_stem.bias.data
    model.features[0][0] = new_stem
    return model

```

### C.2 Generalized Mean (GeM) Spatial Pooling

The learnable generalized mean pooling layer replacing standard Global Average Pooling.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeMPool(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeMPool, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: [B, C, H, W]
        x_clamped = x.clamp(min=self.eps).pow(self.p)
        x_pooled = F.avg_pool2d(x_clamped, (x.size(-2), x.size(-1)))
        return x_pooled.pow(1.0 / self.p).flatten(1)

```

### C.3 False-Negative Masked Semantic InfoNCE Loss

The asymmetric contrastive loss utilizing the multi-label Jaccard (IoU) overlap to mask semantically identical terrain patches.

```python
import torch
import torch.nn.functional as F

def semantic_infonce_loss(online_z, target_z, labels, tau=0.1, tau_iou=0.40):
    """
    online_z: [B, D] normalized embeddings from online encoder (e.g., SAR)
    target_z: [B, D] normalized embeddings from target encoder (e.g., Optical)
    labels: [B, Num_Classes] binary multi-label matrix
    """
    B = online_z.size(0)
    
    # 1. Compute Pairwise Multi-Label IoU
    intersection = torch.mm(labels, labels.t())
    area = labels.sum(dim=1).unsqueeze(1)
    union = area + area.t() - intersection
    iou_matrix = intersection / (union + 1e-8)
    
    # 2. Create False-Negative Mask (Mask out off-diagonal identical scenes)
    off_diagonal_mask = ~torch.eye(B, dtype=torch.bool, device=online_z.device)
    false_negative_mask = (iou_matrix > tau_iou) & off_diagonal_mask
    
    # 3. Compute scaled logits
    logits = torch.mm(online_z, target_z.t()) / tau
    
    # 4. Apply Mask (Set false negatives to -infinity so exp() -> 0)
    logits.masked_fill_(false_negative_mask, -1e9)
    
    # 5. Contrastive Cross Entropy (Targets are the diagonal)
    target_indices = torch.arange(B, device=online_z.device)
    loss = F.cross_entropy(logits, target_indices)
    
    return loss

```
