# Centrifuge analogy demo

In this 2-D great-circle cross-section of $\mathbb{S}^{255}$, the cyan **Online SAR Query Anchor ($u_i$)** is fixed at the North Pole ($\theta_i = 0^\circ$, $\cos\theta = 1.000$), while the four **Momentum Target Key vectors** ($v_i^+, v_j, v_k^-, v_m^-$) move along the perimeter of the circle according to the exact Riemannian gradient of the InfoNCE loss.

## Which Vectors Move Apart (or Together), When, and What It Signifies

### **A. Co-Registered Optical Positive ($v_i^+$, Gold Sphere)**

* **Initial Position:** Northern hemisphere near the anchor ($\theta_i^+ \approx 28^\circ$, $\cos\theta = +0.883$).
* **When It Moves:** Every gradient step, under **both** Armed and Bypassed modes.
* **Direction:** Moves **upward (North) toward $u_i$** ($\theta_i^+$ decreases, $\cos\theta$ approaches $1.0$).
* **Magnitude Changes:**
* **When Gate is ARMED (on $\text{IoU} > 0.40$):** Because the high-similarity neighbor $v_j$ is masked out of the softmax denominator, $P_{i,i}$ is already very close to $1.0$ ($\approx 0.99997$). Since the attraction force is proportional to $(1 - P_{i,i})/\tau$, $v_i^+$ takes a tiny, gentle tightening step.
* **When Gate is BYPASSED (or $\text{IoU} \le 0.40$):** The unmasked neighbor $v_j$ steals a large share of the softmax denominator ($P_{i,j} \approx 0.226$), dropping $P_{i,i}$ to $\approx 0.774$. Consequently, $(1 - P_{i,i})/\tau$ spikes to $\approx 2.26$, pulling $v_i^+$ much harder toward $u_i$ to fight off the perceived competition.


* **What It Signifies:** **Cross-modal alignment.** The model is pulling the optical representation of the exact same geographic patch into alignment with its radar representation.

### **B. Semantic Neighbor / Candidate ($v_j$, Purple/Amber Sphere)**

* **Initial Position:** Northern hemisphere ($\theta_j \approx 40.5^\circ$, $\cos\theta = +0.760$). It sits close to $u_i$ because it represents a *different* geographic tile that shares the same land-cover classes (e.g., another dense Urban patch).
* **Case 1 — Gate ARMED & $\text{IoU}_{i,j} > 0.40$ (e.g., Urban $4/6 = 0.67$ or Forest $3/6 = 0.50$):**
* **Motion:** **Completely stationary ($\Delta\theta_j = 0.0000\text{ rad}$).** It neither moves toward $u_i$ nor moves away from $u_i$.
* **What It Signifies:** **False-negative suppression.** The Jaccard gate recognizes that tile $j$ depicts the same land cover as tile $i$. By setting $l_{i,j} = -10^9$, it zeroes out the softmax probability ($P_{i,j} = 0$) and cuts off the repulsive gradient. The model leaves the existing intra-class cluster intact without artificially forcing two distinct locations to collapse onto the exact same point.

* **Case 2 — Gate BYPASSED (Standard InfoNCE) OR $\text{IoU}_{i,j} \le 0.40$ (e.g., Boundary $2/5 = 0.40$ or Pasture $1/6 = 0.17$):**
* **Motion:** Moves **violently downward (South) away from $u_i$** ($\theta_j$ increases toward $180^\circ$, $\cos\theta_j$ drops rapidly).
* **Why It Moves So Fast:** Because $v_j$ starts close to the anchor ($\cos\theta_j = +0.760$) and temperature is sharp ($\tau = 0.10$), its exponential $\exp(7.60) \approx 1998.2$ is huge compared to distant negatives. Standard InfoNCE treats high-similarity non-anchors as severe **hard negatives** and concentrates nearly all repulsive force onto $v_j$.
* **What It Signifies:**
* When $\text{IoU}_{i,j} > 0.40$ (Bypassed mode): **Pathological cluster fracturing.** The loss is tearing apart two identical urban scenes solely because they have different batch indices ($j \neq i$).
* When $\text{IoU}_{i,j} \le 0.40$ (e.g., Pasture vs. Urban): **Legitimate hard-negative mining.** If an unrelated land-cover patch accidentally sits near the urban cluster, the unmasked gradient rightfully pushes it away toward the southern hemisphere.

### **C. True Negatives ($v_k^-, v_m^-$, Slate/Red Spheres)**

* **Initial Position:** Southern hemisphere ($\cos\theta_k = -0.150$ at $\theta_k \approx 98.6^\circ$, and $\cos\theta_m = -0.520$ at $\theta_m \approx 121.3^\circ$).
* **When They Move:** Every gradient step, moving **southward away from $u_i$**, though by a very small angular increment.
* **Why Their Motion Is Small:** Because they are already in the southern hemisphere ($\cos\theta < 0$), their temperature-scaled logits are negative ($-1.50$ and $-5.20$), making $\exp(l_{i,k}) \approx 0.223$ and $\exp(l_{i,m}) \approx 0.0055$ tiny compared to $\exp(+8.83) \approx 6836.3$ at the north pole.
* **What It Signifies:** **Asymptotic contrastive margin.** In InfoNCE, once true negatives are pushed past the equator into the southern hemisphere, their softmax weight $P_{i,m}$ decays exponentially toward zero—meaning the gradient naturally turns off once negatives are sufficiently separated.

## What Each Column in the Live Telemetry Table Means

Each row $m \in \{v_i^+, v_j, v_k^-, v_m^-\}$ in the **Live $1\times 4$ Softmax & Riemannian Gradient Table** shows the exact chain of derivatives from raw geometry to the angular step on the sphere:

| Column Header | Mathematical Definition | Physical & Optimization Meaning |
| --- | --- | --- |
| **$s_m = u_i^\top v_m$** | $\cos\theta_m \in [-1, 1]$ | **Cosine Similarity:** The dot product between the North Pole anchor $u_i$ and key vector $v_m$. $+1.0$ is identical direction (North Pole), $0.0$ is orthogonal (Equator), and $-1.0$ is antipodal (South Pole). |
| **Logit $l_{i,m}$** | $\begin{cases} -10^9 & \text{if } M_{i,m}=1 \\ s_m / \tau & \text{if } M_{i,m}=0 \end{cases}$ | **Shunted Temperature Logit:** Divides cosine similarity by $\tau = 0.10$ (multiplying by $10\times$) to sharpen the distribution. If the Jaccard gate triggers ($M_{i,j} = 1$), this cell switches to $-10^9$ (`SHUNTED`), cutting it out of the softmax sum. |
| **Softmax $P_{i,m}$** | $\dfrac{\exp(l_{i,m})}{\sum_{r} \exp(l_{i,r})} \in [0, 1]$ | **Contrastive Probability Mass:** How much weight the softmax assigns to key $v_m$. Across all 4 rows, this column always sums to $1.0000$. For negatives ($m \neq i$), $P_{i,m}$ directly measures what fraction of the repulsive budget is aimed at sample $m$. |
| **$\partial\mathcal{L}_i / \partial s_m$** | $\begin{cases} (P_{i,i} - 1)/\tau & m = i \\ +P_{i,m}/\tau & m \neq i \end{cases}$ | **Scalar Similarity Derivative:** The derivative of $\mathcal{L}_i = -\log P_{i,i}$ with respect to cosine similarity $s_m$. Negative values (e.g., $-2.26$ on $v_i^+$) mean decreasing loss requires *increasing* similarity (attraction); positive values (e.g., $+2.26$ on $v_j$) mean decreasing loss requires *decreasing* similarity (repulsion). |
| **$\Vert{}\nabla_{\mathbb{S}^{255}}\mathcal{L}_i\Vert{}$** | $\left\vert{}\dfrac{\partial\mathcal{L}_i}{\partial s_m}\right\vert{} \sin\theta_m$ | **Riemannian Tangent Force Norm:** Projects the Euclidean force onto the surface of the sphere via the $\sin\theta_m = \sqrt{1 - s_m^2}$ Jacobian factor. This is the exact length of the 3D force arrow drawn on the sphere. |
| **Step $\Delta\theta_m$** | $+\eta_{\theta}\,\dfrac{\partial\mathcal{L}_i}{\partial s_m}\sin\theta_m$ | **Geodesic Angular Step (in radians):** The exact angle (with step size $\eta_{\theta} = 0.025\text{ rad}$) by which vector $v_m$ rotates along the circle when you click **Step Gradient (+1)**. Negative $\Delta\theta$ rotates North (toward $0^\circ$); positive $\Delta\theta$ rotates South (toward $180^\circ$); $0.0000$ means the vector is locked in place. |

## What the Top Summary Readouts Mean

Above the table, three counterfactual metrics summarize the state of the step:

1. **Active Loss $\mathcal{L}_i = -\log P_{i,i}$ (Masked vs. Unmasked):** Shows the cross-entropy loss in nats. At Step 0 on Batch A, masking $v_j$ drops the loss from **$0.2564\text{ nats}$** (where $v_j$ is treated as an error) to **$0.00003\text{ nats}$** (recognizing that the positive pair is already well separated from true negatives $v_k^-, v_m^-$).
2. **Candidate Repulsion Share ($P_{i,j}$):** Shows the exact percentage of denominator repulsion directed at $v_j$ ($0.00\%$ when gated vs. $22.62\%$ when unmasked).
3. **Cumulative $\Delta\cos\theta_j$:** Tracks how far the semantic neighbor $v_j$ has drifted from its starting similarity ($+0.760$) across all batches you have stepped so far ($0.000$ when gated vs. negative drift when unmasked).