# Data Engineering, Radiometric Calibration & Pipeline Architecture

## Dataset Curation and Integrity Verification
This project utilizes the BigEarthNet-14K dataset, sourced from Kaggle (`narendraaironi/bigearthnet-14k`)[@sumbul2019]. The archive comprises paired Sentinel-1 Synthetic Aperture Radar (SAR) and Sentinel-2 optical imagery[@sumbul2019]. To ensure absolute alignment across modalities, a strict 1:1 filename mapping convention is enforced, linking the Sentinel-1 TIFF basenames (`s1_name`) to the corresponding Sentinel-2 TIFF basenames (`patch_id`)[@sumbul2019].

The integrity of the dataset is maintained through a programmatic disk-mapping protocol, resulting in a compiled `verified_paired_manifest.csv`. The verified corpus contains exactly 13,683 co-registered S1–S2 patch pairs, partitioned into static, non-overlapping geographic splits to prevent data leakage: 7,180 training pairs, 3,255 validation pairs, and 3,248 test pairs[@sumbul2019].

## Multi-Label Semantic Taxonomy
Unlike standard single-label benchmarks, BigEarthNet-14K annotations are derived from the CORINE Land Cover (CLC) database, providing a dense, multi-label semantic framework[@sumbul2019]. The metadata file (`metadata.parquet`) assigns one or multiple labels from a 43-class taxonomy to each image patch[@sumbul2019]. Parsing these multi-label annotations is critical for evaluating semantic similarity, as it allows the retrieval pipeline to calculate continuous Jaccard (IoU) overlap between disparate geographic scenes rather than relying on binary match/no-match criteria.

## Radiometric Calibration and Normalization
Raw geospatial rasters contain physical units that fall outside the optimal activation ranges of standard neural network backbones. Precise radiometric calibration is applied independently to each modality before tensor fusion.

### Sentinel-1 SAR Calibration
The Sentinel-1 inputs are structured as $2 \times 120 \times 120$ spatial tensors, corresponding to the VV and VH polarizations. The raw backscatter intensity, represented in decibels (dB), exhibits a long-tailed distribution. The values are clipped to physically meaningful lower and upper bounds of $-35.0\text{ dB}$ (noise floor over calm water) and $0.0\text{ dB}$ (strong corner reflectors), followed by linear min-max scaling to project the data into a $[0, 1]$ range:
$$\tilde{\mathbf{X}}_{\text{S1}} = \frac{\text{clip}(\mathbf{X}_{\text{S1}}, -35.0, 0.0) + 35.0}{35.0} \in [0, 1]^{2 \times 120 \times 120}$$

### Sentinel-2 Optical Calibration
The Sentinel-2 inputs are formatted as $10 \times 120 \times 120$ tensors, encompassing the visible, near-infrared, and shortwave-infrared bands. The raw pixel values represent bottom-of-atmosphere surface reflectance scaled by a factor of 10,000. These are normalized by dividing by the scale factor and clamping the upper bound to $1.0$ to handle specular anomalies or saturated cloud pixels:
$$\tilde{\mathbf{X}}_{\text{S2}} = \text{clip}\left(\frac{\mathbf{X}_{\text{S2}}}{10000.0}, 0.0, 1.0\right) \in [0, 1]^{10 \times 120 \times 120}$$

## Co-Registered Geometric Augmentations
To artificially expand the training distribution while strictly preserving the pixel-level spatial alignment between the heterogeneous sensors, synchronized geometric augmentations are applied during the training phase. We utilize the dihedral group $D_4$, which includes horizontal flips, vertical flips, and $90^\circ k$ rotations ($k \in \{0, 1, 2, 3\}$). 

When a random transformation $T \in D_4$ is sampled for a given mini-batch index $i$, it is applied identically to both modalities:
$$(\tilde{\mathbf{X}}_{\text{S1}, i}^{\text{aug}}, \tilde{\mathbf{X}}_{\text{S2}, i}^{\text{aug}}) = (T(\tilde{\mathbf{X}}_{\text{S1}, i}), T(\tilde{\mathbf{X}}_{\text{S2}, i}))$$
This synchronization guarantees that the structural geometry (e.g., the orientation of a river or the layout of an urban grid) remains perfectly matched across the active and passive sensor inputs.