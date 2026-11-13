# Workings of the Sentinel Constellation

The physical dichotomy between active microwave backscatter and passive solar reflectance defines the representational gap in cross-modal remote sensing. Bridging this gap requires a rigorous understanding of the underlying sensor physics, energy propagation, and radiometric calibration processes that ultimately produce the structured tensor spaces processed by metric learning architectures. This chapter details the operational mechanics of the European Space Agency’s (ESA) Sentinel-1 and Sentinel-2 missions.

## Sentinel-1: Active Synthetic Aperture Radar (SAR)

Sentinel-1 operates as an active sensor, carrying a C-band Synthetic Aperture Radar (SAR) instrument operating at a center frequency of 5.405 GHz (corresponding to a wavelength of roughly 5.6 cm). Unlike optical sensors, SAR provides its own source of illumination via microwave pulses, rendering it invariant to solar illumination cycles and largely impervious to cloud cover, aerosols, and atmospheric scattering.

### Microwave Scattering and Dielectric Physics

As Sentinel-1 orbits, its side-looking antenna transmits coherent electromagnetic pulses toward the Earth's surface. The instrument records the amplitude and phase of the backscattered echoes. The intensity of this return signal is governed by two primary surface characteristics:

* **Surface Roughness:** According to the Rayleigh criterion, surfaces that are smooth relative to the 5.6 cm wavelength (e.g., calm water bodies, paved runways) cause specular reflection away from the sensor, yielding a dark radar return. Rough surfaces (e.g., urban fabric, forest canopies) cause diffuse volumetric scattering, returning high-intensity signals to the antenna.
* **Dielectric Constant:** The electrical properties of the target material, heavily influenced by moisture content, dictate radar penetration and reflectivity. Metallic urban structures and wet soils exhibit high dielectric constants, producing strong radar returns, whereas dry soils and vegetation permit deeper microwave penetration.

### Dual-Polarization and Tensor Representation

Sentinel-1 typically operates in Interferometric Wide (IW) swath mode, utilizing dual polarization to capture distinct scattering mechanisms. The sensor transmits a vertically polarized wave and receives both vertically and horizontally polarized returns:

* **VV (Vertical Transmit, Vertical Receive):** Highly sensitive to surface scattering, capturing bare soil interfaces, water boundaries, and direct structural reflections.
* **VH (Vertical Transmit, Horizontal Receive):** Sensitive to volume scattering, capturing depolarization effects caused by complex geometric structures like tree canopies and irregular urban matrices.

In deep learning pipelines, these dual polarizations are stacked to form a 2-channel spatial tensor ($X_{\text{S1}} \in \mathbb{R}^{B \times 2 \times H \times W}$).

### Radiometric Calibration

Raw SAR data is heavily skewed and contains multiplicative speckle noise inherent to coherent imaging. To stabilize the signal for neural network ingestion, the raw complex data is calibrated into a normalized radar cross-section ($\sigma^0$). This backscatter coefficient is typically converted to a logarithmic decibel scale ($\sigma^0_{\text{dB}} = 10 \log_{10} \sigma^0$). Because terrestrial SAR backscatter generally falls within a predictable dynamic range, the data is commonly clipped and min-max normalized (e.g., bounding $\sigma^0_{\text{dB}} \in [-35, 0]$) to map the signal into a strictly positive $[0, 1]$ domain, ensuring stable gradients during early spatial feature extraction.

## Sentinel-2: Passive Multi-Spectral Imaging (MSI)

Sentinel-2 complements the active radar of Sentinel-1 by carrying a passive Multi-Spectral Instrument (MSI). Operating as a push-broom scanner, the MSI relies entirely on external solar illumination, measuring the solar radiation reflected by the Earth's surface across various wavelengths.

### Optical Reflectance and Push-Broom Mechanics

The MSI utilizes a focal plane array of staggered detector modules that simultaneously sweep across a 290 km swath. As the satellite moves along its orbital track, the detectors accumulate photons reflected from the surface. The measured radiance depends on the specific absorption and reflectance spectra of the target materials:

* **Vegetation:** Exhibits strong absorption in the visible red spectrum (due to chlorophyll) and high reflectance in the Near-Infrared (NIR) spectrum (due to internal leaf cell structure), producing the characteristic "red edge" utilized in vegetation indices.
* **Urban and Soil:** Exhibit gradual reflectance increases across the visible and Short-Wave Infrared (SWIR) domains, distinct from the sharp spectral signatures of organic matter.
* **Water:** Strongly absorbs NIR and SWIR radiation, appearing nearly black in these bands, while reflecting shorter visible wavelengths (coastal blue/green).

### Spectral Bands and Tensor Representation

While the MSI captures 13 distinct spectral bands, three atmospheric bands (aerosol, water vapor, and cirrus) are typically discarded for terrestrial mapping. The remaining 10 surface-relevant bands are utilized:

* **Visible (RGB):** Bands 2, 3, and 4 (10m resolution).
* **Near-Infrared (NIR):** Band 8 (10m resolution) and the narrow Red-Edge bands 5, 6, 7, and 8A (20m resolution).
* **Short-Wave Infrared (SWIR):** Bands 11 and 12 (20m resolution).

To construct a unified spatial tensor ($X_{\text{S2}} \in \mathbb{R}^{B \times 10 \times H \times W}$), the 20m bands are typically upsampled to match the 10m spatial resolution of the visible and primary NIR bands, providing a rich, high-dimensional multi-spectral profile for every pixel.

#### 2.3 Radiometric Calibration

Optical data is initially quantified as discrete Digital Numbers (DN). For quantitative applications, these DN values are calibrated to Top-of-Atmosphere (Level-1C) or Bottom-of-Atmosphere (Level-2A) reflectance. Sentinel-2 Level-2A products provide surface reflectance scaled by a factor of 10,000. Dividing the digital numbers by this factor ($\text{DN} / 10000$) maps the physical reflectance directly into a normalized $[0, 1]$ space. Unlike SAR, optical data is bounded and additive, though it is fundamentally vulnerable to cloud obscuration—requiring careful masking or reliance on complementary active sensors.
