# Workings of the Sentinel Constellation

The physical dichotomy between active microwave backscatter and passive solar reflectance defines the representational gap in cross-modal remote sensing. Bridging this gap requires a rigorous understanding of the underlying sensor physics, energy propagation, and radiometric calibration processes that ultimately produce the structured tensor spaces processed by metric learning architectures. This chapter details the operational mechanics of the European Space Agency’s (ESA) Sentinel-1 and Sentinel-2 missions and introduces an interactive 3D visualization of their sensing principles.

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

In deep learning pipelines, these dual polarizations are stacked to form a 2-channel spatial tensor:

$$
X_{\text{S1}} \in \mathbb{R}^{B \times 2 \times H \times W}
$$

### Radiometric Calibration

Raw SAR data is heavily skewed and contains multiplicative speckle noise inherent to coherent imaging. To stabilize the signal for neural network ingestion, the raw complex data is calibrated into a normalized radar cross-section ($\sigma^0$). This backscatter coefficient is typically converted to a logarithmic decibel scale:

$$
\sigma^0_{\text{dB}} = 10 \log_{10} \sigma^0
$$

Because terrestrial SAR backscatter generally falls within a predictable dynamic range, the data is commonly clipped and min-max normalized (e.g., bounding $\sigma^0_{\text{dB}} \in [-35, 0]$) to map the signal into a strictly positive $[0, 1]$ domain, ensuring stable gradients during early spatial feature extraction.

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

To construct a unified spatial tensor:

$$
X_{\text{S2}} \in \mathbb{R}^{B \times 10 \times H \times W}
$$

the 20m bands are typically upsampled to match the 10m spatial resolution of the visible and primary NIR bands, providing a rich, high-dimensional multi-spectral profile for every pixel.

### Radiometric Calibration

Optical data is initially quantified as discrete Digital Numbers (DN). For quantitative applications, these DN values are calibrated to Top-of-Atmosphere (Level-1C) or Bottom-of-Atmosphere (Level-2A) reflectance. Sentinel-2 Level-2A products provide surface reflectance scaled by a factor of 10,000. Dividing the digital numbers by this factor:

$$
\rho = \frac{\text{DN}}{10000}
$$

maps the physical reflectance directly into a normalized $[0, 1]$ space. Unlike SAR, optical data is bounded and additive, though it is fundamentally vulnerable to cloud obscuration—requiring careful masking or reliance on complementary active sensors.

## Interactive 3D Sentinel Physics Explorer

The interactive 3D Sentinel Physics Explorer provides a visual representation of the operating principles of Sentinel-1 and Sentinel-2. It allows users to explore the differences between active microwave radar sensing and passive optical imaging through a 3D satellite-and-Earth visualization.

The model connects the theoretical concepts of microwave transmission, surface backscatter, solar illumination, and optical reflectance with interactive controls and animated visual elements.

### Satellite Visualization and Visibility Controls

The 3D environment represents the two satellite missions as separate observation systems:

* **Sentinel-1:** Represents the active Synthetic Aperture Radar system, which transmits microwave signals toward the Earth's surface and receives the returned backscatter.
* **Sentinel-2:** Represents the passive Multi-Spectral Instrument, which observes solar radiation reflected from the Earth's surface.

The satellite visibility controls allow users to show or hide Sentinel-1 and Sentinel-2 independently. Both satellites can also be displayed together to compare their observation mechanisms in the same 3D environment.

### Environmental Controls

The model includes interactive environmental settings that demonstrate how observation conditions affect the two sensing systems.

* **Cloud Cover:** Demonstrates the effect of cloud obstruction on optical observations compared with the ability of Sentinel-1 radar to operate under most cloud-cover conditions.
* **Night Mode:** Demonstrates the dependence of Sentinel-2 optical observations on solar illumination, while Sentinel-1 can transmit its own microwave energy and operate independently of sunlight.
* **See-Through (Dielectric):** Provides a visual representation of dielectric-related interactions and helps explain how material properties influence microwave behavior.

### Radar Observation: Sentinel-1

The Sentinel-1 visualization demonstrates the active radar observation process:

1. The satellite transmits microwave energy toward the Earth's surface.
2. The transmitted signal interacts with the surface material.
3. The surface reflects and scatters a portion of the incident energy.
4. A portion of the scattered energy returns toward the radar antenna.
5. The returned signal represents the radar backscatter response of the observed surface.

The radar visualization helps explain why different surface characteristics produce different backscatter responses. Surface roughness, moisture, and dielectric properties influence the intensity and behavior of the returned signal.

### Optical Observation: Sentinel-2

The Sentinel-2 visualization demonstrates the passive optical observation process:

1. Solar radiation illuminates the Earth's surface.
2. Different surface materials reflect and absorb energy differently.
3. Reflected radiation travels toward the optical satellite.
4. The Multi-Spectral Instrument records the incoming radiation.
5. The observed reflectance provides information about surface characteristics.

The visualization illustrates the dependence of optical sensing on external illumination and the interaction between solar radiation and surface materials.

### Surface Comparison

The 3D environment provides representative surface types to help explain the different responses of radar and optical sensors.

* **Water:** Smooth water surfaces generally produce weak radar backscatter because much of the incident microwave energy is reflected away from the radar antenna. Water also strongly absorbs near-infrared and short-wave infrared radiation.
* **Forest:** Forest canopies produce complex volume-scattering responses in SAR observations and characteristic spectral reflectance patterns in optical observations.
* **Urban Areas:** Buildings and other structures can produce strong radar returns through surface and double-bounce scattering. Their optical reflectance depends on the composition and properties of the surface materials.

These examples demonstrate why Sentinel-1 and Sentinel-2 provide complementary information about the same geographical region.

### Interactive Model Controls

The visualization provides the following controls for exploring the satellite observation environment:

| Control | Purpose |
|---|---|
| Sentinel-1 visibility | Show or hide the radar satellite representation |
| Sentinel-2 visibility | Show or hide the optical satellite representation |
| Cloud Cover | Examine the effect of cloud conditions on sensor observations |
| Night Mode | Explore the dependence of optical sensing on solar illumination |
| See-Through (Dielectric) | Visualize dielectric-related surface interactions |
| Surface selection | Compare representative surface types |
| Speed control | Adjust the speed of the visualization |
| Reset camera view | Restore the default 3D viewing position |

The interface also provides a sensor guide and a dynamic information display to help users understand the active observation process and the differences between the two satellite systems.

## Relationship Between Sensor Physics and Deep Learning

Sentinel-1 and Sentinel-2 provide complementary measurements of the Earth's surface. Sentinel-1 contributes active microwave backscatter measurements, whereas Sentinel-2 contributes passive multi-spectral reflectance measurements.

Their input representations can be expressed as:

$$
X_{\text{S1}} \in \mathbb{R}^{B \times 2 \times H \times W}
$$

$$
X_{\text{S2}} \in \mathbb{R}^{B \times 10 \times H \times W}
$$

Here:

- $B$ represents the batch size.
- $2$ represents the two Sentinel-1 polarization channels (VV and VH).
- $10$ represents the selected Sentinel-2 spectral bands.
- $H$ and $W$ represent the height and width of the input images.

The complementary information from these modalities can be used in multi-modal learning architectures. However, differences in their physical measurement processes, signal distributions, spatial resolutions, and environmental sensitivities create a cross-modal representation challenge.

Understanding these differences is essential before applying feature extraction, representation alignment, and metric learning methods to satellite imagery.

The Interactive 3D Sentinel Physics Explorer supports this understanding by making the physical differences between active radar and passive optical sensing easier to visualize and explain.
