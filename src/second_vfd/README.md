# :shaved_ice:Freeze-Missing-VF Module

This directory contains the `Freeze-Missing-VF` module, which is part of the Multi-Modal Glaucoma (Multi-Glau) project. This module addresses the challenge of missing data in multi-modal learning for visual field defect (VFD) grading in glaucoma patients.

## :hospital:Background

In clinical practice, especially in rural or remote secondary hospitals, complete data modalities such as visual field (VF) testings and optical coherence tomography (OCT) scans are often unavailable. This module is designed to handle such scenarios by employing a strategy to manage missing modalities effectively without compromising the performance of the model.

## :ice_cream:Architecture

<img src=".\Missing Freeze.png" alt="Missing Freeze" style="zoom: 50%;" />

The `Freeze-Missing-VF` model leverages multiple data inputs, including CSLO fundus images, local optic disc images, RNFL thickness images, and clinical data (e.g., CDR, IOP). These inputs are processed through different branches:
- **Global Flow**: Handles the global view from fundus images.
- **Focused Flow**: Processes local optic disc images.
- **RNFL Flow**: Manages RNFL thickness images.
- **Numerical Flow**: Incorporates clinical data such as CDR and IOP.

### Feature Extraction

The feature extraction network includes an initial feature extraction with a 7x7 convolutional kernel, followed by max-pooling and concatenation with three [Se-Conv](https://github.com/hujie-frank/SENet) blocks. The Se-Conv block includes the Squeeze-and-Excitation (Se) block after the Residual block to model inter-channel dependencies and focus adaptively on significant features.

### Handling Missing Data

To address missing modalities, the module constructs an image generation head that creates a completely black image with three channels when a modality is missing. This ensures that the missing data does not disrupt the feature extraction process of the available data. The features from the black image are multiplied by a zero vector before feature concatenation, maintaining the integrity of the network's learning process.

## :rocket:Performance

The performance of the `Freeze-Missing-VF` model under different missing rates is summarized below:

| Missing Rate | Accuracy        | Sensitivity     | Specificity     | AUC             |
| ------------ | --------------- | --------------- | --------------- | --------------- |
| 1.25%        | 0.8076          | 0.8444          | 0.7373          | 0.8629          |
| 5%           | 0.7813 ± 0.0134 | 0.8151 ± 0.0234 | 0.7169 ± 0.0231 | 0.8530 ± 0.0058 |
| 10%          | 0.7816 ± 0.0083 | 0.8415 ± 0.0275 | 0.6676 ± 0.0372 | 0.8369 ± 0.0099 |
| 15%          | 0.7679 ± 0.0117 | 0.8142 ± 0.0307 | 0.6796 ± 0.0351 | 0.8292 ± 0.0108 |
| 20%          | 0.7693 ± 0.0086 | 0.8479 ± 0.0361 | 0.6196 ± 0.0622 | 0.8172 ± 0.0216 |
| 25%          | 0.7609 ± 0.0124 | 0.8813 ± 0.0376 | 0.5314 ± 0.0427 | 0.7824 ± 0.0108 |
| 30%          | 0.7478 ± 0.0145 | 0.8656 ± 0.0420 | 0.5233 ± 0.0623 | 0.7755 ± 0.0238 |
| 35%          | 0.7455 ± 0.0249 | 0.8769 ± 0.0331 | 0.4949 ± 0.0311 | 0.7406 ± 0.0221 |
| 40%          | 0.7300 ± 0.0157 | 0.8738 ± 0.0201 | 0.4559 ± 0.0351 | 0.7192 ± 0.0319 |
