# Exercise 3 — Multiresolution Image Processing

## Overview

This assignment explores image pyramids and frequency-domain image processing techniques. It implements the core operations for multiresolution image analysis and blending.

## Main Concepts

### Gaussian Pyramid
A series of progressively smaller versions of an image, created by repeatedly downsampling (using `cv.pyrDown`). Each level is half the resolution of the previous level.

### Laplacian Pyramid
Constructed from the Gaussian pyramid by computing the difference between each level of the Gaussian pyramid and the upsampled next coarser level. Represents high-frequency details at each scale.

### Pyramid Reconstruction
Laplacian pyramids can be reconstructed into the original image by starting with the coarsest level and progressively upsampling (using `cv.pyrUp`) while adding each finer level's Laplacian.

### Multiresolution Blending
Two images are blended at each pyramid level using a Gaussian mask. The mask defines a smooth transition between images across scales, producing seamless blends with minimal artifacts at edges.

### Hybrid Images
Combine low-frequency components from one image (extracted via Gaussian blur) with high-frequency components from another image. The result appears different depending on viewing distance.

## Main Source File

- `ex3.py` — Core implementation of pyramid operations, reconstruction, blending, and hybrid image creation

## Dependencies

See `requirements.txt` for dependencies:

```
numpy
matplotlib
opencv
```

## Usage

The main functions are designed to be called directly:

- **`main_blend(image_a_p, image_b_p, mask_p)`** — Blend two images using a Gaussian mask at multiple pyramid levels. Displays pyramid visualizations and saves the result to `pictures/blended.jpg`.

- **`main_hybrid(img_a_p, img_b_p)`** — Create a hybrid image by combining low frequencies from `img_a_p` with high frequencies from `img_b_p`. Saves to `pictures/hybrid_image.jpg` and displays the result.

Example (manual invocation):

```python
from ex3 import main_blend, main_hybrid

main_blend("image_a.jpg", "image_b.jpg", "mask.jpg")
main_hybrid("low_freq.jpg", "high_freq.jpg")
```

Currently, these functions are executed by manually calling them from within a Python environment or Jupyter notebook.
