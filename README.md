# Image Processing & Computer Vision Coursework

Selected assignments from the Hebrew University of Jerusalem's 2024–2025 Image Processing course, implemented primarily in Python using NumPy and OpenCV.

**Note:** The original coursework implementations are preserved and are not presented as production libraries or independent research.

## Highlights

### Exercise 4 — Motion Estimation, Alignment & Panoramas

The standout project: a classical computer-vision pipeline that:

- Detects trackable image features using OpenCV's `goodFeaturesToTrack`
- Tracks features between frames using pyramidal Lucas–Kanade optical flow
- Estimates partial affine transformations robustly using RANSAC
- Represents and composes cumulative transformations using homogeneous coordinates
- Aligns and warps video frames onto a shared canvas
- Constructs stereo panorama frames from aligned image strips

**The student work is the construction and integration of the overall processing pipeline.** Feature detection, optical-flow, affine-estimation, and warping primitives are supplied by OpenCV.

[→ See Exercise 4](./ex4)

### Exercise 3 — Multiresolution Image Processing

Explores image pyramids and frequency-domain blending:

- Gaussian and Laplacian pyramid decomposition
- Reconstruction from a Laplacian pyramid
- Multiresolution image blending using a Gaussian mask
- Hybrid images combining low- and high-frequency components

[→ See Exercise 3](./ex3)

## Assignment Overview

| Directory  | Topic                           | Main concepts                                                              |
| ---------- | ------------------------------- | -------------------------------------------------------------------------- |
| `bootcamp` | Python image-processing warm-up | NumPy arrays, image operations, visualization                              |
| `ex1`      | Video cut detection             | Grayscale conversion, intensity histograms, cumulative histograms          |
| `ex2`      | Audio watermarking              | FFT/STFT analysis, spectral watermark insertion and detection              |
| `ex3`      | Image pyramids and blending     | Gaussian/Laplacian pyramids, reconstruction, hybrid images                 |
| `ex4`      | Video alignment and panoramas   | Feature tracking, optical flow, RANSAC, affine transformations, warping    |
| `ex5`      | Deep image methods              | Notebook-based generative image reconstruction and restoration experiments |

## Technologies

- Python
- NumPy
- OpenCV
- Matplotlib
- MediaPy
- SciPy
- Librosa
- Jupyter Notebook
- PyTorch

## Usage

Exercises may require course-provided input media and command-line arguments. Dependencies are recorded locally where available. Not all exercises run as a unified package; consult individual exercise directories for setup and usage instructions.
