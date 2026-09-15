# Exercise 4 — Motion Estimation, Alignment & Panoramas

## Overview

A complete video processing pipeline that estimates frame-to-frame motion, stabilizes camera shake, aligns frames to a shared canvas, and constructs stereo panorama views from video sequences.

## Pipeline

The processing follows these steps:

1. **Read video frames** from disk into memory using MediaPy
2. **Convert frames to grayscale** for motion estimation
3. **Detect trackable features** using OpenCV's `goodFeaturesToTrack`
4. **Track features between adjacent frames** using pyramidal Lucas–Kanade optical flow via `calcOpticalFlowPyrLK`
5. **Estimate partial affine transformations** with RANSAC using `estimateAffinePartial2D` to recover rotation, scale, and translation
6. **Stabilize selected motion components** by nullifying Y-axis translation and rotation to reduce jitter
7. **Compose cumulative transformations** by multiplying transformation matrices in homogeneous coordinates (3×3 form)
8. **Warp frames onto a shared canvas** using `warpAffine` with backward mapping
9. **Extract aligned strips** from consecutive warped frames
10. **Construct stereo panorama frames** by stitching aligned strips side-by-side to create stereo views

## Implementation Details

The assignment code assembles these operations into a complete processing pipeline:

- **Feature Detection & Tracking:** OpenCV supplies `goodFeaturesToTrack` and `calcOpticalFlowPyrLK`. The assignment selects parameters and chains these calls.
- **Affine Estimation:** OpenCV supplies `estimateAffinePartial2D` with RANSAC. The assignment filters valid feature matches and structures the estimation.
- **Transformation Composition:** Uses matrix multiplication to accumulate transformations across frames, enabling global alignment.
- **Warping:** OpenCV supplies `warpAffine`. The assignment manages the canvas, coordinates, and strip extraction.

## Command-Line Interface

```bash
python ex4.py <input_video> <panorama_method> <output_path>
```

**Arguments:**

1. **`<input_video>`** — Path to the input video file (e.g., `video.mp4`)
2. **`<panorama_method>`** — Panorama construction method:
   - `0` — Stereo panorama (implemented)
   - `1` — Dynamic panorama (not fully implemented)
3. **`<output_path>`** — Path for the output panorama video (e.g., `output.mp4`)

**Note:** The implemented functionality corresponds to method `0` (stereo panorama). The `stitch_dynamic_pano` function is a stub and does not produce output.

## Dependencies

See `requirements.txt` for dependencies:

```
numpy
cv2 (opencv-python)
mediapy
matplotlib
```

## Example

```bash
python ex4.py video.mp4 0 panorama_output.mp4
```
