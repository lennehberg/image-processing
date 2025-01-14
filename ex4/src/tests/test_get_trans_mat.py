# tests/test_get_trans_mat.py
import sys
import os
import numpy as np
import cv2

# Add the src directory to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import ex4.src.ex4 as ex4

raw_frame_a = [[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 255, 255, 0, 0, 0],
               [0, 0, 255, 255, 0, 0, 0],
               [0, 0, 0, 0, 255, 255, 0],
               [0, 0, 0, 0, 255, 255, 0],
               [0, 0, 0, 0, 0, 0, 0]]

raw_frame_b = [[0, 0, 0, 0, 0, 0, 0],
               [0, 255, 255, 0, 0, 0, 0],
               [0, 255, 255, 0, 0, 0, 0],
               [0, 0, 0, 255, 255, 0, 0],
               [0, 0, 0, 255, 255, 0, 0],
               [0, 0, 0, 0, 0, 0, 0]]

# Example usage
frame_a = np.array(raw_frame_a, dtype=np.uint8)

frame_b = np.array(raw_frame_b, dtype=np.uint8)

trans_ = ex4.get_trans_mat(frame_a, frame_b)
print(trans_)
stab_trans_ = ex4.stabilize_transforms([trans_])
print(stab_trans_)

# Compute the inverse matrices for backward warping
inv_trans_ = cv2.invertAffineTransform(trans_)
inv_stab_trans_ = cv2.invertAffineTransform(stab_trans_[0])

# Backward warp both frames using the inverse matrices
warped_a = cv2.warpAffine(frame_a, inv_trans_, (frame_a.shape[1], frame_a.shape[0]), flags=cv2.WARP_INVERSE_MAP)
warped_b = cv2.warpAffine(frame_b, inv_stab_trans_, (frame_b.shape[1], frame_b.shape[0]), flags=cv2.WARP_INVERSE_MAP)

# Display the results
print("Backward Warped Frame A:")
print(warped_a)

print("Backward Warped Frame B:")
print(warped_b)
