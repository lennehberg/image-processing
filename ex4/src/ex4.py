# generalized algorithm for mosaic

# 1. Align consecutive frames (using LK)

# 2. Stabilize rotations and Y translations

# 3. Use motion composition to align frames to same coordinates on canvas

# 4. Create mosaic by pasting strips from images

# 5. Set convergence point

import cv2
import numpy as np
import matplotlib.pyplot as plt
import ex3.src.pyramid_blend as pyramid_blend


# 1. Align consecutive frames:
# a. Select a frame to align towards (Usually the middle one - for simplicity attempt
#                                              on first frame)
# b. Compute the transformation matrices of consecutive frames

def get_trans_mat(frame_a, frame_b):
    """
    align frame b to frame a
    :param frame_a:
    :param frame_b:
    :return: transformation matrix between frame b and frame a.
    """
    # get feature points for lucas kanade
    features1 = cv2.goodFeaturesToTrack(frame_a, maxCorners=100, qualityLevel=0.01, minDistance=1, blockSize=4)

    # get features in frame 2 according to features in frame 1 and track them by lk
    features2, status, error = cv2.calcOpticalFlowPyrLK(frame_a, frame_b, features1, None)

    # filter out the good points
    good_feats1 = features1[status == 1]
    good_feats2 = features2[status == 1]

    # estimate transformation matrix
    # (assuming affine transformation as camera moves in a way that maintains parallel lines)
    trans_mat, _ = cv2.estimateAffinePartial2D(good_feats1, good_feats2)

    return trans_mat


# 2. Stabilize Y translation and rotation
def stabilize_transforms(trans_mats):
    """
    nullify Y translation and rotations of a transformation matrix
    :param trans_mats: list of transformation matrices to stabilize
    :return: list of stabilized transformation matrices
    """
    stabilized_transforms = [np.eye(3)[:2, :]]
    # stab_dy = np.mean([mat[1, 2] for mat in trans_mats])

    for mat in trans_mats:
        dx = mat[0, 2]
        # Stabilize by removing rotation and Y translation
        stable_mat = np.zeros(mat.shape)
        stable_mat[0, 2] = dx  # Keep X translation
        stable_mat[1, 2] = 0  # Neutralize Y translation
        stable_mat[0, 0] = stable_mat[1, 1] = 1  # Neutralize rotation
        stable_mat[0, 1] = stable_mat[1, 0] = 0

        stabilized_transforms.append(stable_mat)

    return np.array(stabilized_transforms)


def display_canvas_with_matplotlib(canvas):
    """
    Displays the canvas using matplotlib.
    :param canvas: The canvas or mosaic image to display.
    """
    # If the canvas is not in the standard 0-255 range, normalize it
    if canvas.dtype != np.uint8:
        canvas = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Display using matplotlib
    plt.imshow(canvas, cmap='gray')  # Use cmap='gray' for grayscale images
    plt.axis('off')  # Hide the axes
    plt.show()


# 3. Use motion composition to align frames on canvas
def warp_frames(vid_frames, stab_trans_mats, transforms, a_frame_ind=0):
    """
    Warp frames according to stabilized transformation matrices and align them on a single canvas.
    Only max_dx is considered for the canvas size.
    :param vid_frames: List of video frames (as numpy arrays).
    :param stab_trans_mats: List of stabilized transformation matrices.
    :param a_frame_ind: index of frame to align to
    :return: A single canvas containing the aligned frames.
    """
    # Get initial frame dimensions
    height, width = vid_frames[0].shape[:2]

    # Calculate max_dx and motion composition from the transformation matrices
    max_dx = 0
    lc_trans_list = []
    rc_trans_list = []

    l_cumulative_transform = np.eye(3)  # Start with identity for the first frame
    r_cumulative_transform = np.eye(3)

    for i in range(a_frame_ind - 1, -1, -1):
        l_cumulative_transform = np.vstack([stab_trans_mats[i], [0, 0, 1]]) @ l_cumulative_transform
        dx = l_cumulative_transform[0, 2]  # X translation
        max_dx = max(max_dx, abs(dx))
        lc_trans_list.append(l_cumulative_transform)

    lc_trans_list.reverse()  # Reverse left cumulative transforms to maintain order

    for j in range(a_frame_ind, len(stab_trans_mats)):
        r_cumulative_transform = r_cumulative_transform @ np.vstack([stab_trans_mats[j], [0, 0, 1]])
        dx = r_cumulative_transform[0, 2]  # X translation
        max_dx = max(max_dx, abs(dx))
        rc_trans_list.append(r_cumulative_transform)

    # Compute canvas width based on max_dx
    canvas_width = width + int(max_dx)
    canvas_height = height  # Y dimension remains unchanged

    # Initialize the canvas
    canvas = [np.zeros((canvas_height, canvas_width, 3), dtype=vid_frames[0].dtype)]

    # Warp frames and paste them onto the canvas
    for inx, frame in enumerate(vid_frames):
        # display_canvas_with_matplotlib(frame)
        if inx < a_frame_ind:
            warp_mat = lc_trans_list[inx][:2, :]
        elif inx > a_frame_ind:
            warp_mat = rc_trans_list[inx - a_frame_ind - 1][:2, :]
        else:
            warp_mat = np.eye(3)[:2, :]

        warped_frame = cv2.warpAffine(frame, warp_mat, (canvas_width, canvas_height), flags=cv2.WARP_INVERSE_MAP)
        # display_canvas_with_matplotlib(warped_frame)
        canvas.append(warped_frame)
    c_trans_list = [np.eye(3)] + lc_trans_list + rc_trans_list
    return canvas, c_trans_list


# 4. Create mosaic
def make_pano(canvas, canvas_shape, strip_center, stab_trans_mats, c_transforms):
    """
    make a pano by pasting strips at offset from center
    :param stab_trans_mats: stabilized matrices of transformations between frames
    :param canvas: canvas of frame (canvas[0] should be empty)
    :param canvas_shape: height and width of canvas
    :param strip_center: distance from center of frame
    :return: panoramic picture created by pasting strips onto canvas
    """
    # create a frame and set strip width to inital 0
    pano_frame = np.zeros(canvas_shape)
    strip_pos = 0
    prev_strip_center = strip_center
    # average the dx to strip width
    # strip_width = abs(int(np.mean([mat[0, 2] for mat in stab_trans_mats])))
    # print(strip_width)
    # update strip width according to dx
    for i in range(len(canvas) - 1):
        strip_width = int(np.ceil(abs(stab_trans_mats[i][0, 2])))

        if strip_width == 0:
            continue

        if strip_width % 2 != 0:
            strip_width += 1

        strip = canvas[i][:, int(strip_center - strip_width // 2): int(strip_center + strip_width // 2)]

        if len(strip[0]) == 0:
            continue

        # display_canvas_with_matplotlib(strip)
        # Define a backward warp (reverse the transformation matrix)
        # inverse_transform = np.linalg.inv(np.vstack([stab_trans_mats[i - 1], [0, 0, 1]]))  # Inverse of the transformation matrix
        print(strip.shape)
        print(strip_width)
        # Warp the strip backward
        warped_strip = cv2.warpAffine(strip, c_transforms[i][:2, :], (pano_frame.shape[1], pano_frame.shape[0]), flags=cv2.WARP_INVERSE_MAP)
        # display_canvas_with_matplotlib(strip)
        # display_canvas_with_matplotlib(warped_strip)
        # # Apply the warped strip to the pano_frame
        pano_frame = np.maximum(warped_strip, pano_frame)

        mask = np.zeros(canvas_shape)
        mask[:, prev_strip_center: strip_center] = 1

        pano_frame = pyramid_blend.blend_images(warped_strip, pano_frame, mask)

        strip_pos += strip_width
        prev_strip_center = strip_center
        strip_center += strip_width

    return pano_frame
