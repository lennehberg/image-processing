import cv2
import numpy as np
import matplotlib.pyplot as plt
import ex3.src.pyramid_blend as pyramid_blend


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


def get_trans_mat(frame_a, frame_b):
    """
    Estimates the transformation matrix between two frames using feature matching and RANSAC.
    :param frame_a: The first grayscale frame.
    :param frame_b: The second grayscale frame.
    :return: The 2x3 affine transformation matrix.
    """
    # Detect good features to track in the first frame
    features1 = cv2.goodFeaturesToTrack(
        frame_a, maxCorners=500, qualityLevel=0.01, minDistance=3, blockSize=7
    )

    # Ensure features were detected
    if features1 is None:
        raise ValueError("No features detected in the first frame.")

    # Calculate optical flow to track these features in the second frame
    features2, status, _ = cv2.calcOpticalFlowPyrLK(frame_a, frame_b, features1, None)

    # Filter only valid points
    valid_features1 = features1[status == 1].reshape(-1, 2)
    valid_features2 = features2[status == 1].reshape(-1, 2)

    # Check if we have enough valid points for estimation
    if len(valid_features1) < 3 or len(valid_features2) < 3:
        raise ValueError("Not enough valid feature matches to estimate the transformation.")

    # Estimate the transformation matrix using RANSAC
    trans_mat, inliers = cv2.estimateAffinePartial2D(
        valid_features1, valid_features2, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )

    # Ensure the transformation matrix was estimated successfully
    if trans_mat is None:
        raise ValueError("Transformation matrix estimation failed.")

    return trans_mat


# 2. Stabilize Y translation and rotation
def stabilize_transforms(trans_mats):
    """
    nullify Y translation and rotations of a transformation matrix
    :param trans_mats: list of transformation matrices to stabilize
    :return: list of stabilized transformation matrices
    """
    stabilized_transforms = []

    for ind, mat in enumerate(trans_mats):
        dx = mat[0, 2]
        dy = mat[1, 2]

        if abs(dy) > 4:
            dy = 0

        # Stabilize by removing rotation and Y translation
        stable_mat = np.zeros(mat.shape)
        stable_mat[0, 2] = dx  # Keep X translation
        stable_mat[1, 2] = 0  # Neutralize Y translation
        stable_mat[0, 0] = stable_mat[1, 1] = 1  # Neutralize rotation
        stable_mat[0, 1] = stable_mat[1, 0] = 0
        stabilized_transforms.append(stable_mat)
    return np.array(stabilized_transforms)


def get_cumulative_transforms(mats):
    """
    return a list of cumulative transforms, with the first transform being the identity
    :param mats: motion transformation between frames
    :return: list of cumulative transforms between frames, represented by 3x3 matrices.
    """
    cumulative_transforms = [np.eye(3)]
    c_trans = cumulative_transforms[0]
    # accumulate the matrices by dot products
    for mat in mats:
        mat_3x3 = np.vstack([mat, [0, 0, 1]])
        c_trans = c_trans @ mat_3x3
        cumulative_transforms.append(c_trans)
    return cumulative_transforms


def get_canvas_dimensions(vid_frames, cumulative_mats):
    height = vid_frames[0].shape[0]
    width = vid_frames[0].shape[1]
    max_dx = int(np.max(np.abs([mat[0, 2] for mat in cumulative_mats])))  # Get the maximum absolute dx

    max_dy = np.max([mat[1, 2] for mat in cumulative_mats])
    min_dy = np.min([mat[1, 2] for mat in cumulative_mats])

    total_dy = int(abs(min_dy) + abs(max_dy))

    return height + total_dy, width + max_dx, 3


def warp_frame(vid_frames, mats, canvas_shape):
    """
    creates a canvas and warps frames onto it using cumulative_mats
    :param canvas_shape:
    :param vid_frames:
    :param mats:
    :return:
    """
    canvas_shape = canvas_shape
    canvas = [np.zeros(canvas_shape, dtype=vid_frames[0].dtype)]
    display_canvas_with_matplotlib(canvas[0])
    # for each frame in vid_frames, warp the frame according to the
    # transform in cumulative transforms (where first us anchor and first mat
    # is the identity, and so on...)
    for ind, frame in enumerate(vid_frames):
        # because cumulative mats are 3x3, only extract the first 2 rows (so mat is 2x3)
        warp_mat = mats[ind][:2, :]
        # backward warp the frame onto the canvas
        warped_frame = cv2.warpAffine(frame, warp_mat, (frame.shape[1], frame.shape[0]), flags=cv2.WARP_INVERSE_MAP)

        # display_canvas_with_matplotlib(warped_frame)
        # append the warped frame to the canvas
        canvas.append(warped_frame)
    return canvas


def get_first_strip(aligned_frame, strip_center):
    end = int(np.ceil(strip_center))
    return aligned_frame[:, :end, :]


def get_strip(aligned_frame, strip_center, l_strip_width, r_strip_width):
    start = int(np.floor(strip_center - l_strip_width))
    end = int(np.ceil(strip_center + r_strip_width))
    return aligned_frame[:, start: end]


def get_last_strip(aligned_frame, strip_center):
    frame_width = aligned_frame.shape[1]
    start = int(np.floor(frame_width - strip_center))
    return aligned_frame[:, start:, :]


def make_pano(canvas, cumulative_mats, stab_mats, strip_center, frame_width):
    """
    Creates a panorama image by backward warping and pasting strips from frames.
    :param canvas: List of warped frames.
    :param cumulative_mats: List of cumulative transformation matrices.
    :param stab_mats: List of stabilized transformation matrices.
    :param strip_center: Initial center of the strip in the frame.
    :param frame_width: Width of the video frames.
    :return: The completed panorama image.
    """
    pano_frame = np.zeros(canvas[0].shape, dtype=canvas[0].dtype)  # Initialize the panorama frame
    strip_pos = 0  # Track the horizontal position to paste strips

    for i in range(1, len(canvas)):  # Start from 1 because canvas[0] is empty
        # Compute the left and right widths of the strip
        l_strip_width = int(np.floor(abs(stab_mats[i - 1][0, 2] / 2)))

        if i < len(canvas) - 1:
            r_strip_width = int(np.ceil(abs(stab_mats[i][0, 2] / 2)))
        else:
            r_strip_width = frame_width - strip_center

        # Extract the strip based on its position
        if i == 1:
            strip = get_first_strip(canvas[i], strip_center)
        elif i < len(canvas) - 1:
            strip = get_strip(canvas[i], strip_center, l_strip_width, r_strip_width)
        else:
            strip = get_last_strip(canvas[i], strip_center)

        # Warp the entire frame to the panorama's coordinate space
        warped_strip = cv2.warpAffine(
            canvas[i], cumulative_mats[i - 1][:2, :],
            (pano_frame.shape[1], pano_frame.shape[0]),
            flags=cv2.WARP_INVERSE_MAP
        )

        # display_canvas_with_matplotlib(warped_strip)
        # Extract only the part of the warped strip corresponding to the current strip
        strip_width = strip.shape[1]
        strip_start = strip_pos
        strip_end = strip_start + strip_width
        strip_segment = warped_strip[:, strip_start:strip_end, :]  # Extract the required part
        # display_canvas_with_matplotlib(strip_segment)

        # Paste the relevant part of the warped strip into the panorama frame
        pano_frame[:, strip_start:strip_end, :] = strip_segment

        # Update the strip position
        strip_pos = strip_end

    return pano_frame
