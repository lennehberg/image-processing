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
    align frame b to frame a
    :param frame_a:
    :param frame_b:
    :return: transformation matrix between frame b and frame a.
    """
    # get feature points for lucas kanade
    features1 = cv2.goodFeaturesToTrack(frame_a, maxCorners=100, qualityLevel=0.01, minDistance=1, blockSize=3)
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
    stabilized_transforms = []
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

    total_dy = int(abs(min_dy - max_dy))

    return height + total_dy, width + max_dx, 3


def warp_frame(vid_frames, mats, canvas_shape):
    """
    creates a canvas and warps frames onto it using cumulative_mats
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


def make_pano(canvas, cumulative_mats, stab_mats, strip_center):
    """
    creates a panorama image by backward warping and pasting strips from canvas
    :param canvas:
    :param cumulative_mats:
    :return:
    """
    # display_canvas_with_matplotlib(canvas[441])
    pano_frame = np.zeros(canvas[0].shape)
    cur_strip_center, prev_strip_center = 0, 0

    # strip_width_r = abs(int(np.mean([mat[0, 2] for mat in stab_mats])))
    for i in range(1, len(canvas)):  # start from 1 because 0 has empty canvas
        # estimate the strip's width by setting it to the dx motion from this frame to the next on the right
        # and the previous frame to this on the left
        if i > 0:
            strip_width_l = abs(stab_mats[i - 1][0, 2])
        else:
            strip_width_l = 0
        if i < len(canvas) - 1:
            strip_width_r = abs(stab_mats[i][0, 2])
        else:
            strip_width_r = 0

        # copy the values from the overlap between the frame and the strip
        strip = canvas[i][:, int(np.floor(strip_center - strip_width_l)): int(np.ceil(strip_center + strip_width_r))]

        # display_canvas_with_matplotlib(strip)
        # backward warp the strip onto the canvas
        warped_strip = cv2.warpAffine(strip, cumulative_mats[i - 1][:2, :], (pano_frame.shape[1], pano_frame.shape[0]),
                                      flags=cv2.WARP_INVERSE_MAP)
        # display_canvas_with_matplotlib(warped_strip)
        if i > 1:
            mask = np.zeros(pano_frame.shape)
            print(prev_strip_center, cur_strip_center)
            mask[:, int(prev_strip_center): int(strip_center), :] = 255
            pano_frame = np.maximum(warped_strip, pano_frame)
            # display_canvas_with_matplotlib(mask)
            pano_frame = pyramid_blend.blend_images(warped_strip, pano_frame, mask)
        else:
            pano_frame = warped_strip
        prev_strip_center = cur_strip_center
        cur_strip_center += strip_width_r
        # display_canvas_with_matplotlib(pano_frame)
    return pano_frame
