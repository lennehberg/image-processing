import os
import sys
import cv2
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
# import ex3.src.pyramid_blend as pyramid_blend


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
        frame_a, maxCorners=0, qualityLevel=0.07,  minDistance=11, blockSize=20
    )

    # Ensure features were detected
    if features1 is None:
        raise ValueError("No features detected in the first frame.")

    # Calculate optical flow to track these features in the second frame
    features2, status, _ = cv2.calcOpticalFlowPyrLK(frame_a, frame_b, features1, None, maxLevel=50)

    # Filter only valid points
    valid_features1 = features1[status == 1].reshape(-1, 2)
    valid_features2 = features2[status == 1].reshape(-1, 2)

    # Check if we have enough valid points for estimation
    if len(valid_features1) < 3 or len(valid_features2) < 3:
        raise ValueError("Not enough valid feature matches to estimate the transformation.")

    # Estimate the transformation matrix using RANSAC
    trans_mat, inliers = cv2.estimateAffinePartial2D(
        valid_features1, valid_features2, method=cv2.RANSAC, ransacReprojThreshold=0.75
    )

    # Ensure the transformation matrix was estimated successfully
    if trans_mat is None:
        raise ValueError("Transformation matrix estimation failed.")

    return trans_mat


# 2. Stabilize Y translation and rotation
def stabilize_transforms(trans_mats, window_size=5):
    """
    nullify Y translation and rotations of a transformation matrix
    :param trans_mats: list of transformation matrices to stabilize
    :return: list of stabilized transformation matrices
    """
    stabilized_transforms = []

    y_translations = [mat[1, 2] for mat in trans_mats]

    # Compute smoothed y-translations using a moving average
    smoothed_y_translations = []
    for i in range(len(y_translations)):
        start = max(0, i - window_size // 2)
        end = min(len(y_translations), i + window_size // 2 + 1)
        smoothed_y_translations.append(np.mean(y_translations[start:end]))

    for ind, mat in enumerate(trans_mats):
        dx = mat[0, 2]
        dy = mat[1, 2]

        # if abs(dy) > 2 or abs(dy) < 1.5:
        #     dy = 0

        # Stabilize by removing rotation and Y translation
        stable_mat = np.zeros(mat.shape)
        stable_mat[0, 2] = dx  # Keep X translation
        if abs(smoothed_y_translations[ind]) > 4:
            smoothed_y_translations[ind] = 0
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
    # display_canvas_with_matplotlib(canvas[0])
    # for each frame in vid_frames, warp the frame according to the
    # transform in cumulative transforms (where first us anchor and first mat
    # is the identity, and so on...)
    for ind, frame in enumerate(vid_frames):
        # because cumulative mats are 3x3, only extract the first 2 rows (so mat is 2x3)
        warp_mat = mats[ind][:2, :]
        # warp_mat[0, 2] = 0  # nullify x movement
        # backward warp the frame onto the canvas
        warped_frame = cv2.warpAffine(frame, warp_mat, (canvas_shape[1], canvas_shape[0]), flags=cv2.WARP_INVERSE_MAP)

        # display_canvas_with_matplotlib(warped_frame)
        # append the warped frame to the canvas
        canvas.append(warped_frame)
    return canvas


def get_first_strip(aligned_frame, strip_center):
    end = 0  # int(np.ceil(strip_center))
    return aligned_frame[:, :end, :]


count = 0


def save_strip_with_box(aligned_frame, start, end):
    """
    Displays the aligned frame with a red box highlighting the strip region and saves the plot.

    :param aligned_frame: The aligned frame (3D array if RGB or 2D for grayscale).
    :param start: The starting column of the strip.
    :param end: The ending column of the strip.
    :param filename: The name of the file to save the plot.
    """
    global count
    # Ensure the start and end values are within the frame dimensions
    start = max(0, start)
    end = min(aligned_frame.shape[1], end)

    filename = f"strip{count}_plot.png"

    count += 1

    # Make a copy of the frame to draw the rectangle
    if aligned_frame.ndim == 2:  # Grayscale
        aligned_frame_display = cv2.cvtColor(aligned_frame, cv2.COLOR_GRAY2BGR)
    else:  # RGB
        aligned_frame_display = aligned_frame.copy()

    # Draw a red rectangle around the strip region
    color = (255, 0, 0)  # Red color in BGR
    cv2.rectangle(aligned_frame_display, (start, 0), (end, aligned_frame.shape[0]), color, thickness=2)

    # Ensure the "plots" folder exists
    os.makedirs("plots", exist_ok=True)

    # Display the frame with the highlighted strip and save the plot
    plt.imshow(cv2.cvtColor(aligned_frame_display, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    plt.title(f"Strip from {start} to {end}")
    plt.axis('off')
    save_path = os.path.join("plots", filename)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plot to avoid display overlap

    # Return the extracted strip for further use
    return aligned_frame[:, start:end, :]


def display_strip_with_box(aligned_frame, start, end):
    """
    Displays the aligned frame with a red box highlighting the strip region.

    :param aligned_frame: The aligned frame (3D array if RGB or 2D for grayscale).
    :param start: The starting column of the strip.
    :param end: The ending column of the strip.
    """
    # Ensure the start and end values are within the frame dimensions
    start = max(0, start)
    end = min(aligned_frame.shape[1], end)

    # Make a copy of the frame to draw the rectangle
    if aligned_frame.ndim == 2:  # Grayscale
        aligned_frame_display = cv2.cvtColor(aligned_frame, cv2.COLOR_GRAY2BGR)
    else:  # RGB
        aligned_frame_display = aligned_frame.copy()

    # Draw a red rectangle around the strip region
    color = (255, 0, 0)  # Red color in BGR
    cv2.rectangle(aligned_frame_display, (start, 0), (end, aligned_frame.shape[0]), color, thickness=2)

    # Display the frame with the highlighted strip
    plt.imshow(cv2.cvtColor(aligned_frame_display, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    plt.title(f"Strip from {start} to {end}")
    plt.axis('off')
    plt.show()

    # Return the extracted strip for further use
    return aligned_frame[:, start:end, :]


def get_strip(aligned_frame, strip_center, l_strip_width, r_strip_width):
    global count
    start = int(np.floor(strip_center - l_strip_width))
    end = int(np.floor(strip_center + r_strip_width))
    # print(start, end)
    count += 1
    # save_strip_with_box(aligned_frame, start, end)
    return aligned_frame[:, start: end, :]


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

    for i in range(1, len(canvas) - 1):  # Start from 1 because canvas[0] is empty
        # Compute the left and right widths of the strip
        # print(strip_center)
        l_strip_width = int(np.round(abs(stab_mats[i - 1][0, 2] / 2)))

        if i < len(canvas) - 1:
            r_strip_width = int(np.round(abs(stab_mats[i][0, 2] / 2)))
            # print(r_strip_width)
        else:
            r_strip_width = frame_width - strip_center

        # Extract the strip based on its position
        if i == 1:
            l_strip_width = strip_center
            # strip = get_first_strip(canvas[i], strip_center)

        if i < len(canvas) - 1:
            # display_canvas_with_matplotlib(canvas[i])
            strip = get_strip(canvas[i], strip_center, l_strip_width, r_strip_width)
        else:
            strip = get_last_strip(canvas[i], strip_center)

        # Warp the entire frame to the panorama's coordinate space
        # warped_strip = cv2.warpAffine(
        #     strip, cumulative_mats[i - 1][:2, :],
        #     (pano_frame.shape[1], pano_frame.shape[0]),
        #     flags=cv2.WARP_INVERSE_MAP
        # )

        # display_canvas_with_matplotlib(strip)
        # Extract only the part of the warped strip corresponding to the current strip
        strip_width = strip.shape[1]
        strip_start = int(np.floor(strip_center - l_strip_width))
        strip_end = int(np.floor(strip_center + r_strip_width))
        strip_segment = strip[:, strip_start:strip_end, :]  # Extract the required part
        # display_strip_with_box(strip, strip_start, strip_end)
        # display_canvas_with_matplotlib(strip_segment)

        # Paste the relevant part of the warped strip into the panorama frame
        if i > 1:
            pano_frame[:, strip_start:strip_end, :] = strip
        # save_strip_with_box(pano_frame, strip_start, strip_end)

        # Update the strip position
        strip_pos = strip_end
        strip_center = strip_end + r_strip_width

    return pano_frame


def make_canvas(vid):
    # Convert video frames to grayscale
    grayscale_vid = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in vid]

    print("generating transforms...")
    # Calculate the transformation matrices
    transforms = [np.eye(3)[:2, :]]
    for i in range(1, len(vid)):
        trans = get_trans_mat(grayscale_vid[i - 1], grayscale_vid[i])
        transforms.append(trans)

    # Stabilize the transformations
    print("stabilizing...")
    stab_trans_no_y = stabilize_transforms(transforms)
    stab_transforms = stab_trans_no_y.copy()

    for i in range(len(transforms)):
        stab_transforms[i][1, 2] = transforms[i][1, 2]

    # stabilize the frames
    print("aligning...")
    aligned_images = warp_frame(vid, stab_transforms, vid[0].shape[:2])
    # get cumulative transforms
    c_transforms = get_cumulative_transforms(stab_transforms)

    # compute canvas shape from cumulative transforms
    canvas_shape = get_canvas_dimensions(vid, c_transforms)
    print(canvas_shape)

    # warp the frames to the canvas
    print("warping canvas...")
    canvas = warp_frame(aligned_images[1:], c_transforms, canvas_shape)

    return canvas, c_transforms, stab_transforms


def stitch_stereo_pano(canvas, c_transforms, stab_transforms, vid):
    stereo_pano = []
    offset = -60
    for i in range(120):
        center_pano = make_pano(canvas, c_transforms, stab_transforms, len(vid[0]) // 2 + offset, vid[0].shape[1])
        offset += 2
        stereo_pano.append(center_pano)
        print(offset)

    stereo_pano_full = stereo_pano.copy()

    stereo_pano.reverse()
    for frame in stereo_pano:
        stereo_pano_full.append(frame)

    return stereo_pano_full


def main(vid_path, pano_method, out_file_path):
    """
    turns a video into a pano by pano_method
    :param vid_path: path to vid
    :param pano_method: 0 - STEREO, 1- DYNAMIC
    :param out_file_path: path for output file
    :return: pano video
    """
    # read video into array
    video = media.read_video(vid_path)

    vid = np.array(video)
    # vid = vid[::-1]
    print("generating canvas...")
    canvas, c_transforms, stab_transforms = make_canvas(vid)
    stereo_pano = None

    print("stitching panorama...")
    if pano_method == '0':
        stereo_pano = stitch_stereo_pano(canvas, c_transforms, stab_transforms, vid)

    print("writing video...")
    try:
        media.write_video(out_file_path, stereo_pano[:10])
    except Exception as e:
        print(f"Error writing video: {e}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
