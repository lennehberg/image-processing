import sys
import mediapy as media
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def power_transform(frame, maxinpval):
    c = 255 / np.log(1 + maxinpval) if 1 + maxinpval != 0 else 1
    gamma = 10
    frame = frame / 255
    frame = (frame ** gamma) * c
    frame = 255 * (frame / np.max(frame))
    return frame


def quantize_histogram(hist, num_levels=16):
    """
    Quantizes the histogram by reducing the number of gray levels.

    :param hist: Input histogram (1D numpy array with 256 bins)
    :param num_levels: The number of gray levels to reduce the histogram to
    :return: Quantized histogram (1D numpy array with reduced bins)
    """
    # Determine the range for each quantization level
    max_value = 255
    step = max_value // num_levels

    # Create an array to hold the quantized histogram
    quantized_hist = np.zeros(num_levels, dtype=np.uint8)

    # Map the original histogram bins to quantized levels
    for i in range(256):
        quantized_level = min(i // step, num_levels - 1)  # Find quantization level
        quantized_hist[quantized_level] += hist[i]  # Accumulate the values into the quantized bin

    return quantized_hist


def convert_vid_arr_to_grayscale(vid_arr, video_type):
    """
    Converts a video array (numpy array) to a grayscale video array (numpy array)
    :param vid_arr: a numpy array representing the video
    :return: a grayscale video numpy array
    """
    # initiate a frames array for grayscale
    grayscale_frames = []

    # iterate over video frame (index 0), convert to grayscale image using PIL
    for frame in vid_arr:
        grayscale_frame = Image.fromarray(frame).convert("L")
        grayscale_frame_arr = np.array(grayscale_frame).astype(np.uint16)
        if video_type == '2':
            grayscale_frame_arr = power_transform(grayscale_frame_arr, np.max(grayscale_frame_arr))
        grayscale_frames.append(grayscale_frame_arr)

    # return frames list
    return grayscale_frames


def apply_lut(hist, lut):
    """ Applies lookup table to grayscale frame to compute new equalized frame"""
    eq_hist = np.zeros(hist.shape)
    # eq_frame[i , j] = lut[grayscale_frame[i, j]]
    for i, val in enumerate(lut):
        eq_hist[int(val)] += hist[i]
    return eq_hist


def calculate_hist(frame, bins_num=256):
    """calculates the histogram of the frame"""
    hist, _ = np.histogram(frame, bins=bins_num, range=(0, bins_num - 1))
    return hist


def normalize_cum_hist(cum_hist, pixel_count, level=256):
    return (level - 1) * cum_hist / pixel_count


def linear_stretch(np_arr, new_min, new_max):
    """Linearly stretch the array to new min and new max"""

    old_min = np.argmax(np_arr != 0)
    old_max = np.max(np_arr)

    numer = np_arr - old_min
    denom = old_max - old_min

    if denom == 0:
        return np.full_like(np_arr, new_min)

    stretched = (numer / denom) * (new_max - new_min) + new_min
    stretched[stretched < 0] = 0
    return stretched


def histogram_difference(hist1, hist2):
    """Calculates the Chi-squared histogram difference."""
    return np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10))


def equalize_frame_histogram(frame, levels=256):
    """Equalize rhe frame's histogram and return the new frame"""
    # compute histogram for frame
    hist = calculate_hist(frame, levels)
    # compute cumulative histogram for cdf
    cum_hist = np.cumsum(hist)
    # normalize cdf
    normed_cdf = normalize_cum_hist(cum_hist, frame.size, levels)
    # perform linear stretching if min value isn't 0 and max value isn't 255
    # if np.min(normed_cdf) != 0 or np.max(normed_cdf) != levels - 1:
    normed_cdf = linear_stretch(normed_cdf, 0, levels - 1)

    # round values in cdf to get mappings for lut
    cdf = np.round(normed_cdf)
    return apply_lut(hist, cdf)


def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected
    (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
    # open video from vido path, make sure valid path
    video = media.read_video(video_path)
    if video is None:
        print(f"{video_path} not found!")

    # convert video to grayscale
    vid_arr = np.array(video)
    show_video_frames(vid_arr)
    grayscale_vid_arr = np.array(convert_vid_arr_to_grayscale(vid_arr, video_type))
    eq_prev_frame = np.zeros(grayscale_vid_arr[0].shape)
    eq_cur_frame = np.zeros(grayscale_vid_arr[0].shape)
    max_diff = 0
    max_cut_ind = 0
    # Equalize first frame and get it's cumulative histogram
    eq_prev_frame = equalize_frame_histogram(grayscale_vid_arr[0])
    print(f"frame size of normalization is: {grayscale_vid_arr[0].size}")
    eq_prev_cum_hist = (np.cumsum(eq_prev_frame))

    # iterate over rest of the frames and get their differences from prev frame
    for i in range(1, len(grayscale_vid_arr)):
        if video_type == '2':
            eq_cur_frame = equalize_frame_histogram(grayscale_vid_arr[i])
            # equalize and quantize (if needed) cur frame
            eq_cur_cum_hist = (np.cumsum(eq_cur_frame))
            # get the cumulative histogram differences
            # show_video_frames([eq_prev_frame, eq_cur_frame])
            hist_diff = histogram_difference(eq_prev_cum_hist,
                                             eq_cur_cum_hist)
            # set prev to cur to check differences between next frame
            eq_prev_frame = eq_cur_frame
            eq_prev_cum_hist = eq_cur_cum_hist
        else:
            hist_diff = histogram_difference(calculate_hist(grayscale_vid_arr[i - 1]),
                                             calculate_hist(grayscale_vid_arr[i]))
        print(f"{hist_diff} at {i - 1, i}")
        # if the diff is bigger than the current maximum diff
        if hist_diff > max_diff:
            # set the max diff to cur diff
            max_diff = hist_diff
            # set the indexes if the frame for the cut
            max_cut_ind = (i, i + 1)

    print(f"cut detected at frames {max_cut_ind} with diff {max_diff}")
    return max_cut_ind

def show_video_frames(vid_arr):
    num_frames = len(vid_arr)
    grid_size = int(np.ceil(np.sqrt(num_frames)))
    # create a plot to display frames
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    # loop over frames of video and show them one by one
    for i, frame in enumerate(vid_arr):
        row = i // grid_size
        col = i % grid_size
        axes[row, col].imshow(frame, cmap="gray")
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Frame {i + 1}")

    for idx in range(num_frames, grid_size * grid_size):
        fig.delaxes(axes.flatten()[idx])

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    return fig


if __name__ == "__main__":
    video_path = sys.argv[1]
    video_type = sys.argv[2]
    # video = media.read_video(video_path)
    # vid_arr = np.array(video)
    # show_video_frames(vid_arr)
    # debug to see where frame cut is
    main(video_path=video_path, video_type=video_type)
