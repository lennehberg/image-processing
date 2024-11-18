import sys
import mediapy as media
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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
    quantized_hist = np.zeros(num_levels, dtype=np.uint64)

    # Map the original histogram bins to quantized levels
    for i in range(256):
        quantized_level = min(i // step, num_levels - 1)  # Find quantization level
        quantized_hist[quantized_level] += hist[i]  # Accumulate the values into the quantized bin

    return quantized_hist


def convert_vid_arr_to_grayscale(vid_arr):
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
        grayscale_frame_arr = np.array(grayscale_frame)
        grayscale_frames.append(grayscale_frame_arr)

    # return frames list
    return grayscale_frames


def create_lut(cum_hist, bins_num=256):
    """Creates a lookup table from cumulative histogram to equalize frame"""
    # create empty lookup table
    lut = np.zeros(bins_num)

    # get the lowest gray level index (first level that is not 0)
    min_gray_level = 0
    for ind, level in enumerate(cum_hist):
        if level > 0:
            min_gray_level = ind
            break # once the first index where the gray level isn't 0 is found, break loop

    # get the highest gray level index (will always be 255 because of cumulative properties)
    max_gray_level = bins_num - 1

    for k in range(bins_num - 1):
        lut[k] = round((bins_num - 1) * ((cum_hist[k] - cum_hist[min_gray_level]) /
                              (cum_hist[max_gray_level] - cum_hist[min_gray_level])))

    return lut


def apply_lut(grayscale_frame, lut):
    """ Applies lookup table to grayscale frame to compute new equalized frame"""
    eq_frame = np.zeros(grayscale_frame.shape)
    # eq_frame[i , j] = lut[grayscale_frame[i, j]]
    for i in range(len(grayscale_frame)):
        for j in range(len(grayscale_frame[i])):
            eq_frame[i][j] = lut[grayscale_frame[i][j]]

    return eq_frame


def calculate_hist(frame, bins_num=256):
    """calculates the histogram of the frame"""
    hist, _ = np.histogram(frame.flatten(), bins=bins_num, range=(0, bins_num - 1))
    return hist


def equalize_frame_histogram(frame, bins_num=256):
    """Equalizes the frame's histogram and returns the new frame"""
    # compute histogram for frame
    hist = calculate_hist(frame)
    # compute cumulative histogram for CDF
    cum_hist = hist.cumsum()
    cum_hist = normalize_cum_hist(cum_hist, cum_hist.size)
    lut = create_lut(cum_hist)

    # apply resulting lookup table to grayscale frame to generate equalized frame
    eq_frame = apply_lut(frame, lut)

    return eq_frame.astype(np.uint8)


def histogram_difference(hist1, hist2):
    """Calculates the Chi-squared histogram difference."""
    # print(hist1, hist2)
    # print(np.abs(hist1 - hist2))
    return np.sum(np.abs(hist1 - hist2))
    # print(hist1)
    # hist1 = hist1.astype(np.int8)
    # print(hist1)
    # hist2 = hist2.astype(np.int8)
    # diffs = []
    # for i in range(len(hist1)):
    #     diff = hist1[i] - hist2[i] if hist1[i] > hist2[i] else hist1[i]-hist2[i]
    #     # print(f"{hist1[i]} - {hist2[i]} = {diff}")
    #     diffs.append(diff.astype(np.uint8))
    # return np.sum(diffs)


def normalize_cum_hist(cum_hist, pixel_count):
    return cum_hist / pixel_count


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
    grayscale_vid_arr = convert_vid_arr_to_grayscale(vid_arr)

    show_video_frames(vid_arr)

    eq_prev_frame = np.zeros(grayscale_vid_arr[0].shape)
    eq_cur_frame = np.zeros(grayscale_vid_arr[0].shape)
    # Equalize first frame and get it's cumulative histogram
    eq_prev_frame = equalize_frame_histogram(grayscale_vid_arr[0])
    if video_type == 1:  # only quantize according to video type
        eq_prev_cum_hist = np.cumsum(calculate_hist(eq_prev_frame))
    else:
        # print("quantizing first frame")
        eq_prev_cum_hist = quantize_histogram(np.cumsum(calculate_hist(eq_prev_frame)))

    # eq_prev_cum_hist = normalize_cum_hist(eq_prev_cum_hist, eq_prev_cum_hist.size)
    # init a max cum hist diff (assume video only has 1 cut)
    max_diff = 0
    max_cut_ind = 0

    # iterate over rest of the frames and get their differences from prev frame
    for i in range(1, len(grayscale_vid_arr)):
        eq_cur_frame = equalize_frame_histogram(grayscale_vid_arr[i])
        # equalize and quantize (if needed) cur frame
        if video_type == 1:
            # get the cur frame's cumulative histogram
            eq_cur_cum_hist = np.cumsum(calculate_hist(eq_cur_frame))
        else:
            # print("quantizing rest of frames")
            # get the cur frame's cumulative histogram
            eq_cur_cum_hist = quantize_histogram(np.cumsum(calculate_hist(eq_cur_frame)))

        # eq_cur_cum_hist = normalize_cum_hist(eq_cur_cum_hist, eq_prev_cum_hist.size)

        # if i == 34 or i == 35 or i == 174 or i == 175:
        #     show_video_frames([eq_prev_frame, eq_cur_frame])
        # fig = show_hists([eq_prev_cum_hist, eq_cur_cum_hist])
        #     plt.pause(0.05)
        #     plt.close(fig)

        # get the cumulative histogram differences
        hist_diff = histogram_difference(eq_prev_cum_hist,
                                         eq_cur_cum_hist)
        print(f"{hist_diff} at {i - 1, i}")
        # if the diff is bigger than the current maximum diff
        if hist_diff > max_diff:
            # set the max diff to cur diff
            max_diff = hist_diff
            # set the indexes if the frame for the cut
            max_cut_ind = (i - 1, i)

        # set prev to cur to check differences between next frame
        eq_prev_frame = eq_cur_frame
        eq_prev_cum_hist = eq_cur_cum_hist

    print(f"cut detected at frames {max_cut_ind} with diff {max_diff}")


def show_hists(hists):
    num_frames = len(hists)
    grid_size = int(np.ceil(np.sqrt(num_frames)))

    bins = np.arange(9)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(30, 15))

    for i, hist in enumerate(hists):
        # Plot the histogram in the corresponding subplot
        row = i // grid_size
        col = i % grid_size
        axes[row, col].bar(bins[:-1], hist, width=1, color='gray')
        axes[row, col].set_title(f"Frame {i+1}")
        axes[row, col].set_xlim(0, 255)
        axes[row, col].set_ylim(0, hist.max())

    # Turn off unused subplots
    for idx in range(num_frames, grid_size * grid_size):
        fig.delaxes(axes.flatten()[idx])

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    return fig


def show_video_frames(vid_arr):
    num_frames = len(vid_arr)
    grid_size = int(np.ceil(np.sqrt(num_frames)))
    # create a plot to display frames
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    # loop over frames of video and show them one by one
    for i, frame in enumerate(vid_arr):
        row = i // grid_size
        col = i % grid_size
        axes[row, col].imshow(frame)
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