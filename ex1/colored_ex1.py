import sys
import mediapy as media
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def split_channels(frame):
    """splits the frame into channels and returns a tuple of (r_frame, g_frame, b_frame)"""
    r_frame = frame[:, :, 0]
    g_frame = frame[:, :, 1]
    b_frame = frame[:, :, 2]

    return r_frame, g_frame, b_frame


def normalize_cum_hist(cum_hist, pixel_count):
    return 255 * cum_hist / pixel_count


def log_transform(video):
    c = 255 / np.log(1 + 255)
    # Iterate over each frame in the video
    for f in range(video.shape[0]):  # f is the frame index
        # Iterate over each row (height) in the frame
        for h in range(video.shape[1]):  # h is the height index
            # Iterate over each column (width) in the frame
            for w in range(video.shape[2]):  # w is the width index
                # Iterate over each color channel (RGB)
                for c in range(video.shape[3]):  # c is the channel index (0: Red, 1: Green, 2: Blue)
                    # print("true" if video[f, h, w, c] + 1 == 0 else "")
                    if video[f, h, w, c] + 1 > 0:
                        transformed = np.round(c * np.log(1 + video[f, h, w, c]))
                        if transformed > 255:
                            transformed = 255
                        elif transformed < 0:
                            transformed = 0
                        video[f, h, w, c] = int(transformed)
    print("finished log transforming!")

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
    hist, _ = np.histogram(frame, bins=bins_num, range=(0, bins_num))
    return hist


def equalize_frame_histogram(frame, bins_num=256):
    """Equalizes the frame's histogram and returns the new frame"""
    # compute histogram for frame
    hist = calculate_hist(frame, bins_num)
    hist = hist / np.sum(hist)
    # compute cumulative histogram for CDF
    cum_hist = hist.cumsum()
    # print(frame.size, frame.shape)
    cum_hist = normalize_cum_hist(cum_hist, frame.size)
    lut = create_lut(cum_hist)

    # apply resulting lookup table to grayscale frame to generate equalized frame
    eq_frame = apply_lut(frame, lut)

    return eq_frame.astype(np.uint8)


def histogram_difference(hist1, hist2):
    """Calculates the Chi-squared histogram difference."""
    epsilon = 1e-10
    return np.sum(np.abs(hist1 - hist2))


def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected
    (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
    # open video file and convert to numpy array
    video = media.read_video(video_path)
    vid_arr = np.array(video)
    # log_transform(vid_arr)
    # show_video_frames(vid_arr)

    # split the first frame into channels and equalize each channel
    prev_rgb_frame = split_channels(vid_arr[0])
    prev_r_eq_cum = np.cumsum(
                        calculate_hist(
                            equalize_frame_histogram(prev_rgb_frame[0])))
    prev_g_eq_cum = np.cumsum(
                        calculate_hist(
                            equalize_frame_histogram(prev_rgb_frame[1])))
    prev_b_eq_cum = np.cumsum(
                        calculate_hist(
                            equalize_frame_histogram(prev_rgb_frame[2])))

    # going from the next frame,
    # do the same as above and get distance for each channel
    max_diff = 0
    max_cut_ind = 0
    for i in range(1, len(vid_arr)):
        # split frame into channels
        cur_rgb_frame = split_channels(vid_arr[i])
        # equalize histograms for each channel
        cur_r_eq_cum = np.cumsum(
                            calculate_hist(
                                equalize_frame_histogram(cur_rgb_frame[0])))
        cur_g_eq_cum = np.cumsum(
                            calculate_hist(
                                equalize_frame_histogram(cur_rgb_frame[1])))
        cur_b_eq_cum = np.cumsum(
                            calculate_hist(
                                equalize_frame_histogram(cur_rgb_frame[2])))
        # if i == 149 or i == 150 or i == 249:
        #     show_hists([prev_r_eq_cum, cur_r_eq_cum, prev_g_eq_cum, cur_g_eq_cum, prev_b_eq_cum, cur_b_eq_cum])

        # check if the hist difference is bigger than the previous one -
        # maximal difference = cut
        hist_diff = (histogram_difference(prev_r_eq_cum, cur_r_eq_cum) +
                     histogram_difference(prev_g_eq_cum, cur_g_eq_cum) +
                     histogram_difference(prev_b_eq_cum, cur_b_eq_cum))
        print(f"{hist_diff} at {i - 1, i}")
        if hist_diff > max_diff:
            max_diff = hist_diff
            max_cut_ind = (i, i + 1)  # indexed 0

        # set prevs to curs and continue checking distances
        prev_r_eq_cum = cur_r_eq_cum
        prev_g_eq_cum = prev_g_eq_cum
        prev_b_eq_cum = prev_b_eq_cum

    print(f"cut detected at frames {max_cut_ind} with diff {max_diff}")


def show_hists(hists):
    num_frames = len(hists)
    grid_size = int(np.ceil(np.sqrt(num_frames)))

    bins = np.arange(257)

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
