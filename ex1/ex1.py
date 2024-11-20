import sys
import mediapy as media
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def __power_transform(frame, maxinpval):
    """
    Power transform on a single frame, gamma = 7 for very bright frames,
    assuming 255 intensity levels
    """
    # power transform algorithm: r = c * (s ** gamma)
    c = 255 / np.log(1 + maxinpval) if 1 + maxinpval != 0 else 1
    gamma = 7
    frame = frame / 255
    frame = (frame ** gamma) * c
    frame = 255 * (frame / np.max(frame))
    return frame


def convert_vid_arr_to_grayscale(vid_arr, video_type):
    """
    Converts a video array (numpy array) to a grayscale video array (numpy array)
    if video is of type 2, run a quick power transform on the frame to equalize the intensities
    :param vid_arr: a numpy array representing the video
    :param video_type: type of video to convert
    :return: a grayscale video numpy array
    """
    # initiate a frames array for grayscale
    grayscale_frames = []

    # iterate over video frame (index 0), convert to grayscale image using PIL
    for frame in vid_arr:
        grayscale_frame = Image.fromarray(frame).convert("L")
        grayscale_frame_arr = np.array(grayscale_frame).astype(np.uint16)
        if video_type == '2':
            grayscale_frame_arr = __power_transform(grayscale_frame_arr, np.max(grayscale_frame_arr))
        grayscale_frames.append(grayscale_frame_arr)

    # return frames list
    return grayscale_frames


def __apply_lut(hist, lut):
    """ Applies lookup table to grayscale frame to compute new equalized frame"""
    eq_hist = np.zeros(hist.shape)
    # eq_frame[i , j] = lut[grayscale_frame[i, j]]
    for i, val in enumerate(lut):
        eq_hist[int(val)] += hist[i]
    return eq_hist


def __calculate_hist(frame, bins_num=256):
    """calculates the histogram of the frame"""
    hist, _ = np.histogram(frame, bins=bins_num, range=(0, bins_num - 1))
    return hist


def __normalize_cum_hist(cum_hist, pixel_count, level=256):
    return (level - 1) * cum_hist / pixel_count


def __linear_stretch(np_arr, new_min, new_max):
    """Linearly stretch the array to new min and new max"""
    # stretched[k] = (new_max - new_min) * (a[k] - a[m]) / (a[M] - a[m]) + new_min
    old_min = np.argmax(np_arr != 0)
    old_max = np.max(np_arr)

    numer = np_arr - old_min
    denom = old_max - old_min

    # math for uniform frames
    if denom == 0:
        return np.full_like(np_arr, new_min)

    stretched = (numer / denom) * (new_max - new_min) + new_min
    # make sure values under 0 are 0
    stretched[stretched < 0] = 0
    return stretched


def __histogram_difference(hist1, hist2):
    """Calculates the vector distance of the histograms difference."""
    return np.sum(np.abs(hist1 - hist2))


def __equalize_frame_histogram(frame, levels=256):
    """Equalize rhe frame's histogram and return the new frame"""
    # compute histogram for frame
    hist = __calculate_hist(frame, levels)
    # compute cumulative histogram for cdf
    cum_hist = np.cumsum(hist)
    # normalize cdf
    normed_cdf = __normalize_cum_hist(cum_hist, frame.size, levels)
    # perform linear stretching if min value isn't 0 and max value isn't 255
    # if np.min(normed_cdf) != 0 or np.max(normed_cdf) != levels - 1:
    normed_cdf = __linear_stretch(normed_cdf, 0, levels - 1)

    # round values in cdf to get mappings for lut
    cdf = np.round(normed_cdf)
    return __apply_lut(hist, cdf)


def detect_cuts(grayscale_vid_arr, video_type):
    """
    Detect a cut in a video by finding the maximum difference between cumulative histograms
    :param grayscale_vid_arr: grayscale numpy array of video to find cut
    :param video_type: type of video to find cut
    :return: a tuple of the indexes where the cut happened
    """
    first_frame = grayscale_vid_arr[0]
    frame_shape = grayscale_vid_arr[0].shape
    max_diff = 0
    max_cut_ind = (0, 0)
    # get the equalized cumulative histogram of the first frame (needed for category 2)
    eq_prev_hist = __equalize_frame_histogram(first_frame)
    prev_cdf = np.cumsum(eq_prev_hist)

    # iterate over rest of the frame and get their differences from previous frames
    for i in range(1, len(grayscale_vid_arr)):
        # if video type is 1, just check differences between cumulative histograms
        if video_type == '1':
            hist_diff = __histogram_difference(np.cumsum(__calculate_hist(grayscale_vid_arr[i - 1])),
                                               np.cumsum(__calculate_hist(grayscale_vid_arr[i])))
        else:  # if video type is 2, compare equalized cumulative histograms
            # equalize the current frame's histogram, and get it's cumulative histogram
            eq_cur_hist = __equalize_frame_histogram(grayscale_vid_arr[i])
            cur_cdf = np.cumsum(eq_cur_hist)
            # get the difference between cur cumulative histogram and prev cumulative histogram
            hist_diff = __histogram_difference(prev_cdf, cur_cdf)
            # set prev cdf to cur cdf and continue to rest of frames
            prev_cdf = cur_cdf

        # debug print: check histogram differences between frames
        # print(f"{hist_diff} at {i - 1, i}")
        # check if cdf distance is bigger than max
        if hist_diff > max_diff:
            # set max to current cdf distance
            max_diff = hist_diff
            # set the max cut index to cur and prev index of frames
            max_cut_ind = (i - 1, i)

    # debug print: check where cut was found and at what distance
    # print(f"cut detected at frames {max_cut_ind} with diff {max_diff}")
    return max_cut_ind


def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected
    (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
    # open video from vido path
    video = media.read_video(video_path)
    # convert video to grayscale
    vid_arr = np.array(video)
    grayscale_vid_arr = np.array(convert_vid_arr_to_grayscale(vid_arr, video_type))
    # detect a cut in video by checking for the maximal difference between histograms
    # of consecutive frames
    return detect_cuts(grayscale_vid_arr, video_type)


# if __name__ == "__main__":
#     video_path = sys.argv[1]
#     video_type = sys.argv[2]
#     # video = media.read_video(video_path)
#     # vid_arr = np.array(video)
#     # show_video_frames(vid_arr)
#     # debug to see where frame cut is
#     print(main(video_path=video_path, video_type=video_type))
