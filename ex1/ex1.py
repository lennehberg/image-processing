import sys
import mediapy as media
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def convert_vid_arr_to_grayscale(vid_arr):
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
        grayscale_frames.append(grayscale_frame_arr)

    # return frames list
    return grayscale_frames


def __calculate_hist(frame, bins_num=256):
    """calculates the histogram of the frame"""
    hist, _ = np.histogram(frame, bins=bins_num, range=(0, bins_num - 1))
    return hist


def __histogram_difference(hist1, hist2):
    """Calculates the vector distance of the histograms difference."""
    return np.sum(np.abs(hist1 - hist2))


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

    # iterate over the frame and get their differences from previous frames
    for i in range(1, len(grayscale_vid_arr)):
        # if video type is 1, just check differences between cumulative histograms
        if video_type == '1':
            hist_diff = __histogram_difference((__calculate_hist(grayscale_vid_arr[i - 1])),
                                                (__calculate_hist(grayscale_vid_arr[i])))
        else:  # if video type is 2, compare equalized cumulative histograms
            hist_diff = __histogram_difference(np.cumsum(__calculate_hist(grayscale_vid_arr[i - 1])),
                                               np.cumsum(__calculate_hist(grayscale_vid_arr[i])))

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
    grayscale_vid_arr = np.array(convert_vid_arr_to_grayscale(vid_arr))
    # detect a cut in video by checking for the maximal difference between histograms
    # of consecutive frames
    return detect_cuts(grayscale_vid_arr, video_type)


if __name__ == "__main__":
    video_path = sys.argv[1]
    video_type = sys.argv[2]
    # video = media.read_video(video_path)
    # vid_arr = np.array(video)
    # show_video_frames(vid_arr)
    # debug to see where frame cut is
    print(main(video_path=video_path, video_type=video_type))
