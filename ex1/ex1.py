import mediapy as media
import numpy as np
from PIL import Image


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
        grayscale_frames.append(grayscale_frame)

    # return frames list
    return grayscale_frames

def generate_cum_hists(grayscale_vid_arr):
    """
    generate a Cumulative hists list from a numpy grayscale array
    :param grayscale_vid_arr: a numpy array representing a grayscale video
    :return: cumulative histogram list by frames
    """
    cum_hists = []
    # iterate over frames and generate a cumulative histogram for each frame
    for frame in grayscale_vid_arr:
        temp_hist, _ = np.histogram(frame, bins=256, range=(0, 255))
        cum_hist = np.cumsum(temp_hist)

def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
    # open video from video path and make sure it's a valid path
    video = media.read_video(video_path)
    if video is None:
        print(f"{video_path} not found")
    # convert the video to grayscale for easier histogram manipulation
    vid_arr = np.array(video)
    grayscale_vid_arr = convert_vid_arr_to_grayscale(vid_arr)
    # generate a cumulative histogram array from frames

    # run a histogram equalization

    # compute histogram differences for pairs in video

    # check where the highest changes in histograms are

    # return (first scene index, last scene index)
    pass
