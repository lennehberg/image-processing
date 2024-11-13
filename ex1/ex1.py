import sys
import mediapy as media
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

def generate_cum_hists(grayscale_vid_arr):
    """
    generate a Cumulative hists list from a numpy grayscale array
    :param grayscale_vid_arr: a numpy array representing a grayscale video
    :return: cumulative histogram list by frames
    """
    hists_and_bins = []
    cum_hists = []
    # iterate over frames and generate a cumulative histogram for each frame
    for frame in grayscale_vid_arr:
        temp_hist, bins = np.histogram(frame, bins=256, range=(0, 255))
        hists_and_bins.append((temp_hist, bins))
        cum_hist = np.cumsum(temp_hist)
        # append cum hist to cum hists list
        cum_hists.append(cum_hist)

    return hists_and_bins, cum_hists


def create_lut(cum_hist, N):
    # create an empty lookup table
    lut = np.zeros(256)
    # using T(k) = round(N * ((C(k) - C(m)) / ((C(N) - c(m)))
    for k in range(256):
        lut[k] = round(N * ((cum_hist[k] - cum_hist[np.argmax(cum_hist != 0)]) /
                            (cum_hist[N] - cum_hist[np.argmax(cum_hist != 0)])))
        # normalize values in case of overflow
        if lut[k] < 0:
            lut[k] = 0

    return lut


def apply_lut(frame, lut):
    return np.interp(frame.flatten(), np.arange(256), lut).reshape(frame.shape)


def equalize_histogram(gray_frame, cum_hist):
    """
    run a simple histogram equalization algorithm
    :param gray_frame: original grayscale frame to equalize
    :param cum_hist: cumulative frame histogram
    :return: equalized frame
    """
    # normalize the cum_hists to get a normalized cdf
    normed_cdf = cum_hist * (255 / cum_hist[-1])
    # create a look-up table and map values accordingly
    lut = create_lut(cum_hist, 255)
    eq_frame = apply_lut(gray_frame, lut)
    # return frame to uint8 type
    eq_frame = eq_frame.astype(np.uint8)
    eq_hist, _ = np.histogram(eq_frame, bins=256, range=(0, 256))
    return eq_frame, eq_hist


def equalize_video_frame(grayscale_vid_arr, cum_hists):
    """
    iterate over all frame in grayscale video array and run the equalization algorithm
    :param grayscale_vid_arr:
    :param cum_hists:
    :return: array of equalized frames
    """
    # init a frame count to keep track of needed hists
    frame_count = 0
    eq_frames = []
    eq_hists = []
    for frame in grayscale_vid_arr:
        # run the equalization algorithm on each frame
        eq_frame, eq_hist = equalize_histogram(frame, cum_hists[frame_count])
        # append generated frame to frame list
        eq_frames.append(eq_frame)
        eq_hists.append(eq_hist)
        # increment frame count for next frame
        frame_count += 1

    return eq_frames, eq_hists


def get_hist_diff(l_hist, r_hist):
    return np.sum(np.abs(l_hist - r_hist))


def get_hist_paris_diff(eq_hists):
    """
    get the hist differences of consecutive frame
    :param eq_hists:
    :return: array[{(prev_frame_num, cur_frame_num), diff}]
    """
    pairs_diff = []
    for k in range(len(eq_hists) - 2):
        # compute the diff and store in paris_diff
        pairs_diff.append(((k, k + 1), get_hist_diff(eq_hists[k], eq_hists[k + 1])))
    return pairs_diff


def get_cut_diff_indexes(hists_pairs_diffs):
    cut_indexes = []
    max_diff = 0
    max_pair = 0
    for pair in hists_pairs_diffs:
        print(f"diff is {pair[1]}")
        if pair[1] > max_diff:
            max_pair = pair
            max_diff = pair[1]

    return max_pair



def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected
    (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
    print("starting cut detection...")
    # open video from video path and make sure it's a valid path
    video = media.read_video(video_path)
    if video is None:
        print(f"{video_path} not found")
    print("video found!")
    # convert the video to grayscale for easier histogram manipulation
    vid_arr = np.array(video)
    grayscale_vid_arr = convert_vid_arr_to_grayscale(vid_arr)
    # generate a cumulative histogram array from frames
    hists_and_bins, cum_hists = generate_cum_hists(grayscale_vid_arr)
    # run a histogram equalization
    eq_vid, eq_vid_hists = equalize_video_frame(grayscale_vid_arr, cum_hists)
    # compute histogram differences for pairs in video
    hists_pairs_diffs = get_hist_paris_diff(eq_vid_hists)
    # check where the highest changes in histograms are
    cut_indexes = get_cut_diff_indexes(hists_pairs_diffs)
    # return (first scene index, last scene index)
    # print(f"cuts at: {', '.join([str(index) for index in cut_indexes])}")
    grayscale_vid_arr = np.array(grayscale_vid_arr)
    print(grayscale_vid_arr.shape)
    print(cut_indexes)
    # Create a figure with 1 row and 2 columns
    plt.figure(figsize=(10, 5))

    # First subplot (first image)
    plt.subplot(1, 2, 1)
    plt.imshow(grayscale_vid_arr[cut_indexes[0][0]], cmap='gray')  # Use cmap='gray' for grayscale images
    plt.title('Image 1')
    plt.axis('off')  # Turn off axis

    # Second subplot (second image)
    plt.subplot(1, 2, 2)
    plt.imshow(grayscale_vid_arr[cut_indexes[0][1]], cmap='gray')  # Use cmap='gray' for grayscale images
    plt.title('Image 2')
    plt.axis('off')  # Turn off axis

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    video_path = sys.argv[1]
    main(video_path=video_path, video_type=1)
