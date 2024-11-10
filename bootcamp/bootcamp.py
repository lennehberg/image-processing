import numpy as nm
import mediapy as media
import matplotlib.pylab as plt
from PIL import Image

def main():
    # create a random number generator instance
    rng = nm.random.default_rng()

    # generate a random array of size (9,1)
    rand_arr = rng.integers(10, size=(9,1))
    print(rand_arr)

    # reshape the array to a 3x3 matrix
    rand_mat = rand_arr.reshape(3, 3)
    print(rand_mat)
    print(rand_mat.shape)
    print(rand_mat.dtype)

    # multiply first column by third column
    res_vec = rand_mat[:, 0] * rand_mat[:, 2]
    print(res_vec)

    # create a new 9x9 matrix and fill it with res_vec
    mat_9x9 = nm.tile(res_vec, (9, 3))
    print(mat_9x9)

    # create a random 9x9 matrix
    rand_mat_9x9 = rng.integers(10, size=(9, 9))
    print(rand_mat_9x9)

    # stack matrices to create tensor
    tensor = nm.stack((mat_9x9, rand_mat_9x9), axis=0)
    print(tensor)
    print(nm.sum(tensor), nm.mean(tensor), nm.median(tensor))

    tensor[tensor > nm.mean(tensor)] = 0
    print(tensor)

    # using mediapy, open video in bootcamp_images
    video = media.read_video("bootcamp_images/short_video.mp4")
    # convert to numpy array
    vid_arr = nm.array(video)

    # first_frame = Image.fromarray(vid_arr[0]).convert("L")
    # grayscale_frame = nm.array(first_frame)
    # plt.imshow(grayscale_frame, cmap="gray")
    # plt.show()

    grayscale_frames = []
    # convert video frame by frame to grayscale
    for frame in vid_arr:
        grayscale_frame = Image.fromarray(frame).convert("L")
        grayscale_frame_arr = nm.array(grayscale_frame)
        grayscale_frames.append(grayscale_frame_arr)

    grayscale_vid_arr = nm.stack(grayscale_frames)
    print(grayscale_vid_arr)

    media.write_video("bootcamp_images/short_video_grayscale.mp4", grayscale_vid_arr, fps=22)
    media.write_video("bootcamp_images/short_video_grayscale_half_frames.mp4", grayscale_vid_arr, fps=11)

    fps = len(vid_arr) / 15

    mean_orig = [nm.mean(frame) for frame in vid_arr]
    sum_orig = [nm.sum(frame) for frame in vid_arr]
    std_orig = [nm.std(frame) for frame in vid_arr]
    max_orig = [nm.max(frame) for frame in vid_arr]

    mean_gray = [nm.mean(frame) for frame in grayscale_vid_arr]
    std_gray = [nm.std(frame) for frame in grayscale_vid_arr]
    sum_gray = [nm.sum(frame) for frame in grayscale_vid_arr]
    max_gray = [nm.max(frame) for frame in grayscale_vid_arr]

    time = nm.arange(len(vid_arr)) / fps

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(time, mean_orig, label="Original", color="blue")
    axes[0].plot(time, mean_gray, label='Grayscale', color="gray")
    axes[0].set_ylabel("Mean")
    axes[0].legend()

    # Sum per frame
    axes[1].plot(time, sum_orig, label="Original", color="blue")
    axes[1].plot(time, sum_gray, label="Grayscale", color="gray")
    axes[1].set_ylabel("Sum")

    # Standard deviation per frame
    axes[2].plot(time, std_orig, label="Original", color="blue")
    axes[2].plot(time, std_gray, label="Grayscale", color="gray")
    axes[2].set_ylabel("Standard Deviation")

    # Max value per frame
    axes[3].plot(time, max_orig, label="Original", color="blue")
    axes[3].plot(time, max_gray, label="Grayscale", color="gray")
    axes[3].set_ylabel("Max Value")
    axes[3].set_xlabel("Time (s)")

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
