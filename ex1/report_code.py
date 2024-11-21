import matplotlib.pyplot as plt
import ex1 as cutDetect


def procure_category_1():
    # detect cut on the category 1 videos using type 1, and save their cut with the frame index
    vid1_path = "videos/video1_category1.mp4"
    vid2_path = "videos/video2_category1.mp4"

    plot_results(vid1_path, vid2_path, '1')


def procure_category_2():
    # detect cut on the category 1 videos using type 1, and save their cut with the frame index
    vid1_path = "videos/video3_category2.mp4"
    vid2_path = "videos/video4_category2.mp4"

    plot_results(vid1_path, vid2_path, '2')
    plot_results(vid1_path, vid2_path, '1')


def plot_results(vid1_path, vid2_path, video_type):
    # read videos using mediapy
    vid1 = cutDetect.media.read_video(vid1_path)
    vid2 = cutDetect.media.read_video(vid2_path)

    vid1_cut_ind = cutDetect.main(vid1_path, video_type)
    vid2_cut_ind = cutDetect.main(vid2_path, video_type)

    vid1_arr = cutDetect.np.array(vid1)
    vid2_arr = cutDetect.np.array(vid2)

    # show the cumulative histogram of the frame below the frame
    grayscale_frames = cutDetect.convert_vid_arr_to_grayscale([vid1_arr[vid1_cut_ind[0]],
                                                               vid1_arr[vid1_cut_ind[1]]], video_type)
    cdf1 = cutDetect.np.cumsum(cutDetect.__calculate_hist(grayscale_frames[0]))
    cdf2 = cutDetect.np.cumsum(cutDetect.__calculate_hist(grayscale_frames[1]))
    bins = cutDetect.np.arange(257)
    # plot the frames using pyplot and save the config for report
    fig1, axes1 = cutDetect.plt.subplots(2, 2, figsize=(10, 5))

    # show previous frame in position 0 and next frame in position 1
    axes1[0][0].imshow(vid1_arr[vid1_cut_ind[0]])
    axes1[0][0].set_title(f"Frame {vid1_cut_ind[0]}")
    axes1[0][0].axis("off")

    # plot the first frame's cumulative histogram
    axes1[1][0].bar(bins[:-1], cdf1, width=1, color="black")
    axes1[1][0].set_title("CDF (Grayscale)")
    axes1[1][0].set_xlabel("Pixel Intensity")
    axes1[1][0].set_ylabel("Cumulative Frequency")

    # show next frame in position 1
    axes1[0][1].imshow(vid1_arr[vid1_cut_ind[1]])
    axes1[0][1].set_title(f"Frame {vid1_cut_ind[1]}")
    axes1[0][1].axis("off")

    # plot the second frame's cumulative histogram
    axes1[1][1].bar(bins[:-1], cdf2, width=1, color="black")
    axes1[1][1].set_title("CDF (Grayscale / Intensity corrected)")
    axes1[1][1].set_xlabel("Pixel Intensity")
    axes1[1][1].set_ylabel("Cumulative Frequency")

    # save figure to disk
    plt.savefig("report_extras/vid1_cut.png", dpi=300, bbox_inches="tight")

    plt.tight_layout()
    plt.show()

    # do the same for video 2
    fig2, axes2 = cutDetect.plt.subplots(2, 2, figsize=(10, 5))

    # show the cumulative histogram of the frame below the frame
    grayscale_frames = cutDetect.convert_vid_arr_to_grayscale([vid2_arr[vid2_cut_ind[0]],
                                                               vid2_arr[vid2_cut_ind[1]]], 1)
    cdf1 = cutDetect.np.cumsum(cutDetect.__calculate_hist(grayscale_frames[0]))
    cdf2 = cutDetect.np.cumsum(cutDetect.__calculate_hist(grayscale_frames[1]))

    # show previous frame in position 0 and next frame in position 1
    axes2[0][0].imshow(vid2_arr[vid2_cut_ind[0]])
    axes2[0][0].set_title(f"Frame {vid2_cut_ind[0]}")
    axes2[0][0].axis("off")

    # plot the first frame's cumulative histogram
    axes2[1][0].bar(bins[:-1], cdf1, width=1, color="black")
    axes2[1][0].set_title("CDF (Grayscale)")
    axes2[1][0].set_xlabel("Pixel Intensity")
    axes2[1][0].set_ylabel("Cumulative Frequency")

    # show next frame in position 1
    axes2[0][1].imshow(vid2_arr[vid2_cut_ind[1]])
    axes2[0][1].set_title(f"Frame {vid2_cut_ind[1]}")
    axes2[0][1].axis("off")

    # plot the second frame's cumulative histogram
    axes2[1][1].bar(bins[:-1], cdf2, width=1, color="black")
    axes2[1][1].set_title("CDF (Grayscale / Intensity Corrected)")
    axes2[1][1].set_xlabel("Pixel Intensity")
    axes2[1][1].set_ylabel("Cumulative Frequency")

    # save figure to disk
    plt.savefig("report_extras/vid2_cut.png", dpi=300, bbox_inches="tight")

    plt.tight_layout()
    plt.show()


def main():
    """Main entry point for generating figures for report for ex1"""
    procure_category_1()
    procure_category_2()

if __name__ == "__main__":
    main()
