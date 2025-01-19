import matplotlib.pyplot as plt
import numpy as np
import cv2
import mediapy as media
import ex4.src.ex4 as ex4
import ex4.src.ex4_reworked as reworked

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


raw_frame_a = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 255, 255, 0, 0, 0, 0],
               [0, 0, 255, 255, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

raw_frame_b = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 255, 255, 0, 0, 0],
               [0, 0, 0, 255, 255, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8
)
raw_frame_c = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 255, 255, 0, 0],
               [0, 0, 0, 0, 255, 255, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8
)
raw_frame_d = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 255, 255, 0],
               [0, 0, 0, 0, 0, 255, 255, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8
)
syn_vid = [raw_frame_a, raw_frame_b, raw_frame_c, raw_frame_d]

# Read the video
video = media.read_video("../../ex4-vids/boat.mp4")
vid = np.array(video)

# Convert video frames to grayscale
grayscale_vid = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in vid]

# Calculate the transformation matrices
transforms = []
for i in range(1, len(vid)):
    trans = reworked.get_trans_mat(grayscale_vid[i - 1], grayscale_vid[i])
    transforms.append(trans)

# Stabilize the transformations
stab_transforms = reworked.stabilize_transforms(transforms)

c_transforms = reworked.get_cumulative_transforms(stab_transforms)

# for i in range(len(transforms)):
#     c_transforms[i + 1][1, 2] = transforms[i][1, 2]

# Warp frames onto the canvas using stabilized transformations
canvas = reworked.warp_frame(vid, c_transforms)

# stab_transforms = np.insert(stab_transforms, 0, np.eye(3)[:2, :])
# c_transforms = np.insert(c_transforms, 0, np.eye(3))

center_pano = reworked.make_pano(canvas, c_transforms, stab_transforms, len(vid[0]) // 2)

display_canvas_with_matplotlib(center_pano)
# # Display the resulting canvas
# for frame in canvas:
#     display_canvas_with_matplotlib(frame)
