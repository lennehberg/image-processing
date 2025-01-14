import matplotlib.pyplot as plt
import numpy as np
import cv2
import mediapy as media
import ex4.src.ex4 as ex4

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

# Read the video
video = media.read_video("../../ex4-vids/boat.mp4")
vid = np.array(video)

# Convert video frames to grayscale
grayscale_vid = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in vid]

# Calculate the transformation matrices
transforms = []
for i in range(1, len(grayscale_vid)):
    trans = ex4.get_trans_mat(grayscale_vid[i - 1], grayscale_vid[i])
    transforms.append(trans)

# Stabilize the transformations
stab_transforms = ex4.stabilize_transforms(transforms)

# Warp frames onto the canvas using stabilized transformations
canvas = ex4.warp_frames(vid, stab_transforms)

center_pano = ex4.make_pano(canvas, canvas[0].shape, len(vid[0]) // 2, stab_transforms)

display_canvas_with_matplotlib(center_pano)
# # Display the resulting canvas
# for frame in canvas:
#     display_canvas_with_matplotlib(frame)
