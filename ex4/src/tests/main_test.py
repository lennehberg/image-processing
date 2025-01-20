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
transforms = [np.eye(3)[:2, :]]
for i in range(1, len(vid)):
    trans = reworked.get_trans_mat(grayscale_vid[i - 1], grayscale_vid[i])
    transforms.append(trans)

# Stabilize the transformations
stab_transforms = reworked.stabilize_transforms(transforms)

c_transforms = reworked.get_cumulative_transforms(stab_transforms)

# for i in range(len(transforms)):
#     c_transforms[i + 1][1, 2] = transforms[i][1, 2]

canvas_shape = reworked.get_canvas_dimensions(vid, c_transforms)

print(canvas_shape)

# Warp frames onto the canvas using stabilized transformations
canvas = reworked.warp_frame(vid, transforms, canvas_shape)

# stab_transforms = np.insert(stab_transforms, 0, np.eye(3)[:2, :])
# c_transforms = np.insert(c_transforms, 0, np.eye(3))

center_pano = reworked.make_pano(canvas, c_transforms, transforms, len(vid[0]) // 2, vid[0].shape[1])

display_canvas_with_matplotlib(center_pano)
# # Display the resulting canvas
# for frame in canvas:
#     display_canvas_with_matplotlib(frame)


# anchoring to middle frame

l_frames = vid[:len(vid) // 2]
l_frames = l_frames[::-1]  # reverse the array for proper matrix calulations
r_frames = vid[len(vid) // 2:]

l_gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in l_frames]
r_gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in r_frames]

l_transforms = [np.eye(3)[:2, :]]
for i in range(1, len(l_frames)):
    trans = reworked.get_trans_mat(l_gray[i - 1], l_gray[i])
    l_transforms.append(trans)


r_transforms = [np.eye(3)[:2, :]]
for i in range(1, len(r_frames)):
    trans = reworked.get_trans_mat(r_gray[i - 1], r_gray[i])
    r_transforms.append(trans)

# Stabilize the transformations
l_stab_transforms = reworked.stabilize_transforms(l_transforms)
r_stab_transforms = reworked.stabilize_transforms(r_transforms)

l_c_transforms = reworked.get_cumulative_transforms(l_stab_transforms)
r_c_transforms = reworked.get_cumulative_transforms(r_stab_transforms)

# for i in range(len(transforms)):
#     c_transforms[i + 1][1, 2] = transforms[i][1, 2]

l_canvas_shape = reworked.get_canvas_dimensions(l_frames, l_c_transforms)
r_canvas_shape = reworked.get_canvas_dimensions(r_frames, r_c_transforms)

print(l_canvas_shape, r_canvas_shape)

# Warp frames onto the canvas using stabilized transformations
l_canvas = reworked.warp_frame(l_frames, l_transforms, l_canvas_shape)
r_canvas = reworked.warp_frame(r_frames, r_transforms, r_canvas_shape)

# stab_transforms = np.insert(stab_transforms, 0, np.eye(3)[:2, :])
# c_transforms = np.insert(c_transforms, 0, np.eye(3))

l_center_pano = reworked.make_pano(l_canvas, l_c_transforms, l_transforms, len(l_frames[0]) // 2, l_frames[0].shape[1])
r_center_pano = reworked.make_pano(r_canvas, r_c_transforms, r_transforms, len(r_frames[0]) // 2, r_frames[0].shape[1])

display_canvas_with_matplotlib(l_center_pano)
display_canvas_with_matplotlib(r_center_pano)
