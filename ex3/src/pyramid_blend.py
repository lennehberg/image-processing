import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

PYRAMID_LEVELS = 8


def print_pyramid_shape(pyr):
    for i in range(len(pyr)):
        print(pyr[i].shape)


def display_pyramids(pyr, title="Pyramid"):
    plt.figure(figsize=(15, 15))
    for i, level in enumerate(pyr):
        plt.subplot(1, len(pyr), i + 1)
        plt.imshow(cv.cvtColor(level, cv.COLOR_BGR2RGB) if len(level.shape) == 3 else level, cmap="gray")
        plt.title(f"Level {i}")
        plt.axis("off")
    plt.show()


def restore_image(l_pyr):
    recon_img = l_pyr[0]
    for i in range(1, PYRAMID_LEVELS):
        recon_img = cv.pyrUp(recon_img)
        recon_img = cv.add(recon_img, l_pyr[i])

    return recon_img


def merge_pyramids(l_pyr_a, l_pyr_b, g_pyr_mask):
    merged = [[] for _ in range(PYRAMID_LEVELS)]
    for i in range(PYRAMID_LEVELS):
        merged[i] = g_pyr_mask[PYRAMID_LEVELS - 1 - i] * l_pyr_a[i] + (1 - g_pyr_mask[PYRAMID_LEVELS - 1 - i]) * \
                    l_pyr_b[i]

    return merged


def get_g_pyr(img):
    pyr_head = img.copy()
    g_pyr = [pyr_head]

    for i in range(PYRAMID_LEVELS):
        pyr_head = cv.pyrDown(pyr_head)
        g_pyr.append(pyr_head)

    return g_pyr


def get_l_pyr(img):
    g_pyr = get_g_pyr(img)
    l_pyr = [g_pyr[PYRAMID_LEVELS - 1]]
    for i in range(PYRAMID_LEVELS - 1, 0, -1):
        expanded_g = cv.pyrUp(g_pyr[i])
        subtracted_l = cv.subtract(g_pyr[i - 1], expanded_g)
        l_pyr.append(subtracted_l)

    return l_pyr


def blend_images(img_a, img_b, mask):
    l_pyr_a = get_l_pyr(img_a)
    l_pyr_b = get_l_pyr(img_b)
    mask = mask / 255.0
    m_pyr = get_g_pyr(mask)
    merged = merge_pyramids(l_pyr_a, l_pyr_b, m_pyr)
    return restore_image(merged)


def main(image_a_p, image_b_p, mask_p):
    img_a = cv.imread(image_a_p)
    img_b = cv.imread(image_b_p)
    mask = cv.imread(mask_p, cv.IMREAD_GRAYSCALE)  # Ensure mask is grayscale

    # Display pyramids for debugging and visualization
    print("Gaussian Pyramid of Image A")
    g_pyr_a = get_g_pyr(img_a)
    display_pyramids(g_pyr_a, "Gaussian Pyramid A")

    print("Gaussian Pyramid of Image B")
    g_pyr_b = get_g_pyr(img_b)
    display_pyramids(g_pyr_b, "Gaussian Pyramid B")

    print("Gaussian Pyramid of Mask")
    g_pyr_mask = get_g_pyr(mask)
    display_pyramids(g_pyr_mask, "Gaussian Pyramid Mask")

    print("Laplacian Pyramid of Image A")
    l_pyr_a = get_l_pyr(img_a)
    display_pyramids(l_pyr_a, "Laplacian Pyramid A")

    print("Laplacian Pyramid of Image B")
    l_pyr_b = get_l_pyr(img_b)
    display_pyramids(l_pyr_b, "Laplacian Pyramid B")

    # Blend images
    blended = blend_images(img_a, img_b, mask)
    cv.imwrite("pictures/blended.jpg", blended)
    print("Blended image saved to pictures/blended.jpg")


if __name__ == "__main__":
    image_a_path = sys.argv[1]
    image_b_path = sys.argv[2]
    mask_path = sys.argv[3]
    main(image_a_path, image_b_path, mask_path)
