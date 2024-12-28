import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

PYRAMID_LEVELS = 6


def print_pyramid_shape(pyr):
    for i in range(PYRAMID_LEVELS):
        print(pyr[i].shape)


def restore_image(l_pyr):
    recon_img = l_pyr[0]
    for i in range(1, PYRAMID_LEVELS):
        recon_img = cv.pyrUp(recon_img)
        recon_img = cv.add(recon_img, l_pyr[i])

    return recon_img


def merge_pyramids(l_pyr_a, l_pyr_b, g_pyr_mask):
    merged = [[] for _ in range(PYRAMID_LEVELS)]
    print_pyramid_shape(g_pyr_mask)
    # g_pyr_mask.reverse()
    # print_pyramid_shape(l_pyr_a)
    # print_pyramid_shape(l_pyr_b)
    print_pyramid_shape(g_pyr_mask)
    for i in range(PYRAMID_LEVELS):
        # print(i, g_pyr_mask[PYRAMID_LEVELS - 1 - i].shape)
        merged[i] = g_pyr_mask[PYRAMID_LEVELS - 1 - i] * l_pyr_a[i] + (1 - g_pyr_mask[PYRAMID_LEVELS - 1 - i]) * l_pyr_b[i]

    return merged


def get_g_pyr(img):
    pyr_head = img.copy()
    # print(pyr_head.shape)
    g_pyr = [pyr_head]

    for i in range(PYRAMID_LEVELS):
        pyr_head = cv.pyrDown(pyr_head)
        # print(pyr_head.shape)
        g_pyr.append(pyr_head)
    # print_pyramid_shape(g_pyr)
    # print(g_pyr)
    return g_pyr


def get_l_pyr(img):
    g_pyr = get_g_pyr(img)
    l_pyr = [g_pyr[PYRAMID_LEVELS - 1]]
    for i in range(PYRAMID_LEVELS - 1, 0, -1):
        # print(i)
        # print(len(g_pyr))
        expanded_g = cv.pyrUp(g_pyr[i])
        # print(expanded_g.shape, g_pyr[i - 1].shape)
        subtracted_l = cv.subtract(g_pyr[i - 1], expanded_g)
        l_pyr.append(subtracted_l)

    return l_pyr


def blend_images(img_a, img_b, mask):
    # get the laplacian pyramids of both images
    l_pyr_a = get_l_pyr(img_a)
    l_pyr_b = get_l_pyr(img_b)
    # normalize mask and get the gaussian pyramid of the mask
    mask = mask / 255.0
    m_pyr = get_g_pyr(mask)
    # print_pyramid_shape(m_pyr)
    # merge pyramids
    merged = merge_pyramids(l_pyr_a, l_pyr_b, m_pyr)
    # reconstruct image from merged pyramids
    return restore_image(merged)


def main(image_a_p, image_b_p, mask_p):
    img_a = cv.imread(image_a_p)
    img_b = cv.imread(image_b_p)
    mask = cv.imread(mask_p)
    # print(img_b.shape, mask.shape)
    # print(img_a.shape, img_b.shape, mask.shape)
    blended = blend_images(img_a, img_b, mask)
    cv.imwrite("pictures/blended.jpg", blended)


if __name__ == "__main__":
    image_a_path = sys.argv[1]
    image_b_path = sys.argv[2]
    mask_path = sys.argv[3]
    main(image_a_path, image_b_path, mask_path)
