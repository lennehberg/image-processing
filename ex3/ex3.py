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


def main_blend(image_a_p, image_b_p, mask_p):
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


def show_image(img, title="Image"):
    """
    Displays an image using matplotlib.

    Parameters:
    - img: The image to display. Can be a grayscale or color image.
    - title: Title for the displayed image.
    """
    plt.figure(figsize=(6, 6))
    plt.title(title)
    cmap = "gray" if len(img.shape) == 2 else None
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.show()


def apply_gaussian_blur(img, kernel_size=(5, 5), sigma=0):
    """
    Applies a Gaussian blur to remove high frequencies from the image.

    Parameters:
    - img: Input grayscale image.
    - kernel_size: Size of the Gaussian kernel (must be odd numbers).
    - sigma: Standard deviation for Gaussian kernel.

    Returns:
    - Blurred image (low frequencies).
    """
    return cv.GaussianBlur(img, kernel_size, sigma)


def extract_high_frequencies(img, kernel_size=(5, 5), sigma=0):
    """
    Extracts the high-frequency component of an image.

    Parameters:
    - img: Input grayscale image.
    - kernel_size: Size of the Gaussian kernel (must be odd numbers).
    - sigma: Standard deviation for Gaussian kernel.

    Returns:
    - High-frequency image (original image minus blurred image).
    """
    low_frequencies = apply_gaussian_blur(img, kernel_size, sigma)
    return img - low_frequencies


def create_hybrid_image(img_a, img_b, kernel_size=(31, 31), sigma=5):
    """
    Creates a hybrid image by combining the low frequencies of one image and
    the high frequencies of another.

    Parameters:
    - img_a: Grayscale image for low frequencies.
    - img_b: Grayscale image for high frequencies.
    - kernel_size: Size of the Gaussian kernel.
    - sigma: Standard deviation for Gaussian kernel.

    Returns:
    - Hybrid image.
    """
    # Extract low frequencies from image A
    low_frequencies = apply_gaussian_blur(img_a, kernel_size, sigma)

    # Extract high frequencies from image B
    high_frequencies = extract_high_frequencies(img_b, kernel_size, sigma)

    # Combine low and high frequencies
    hybrid_image = low_frequencies + high_frequencies

    # Normalize the image to ensure values are in a displayable range
    hybrid_image = np.clip(hybrid_image, 0, 255).astype(np.uint8)
    return hybrid_image


def main_hybrid(img_a_p, img_b_p):
    """
    Main function to create and save a hybrid image.

    Parameters:
    - img_a_p: Path to the first image (used for low frequencies).
    - img_b_p: Path to the second image (used for high frequencies).
    """
    # Read input images
    img_a = cv.imread(img_a_p, cv.IMREAD_GRAYSCALE)
    img_b = cv.imread(img_b_p, cv.IMREAD_GRAYSCALE)

    if img_a is None or img_b is None:
        print("Error: One or both input images could not be loaded.")
        return

    # Create the hybrid image
    hybrid = create_hybrid_image(img_a, img_b)

    # Save and display the hybrid image
    output_path = "pictures/hybrid_image.jpg"
    cv.imwrite(output_path, hybrid)
    print(f"Hybrid image saved to {output_path}")

    # Show the hybrid image
    show_image(hybrid, title="Hybrid Image")



