import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


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


def main(img_a_p, img_b_p):
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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python hybrid_image.py <image_a_path> <image_b_path>")
    else:
        img_a_path = sys.argv[1]
        img_b_path = sys.argv[2]
        main(img_a_path, img_b_path)