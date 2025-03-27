import cv2
import numpy as np

def shades_of_gray_color_correction(image, p=6):
    """
    Apply Shades of Gray color constancy algorithm to balance colors.

    Parameters:
        image: Input BGR image (uint8).
        p: Minkowski norm parameter (default is 6).

    Returns:
        Corrected image (uint8).
    """
    # Convert to float32 to avoid overflow during calculations
    image = image.astype(np.float32)

    # Compute per-channel Minkowski norm
    norm_r = np.power(np.mean(np.power(image[:, :, 0], p)), 1/p)
    norm_g = np.power(np.mean(np.power(image[:, :, 1], p)), 1/p)
    norm_b = np.power(np.mean(np.power(image[:, :, 2], p)), 1/p)

    # Compute the mean of the three norms (gray reference)
    mean_gray = (norm_r + norm_g + norm_b) / 3.0

    # Scale each channel based on the ratio to the gray mean
    image[:, :, 0] *= (mean_gray / norm_r)
    image[:, :, 1] *= (mean_gray / norm_g)
    image[:, :, 2] *= (mean_gray / norm_b)

    # Clip values and convert back to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image

# Load an image
image = cv2.imread("target.jpg")

# Apply Shades of Gray color correction
corrected_image = shades_of_gray_color_correction(image, p=6)

# Save and display the result
cv2.imwrite("corrected_image.jpg", corrected_image)
# cv2.imshow("Corrected Image", corrected_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
