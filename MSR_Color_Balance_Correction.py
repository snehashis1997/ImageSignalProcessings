import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def multi_scale_retinex(image, sigmas=[15, 80, 250], weights=None):
    """
    Applies Multi-Scale Retinex (MSR) for intrinsic image decomposition.
    """
    image = image.astype(np.float32) + 1e-6  # Avoid division by zero
    log_image = np.log(image)

    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)

    illumination = np.zeros_like(image)
    for sigma, weight in zip(sigmas, weights):
        smoothed = gaussian_filter(log_image, sigma=(sigma, sigma, 0))  
        illumination += weight * smoothed

    illumination = np.exp(illumination)  
    reflectance = image / (illumination + 1e-6)

    reflectance = np.clip(reflectance * 255 / reflectance.max(), 0, 255).astype(np.uint8)
    return reflectance, illumination


def color_balance_correction(image):
    """
    Applies color balance correction using White Balance (Gray World Assumption).
    """
    balanced_image = image.astype(np.float32)
    
    # Compute mean for each channel
    mean_r = np.mean(balanced_image[:, :, 2])  # Red
    mean_g = np.mean(balanced_image[:, :, 1])  # Green
    mean_b = np.mean(balanced_image[:, :, 0])  # Blue

    mean_gray = (mean_r + mean_g + mean_b) / 3

    # Scale each channel to match the mean gray value
    balanced_image[:, :, 2] *= mean_gray / mean_r  # Red
    balanced_image[:, :, 1] *= mean_gray / mean_g  # Green
    balanced_image[:, :, 0] *= mean_gray / mean_b  # Blue

    # Normalize and clip
    balanced_image = np.clip(balanced_image, 0, 255).astype(np.uint8)
    
    return balanced_image


# Load an image
image = cv2.imread("3.jpg")

# Apply Multi-Scale Retinex
reflectance, illumination = multi_scale_retinex(image)

# Apply Color Balance Correction
balanced_reflectance = color_balance_correction(reflectance)

# Save and display results
cv2.imwrite("balanced_reflectance.jpg", balanced_reflectance)
cv2.imshow("Original Image", image)

# cv2.imshow("Reflectance (MSR)", reflectance)
# cv2.imshow("Color Balanced Reflectance", balanced_reflectance)
# cv2.waitKey(0)
# cv2.destroyAllWindows()