import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def multi_scale_retinex(image, sigmas=[15, 80, 250], weights=None):
    """
    Applies Multi-Scale Retinex (MSR) for intrinsic image decomposition.

    Parameters:
        image (numpy.ndarray): Input image (BGR format).
        sigmas (list): List of sigma values for multi-scale Retinex.
        weights (list): List of weights for each scale (default: equal weights).

    Returns:
        reflectance (numpy.ndarray): Reflectance component.
        illumination (numpy.ndarray): Illumination component.
    """
    image = image.astype(np.float32) + 1e-6  # Avoid division by zero
    log_image = np.log(image)

    # Default equal weights if not specified
    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)

    # Multi-scale illumination estimation
    illumination = np.zeros_like(image)
    for sigma, weight in zip(sigmas, weights):
        smoothed = gaussian_filter(log_image, sigma=(sigma, sigma, 0))  # Per-channel filtering
        illumination += weight * smoothed

    illumination = np.exp(illumination)  # Convert back from log domain

    # Compute reflectance
    reflectance = image / (illumination + 1e-6)

    # Normalize reflectance for visualization
    reflectance = np.clip(reflectance * 255 / reflectance.max(), 0, 255).astype(np.uint8)
    
    return reflectance, illumination

# Load an image
image = cv2.imread("3.jpg")

# Apply Multi-Scale Retinex
reflectance, illumination = multi_scale_retinex(image, sigmas=[15, 80, 250])

# Save results
cv2.imwrite("reflectance_msr.jpg", reflectance)
cv2.imwrite("illumination_msr.jpg", illumination.astype(np.uint8))

# Display results
cv2.imshow("Original Image", image)
cv2.imshow("Reflectance (MSR)", reflectance)
cv2.imshow("Illumination (MSR)", illumination.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
