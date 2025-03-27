import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def intrinsic_decomposition(image, sigma=10):
    """
    Decomposes an image into reflectance and illumination using a Retinex-based method.
    
    Parameters:
        image (numpy.ndarray): Input color image.
        sigma (int): Standard deviation for Gaussian filtering.

    Returns:
        reflectance (numpy.ndarray): Reflectance component.
        illumination (numpy.ndarray): Illumination component.
    """
    # Convert to grayscale for illumination estimation
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) + 1e-6  # Avoid log(0)
    
    # Apply logarithm to avoid multiplicative effects
    log_image = np.log(grayscale)
    
    # Estimate illumination using a Gaussian filter
    smooth_log = gaussian_filter(log_image, sigma=sigma)
    illumination = np.exp(smooth_log)

    # Compute reflectance
    reflectance = image / (illumination[..., None] + 1e-6)  # Ensure per-channel division

    # Normalize reflectance for better visualization
    reflectance = np.clip(reflectance * 255 / reflectance.max(), 0, 255).astype(np.uint8)
    
    return reflectance, illumination

# Load an image
image = cv2.imread("3.jpg")

# Perform intrinsic decomposition
reflectance, illumination = intrinsic_decomposition(image, sigma=10)

# Save and display results
cv2.imwrite("reflectance.jpg", reflectance)
cv2.imwrite("illumination.jpg", illumination)

# Create a window
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original Image", 640, 480)
cv2.imshow("Original Image", image)

cv2.namedWindow("Reflectance", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Reflectance", 640, 480)
cv2.imshow("Reflectance", reflectance)

cv2.namedWindow("illumination", cv2.WINDOW_NORMAL)
cv2.resizeWindow("illumination", 640, 480)
cv2.imshow("illumination", illumination.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()
