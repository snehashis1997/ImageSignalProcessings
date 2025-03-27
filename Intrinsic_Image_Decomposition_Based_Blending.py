import cv2
import numpy as np
from skimage import color, restoration
from scipy.ndimage import gaussian_filter

def intrinsic_decomposition(image):
    """
    Decomposes an image into reflectance and illumination using Retinex-based method.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) + 1e-6  # Avoid log(0)
    log_image = np.log(grayscale)
    smooth_log = gaussian_filter(log_image, sigma=5)
    illumination = np.exp(smooth_log)
    reflectance = image / (illumination[..., None] + 1e-6)
    return reflectance, illumination

def poisson_blending(source, target, mask, offset=(0, 0)):
    """
    Blends the source image into the target using Poisson image editing.
    """
    center = (offset[0] + source.shape[1] // 2, offset[1] + source.shape[0] // 2)
    blended = cv2.seamlessClone(source, target, mask, center, cv2.NORMAL_CLONE)
    return blended

# Load images
# background = cv2.imread("background.jpg")
# object_img = cv2.imread("object.jpg")
# mask = cv2.imread("mask.jpg", 0)  # Binary mask of object

# Create a plain background (e.g., gradient)
background = np.zeros((400, 600, 3), dtype=np.uint8)
for i in range(600):
    background[:, i] = (i // 3, i // 2, i // 4)  # Creating a gradient effect

# Create an object (e.g., a red circle on a white background)
object_img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White square
cv2.circle(object_img, (50, 50), 40, (0, 0, 255), -1)  # Red circle

# Create a binary mask for the object (white where object is present)
mask = np.zeros((100, 100), dtype=np.uint8)
cv2.circle(mask, (50, 50), 40, 255, -1)  # White circle mask

# Intrinsic decomposition
obj_reflectance, obj_illumination = intrinsic_decomposition(object_img)
bg_reflectance, bg_illumination = intrinsic_decomposition(background)

# Adjust illumination of the object to match the background
adjusted_illumination = bg_illumination * (obj_illumination.mean() / bg_illumination.mean())
harmonized_object = (obj_reflectance * adjusted_illumination[..., None]).clip(0, 255).astype(np.uint8)

# Blend the adjusted object with the background
result = poisson_blending(harmonized_object, background, mask)

# Save or display the result
cv2.imwrite("harmonized_output.jpg", result)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
