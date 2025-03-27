import cv2
import numpy as np

def white_patch_retinex(image):
    max_vals = np.max(image, axis=(0, 1))  # Find max intensity for each channel
    normalized = (image / max_vals) * 255  # Normalize each channel
    return np.clip(normalized, 0, 255).astype(np.uint8)

# Load image
image = cv2.imread("3.jpg")

# Apply White Patch Retinex
output_img = white_patch_retinex(image)

cv2.imwrite("white_patch_corrected.jpg", output_img)
