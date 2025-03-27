import cv2
import numpy as np

def gradient_domain_tone_mapping(hdr, alpha=0.1):
    """Performs gradient domain compression."""
    laplacian = cv2.Laplacian(hdr, cv2.CV_32F)
    compressed_hdr = hdr - alpha * laplacian  # Reduce large gradients
    return np.clip(compressed_hdr, 0, 1)

hdr_image = cv2.imread("hdr_mann.hdr", cv2.IMREAD_ANYDEPTH)
hdr_image = cv2.normalize(hdr_image, None, 0, 1, cv2.NORM_MINMAX)

ldr_gradient = gradient_domain_tone_mapping(hdr_image)
ldr_gradient = (ldr_gradient * 255).astype(np.uint8)
cv2.imwrite("ldr_gradient.jpg", ldr_gradient)