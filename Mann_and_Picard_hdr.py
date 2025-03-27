import cv2
import numpy as np

def compute_hdr_mann(images, exposure_times):
    """Mann and Picard's logarithmic HDR reconstruction."""
    images = [img.astype(np.float32) / 255.0 for img in images]
    log_exposures = np.log(exposure_times[:, None, None, None])

    # Compute weighted sum
    hdr = np.zeros_like(images[0])
    weight_sum = np.zeros_like(images[0])

    for img, log_exp in zip(images, log_exposures):
        weight = np.clip(img, 0.01, 0.99)  # Avoid division by zero
        hdr += weight * (img - log_exp)
        weight_sum += weight

    hdr /= weight_sum
    return np.exp(hdr)

# Example usage
exposure_times = np.array([1/30, 1/8, 1/2, 1], dtype=np.float32)
image_filenames = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
images = [cv2.imread(img).astype(np.float32) for img in image_filenames]

hdr_image = compute_hdr_mann(images, exposure_times)
cv2.imwrite("hdr_mann.hdr", hdr_image * 255)