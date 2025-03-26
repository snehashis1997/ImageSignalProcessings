import cv2
import numpy as np
import matplotlib.pyplot as plt

def notch_filter(shape, notch_points, notch_size=10):
    """
    Create a notch filter mask to remove specific frequency components.
    """
    mask = np.ones(shape, np.uint8)
    rows, cols = shape
    center = (rows // 2, cols // 2)

    for point in notch_points:
        x, y = center[0] + point[0], center[1] + point[1]
        cv2.circle(mask, (x, y), notch_size, 0, -1)
        cv2.circle(mask, (center[0] - point[0], center[1] - point[1]), notch_size, 0, -1)

    return mask

# Load a noisy image
image = cv2.imread("noisy_image.jpg", cv2.IMREAD_GRAYSCALE)

# Apply FFT
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# Create notch filter mask
notch_points = [(50, 0), (-50, 0)]  # Example periodic noise frequencies
mask = notch_filter(image.shape, notch_points)

# Apply filter
fshift_filtered = fshift * mask

# Inverse FFT
f_ishift = np.fft.ifftshift(fshift_filtered)
image_filtered = np.fft.ifft2(f_ishift)
image_filtered = np.abs(image_filtered)

# Show results
plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.imshow(image, cmap='gray'), plt.title("Original Image with Periodic Noise")
plt.subplot(1,2,2), plt.imshow(image_filtered, cmap='gray'), plt.title("Filtered Image (Notch Filter)")
plt.show()