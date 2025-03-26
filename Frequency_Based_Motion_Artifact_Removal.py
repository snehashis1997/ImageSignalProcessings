import cv2
import numpy as np

# Load an image with periodic motion artifacts
image = cv2.imread("motion_artifact_image.jpg", cv2.IMREAD_GRAYSCALE)

# Apply 2D Fourier Transform
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# Create a notch filter to remove high-frequency periodic noise
rows, cols = image.shape
mask = np.ones((rows, cols), np.uint8)
r, c = rows//2, cols//2
mask[r-10:r+10, c-10:c+10] = 0  # Remove central high-frequency noise

# Apply the filter
f_filtered = fshift * mask
f_ishift = np.fft.ifftshift(f_filtered)
image_filtered = np.fft.ifft2(f_ishift)
image_filtered = np.abs(image_filtered)

# Display results
cv2.imshow("Original Image", image)
cv2.imshow("Filtered Image", np.uint8(image_filtered))
cv2.waitKey(0)
cv2.destroyAllWindows()
