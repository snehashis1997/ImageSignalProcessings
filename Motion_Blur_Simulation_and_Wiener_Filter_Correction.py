
import cv2
import numpy as np

# Load the blurred image
image = cv2.imread("blurred_image.jpg", cv2.IMREAD_GRAYSCALE)

# Define a motion blur kernel (simulating hand tremor)
kernel_size = 15
kernel = np.zeros((kernel_size, kernel_size))
kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
kernel /= kernel_size

# Apply Wiener deconvolution to remove motion blur
def wiener_filter(img, kernel, K=0.02):
    img_fft = np.fft.fft2(img)
    kernel_fft = np.fft.fft2(kernel, s=img.shape)
    kernel_fft = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)
    result = np.fft.ifft2(img_fft * kernel_fft)
    return np.abs(result)

corrected_image = wiener_filter(image, kernel)

# Show the results
cv2.imshow("Original", image)
cv2.imshow("Corrected", corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
