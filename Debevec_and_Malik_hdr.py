import cv2
import numpy as np

# List of exposure times (corresponding to input images)
exposure_times = np.array([1/30, 1/8, 1/2, 1], dtype=np.float32)

# Load images with different exposures
image_filenames = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
images = [cv2.imread(img) for img in image_filenames]

# Convert images to floating point format
images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for img in images]

# Create a Debevec & Malik method calibrator
calibrate = cv2.createCalibrateDebevec()
response = calibrate.process(images, exposure_times)

# Merge images into an HDR radiance map
merge_debevec = cv2.createMergeDebevec()
hdr_image = merge_debevec.process(images, exposure_times, response)

# Save the HDR image
cv2.imwrite("hdr_image.hdr", hdr_image)

# Convert HDR to displayable format using tone mapping
tonemap = cv2.createTonemapDurand(gamma=2.2)
ldr_image = tonemap.process(hdr_image)

# Scale LDR image for proper display
ldr_image = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)

# Save the LDR image
cv2.imwrite("ldr_image.jpg", cv2.cvtColor(ldr_image, cv2.COLOR_RGB2BGR))

# Show results
cv2.imshow("HDR Image", ldr_image)
cv2.waitKey(0)
cv2.destroyAllWindows()