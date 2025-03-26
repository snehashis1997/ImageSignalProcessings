
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load stereo image pair (grayscale)
left_image = cv2.imread("left.JPG", cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread("right.JPG", cv2.IMREAD_GRAYSCALE)

# Create a Stereo Block Matching (SBM) object
stereo = cv2.StereoBM_create(numDisparities=16*3, blockSize=15)

# Compute disparity map
disparity_map = stereo.compute(left_image, right_image)

# Normalize for visualization
disparity_map = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_map = np.uint8(disparity_map)

# Display the disparity image
plt.imshow(disparity_map, cmap='gray')
plt.title("Disparity Image")
plt.colorbar()
plt.show()
