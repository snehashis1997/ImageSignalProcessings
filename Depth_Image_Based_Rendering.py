
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load RGB image and depth map
image = cv2.imread("img00002.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

depth_map = cv2.imread("img00002.png", cv2.IMREAD_GRAYSCALE)  # Load depth as grayscale

# Normalize depth map to range [0, 1]
depth_map = depth_map.astype(np.float32) / 255.0

# Display the input image and depth map
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(image)
# ax[0].set_title("Original Image")
# ax[1].imshow(depth_map, cmap="gray")
# ax[1].set_title("Depth Map")
# plt.show()


def warp_image(image, depth_map, shift_amount=20):
    """
    Performs depth-based pixel warping to create a new viewpoint.
    
    Parameters:
    - image: RGB image
    - depth_map: Normalized depth map (0 to 1)
    - shift_amount: Maximum pixel shift for warping
    
    Returns:
    - Warped image simulating a new viewpoint
    """
    height, width, _ = image.shape
    warped_image = np.zeros_like(image)  # Initialize empty image

    for y in range(height):
        for x in range(width):
            # Compute disparity shift (farther objects shift less)
            shift = int(shift_amount * (1 - depth_map[y, x]))  # Inversely proportional to depth

            new_x = x + shift  # Shift pixel to the right
            if 0 <= new_x < width:
                warped_image[y, new_x] = image[y, x]  # Assign new pixel position

    return warped_image


# Apply warping to simulate a new viewpoint
warped_image = warp_image(image, depth_map, shift_amount=200)

# Display the warped image
plt.figure(figsize=(6, 6))
plt.imshow(warped_image)
plt.title("Warped Viewpoint Image")
plt.show()


def dibr(image, depth_map, shift_amount=20):
    """
    Depth-Image-Based Rendering (DIBR) using backward warping.
    
    Parameters:
    - image: RGB input image
    - depth_map: Normalized depth map
    - shift_amount: Maximum pixel shift based on depth
    
    Returns:
    - New viewpoint image with DIBR
    """
    height, width, _ = image.shape
    dibr_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            shift = int(shift_amount * (1 - depth_map[y, x]))  # Compute shift
            new_x = x - shift  # Move pixels left (inverse mapping)
            
            if 0 <= new_x < width:
                dibr_image[y, x] = image[y, new_x]  # Assign pixel from the new position

    return dibr_image


# Apply DIBR to generate a novel view
dibr_result = dibr(image, depth_map, shift_amount=200)

# Display the DIBR result
plt.figure(figsize=(6, 6))
plt.imshow(dibr_result)
plt.title("DIBR Synthesized View")
plt.show()
