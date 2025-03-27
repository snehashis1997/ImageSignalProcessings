import cv2
import numpy as np

def gaussian_pyramid(image, levels):
    """ Generate Gaussian Pyramid """
    gp = [image]
    for _ in range(levels):
        image = cv2.pyrDown(image)
        gp.append(image)
    return gp

def laplacian_pyramid(image, levels):
    """ Generate Laplacian Pyramid """
    gp = gaussian_pyramid(image, levels)
    lp = [gp[-1]]  # Last level is the same
    for i in range(levels, 0, -1):
        upsampled = cv2.pyrUp(gp[i], dstsize=(gp[i-1].shape[1], gp[i-1].shape[0]))
        laplacian = cv2.subtract(gp[i-1], upsampled)
        lp.append(laplacian)
    return lp

def blend_pyramids(lp1, lp2, gp_mask):
    """ Blend two Laplacian pyramids using a Gaussian mask pyramid """
    blended_pyramid = []
    for l1, l2, gm in zip(lp1, lp2, gp_mask):
        blended = l1 * (gm / 255.0) + l2 * (1 - (gm / 255.0))
        blended_pyramid.append(blended.astype(np.uint8))
    return blended_pyramid

def reconstruct_from_pyramid(lp):
    """ Reconstruct image from Laplacian Pyramid """
    image = lp[0]
    for i in range(1, len(lp)):
        image = cv2.pyrUp(image, dstsize=(lp[i].shape[1], lp[i].shape[0]))
        image = cv2.add(image, lp[i])
    return image

# Load images (ensure same size)
image1 = cv2.imread("1.JPG")
image2 = cv2.imread("2.png")
mask = cv2.imread("mask.jpg", 0)  # Grayscale mask

# Resize images if necessary to match dimensions
image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
mask = cv2.resize(mask, (image1.shape[1], image1.shape[0]))

# Define pyramid levels
levels = 5

# Generate pyramids
gp_mask = gaussian_pyramid(mask, levels)
lp1 = laplacian_pyramid(image1, levels)
lp2 = laplacian_pyramid(image2, levels)

# Blend the pyramids
blended_pyramid = blend_pyramids(lp1, lp2, gp_mask)

# Reconstruct final blended image
blended_image = reconstruct_from_pyramid(blended_pyramid)

# Save and display result
cv2.imwrite("blended_result.jpg", blended_image)
cv2.imshow("Blended Image", blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()