import cv2
import numpy as np

def match_color_moments(input_img, reference_img):
    input_img = input_img.astype(np.float32)
    reference_img = reference_img.astype(np.float32)
    
    for i in range(3):  # For R, G, B channels
        mean_input, std_input = cv2.meanStdDev(input_img[:, :, i])
        mean_ref, std_ref = cv2.meanStdDev(reference_img[:, :, i])
        
        input_img[:, :, i] = (input_img[:, :, i] - mean_input) * (std_ref / std_input) + mean_ref

    return np.clip(input_img, 0, 255).astype(np.uint8)

# Load images
input_img = cv2.imread("target2.jpg")
reference_img = cv2.imread("sample.jpg")

# Apply Color Moments Matching
output_img = match_color_moments(input_img, reference_img)

cv2.imwrite("color_matched.jpg", output_img)