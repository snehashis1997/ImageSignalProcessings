import cv2
import numpy as np

image = cv2.imread("target4.jpeg")
b,g,r = cv2.split(image)

mu_b = np.mean(b)
mu_g = np.mean(g)
mu_r = np.mean(r)
mu_gray = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

b_ = b * mu_gray / mu_b
g_ = g * mu_gray / mu_g
r_ = r * mu_gray / mu_r

new_image_balanced = cv2.merge((b_, g_, r_))

cv2.imwrite("new_image_balanced.jpg", new_image_balanced)