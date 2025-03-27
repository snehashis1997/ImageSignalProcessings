
import cv2
import numpy as np


hdr_image = cv2.imread("hdr_mann.hdr", cv2.IMREAD_ANYDEPTH)
tonemap_reinhard = cv2.createTonemapReinhard(1.0, 0.0, 0.0, 0.0)  # Default parameters
ldr_global = tonemap_reinhard.process(hdr_image)

ldr_global = np.clip(ldr_global * 255, 0, 255).astype(np.uint8)
cv2.imwrite("ldr_global.jpg", ldr_global)
