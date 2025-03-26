import cv2
import numpy as np

cap = cv2.VideoCapture("vibrate_randomly.mp4")
stabilizer = cv2.createOptFlow_DualTVL1()  # Optical flow-based stabilization

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale for optical flow
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute motion vectors & stabilize frame
    stabilized_frame = stabilizer.calc(gray, gray, None)
    
    # Display stabilized video
    cv2.imshow("Stabilized Video", stabilized_frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()