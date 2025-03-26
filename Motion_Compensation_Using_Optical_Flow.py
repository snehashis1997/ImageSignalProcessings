import cv2
import numpy as np

cap = cv2.VideoCapture("vibrate_randomly.mp4")
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Compute optical flow (motion estimation)
    flow = optical_flow.calc(prev_gray, gray, None)
    
    # Warp frame using estimated motion
    h, w = gray.shape
    flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h)))
    flow_map = (flow_map + flow.reshape(-1, 2)).astype(np.float32)
    stabilized = cv2.remap(frame, flow_map[:, 0], flow_map[:, 1], interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Motion Compensated Video", stabilized)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break
    
    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
