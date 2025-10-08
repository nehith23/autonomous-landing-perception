import cv2
import time
import numpy as np

# Path to video file
video_path = "/Users/julianmarchington/Desktop/Comp0241-Coursework/Comp0241-Coursework/Task 3/video.mov"

cap = cv2.VideoCapture(video_path)

# Read the first frame and select a region as the template
ret, first_frame = cap.read()

cv2.imshow("Select Template", first_frame)
bbox = cv2.selectROI("Select Template", first_frame, fromCenter=False, showCrosshair=True)
template = first_frame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
cv2.destroyWindow("Select Template")

# Initialize variables
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
timestamps = []
last_match_time = None
match_threshold = 0.8  # Adjust as needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Match template
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= match_threshold:  # If match is strong enough
        current_time = time.time()
        if last_match_time is None or current_time - last_match_time > 1.0:  # Prevent double-counting
            timestamps.append(current_time)
            last_match_time = current_time
            # Highlight the match on the frame
            top_left = max_loc
            bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            print(f"Match found at time: {current_time:.2f}s")
            print(f"Frame rate: {frame_rate} fps")
            print(f"Frame number: {cap.get(cv2.CAP_PROP_POS_FRAMES)}")

    # Display frame
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calculate rotation periods
rotation_periods = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
print(f"Rotation Periods (seconds): {rotation_periods}")
