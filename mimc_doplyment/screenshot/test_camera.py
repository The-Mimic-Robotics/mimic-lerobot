import cv2
import numpy as np

# Camera indices
cam1_index = 0  # /dev/video0
cam2_index = 2  # /dev/video2
cam3_index = 4  # /dev/video4

# Open the three cameras
cap1 = cv2.VideoCapture(cam1_index)
cap2 = cv2.VideoCapture(cam2_index)
cap3 = cv2.VideoCapture(cam3_index)

# Check if they opened successfully
if not cap1.isOpened():
    print(f"Warning: Could not open /dev/video{cam1_index}")
if not cap2.isOpened():
    print(f"Warning: Could not open /dev/video{cam2_index}")
if not cap3.isOpened():
    print(f"Warning: Could not open /dev/video{cam3_index}")

# Optional: Set resolution (helps with some virtual devices)
width, height = 640, 480
for cap in [cap1, cap2, cap3]:
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

print("Displaying three cameras: video0 (left) | video2 (middle) | video4 (right)")
print("Press 'q' to quit.")

while True:
    # Read frames
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()

    # Create placeholder frames if capture fails
    placeholder = np.zeros((height, width, 3), dtype=np.uint8)

    if not ret1 or frame1 is None:
        frame1 = placeholder.copy()
        cv2.putText(frame1, f"video{cam1_index} FAILED", (80, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if not ret2 or frame2 is None:
        frame2 = placeholder.copy()
        cv2.putText(frame2, f"video{cam2_index} FAILED", (80, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if not ret3 or frame3 is None:
        frame3 = placeholder.copy()
        cv2.putText(frame3, f"video{cam3_index} FAILED", (80, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Resize all frames to the same size
    frame1 = cv2.resize(frame1, (width, height))
    frame2 = cv2.resize(frame2, (width, height))
    frame3 = cv2.resize(frame3, (width, height))

    # Add labels on top
    cv2.putText(frame1, "/dev/video0", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame2, "/dev/video2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame3, "/dev/video4", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Combine horizontally: left | middle | right
    combined = np.hstack((frame1, frame2, frame3))

    # Show the combined view
    cv2.imshow("Triple Camera View (video0 | video2 | video4)", combined)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap1.release()
cap2.release()
cap3.release()
cv2.destroyAllWindows()