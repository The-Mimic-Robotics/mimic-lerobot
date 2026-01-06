import cv2

def main():
    # Use index 4 as before
    cap = cv2.VideoCapture(4)

    # --- ATTEMPT TO SET HD720 RESOLUTION ---
    # We ask for 2560x720. 
    # If you are on USB 2.0, this might fail and revert to VGA.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # --- SANITY CHECK ---
    # Let's see what the camera actually gave us
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"DEBUG: Camera resolution is currently: {int(actual_width)} x {int(actual_height)}")

    if actual_width == 1344 or actual_width == 672:
        print("WARNING: You are getting low-res VGA! Check your USB connection.")

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Split the frame
        height, width, _ = frame.shape
        left_view = frame[:, :width//2]
        right_view = frame[:, width//2:]

        cv2.imshow('ZED Left Eye (High Res?)', left_view)
        cv2.imshow('ZED Right Eye (High Res?)', right_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()