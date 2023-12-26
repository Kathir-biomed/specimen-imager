import cv2
import numpy as np
import pyrealsense2 as rs
import os

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Create a directory to save images
output_directory = 'NewCalibration\captured_images'
os.makedirs(output_directory, exist_ok=True)

# Image counter
image_counter = 0

try:
    while True:
        # Wait for a coherent color frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert color frame to OpenCV format
        frame = np.asanyarray(color_frame.get_data())

        # Display the frame
        cv2.imshow("Intel RealSense Camera", frame)

        # Capture an image when the 'c' key is pressed
        key = cv2.waitKey(1)
        if key == ord('c'):
            # Save the frame as an image
            image_path = os.path.join(output_directory, f'image_{image_counter}.png')
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")

            image_counter += 1
            cv2.putText(frame, 'Image Captured!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Intel RealSense Camera", frame)
            cv2.waitKey(1000)  # Display "Image Captured!" for 1 second

        # Break the loop if 'q' key is pressed
        elif key == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
