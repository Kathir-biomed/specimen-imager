#pose estimation. uses realsense cam. 
import cv2 as cv
import numpy as np
import pyrealsense2 as rs

# Use getPredefinedDictionary instead of Dictionary_get
marker_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

# Create an instance of DetectorParameters
param_markers = cv.aruco.DetectorParameters()

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

while True:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    # Convert color frame to OpenCV format
    frame = np.asanyarray(color_frame.get_data())

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = cv.aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )

    if marker_corners:
        for ids, corners in zip(marker_IDs, marker_corners):
            cv.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()

            # Print id and coordinates on top of the marker
            text = f"id: {ids[0]}, coords: {tuple(top_right)}"
            cv.putText(
                frame,
                text,
                tuple(top_right),
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (200, 100, 0),
                2,
                cv.LINE_AA,
            )

    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == ord("q"):
        break

# Stop streaming
pipeline.stop()
cv.destroyAllWindows()
