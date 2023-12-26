import cv2 as cv
import numpy as np
import pyrealsense2 as rs

# Load calibration data
calibration_data = np.load('calibration_data.npz')
cameraMatrix = calibration_data['cameraMatrix']
distCoeffs = calibration_data['distCoeffs']

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
    marker_corners, marker_IDs, _ = cv.aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )

    if marker_corners:
        # Draw detected markers
        cv.aruco.drawDetectedMarkers(frame, marker_corners, marker_IDs)

        for i, corners in enumerate(marker_corners):
            if len(corners) >= 4:  # Ensure there are enough points for pose estimation
                # Estimate pose
                rvec, tvec, _ = cv.aruco.estimatePoseSingleMarkers(
                    corners, 1.0, cameraMatrix, distCoeffs
                )

                # Draw coordinate axes on the marker
                cv.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 1.0)

                # Print marker ID and pose information
                print(f"Marker {marker_IDs[i][0]} - rvec: {rvec}, tvec: {tvec}")

    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == ord("q"):
        break

# Stop streaming
pipeline.stop()
cv.destroyAllWindows()
