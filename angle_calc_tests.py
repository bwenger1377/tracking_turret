# Written by Benjamin Wenger on 8-10-26
# Angle Calculations w/ OpenCV
# Last Revision 8-10-26

# Import libraries
import cv2 # OpenCV
import numpy as np
import time # As needed

# Specify camera resolution
HD = (1280, 720)
resolution = HD

# Define constant values for frame coordinates
CENTER_X = resolution[0] / 2
CENTER_Y = resolution[1] / 2

# Open camera for recording
camera = cv2.VideoCapture(0) # 1 corresponds to external webcam
if camera.isOpened():
    print("Camera operational.")
else:
    print("Camera did not open.")

# Main loop: Capture a frame and mark it up
while True:
    success, frame = camera.read()
    if not success:
        print("Frame could not be captured.")
        break

    frame = cv2.resize(frame, resolution) # Resize the frame to a higher resolution
    frame = cv2.flip(frame, flipCode=1) # Flip the flame L-R
    cv2.circle(frame, (int(CENTER_X), int(CENTER_Y)), 5, (0,0,255), thickness=2) # Add a circle to the center of the frame
    cv2.imshow("Channel 1", frame) # Display the current frame on the screen

    # Exit video capture loop if ESC key pressed
    if cv2.waitKey(1) == 27: # 27 is ASCII for ESC key
        break

# Close video capture and end program
camera.release()
cv2.destroyAllWindows()