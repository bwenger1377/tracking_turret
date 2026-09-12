# Written by Benjamin Wenger on 3-13-26
# Last Revision 8-10-26
# Gesture Detection Program

# Import Necessary Libraries
import cv2 # OpenCV
import numpy as np
from telemetrix import telemetrix # Arduino connection
import time

# Specify camera resolution
SD = (640, 480)
HD = (1280, 720)
resolution = HD

# Define constant values for frame coordinates
CENTER_X = resolution[0] / 2
CENTER_Y = resolution[1] / 2

# Define constants for OpenCV colors (BGR)
BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)

# Initialize Arduino
board = telemetrix.Telemetrix(com_port='COM3')

# Set pin modes
board.set_pin_mode_digital_output(13)
board.set_pin_mode_digital_output(2)
board.set_pin_mode_digital_output(3)
board.set_pin_mode_digital_output(4)
board.set_pin_mode_digital_output(5)
board.set_pin_mode_digital_output(6)

# Set up camera for video capture
cam = cv2.VideoCapture(1)
if cam.isOpened():
    print("Camera operational.")
else:
    print("Camera did not open.")

# Set initial time and number of frames captured
t_ref = time.perf_counter()
t_lastprint = time.perf_counter()
num_frames = 0

# Centroid smoothing variables
cx_prev = SD[0] / 2 # Initialize the values of cx and cy to middle of the screen for smoothing
cy_prev = SD[1] / 2
alpha_c = 0.25 # alpha for EMA

# Define additional needed variables
motion_history = None
vx = None
vy = None
vx_prev = 0.
vy_prev = 0.
ax = 0.
ay = 0.

# Capture initial frame
successful, frame = cam.read() # capture the frame
if not successful or frame is None:
    print("Unsuccessful capture")
gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert frame to grayscale

while True:
    # Measure the time through one loop (for velocity measurement)
    t_elapsed = time.perf_counter() - t_ref
    t_ref = time.perf_counter()

    # Read a frame from the camera
    successful, frame = cam.read()
    if not successful or frame is None:
        print("Unsuccessful capture")
        break

    # Convert to grayscale and add frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Get HSV frame for color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (23, 23), 0)

    # Threshold the HSV frame to include only color(s) of interest
    lower = np.array([95, 100, 50], dtype=np.uint8) # was 0, 70, 70 for skin
    upper = np.array([130, 255, 255], dtype=np.uint8) # was 25, 150, 255 for skin
    skin_mask = cv2.inRange(hsv, lower, upper)

    # Clean the skin mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_close = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)     
    skin_clean = cv2.morphologyEx(skin_close, cv2.MORPH_OPEN, kernel)
    skin_clean_frame = cv2.cvtColor(skin_clean, cv2.COLOR_GRAY2BGR)

    # Find contours
    contours, hierarchy = cv2.findContours(skin_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea) # find the largest contour
        if cv2.contourArea(largest_contour) <= 500:
            continue
    else:
        continue

    # Find the centroid of the largest contour
    moments = cv2.moments(largest_contour) # get contour moments
    if moments['m00'] != 0:
        # Locate centroid
        cx = moments['m10'] / moments['m00'] # x-coord of contour centroid
        cy = moments['m01'] / moments['m00'] # y-coord of contour centroid

        # Apply EMA to eliminate noise
        cx_smooth = alpha_c * cx + (1-alpha_c) * cx_prev
        cy_smooth = alpha_c * cy + (1-alpha_c) * cy_prev

        # Find centroid velocity
        vx = (cx_smooth - cx_prev)/t_elapsed
        vy = (cy_smooth - cy_prev)/t_elapsed

        # Smooth velocity using same EMA as for position
        vx = alpha_c * vx + (1 - alpha_c) * vx_prev
        vy = alpha_c * vy + (1 - alpha_c) * vy_prev
        
        # Calculate speed (magnitude of velocities)
        speed = np.sqrt(vx**2 + vy**2)

        # Determine centroid acceleration
        ax = vx - vx_prev
        ay = vy - vy_prev

        # Get cx_prev, cy_prev, vx_prev, and vy_prev
        cx_prev = cx_smooth
        cy_prev = cy_smooth
        vx_prev = vx
        vy_prev = vy

    # Draw a circle to indicate centroid location
    cv2.circle(frame, (int(cx_smooth), int(cy_smooth)), 10, PURPLE, 2)

    # Find coordinates of a bounding rectangle around the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Define rectangle corners
    bot_L = (x, y)
    top_R = (x + w, y + h)

    # Plot rectangle on the frame
    cv2.rectangle(frame, bot_L, top_R, BLACK, 2)
    cv2.putText(frame, f"Largest Contour", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, BLACK, 1)
    cv2.putText(frame, f"Speed: {speed}", (x+w, y+h), cv2.FONT_HERSHEY_TRIPLEX, 1.0, BLACK, 1)

    # Combine the frames and resize the result to fit on the screen
    combined = np.hstack((np.vstack((frame, skin_clean_frame)), np.vstack((gray_frame, skin_clean_frame)), np.vstack((gray_frame, skin_clean_frame))))
    combined_frame = cv2.resize(combined, resolution)

    # Show the combined frame on the screen
    cv2.imshow('Channel 1', combined_frame)

    # Increment the frame value
    num_frames += 1

    # ===================================================== DECISION LOGIC ==============================================================
    # Based on the tracking information, determine the specific gestures the hand is making.
    
    # 1. Swipes in one of the four cardinal directions (one x case and one y case can be true at once)
    # X Direction (L/R)
    if vx > 0 and ax > 50:
        swipe_left = True
        board.digital_write(2, 1)
    elif vx < 0 and ax < -50:
        swipe_right = True
        board.digital_write(3, 1)
    else:
        board.digital_write(2, 0)
        board.digital_write(3, 0)
    # Y Direction (U/D)
    if vy > 0 and ay > 50:
        swipe_down = True
        board.digital_write(4, 1)
    elif vy < 0 and ay < -50:
        swipe_up = True
        board.digital_write(5, 1)
    else: # if neither swiping up or swiping down,
        board.digital_write(4, 0)
        board.digital_write(5, 0)

    # 2. Hand positions (closed fist vs. open palm)
    rectArea = w * h
    if rectArea >= 37500:
        fist = False
        board.digital_write(6, 0)
    else:
        fist = True
        board.digital_write(6, 1)

    # Determine elapsed time and compare to reference time; reset  and print fps if more than 1 sec
    t_elapsedprint = time.perf_counter() - t_lastprint
    if t_elapsedprint >= 1.0:
        fps = num_frames / t_elapsedprint
        print(f"{fps:.1f} FPS")
        num_frames = 0
        t_lastprint = time.perf_counter()

    # If the 'esc' key is pressed, exit the while loop
    if cv2.waitKey(1) == 27: # 27 is ASCII for 'esc' key
        for ii in range(2, 7):
            board.digital_write(ii, 0)
        break

cam.release()
cv2.destroyAllWindows()