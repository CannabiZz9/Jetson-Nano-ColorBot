import cv2
import numpy as np


# Function to filter magenta-purple color
def filter_magenta_purple(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the range for magenta-purple color in HSV
    lower_magenta = np.array([160, 50, 50])
    upper_magenta = np.array([180, 255, 255])
    
    # Create a mask for the magenta-purple color
    mask = cv2.inRange(hsv, lower_magenta, upper_magenta)
    
    # Apply the mask to get the magenta-purple parts of the image
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    return result

# Open a connection to the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Filter the magenta-purple color from the frame
    filtered_frame = filter_magenta_purple(frame)
    
    # Display the resulting frame
    cv2.imshow('Webcam Feed', filtered_frame)
    cv2.imshow('Webcam Feed', cap)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
