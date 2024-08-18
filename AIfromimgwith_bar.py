import cv2
import numpy as np

# Callback function for the trackbars (it does nothing but is required for createTrackbar)
def nothing(x):
    pass

# Create a named window to attach the sliders
cv2.namedWindow('Trackbars')

# Create trackbars for adjusting HSV values
cv2.createTrackbar('Lower H', 'Trackbars', 120, 179, nothing)
cv2.createTrackbar('Lower S', 'Trackbars', 40, 255, nothing)
cv2.createTrackbar('Lower V', 'Trackbars', 130, 255, nothing)
cv2.createTrackbar('Upper H', 'Trackbars', 255, 179, nothing)
cv2.createTrackbar('Upper S', 'Trackbars', 185, 255, nothing)
cv2.createTrackbar('Upper V', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('Max Area', 'Trackbars', 100, 1000, nothing)

# Read the image
image = cv2.imread('test2.jpg')

while True:
    # Get current positions of all trackbars
    lower_h = cv2.getTrackbarPos('Lower H', 'Trackbars')
    lower_s = cv2.getTrackbarPos('Lower S', 'Trackbars')
    lower_v = cv2.getTrackbarPos('Lower V', 'Trackbars')
    upper_h = cv2.getTrackbarPos('Upper H', 'Trackbars')
    upper_s = cv2.getTrackbarPos('Upper S', 'Trackbars')
    upper_v = cv2.getTrackbarPos('Upper V', 'Trackbars')
    max_area = cv2.getTrackbarPos('Max Area', 'Trackbars')

    # Define range of purple color in HSV using the trackbar values
    lower_purple = np.array([lower_h, lower_s, lower_v])
    upper_purple = np.array([upper_h, upper_s, upper_v])

    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only purple colors
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new mask to include only small objects
    filtered_mask = np.zeros_like(mask)

    for contour in contours:
        if cv2.contourArea(contour) < max_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Convert the mask to a 3-channel image for proper display in grayscale
    filtered_mask_rgb = cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2BGR)

    # Display the filtered mask in grayscale
    cv2.imshow('Filtered Mask (Grayscale)', filtered_mask)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release all OpenCV windows
cv2.destroyAllWindows()
