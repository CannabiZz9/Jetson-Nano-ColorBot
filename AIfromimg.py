import cv2
import matplotlib.pyplot as plt
import numpy as np

# Reading the image
image = cv2.imread('test2.jpg')

# Converting the image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range of purple color in HSV
lower_purple = np.array([123, 19, 176])
upper_purple = np.array([153, 255, 255])

# Threshold the HSV image to get only purple colors
mask = cv2.inRange(hsv, lower_purple, upper_purple)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Set a maximum area threshold (tune this value according to your needs)
max_area = 100  # Adjust this value based on your image size and object size

# Create a new mask to include only small objects
filtered_mask = np.zeros_like(mask)

for contour in contours:
    if cv2.contourArea(contour) < max_area:
        cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

# Draw a plus sign in the middle of the original image
center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
line_length = 15  # Length of each line in the plus sign

# Horizontal line
cv2.line(image, (center_x - line_length, center_y), (center_x + line_length, center_y), (0, 255, 0), 3)

# Vertical line
cv2.line(image, (center_x, center_y - line_length), (center_x, center_y + line_length), (0, 255, 0), 3)

# Display the original image with the plus sign and the filtered mask
plt.figure(figsize=(10, 10))

plt.subplot(121)
plt.imshow(image[..., ::-1])
plt.title("Original Image with Plus Sign", fontdict={'fontsize': 25})
plt.axis('off')

plt.subplot(122)
plt.imshow(filtered_mask, cmap='gray')
plt.title("Filtered Mask", fontdict={'fontsize': 25})
plt.axis('off')

plt.show()
