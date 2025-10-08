import cv2
from matplotlib import pyplot as plt
import numpy as np

# Intrinsic parameters (camera matrix components)
fx = 900.2606299
fy = 904.38733995
cx = 689.49222
cy = 381.58845892

# Paths to the input images
ground_level_image_path = '/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/atrium.jpg'
height_level_image_path = '/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/Image.png'

# Function to detect the centroid and radius of the sphere
def getCentroidAndRadius(image_path):
    lower_blue = np.array([100, 50, 50]) 
    upper_blue = np.array([140, 255, 255])

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroid_x, centroid_y, radius = None, None, None
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (center_x, center_y), radius = cv2.minEnclosingCircle(largest_contour)
        centroid_x, centroid_y = int(center_x), int(center_y)
        radius = int(radius)
    else:
        raise ValueError("No contours found in the mask.")

    return centroid_x, centroid_y, radius, cleaned_mask, image

# Function to calculate the relative height
def calculateRelativeHeight(r1, r2, Z1):
    """
    :param r1: Apparent radius of the sphere in the ground-level image (in pixels)
    :param r2: Apparent radius of the sphere in the height-level image (in pixels)
    :param Z1: Initial distance to the sphere at ground level (in meters)
    :return: Relative height above the ground (in meters)
    """
    if r2 == 0:
        raise ValueError("Detected radius in the second image is zero.")
    
    # Calculate the relative height
    delta_y = Z1 * (r1 / r2 - 1)
    return delta_y

# Detect the sphere in both images
centroid_x1, centroid_y1, radius1, mask1, image1 = getCentroidAndRadius(ground_level_image_path)
centroid_x2, centroid_y2, radius2, mask2, image2 = getCentroidAndRadius(height_level_image_path)

# Known distance to the sphere at ground level (in meters, estimated or measured)
Z1 = 2.0  # Replace with the actual value

# Calculate the relative height above the ground
try:
    relative_height = calculateRelativeHeight(radius1, radius2, Z1)
    print(f"Estimated relative height above ground: {relative_height:.2f} meters")
except ValueError as e:
    print(e)

# Display the results
plt.figure(figsize=(12, 6))

# Ground-level image
plt.subplot(1, 2, 1)
plt.title("Ground-Level Detection")
plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.axis("off")

# Height-level image
plt.subplot(1, 2, 2)
plt.title("Height-Level Detection")
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()