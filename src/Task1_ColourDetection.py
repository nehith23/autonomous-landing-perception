import numpy as np
import cv2
from matplotlib import pyplot as plt

# Attempting to retrieve binary mask from image
image = cv2.imread('/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/images/000099.png')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define ranges (found through ColourRangeDetector.py)
lower_blue = np.array([100, 50, 50]) 
upper_blue = np.array([140, 255, 255])

lower_clouds = np.array([85, 5, 180]) 
upper_clouds = np.array([135, 80, 255])

lower_bound1 = np.array([60, 0, 0])
upper_bound1 = np.array([179, 173, 134])

lower_bound2 = np.array([69, 20, 0])
upper_bound2 = np.array([179, 255, 255])

# Create masks
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
white_mask = cv2.inRange(hsv, lower_clouds, upper_clouds)
another_mask = cv2.inRange(hsv, lower_bound1, upper_bound1)
one_more_mask = cv2.inRange(hsv, lower_bound2, upper_bound2)

# Combine the masks
combined_mask = cv2.bitwise_or(blue_mask, white_mask)
combined_mask = cv2.bitwise_or(combined_mask, another_mask)
combined_mask = cv2.bitwise_or(combined_mask, one_more_mask)

# Apply morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

# Create final mask
final_mask = cv2.bitwise_and(image, image, mask=cleaned_mask)

#cv2.imshow('Segmented AO', final_mask)
#cv2.imshow('Final Binary Mask', cleaned_mask)

plt.subplot(1, 3, 1)
plt.title(f'Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f'Mask')
plt.imshow(cleaned_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title(f'Final Output')
plt.imshow(cv2.cvtColor(final_mask, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()