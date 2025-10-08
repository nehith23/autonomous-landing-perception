import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your input image
image_path = '/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/Image.png'
img = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a blur to reduce noise and improve circle detection
blurred_mask = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circles using Hough transform
circles = cv2.HoughCircles(
    blurred_mask, 
    cv2.HOUGH_GRADIENT, 
    dp=1.1, 
    minDist=20, 
    param1=50, 
    param2=30, 
    minRadius=0, 
    maxRadius=0
)

# Initialize a mask the same size as the image
circle_mask = np.zeros_like(blurred_mask)

if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :1]:  # Only takes the first detected circle
            center_x, center_y, radius = i
            cv2.circle(circle_mask, (center_x, center_y), radius, (255, 255, 255), thickness =-1)

# Now mask the image to isolate the globe
final_mask = cv2.bitwise_and(img, img, mask=circle_mask)

plt.subplot(1, 3, 1)
plt.title(f'Original Image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f'Mask')
plt.imshow(circle_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title(f'Final Output')
plt.imshow(cv2.cvtColor(final_mask, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()