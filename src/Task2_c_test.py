import cv2
import numpy as np
import matplotlib.pyplot as plt

def getMask(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise on the cleaned mask directly
    blurred_mask = cv2.GaussianBlur(gray, (9, 9), 2)

    # Use Hough Circle Transform to detect circles in the blurred mask
    circles = cv2.HoughCircles(blurred_mask, 
                               cv2.HOUGH_GRADIENT, 
                               dp=1.1, 
                               minDist=20, 
                               param1=50, 
                               param2=30, 
                               minRadius=0, 
                               maxRadius=0)

    # Create a mask for the detected circles
    circle_mask = np.zeros_like(blurred_mask)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :1]:  # Only takes the first detected circle
            center_x, center_y, radius = i
            cv2.circle(circle_mask, (center_x, center_y), radius, (255, 255, 255), thickness =-1)

    return circle_mask
def getDiametre(mask):
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in the mask.")
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(largest_contour)
    return max(width, height)

# Camera parameters
fx, fy = 911.52139558, 914.77008904
cx, cy = 707.55308076, 360.72439761

delta_x = 1.0  # Increase distance moved for better measurement

mask1 = getMask('/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/far.jpg')
mask2 = getMask('/Users/julianmarchington/Desktop/Comp0241-Coursework/Dataset/close.jpg')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('First Image')
plt.imshow(mask1, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Second Image')
plt.imshow(mask2, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

d1 = getDiametre(mask1)
d2 = getDiametre(mask2)

if d2 <= d1:
    raise ValueError("Second diameter must be larger than the first.")

X1 = (delta_x * d1) / (d2 - d1)

angular_size1 = d1 / fx
D = 2 * X1 * np.tan(angular_size1 / 2)

print(f"Diamter of the sphere in the first image: {d1:.2f} pixels")
print(f"Diamter of the sphere in the second image: {d2:.2f} pixels")
print(f"Estimated distance to the sphere from the first image: {X1:.2f} meters")
print(f"Estimated real diameter of the sphere: {D:.2f} meters")