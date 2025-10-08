import cv2
import numpy as np


def getDiametre(mask):

    # Ensure the mask is binary
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Step 2: Find contours of the sphere
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming there is only one sphere, take the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Step 3: Fit a bounding box and compute the diameter
    x, y, width, height = cv2.boundingRect(largest_contour)
    diameter = max(width, height)  # Diameter is the larger dimension

    # Step 4 (optional): Fit a minimum enclosing circle
    # This is useful if the sphere is irregularly segmented
    (x_center, y_center), radius = cv2.minEnclosingCircle(largest_contour)
    circle_diameter = 2 * radius

    # Print the results
    print(f"Bounding box diameter: {diameter} pixels")
    print(f"Circle-based diameter: {circle_diameter:.2f} pixels")

    # (Optional) Visualize the results
    import matplotlib.pyplot as plt

    # Draw the bounding box and enclosing circle on the mask
    output_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(output_image, (x, y), (x + width, y + height), (0, 255, 0), 2)  # Bounding box
    cv2.circle(output_image, (int(x_center), int(y_center)), int(radius), (255, 0, 0), 2)  # Circle

    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title("Bounding Box and Enclosing Circle")
    plt.axis('off')
    plt.show()

    return diameter