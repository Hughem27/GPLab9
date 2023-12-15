import cv2
import numpy as np
from matplotlib import pyplot as plt

dst = cv2.cornerHarris(imgGray, blockSize, aperture_size, k)

# Create a deep copy of the original image
# Create a deep copy of the original image for drawing Harris circles
imgHarris = copy.deepcopy(imgOrig)

# Set the threshold for corner detection
threshold = 0.01
# Set the threshold for Harris corner detection
threshold_harris = 0.01

# Loop through every element in the dst matrix
for i in range(len(dst)):
    for j in range(len(dst[i])):
        # Check if the element is greater than the threshold
        if dst[i][j] > (threshold * dst.max()):
            # Draw a circle on the imgHarris image
            cv2.circle(imgHarris, (j, i), 3, (0, 0, 255), -1)  # Red color for circles
        if dst[i][j] > (threshold_harris * dst.max()):
            # Draw a circle on the imgHarris image for Harris corners
            cv2.circle(imgHarris, (j, i), 3, (0, 0, 255), -1)  # Red color for Harris circles

# Plot the image with Harris corners
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))
plt.title('Harris Corners')
plt.show()

# Plot the result
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))
plt.title('Harris Corner Detection with Circles')


# Perform corner detection using Shi-Tomasi algorithm
maxCorners = 100  # Experiment with different values
qualityLevel = 0.01
minDistance = 10
corners = cv2.goodFeaturesToTrack(imgGray, maxCorners, qualityLevel, minDistance)

# Create a deep copy
imgGFTT = copy.deepcopy(imgOrig)

# Convert to ints
corners = np.int0(corners)

# Draw circles Shi-Tomasi
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(imgGFTT, (x, y), 3, (0, 255, 0), -1)  # Green color for GFTT circles

# Plot the image with Shi-Tomasi corners
plt.imshow(cv2.cvtColor(imgGFTT, cv2.COLOR_BGR2RGB))
plt.title('Shi-Tomasi Corners')
plt.show()