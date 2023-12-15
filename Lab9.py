import numpy as np
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
imgOrig = cv2.imread(r'C:\Lab9\ATU2.jpg')

# Check if the image is loaded successfully
if imgOrig is None:
    print("Error: Could not open or read the image.")
else:
    print("Image loaded successfully.")

# Convert the original image to grayscale
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

# Blur the grayscale image
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)

# Create a subplot with 2 rows and 3 columns
plt.subplot(2, 3, 1), plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])
# Load the images
img1 = cv2.imread(r'C:\Lab9\House.jpg')  # Update with the correct path if needed
img2 = cv2.imread(r'C:\Lab9\ATU2.jpg')

plt.subplot(2, 3, 2), plt.imshow(imgGray, cmap='gray')
plt.title('Grayscale'), plt.xticks([]), plt.yticks([])
# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

plt.subplot(2, 3, 3), plt.imshow(imgBlur, cmap='gray')
plt.title('Blurred'), plt.xticks([]), plt.yticks([])

# Apply Sobel detector to the grayscale image
sobelx = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(imgGray, cv2.CV_64F, 0, 1, ksize=5)
imgSobel = np.sqrt(sobelx**2 + sobely**2)

plt.subplot(2, 3, 4), plt.imshow(imgSobel, cmap='gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])

# Apply Canny detector to the grayscale image
imgCanny = cv2.Canny(imgGray, 100, 200)

plt.subplot(2, 3, 5), plt.imshow(imgCanny, cmap='gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])

# ORB (Oriented FAST and Rotated BRIEF)
# Initiate ORB detector
orb = cv2.ORB_create()

# Find the keypoints with ORB
kp = orb.detect(imgGray, None)

# Compute the descriptors with ORB
kp, des = orb.compute(imgGray, kp)
# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Draw only keypoints location, not size and orientation
img_with_keypoints = cv2.drawKeypoints(imgGray, kp, None, color=(0, 255, 0), flags=0)
# Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_bf = bf.match(des1, des2)

plt.subplot(2, 3, 6), plt.imshow(img_with_keypoints, cmap='gray')
plt.title('ORB Keypoints'), plt.xticks([]), plt.yticks([])
# Sort them in ascending order of distance
matches_bf = sorted(matches_bf, key=lambda x: x.distance)

# Adjust layout for better display
plt.tight_layout()
# Draw matches
img_matches_bf = cv2.drawMatches(img1, kp1, img2, kp2, matches_bf[:10], None, flags=2)

# Show the plot
# Display the result
plt.imshow(img_matches_bf)
plt.title('BruteForceMatcher')
plt.show()