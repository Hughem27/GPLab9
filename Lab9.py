import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the images
img1 = cv2.imread(r'C:\Lab9\OperaHouse.jpg')  # Update with the correct path if needed
img2 = cv2.imread(r'C:\Lab9\ATU2.jpg')
# Load the image
img = cv2.imread(r'C:\Lab9\OperaHouse.jpg')  # Update with the correct path to your image

# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# Split the image into channels
b, g, r = cv2.split(img)

# Initiate ORB detector
orb = cv2.ORB_create()
# Create a Matplotlib subplot
plt.figure(figsize=(10, 4))

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)
# Display the original image
plt.subplot(1, 4, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.axis('off')

# Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_bf = bf.match(des1, des2)
# Display the Red channel
plt.subplot(1, 4, 2), plt.imshow(r, cmap='gray'), plt.title('Red Channel')
plt.axis('off')

# Sort them in ascending order of distance
matches_bf = sorted(matches_bf, key=lambda x: x.distance)
# Display the Green channel
plt.subplot(1, 4, 3), plt.imshow(g, cmap='gray'), plt.title('Green Channel')
plt.axis('off')

# Draw matches
img_matches_bf = cv2.drawMatches(img1, kp1, img2, kp2, matches_bf[:10], None, flags=2)
# Display the Blue channel
plt.subplot(1, 4, 4), plt.imshow(b, cmap='gray'), plt.title('Blue Channel')
plt.axis('off')

# Display the result for the first pair of images
plt.figure(figsize=(8, 4))
plt.subplot(121), plt.imshow(img_matches_bf), plt.title('BruteForceMatcher')

# Load new images
img1_new = cv2.imread(r'C:\Lab9\castle.jpg')
img2_new = cv2.imread(r'C:\Lab9\Sky.jpg')

# Convert the new images to grayscale
gray1_new = cv2.cvtColor(img1_new, cv2.COLOR_BGR2GRAY)
gray2_new = cv2.cvtColor(img2_new, cv2.COLOR_BGR2GRAY)

# Find the keypoints and descriptors with ORB for the new images
kp1_new, des1_new = orb.detectAndCompute(gray1_new, None)
kp2_new, des2_new = orb.detectAndCompute(gray2_new, None)

# Brute-Force Matcher for the new images
matches_bf_new = bf.match(des1_new, des2_new)

# Sort them in ascending order of distance
matches_bf_new = sorted(matches_bf_new, key=lambda x: x.distance)

# Draw matches for the new images
img_matches_bf_new = cv2.drawMatches(img1_new, kp1_new, img2_new, kp2_new, matches_bf_new[:10], None, flags=2)

# Display the result for the second pair of images
plt.subplot(122), plt.imshow(img_matches_bf_new), plt.title('BruteForceMatcher')

# Adjust layout for better display
plt.tight_layout()

# Show the plots
# Show the plot
plt.show()