import cv2
import os

# Suppress libpng warnings
os.environ['OPENCV_IO_ENABLE_WARNINGS'] = 'FALSE'
# Load images
image1 = cv2.imread('images/1.png')
image2 = cv2.imread('images/2.jpg')

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Draw keypoints on images
image1_with_keypoints = cv2.drawKeypoints(image1, kp1, None, color=(0,255,0), flags=0)
image2_with_keypoints = cv2.drawKeypoints(image2, kp2, None, color=(0,255,0), flags=0)

# Display images with keypoints
cv2.imshow("Image 1 with Keypoints", image1_with_keypoints)
cv2.imshow("Image 2 with Keypoints", image2_with_keypoints)
cv2.waitKey(0)

# Match keypoints between images
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches
matched_image = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display matched image
cv2.imshow("Matched Keypoints", matched_image)
cv2.waitKey(0)

# Calculate number of matches
print("Number of matches:", len(matches))

# Calculate performance metrics if ground truth keypoints are available
# precision = true positives / (true positives + false positives)
# recall = true positives / (true positives + false negatives)
# F1-score = 2 * (precision * recall) / (precision + recall)

# Release resources
cv2.destroyAllWindows()
