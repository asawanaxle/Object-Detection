#!/usr/bin/python3

import numpy as np
import cv2
from picamera2 import Picamera2

def ORB_detector(new_image, image_template):
    # Check if the image is in RGB format
    if len(new_image.shape) == 3 and new_image.shape[2] == 3:
        # Convert image to grayscale
        image1 = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
    else:
        print("Error: Image shape is", new_image.shape)
        print("Content of the captured image array:")
        print(new_image)
        return None

    # Create ORB detector with 1000 keypoints with a scaling pyramid factor of 1.2
    orb = cv2.ORB_create(1000, 1.2)

    # Detect keypoints of original image
    (kp1, des1) = orb.detectAndCompute(image1, None)

    # Detect keypoints of template image
    (kp2, des2) = orb.detectAndCompute(image_template, None)

    # Create matcher 
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Do matching
    matches = bf.match(des1, des2)

    # Sort the matches based on distance.  Least distance is better
    matches = sorted(matches, key=lambda val: val.distance)
    return len(matches)

# Start PiCamera
picam2 = Picamera2()
picam2.start()

# Load our image template, this is our reference image
image_template = cv2.imread('images/simple.png', 0) 

while True:
    # Get image from PiCamera
    im = picam2.capture_array()

    # Print the shape of the captured image array
    print("Shape of captured image array:", im.shape)

    # Convert image to RGB format
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Convert image to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    # Get number of ORB matches 
    matches = ORB_detector(gray, image_template)

    # Display status string showing the current no. of matches 
    output_string = "# of Matches = " + str(matches)
    cv2.putText(im, output_string, (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 0, 0), 2)

    # Our threshold to indicate object detection
    threshold = 200

    # If matches exceed our threshold then object has been detected
    if matches and matches > threshold:
        cv2.putText(im, 'Object Found', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

    cv2.imshow("Camera", im)
    if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
        break

# Stop PiCamera
picam2.stop()
cv2.destroyAllWindows()
