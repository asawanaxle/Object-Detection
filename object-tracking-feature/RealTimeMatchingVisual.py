import cv2
from picamera2 import Picamera2

def ORB_detector(new_image, image_template):
    try:
        if len(new_image.shape) == 2:  # Grayscale image
            image1 = new_image
        elif len(new_image.shape) == 3 and new_image.shape[2] == 3:  # BGR image
            # Convert image to grayscale
            image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        else:
            print("Error: Unexpected image format")
            return None, None, None

        # Create ORB detector with 1000 keypoints with a scaling pyramid factor of 1.2
        orb = cv2.ORB_create(1000, 1.2)

        # Detect keypoints of original image
        kp1, des1 = orb.detectAndCompute(image1, None)

        # Detect keypoints of template image
        kp2, des2 = orb.detectAndCompute(image_template, None)

        # Create matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Do matching
        matches = bf.match(des1, des2)

        # Sort the matches based on distance.  Least distance is better
        matches = sorted(matches, key=lambda val: val.distance)
        return matches, kp1, kp2
    except Exception as e:
        print("Error in ORB_detector:", e)
        return None, None, None



# Start PiCamera
picam2 = Picamera2()
picam2.start()

# Load our image template, this is our reference image
image_template = cv2.imread('images/simple.png', 0)

while True:
    # Capture an image from the camera
    im = picam2.capture_array()

    # Convert image to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Get matches and keypoints
    matches, kp1, kp2 = ORB_detector(gray, image_template)

    if matches is not None:
        # Draw matches
        im_matches = cv2.drawMatches(im, kp1, image_template, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Display the matches
        cv2.imshow("Preview", im_matches)
    else:
        # Display the image without matches
        cv2.imshow("Preview", im)

    # Check for key press to exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Stop PiCamera
picam2.stop()
cv2.destroyAllWindows()