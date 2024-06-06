import cv2
import numpy as np
from picamera2 import Picamera2

# Define preprocess_image function before it's called
def preprocess_image(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(image)

# Preprocess the templates once before the loop
templates = [cv2.imread('images/1.png', 0), cv2.imread('images/2.jpg', 0)]
templates = [preprocess_image(template) for template in templates]

def ORB_detector(new_image, templates):
    image1 = preprocess_image(new_image)
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, edgeThreshold=15)
    kp1, des1 = orb.detectAndCompute(image1, None)

    final_matches = []
    for template in templates:
        if template is not None:
            kp2, des2 = orb.detectAndCompute(template, None)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            good_matches = [m for m in matches if m.distance < 50]

            if len(good_matches) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                if M is not None and np.linalg.det(M) > 0.1 and mask.sum() > 10:
                    final_matches.append((good_matches, kp1, kp2, template, M, mask))
                else:
                    print(f"Homography computation failed or insufficient inliers for template.")
            else:
                print("Not enough good matches for template.")
        else:
            print("Template image is None, skipped.")
    return final_matches

THRESHOLD = 15  # Adjust as needed

def draw_bounding_boxes(image, matches_info):
    object_detected = False
    for matches, kp1, kp2, template, M, mask in matches_info:
        if M is not None:
            h, w = template.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            x, y, w, h = cv2.boundingRect(np.int32(dst))
            if len(matches) > THRESHOLD:
                object_detected = True
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(image, f"Matches: {len(matches)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(image, "Insufficient inliers or no homography.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    if object_detected:
        print("Object detected on screen!")
    
    return image

picam2 = Picamera2()
picam2.start()

while True:
    im = picam2.capture_array()
    matches_info = ORB_detector(im, templates)
    im_with_info = draw_bounding_boxes(im, matches_info)
    cv2.imshow("Preview", im_with_info)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
