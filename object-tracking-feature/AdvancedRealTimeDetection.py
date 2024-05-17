import cv2
from picamera2 import Picamera2
import numpy as np

def ORB_detector(new_image, templates):
    if new_image.ndim == 3:
        image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    else:
        image1 = new_image

    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, edgeThreshold=15)
    kp1, des1 = orb.detectAndCompute(image1, None)

    flann_index = 6
    index_params = dict(algorithm=flann_index, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    final_matches = []
    for template in templates:
        if template.ndim == 3:
            image2 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            image2 = template

        kp2, des2 = orb.detectAndCompute(image2, None)
        matches = flann.knnMatch(des1, des2, k=2)
        # Ensure each match list has two matches and apply ratio test, then select the first match
        good_matches = [m_n[0] for m_n in matches if len(m_n) == 2 and m_n[0].distance < 0.60 * m_n[1].distance]
        final_matches.append((good_matches, kp1, kp2, template))

    return final_matches

def draw_bounding_boxes(image, matches_info):
    for matches, kp1, kp2, template in matches_info:
        if len(matches) >= 15:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.5)  # Adjusted RANSAC threshold for robustness
            if M is not None and mask.sum() > 10:  # Ensure a minimum number of inliers
                h, w = template.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                cv2.polylines(image, [np.int32(dst)], True, (0, 255, 0), 3)
                cv2.putText(image, f"Matches: {len(matches)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(image, "Insufficient inliers", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return image

# Load multiple templates
templates = [cv2.imread('images/1.png', 0), cv2.imread('images/2.jpg', 0)]

# Initialize camera
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
