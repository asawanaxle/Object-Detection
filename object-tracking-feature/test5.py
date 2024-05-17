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
    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.3, edgeThreshold=10)
    kp1, des1 = orb.detectAndCompute(image1, None)

    flann_index = 6
    index_params = dict(algorithm=flann_index, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    final_matches = []
    for template in templates:
        if template is not None:
            kp2, des2 = orb.detectAndCompute(template, None)

            matches = flann.knnMatch(des1, des2, k=2)
            good_matches = [m_n[0] for m_n in matches if len(m_n) == 2 and m_n[0].distance < 0.70 * m_n[1].distance]

            if len(good_matches) > 15:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                if M is not None and np.linalg.det(M) > 0.1 and mask.sum() > 10:
                    final_matches.append((good_matches, kp1, kp2, template, M, mask))
                else:
                    print(f"Homography computation failed or insufficient inliers for template.")
        else:
            print("Template image is None, skipped.")
    return final_matches

def draw_bounding_boxes(image, matches_info):
    for matches, kp1, kp2, template, M, mask in matches_info:
        if M is not None:
            h, w = template.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            cv2.polylines(image, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(image, f"Matches: {len(matches)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(image, "Insufficient inliers or no homography.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
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
