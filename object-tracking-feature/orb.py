from flask import Flask, Response
import cv2
import numpy as np 
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load and preprocess the template images
def preprocess_image(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(image)

templates = [cv2.imread('images/1.png', 0), cv2.imread('images/2.jpg', 0)]
templates = [preprocess_image(template) for template in templates]

def ORB_detector(new_image, templates):
    image1 = preprocess_image(new_image)
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, edgeThreshold=15)
    kp1, des1 = orb.detectAndCompute(image1, None)

    if len(kp1) == 0:
        print("No keypoints found in the image.")
        return []

    final_matches = []
    for template in templates:
        if template is not None:
            kp2, des2 = orb.detectAndCompute(template, None)

            if len(kp2) == 0:
                print("No keypoints found in the template.")
                continue

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            print(f"Total matches found: {len(matches)}")

            # Loosen the distance threshold to consider more matches as good
            good_matches = [m for m in matches if m.distance < 75]
            print(f"Good matches found: {len(good_matches)}")

            if len(good_matches) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None and np.linalg.det(M) > 0.1 and mask is not None and mask.sum() > 5:
                    print("Valid homography found.")
                    final_matches.append((good_matches, kp1, kp2, template, M, mask))
                else:
                    print("Homography is invalid or has too few inliers.")
                    print(f"Det(M): {np.linalg.det(M) if M is not None else 'None'}, Mask Sum: {mask.sum() if mask is not None else 'None'}")
            else:
                print("Insufficient good matches; skipping this template.")

    if not final_matches:
        print("No matches found; matches_info is empty.")

    return final_matches

THRESHOLD = 15

def draw_bounding_boxes(image, matches_info):
    object_detected = False 
    print("Entered draw_bounding_boxes function.") 

    for matches, kp1, kp2, template, M, mask in matches_info:
        if M is not None:
            h, w = template.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            x, y, w, h = cv2.boundingRect(np.int32(dst))
            
            # Debug output
            print(f"Debug: Keypoints in image: {len(kp1)}")
            print(f"Debug: Keypoints in template: {len(kp2)}")
            print(f"Debug: Number of matches: {len(matches)}")
            print(f"Debug: Homography matrix: {M}")
            print(f"Debug: Bounding box coordinates: x={x}, y={y}, w={w}, h={h}")

            if len(matches) > THRESHOLD:
                object_detected = True
                # Ensure coordinates are within image bounds
                if x >= 0 and y >= 0 and (x + w) <= image.shape[1] and (y + h) <= image.shape[0]:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(image, f"Matches: {len(matches)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    print(f"Object detected! Bounding box: ({x}, {y}, {x+w}, {y+h})")
                else:
                    print("Bounding box coordinates out of image bounds.")
        else:
            print(f"Homography computation failed or insufficient inliers. Mask sum: {mask.sum()}")

    if object_detected:
        print("Object detected on screen!")
    else:
        print("No object detected.")
    
    return image

@app.route('/events')
def events():
    def generate():
        cap = cv2.VideoCapture(0)  # Use 0 for default webcam
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            matches_info = ORB_detector(frame, templates)
            frame_with_boxes = draw_bounding_boxes(frame, matches_info)
            
            _, img_encoded = cv2.imencode('.jpg', frame_with_boxes)
            img_bytes = img_encoded.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + img_bytes + b"\r\n")
        cap.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    def generate():
        cap = cv2.VideoCapture(0)  # Use 0 for default webcam
        last_detection_time = None
        consistent_detections = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            matches_info = ORB_detector(frame, templates)
            object_detected = any(len(matches) > THRESHOLD for matches, _, _, _, _, _ in matches_info)
            
            if object_detected:
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                if last_detection_time is None or current_time - last_detection_time > 1.0:
                    consistent_detections = 1
                else:
                    consistent_detections += 1
                last_detection_time = current_time
            else:
                consistent_detections = 0

            status = 'Object detected on the UI' if consistent_detections > 0 else 'Object Not Found'
            
            yield f"data: {status}\n\n"
        cap.release()
    return Response(generate(), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
