import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Loading the YOLOv8 model
model = YOLO('yolov8s.pt')

cap = cv2.VideoCapture("Test Videos/thief_video2.mp4")

target_classes = ['car', 'bus', 'truck', 'person']

# Polygon points
pts = []

# Function to draw polygon (ROI)
def draw_polygon(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        pts.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        pts = []

# Function to check if a point is inside a polygon
def inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, (point[0], point[1]), False) == 1

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_polygon)

def preprocess(img):
    height, width = img.shape[:2]
    ratio = height / width
    return cv2.resize(img, (640, int(640 * ratio)))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = preprocess(frame)
    results = model(frame)[0]
    
    count = 0  # Counter for objects inside the polygon
    
    # Process detected objects
    for result in results.boxes.data:
        x1, y1, x2, y2, conf, cls = result.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        name = model.names[int(cls)]
        
        if name in target_classes:
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Check if the object is inside the polygon
            if len(pts) >= 4 and inside_polygon((center_x, center_y), np.array([pts])):
                count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Draw the polygon
    if len(pts) >= 4:
        frame_copy = frame.copy()
        cv2.fillPoly(frame_copy, np.array([pts]), (0, 255, 0))
        frame = cv2.addWeighted(frame_copy, 0.1, frame, 0.9, 0)
    
    # Display count on top
    cv2.putText(frame, f'Count: {count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()