import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def yolo_detection(image, cfg_path="yolo/yolov4.cfg", weights_path="yolo/yolov4.weights", names_path="yolo/coco.names", confidence_threshold=0.27):
    # Load the classes
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Load YOLO
    net = cv2.dnn.readNet(weights_path, cfg_path)
    layer_names = net.getUnconnectedOutLayersNames()

    # Load a test image
    height, width = image.shape[:2]

    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Execute neural network
    outputs = net.forward(layer_names)

    # analyze the results
    cut_image = image.copy()
    for output in outputs:
        for detection in output:
            scores = detection[5:]  # Probability of each class
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")

                # Compute the top left angle
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Correct the coordinates to ensure they are within the image bounds
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = max(1, min(w, width - x))
                h = max(1, min(h, height - y))

                # Cut the image
                cropped = image[y:int(y+h*0.5), int(x*0.9):int(x+w*1.15)]
                x = int(x * 0.9)  # Offset x
                return cropped, x, y

    return None, None, None

frame1 = cv2.imread("outputFolder/frame_0013.png")
if frame1 is None:
    print("Image not found")
    exit()

yolo_frame1, offset_x, offset_y = yolo_detection(frame1)
if yolo_frame1 is None or isinstance(yolo_frame1, int):
    print("Car not detected")
    exit()

yolo_resized = cv2.resize(yolo_frame1, (yolo_frame1.shape[1] // 3, yolo_frame1.shape[0] // 3), interpolation=cv2.INTER_AREA)
cv2.imshow("YOLO Detection", yolo_resized)
cv2.waitKey(0)

# Convert to grayscale and apply Gaussian blur
gray = cv2.cvtColor(yolo_frame1, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, threshold1=150, threshold2=250)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_len = 100  # Minimum length of contours to consider
long_contours = [cnt for cnt in contours if cv2.arcLength(cnt, closed=False) > min_len]

# Draw long contours on a black background
result = np.zeros_like(edges)
cv2.drawContours(result, long_contours, -1, 255, 1)

result_resized= cv2.resize(result, (result.shape[1] // 3, result.shape[0] // 3), interpolation=cv2.INTER_AREA)
cv2.imshow("Contorni lunghi", result_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find the extreme point with the maximum x-coordinate
max_x = -1
extreme_point = None

for cnt in long_contours:
    for point in cnt:
        x, y = point[0]
        if x > max_x:
            max_x = x
            extreme_point = (x, y)

if not extreme_point:
    print("Mirror not detected in real image")
    exit()
    
mirror_point = (extreme_point[0] + offset_x, extreme_point[1] + offset_y)
cv2.circle(frame1, mirror_point, 10, (0, 0, 255), -1)
frame1 = cv2.resize(frame1, (frame1.shape[1] // 4, frame1.shape[0] // 4), interpolation=cv2.INTER_AREA)
cv2.imshow("Image with Mirror Detected", frame1)
cv2.waitKey(0)
