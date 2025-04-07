import cv2
import numpy as np
from sklearn.cluster import DBSCAN

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

                # Draw the Bounding box
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = max(0, min(w, width - x - 1))
                h = max(0, min(h, height - y - 1))

                # Hide all pixels that are not in the bounding box
                cut_image[0:int(y+h*0.3), 0:width] = 0 # Above the bounding box
                cut_image[int(y+h*0.85):height, 0:width] = 0 # Under the bounding box
                cut_image[y:y+h, 0:int(x-w*0.05)] = 0 # On the left of the bounding box
                cut_image[y:y+h, int(x+w*1.05):width] = 0 # On the right of the bounding box

    # Save the image
    show_image = cut_image.copy()
    show_image = cv2.resize(show_image, (show_image.shape[1] // 4, show_image.shape[0] // 4), interpolation=cv2.INTER_AREA)
    cv2.imshow("Cut Image", show_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cut_image


def detect_red_lights(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red hue range (focused on core light, less aura)
    lower_red1 = np.array([0, 120, 120])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 120, 120])
    upper_red2 = np.array([180, 255, 255])

    # red mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Get coordinates of red pixels
    y_coords, x_coords = np.where(red_mask > 0)
    pixel_coords = np.column_stack((x_coords, y_coords))
    
    if len(pixel_coords) > 0:
        # Apply DBSCAN clustering to group different red light sources
        clustering = DBSCAN(eps=20, min_samples=5).fit(pixel_coords)
        labels = clustering.labels_
        
        unique_labels = set(labels)
        red_centers = []
        
        for label in unique_labels:
            if label == -1:
                continue  # Ignore noise points
            cluster_points = pixel_coords[labels == label]
            mean_x = int(np.mean(cluster_points[:, 0]))
            mean_y = int(np.mean(cluster_points[:, 1]))
            red_centers.append((mean_x, mean_y))
    else:
        red_centers = []
        print("No red pixels detected")
    
    # Display detected red pixels and cluster centers
    red_detected = cv2.bitwise_and(image, image, mask=red_mask)
    
    for center in red_centers:
        cv2.circle(red_detected, center, 5, (0, 255, 0), -1)
    
    red_detected = cv2.resize(red_detected, (red_detected.shape[1] // 4, red_detected.shape[0] // 4), interpolation=cv2.INTER_AREA)
    cv2.imshow('Detected Red Pixels', red_detected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return red_centers



frame1 = cv2.imread("outputFolder/frame_07.png")
frame2 = cv2.imread("outputFolder/frame_10.png")

yolo_frame1 = yolo_detection(frame1)
yolo_frame2 = yolo_detection(frame2)

lights1 = detect_red_lights(yolo_frame1)
lights2 = detect_red_lights(yolo_frame2)

print("Lights 2:", lights2)
#print original image with red lights
for light in lights1:
    cv2.circle(frame1, light, 5, (0, 255, 0), -1)

#print line that connects the two red lights
L1, R1 = lights1[0], lights1[1]
cv2.line(frame1, L1, R1, (255, 0, 0), 5)

for light in lights2:
    cv2.circle(frame2, light, 5, (0, 255, 0), -1)

L2, R2 = lights2[0], lights2[1]
cv2.line(frame2, L2, R2, (255, 0, 0), 5)

frame1 = cv2.resize(frame1, (frame1.shape[1] // 4, frame1.shape[0] // 4), interpolation=cv2.INTER_AREA)
frame2 = cv2.resize(frame2, (frame2.shape[1] // 4, frame2.shape[0] // 4), interpolation=cv2.INTER_AREA)

cv2.imshow("Frame 1", frame1)
cv2.imshow("Frame 2", frame2)
cv2.waitKey(0)
cv2.destroyAllWindows()