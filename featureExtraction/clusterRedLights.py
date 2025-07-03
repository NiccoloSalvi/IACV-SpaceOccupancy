import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def yolo_detection(image, cfg_path="featureExtraction/yolo/yolov4.cfg", weights_path="featureExtraction/yolo/yolov4.weights", names_path="featureExtraction/yolo/coco.names", confidence_threshold=0.27):
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
        clustering = DBSCAN(eps=20, min_samples=5).fit(pixel_coords)
        labels = clustering.labels_
        unique_labels = [label for label in set(labels) if label != -1]

        if len(unique_labels) < 2:
            print("Less than two clusters detected")
            return []

        # Sort clusters from left to right based on mean X
        cluster_means = []
        for label in unique_labels:
            cluster = pixel_coords[labels == label]
            cluster_means.append((label, np.mean(cluster[:, 0])))

        sorted_clusters = sorted(cluster_means, key=lambda x: x[1])
        left_cluster = pixel_coords[labels == sorted_clusters[0][0]]
        right_cluster = pixel_coords[labels == sorted_clusters[1][0]]

        # From the left cluster: get the point with the largest X (rightmost point in that cluster)
        left_internal = tuple(left_cluster[np.argmax(left_cluster[:, 0])])

        # From the right cluster: get the point with the smallest X (leftmost point in that cluster)
        right_internal = tuple(right_cluster[np.argmin(right_cluster[:, 0])])

        red_centers = [left_internal, right_internal]
    else:
        red_centers = []
        print("No red pixels detected")
    
    # Display detected red pixels and cluster centers
    red_detected = cv2.bitwise_and(image, image, mask=red_mask)
    
    for center in red_centers:
        cv2.circle(red_detected, center, 10, (0, 255, 0), -1)
    
    red_detected = cv2.resize(red_detected, (red_detected.shape[1] // 4, red_detected.shape[0] // 4), interpolation=cv2.INTER_AREA)
    cv2.imshow('Detected Red Pixels', red_detected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return red_centers


def lights_internal(image):

    frame1 = cv2.imread(image)

    yolo_frame1 = yolo_detection(frame1)

    lights = detect_red_lights(yolo_frame1)

    print("Lights :", lights)
    #print original image with red lights
    for light in lights:
        cv2.circle(frame1, light, 15, (0, 255, 0), -1)

    cv2.line(frame1, lights[0], lights[1], (255, 0, 0), 5)

    frame1 = cv2.resize(frame1, (frame1.shape[1] // 4, frame1.shape[0] // 4), interpolation=cv2.INTER_AREA)

    cv2.imshow("Frame 1", frame1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return lights[0], lights[1]  # Return the internal points of the left and right lights

if __name__ == "__main__":
    lights_internal("outputFolder/frame_02.png")
