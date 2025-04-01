import cv2
import numpy as np

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

            if confidence > confidence_threshold:  # Se è abbastanza sicuro
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


def detect_red_lights(image, min_area=3):
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

    # clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)

    # find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    red_lights = []
    boxes = []

    for cnt in contours[:2]:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)

            # inner box center
            inner_x = x + int(w * 0.15)
            inner_y = y + int(h * 0.15)
            inner_w = int(w * 0.7)
            inner_h = int(h * 0.7)

            boxes.append((inner_x, inner_y, inner_w, inner_h))

    boxes = sorted(boxes, key=lambda b: b[0])

    for idx, (x, y, w, h) in enumerate(boxes):
        # get outer edge center
        if idx == 0:
            cx = x
        else:
            cx = x + w
        cy = y + h // 2
        red_lights.append((cx, cy))

    return red_lights, boxes


def detect_license_plate(image, lights, min_plate_area=200):

    (lx, ly), (rx, ry) = lights[0], lights[-1]

    dx, dy = rx - lx, ry - ly
    angle = np.degrees(np.arctan2(dy, dx))

    # calculate rotation center
    center = ((lx + rx) // 2, (ly + ry) // 2)

    # get rotation matrix and rotate image
    rot_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))

    def rotate_point(x, y, M):
        px = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        py = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        return int(px), int(py)

    new_lx, new_ly = rotate_point(lx, ly, rot_matrix)
    new_rx, new_ry = rotate_point(rx, ry, rot_matrix)

    # Define ROI in rotated image
    roi_margin = 20
    roi_top = min(new_ly, new_ry) - 10
    roi_bottom = max(new_ly, new_ry) + 60
    roi_left = new_lx - roi_margin
    roi_right = new_rx + roi_margin

    h, w = rotated_image.shape[:2]
    roi_left = max(0, roi_left)
    roi_right = min(w, roi_right)
    roi_top = max(0, roi_top)
    roi_bottom = min(h, roi_bottom)

    roi = rotated_image[roi_top:roi_bottom, roi_left:roi_right]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # white mask for white parts of the plate
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 60, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # blue mask for blue part of the plate
    # we keep this if we want the whole plate to be taken into account
    lower_blue = np.array([100, 90, 90])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    plate_mask = cv2.bitwise_or(white_mask, blue_mask)
    #plate_mask = white_mask

    # Clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_plate = None
    best_score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_plate_area:
            continue
        x, y, w_box, h_box = cv2.boundingRect(cnt)

        aspect_ratio = w_box / float(h_box)
        if 1.5 < aspect_ratio < 5.5:
            score = area
            if score > best_score:
                best_score = score
                best_plate = (x + roi_left, y + roi_top, w_box, h_box)

    return best_plate



frame1 = cv2.imread("outputFolder/frame_02.png")
frame2 = cv2.imread("outputFolder/frame_10.png")

yolo_frame1 = yolo_detection(frame1)
yolo_frame2 = yolo_detection(frame2)

lights1, box1 = detect_red_lights(yolo_frame1)
lights2, box2 = detect_red_lights(yolo_frame2)

# draw detections
for pt in lights1:
    cv2.circle(frame1, pt, 10, (0, 255, 0), -1)
    #cv2.putText(frame1, f"{pt}", (pt[0]+6, pt[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

for pt in lights2:
    cv2.circle(frame2, pt, 10, (0, 255, 0), -1)
    cv2.putText(frame2, f"{pt}", (pt[0]+6, pt[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

L1, R1 = lights1[0], lights1[1]
cv2.line(frame1, L1, R1, (255, 0, 0), 5)

L2, R2 = lights2[0], lights2[1]
cv2.line(frame2, L2, R2, (255, 0, 0), 5)

frame1 = cv2.resize(frame1, (frame1.shape[1] // 4, frame1.shape[0] // 4), interpolation=cv2.INTER_AREA)
frame2 = cv2.resize(frame2, (frame2.shape[1] // 4, frame2.shape[0] // 4), interpolation=cv2.INTER_AREA)

cv2.imshow("Frame 1", frame1)
cv2.imshow("Frame 2", frame2)
cv2.waitKey(0)
cv2.destroyAllWindows()

frame = cv2.imread("outputFolder/frame_10.png")
plate_box = detect_license_plate(frame, lights2)

if plate_box:
    x, y, w, h = plate_box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 5)
    #cv2.putText(frame, "License Plate", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
else:
    print("No license plate detected")

frame = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4), interpolation=cv2.INTER_AREA)
cv2.imshow("Frame with Plate", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()