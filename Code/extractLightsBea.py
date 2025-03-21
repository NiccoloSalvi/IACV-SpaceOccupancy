import cv2
import os
import numpy as np

def detect_red_lights(image, min_area=5):
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

frame1 = cv2.imread("outputFolder/frame_02.png")
frame2 = cv2.imread("outputFolder/frame_03.png")

lights1, box1 = detect_red_lights(frame1)
lights2, box2 = detect_red_lights(frame2)

# draw detections
for pt in lights1:
    cv2.circle(frame1, pt, 6, (0, 255, 0), -1)
    cv2.putText(frame1, f"{pt}", (pt[0]+6, pt[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

for pt in lights2:
    cv2.circle(frame2, pt, 6, (0, 255, 0), -1)
    cv2.putText(frame2, f"{pt}", (pt[0]+6, pt[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# draw light segments
if len(lights1) >= 2:
    L1, R1 = lights1[0], lights1[1]
    cv2.line(frame1, L1, R1, (255, 0, 0), 2)
else:
    print("Not enough lights in Frame 1")

if len(lights2) >= 2:
    L2, R2 = lights2[0], lights2[1]
    cv2.line(frame2, L2, R2, (255, 0, 0), 2)
else:
    print("Not enough lights in Frame 2")

cv2.imshow("Frame 1 - Light Segment (Outer Edge Center)", frame1)
cv2.imshow("Frame 2 - Light Segment (Outer Edge Center)", frame2)
cv2.waitKey(0)
cv2.destroyAllWindows()