import cv2
import os

# Store clicked points
points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

# Load your image
image_path = "outputFolder/frame_02.png"  # Change this to your image path
image = cv2.imread(image_path)
image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), interpolation=cv2.INTER_AREA)

if image is None:
    raise FileNotFoundError("Image not found. Please check the path.")

clone = image.copy()

cv2.namedWindow("Select 4 Points")
cv2.setMouseCallback("Select 4 Points", mouse_callback)

while True:
    temp_image = clone.copy()
    for pt in points:
        cv2.circle(temp_image, pt, 5, (0, 255, 0), -1)

    cv2.imshow("Select 4 Points", temp_image)
    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to quit after selecting points
    if key == ord("q") or len(points) == 4:
        break

cv2.destroyAllWindows()

for i, pt in enumerate(points):
    # Scale back to original size
    x, y = pt
    original_x = x * 4  # Assuming the original image was 4 times larger
    original_y = y * 4
    points[i] = (original_x, original_y)
    print(f"Point {i + 1} in original size: ({original_x}, {original_y})")
