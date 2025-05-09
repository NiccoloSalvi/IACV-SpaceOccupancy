import cv2

# List to store the clicked points
points = []

# Mouse callback function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Image', img)

# Load the image
img = cv2.imread('dayImg.jpeg')  # Replace with your image file path
cv2.imshow('Image', img)

# Set mouse callback
cv2.setMouseCallback('Image', click_event)

print("Click on 4 points in the image...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print final coordinates
print("Selected Points:")
for i, point in enumerate(points):
    print(f"Point {i + 1}: {point}")