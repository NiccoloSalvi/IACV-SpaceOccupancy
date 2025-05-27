import cv2
import numpy as np

# --- PARAMETERS ---
IMAGE_PATH = "sunnyFrame.png"     # path to your 3840×2160 image
DISPLAY_SCALE_INIT = 0.25              # initial display scale
SCALE_STEP = 1.25                      # zoom factor per key press
SCALE_MIN = 0.1                        # minimum zoom level
SCALE_MAX = 4.0                        # maximum zoom level
# -------------------

# Load full‑res frame
def load_frame(path: str) -> np.ndarray:
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Could not load {path}")
    return frame

frame = load_frame(IMAGE_PATH)
orig_h, orig_w = frame.shape[:2]

# Current zoom scale and resized image
scale = DISPLAY_SCALE_INIT
resized = cv2.resize(
    frame,
    (int(orig_w * scale), int(orig_h * scale)),
    interpolation=cv2.INTER_AREA
)

# Store clicked points in original coords
orig_points = []  # list of (x, y) in original image space

# Mouse callback: record original coords

def on_mouse(event, x, y, flags, param):
    global orig_points, scale
    if event == cv2.EVENT_LBUTTONDOWN:
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        orig_points.append((orig_x, orig_y))
        print(f"Clicked: display=({x}, {y}) -> original=({orig_x}, {orig_y})")

cv2.namedWindow("Click to select", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Click to select", on_mouse)

print("Instructions:")
print("  Click to select points.")
print("  Press '+' or '=' to zoom in.")
print("  Press '-' to zoom out.")
print("  Press ESC to exit.")

while True:
    # Draw circles on a copy of the resized image
    display = resized.copy()
    for ox, oy in orig_points:
        dx = int(ox * scale)
        dy = int(oy * scale)
        cv2.circle(display, (dx, dy), radius=5, color=(0, 0, 255), thickness=-1)

    cv2.imshow("Click to select", display)
    key = cv2.waitKey(20) & 0xFF

    if key == 27:  # ESC
        break
    elif key in (ord('+'), ord('=')):
        # Zoom in
        scale = min(scale * SCALE_STEP, SCALE_MAX)
        resized = cv2.resize(
            frame,
            (int(orig_w * scale), int(orig_h * scale)),
            interpolation=cv2.INTER_AREA
        )
        print(f"Zoom: scale set to {scale:.2f}")
    elif key == ord('-'):
        # Zoom out
        scale = max(scale / SCALE_STEP, SCALE_MIN)
        resized = cv2.resize(
            frame,
            (int(orig_w * scale), int(orig_h * scale)),
            interpolation=cv2.INTER_AREA
        )
        print(f"Zoom: scale set to {scale:.2f}")

cv2.destroyAllWindows()
