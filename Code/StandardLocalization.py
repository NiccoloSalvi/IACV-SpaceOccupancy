import cv2
import numpy as np

def project_point(X_world, K, R, t):
    X_world = np.array(X_world).reshape((3, 1))  # 3D column vector
    Rt = np.hstack((R, t))  # 3x4
    X_world_h = np.vstack((X_world, [1]))  # Make it 4x1 homogeneous
    x_img_h = K @ Rt @ X_world_h  # 3x1

    x = x_img_h[0, 0] / x_img_h[2, 0]
    y = x_img_h[1, 0] / x_img_h[2, 0]
    return (int(round(x)), int(round(y)))


frame = cv2.imread("outputFolder/frame_10.png")

# Real world points (x, y, z) in millimeters
object_points = np.array([
    [606, 820.6, 632.1],     # LeftLicensPlate 700.9 y
    [1126, 820.6, 632.1],    # RightLicensPlate
    [137.5, 791.7, 683.2],   # LeftLight
    [1594.5, 791.7, 683.2]   # RightLight
], dtype=np.float32)

# Real world points (x, y, z) in millimeters
new_object_points = np.array([
    [589.34, 855.35, 129.0],     # LeftLicensPlate 700.9 y
    [1120.24, 855.35, 129.0],    # RightLicensPlate
    [188.04, 819.79, 188.18],   # LeftLight
    [1548.9, 819.79, 188.18]   # RightLight
], dtype=np.float32)


# image points (x, y) in pixels
image_points = np.array([
    [1325, 1500],     # LeftLicensPlate
    [1556, 1500],     # RightLicensPlate
    [1176, 1528],     # LeftLight
    [1748, 1516]      # RightLight
], dtype=np.float32)

# Convert to numpy arrays
# K = IOS API

K = np.array([
    [1399.6365, 0, 961.83606],
    [0, 1399.6365, 538.7572],
    [0, 0, 1]
], dtype=np.float32)

# K = OPENCV
"""
K = np.array([
    [833.38017686,	0.0,	399.68083124],
	[0.0,	833.44962868,	550.39137349],
	[0.0,	0.0,	1.0]
], dtype=np.float32)
"""

# K =  API 4K
"""
K = np.array([
    [2805.4324, 0.0, 0.0],
    [0.0, 2805.4324, 0.0],
    [1919.5735, 1077.1753, 1.0]
], dtype=np.float32)
"""

# Solve for rotation and translation
success, rvec, tvec = cv2.solvePnP(
    new_object_points,
    image_points,
    K,
    distCoeffs=None
)

# Convert rvec in rotation matrix
R, _ = cv2.Rodrigues(rvec)

# Projection matrix: P = K * [R | t]
Rt = np.hstack((R, tvec))  # 3x4
P = K @ Rt

print("R:\n", R)
print("t:\n", tvec)
print("Projection P:\n", P)

# Bounding box points in 3D space (in millimeters)
points_3D = [
    [0, 0, 0],         # leftBottom
    [1732, 0, 0],      # rightBottom
    [1732, 1519, 0],   # rightTop
    [0, 1519, 0]       # leftTop
]

points_3Dfront = [
    [0, 0, -3997],         # leftBottom
    [1732, 0, -3997],      # rightBottom
    [1732, 1519, -3997],   # rightTop
    [0, 1519, -3997]       # leftTop
]

# Project 3D points to 2D image plane
projected_pts_front = [project_point(pt, K, R, tvec) for pt in points_3Dfront]
print("Projected points:", projected_pts_front)

for i in range(4):
    pt1 = projected_pts_front[i]
    pt2 = projected_pts_front[(i + 1) % 4]
    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)


# Project 3D points to 2D image plane
projected_pts = [project_point(pt, K, R, tvec) for pt in points_3D]
print("Projected points:", projected_pts)

for i in range(4):
    pt1 = projected_pts[i]
    pt2 = projected_pts[(i + 1) % 4]
    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)


# Show the projected rectangle on the image
frame_resized = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4), interpolation=cv2.INTER_AREA)

cv2.imshow("Bounding box", frame_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


