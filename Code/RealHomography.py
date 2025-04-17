import cv2
import numpy as np

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

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
new_object_points = np.array([
    [150, 800, 20],       # FaroSinistra
    [1500, 800, 20],      # FaroDestro
    # [606, 750, 100],       # TargaAltoSinistra
    [250, 1400, 550],       # TettoAltoSinistra
    [1126, 640, 100],        # TargaBassoDestra
    [1126, 750, 100],       # TargaAltoDestra
    # [606, 640, 100],       # TargaBassoSinistra
    # [1845, 1100, 2293],         # SpecchiettoDestra
], dtype=np.float32)

points = np.array([
    [1204, 1500],       #faroSinistra
    [1708, 1492],       #faroDestra
    # [1332, 1512],       #TargaAltoSinistra 
    [1276, 1260],       #tettoAltoSinistra
    [1548, 1564],        #TargaBassoDestra
    [1560, 1508],       #TargaAltoDestra
    # [1324, 1568],       #TargaBassoSinistra
    # [1956, 1256]        #SpecchiettoDestra
], dtype=np.float32)


#K = OPENCV
K = np.array([
    [3.31565306e+03, 0.00000000e+00, 1.91087753e+03],
    [0.00000000e+00, 3.31989314e+03, 1.07238815e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float32)


# # K =  API 4K
# K = np.array([
#     [2805.4324, 0.0, 1919.5735],
#     [0.0, 2805.4324, 1077.1753],
#     [0.0, 0.0, 1.0]
# ], dtype=np.float32)

distCoeffs = np.array([[ 2.65104301e-01, -1.78436004e+00,  2.42978100e-03,  1.18030874e-04, 3.77074289e+00]],dtype=np.float32)
# distCoeffs = None

# Solve for rotation and translation
# success, rvec, tvec, inliers = cv2.solvePnPRansac(
#     new_object_points,
#     image_points,
#     K,
#     distCoeffs=distCoeffs,
#     flags=cv2.SOLVEPNP_ITERATIVE,  # or cv2.SOLVEPNP_AP3P / EPNP if you want to test alternatives
#     iterationsCount=100           # Number of RANSAC iterations
# )

undistorted = cv2.undistort(frame, K, distCoeffs=distCoeffs)

success, rvec, tvec, inliers = cv2.solvePnPRansac(
    new_object_points,
    points,
    K,
    distCoeffs=distCoeffs,
    reprojectionError=800,
    iterationsCount=10000,
    confidence=0.99
)

print(inliers)
print("Success:", success)

# Convert rvec in rotation matrix
R, _ = cv2.Rodrigues(rvec)

# Projection matrix: P = K * [R | t]
Rt = np.hstack((R, tvec))  # 3x4
P = K @ Rt

targa = np.array([
    #[606, 750, 100],       # LeftLowLicensPlate
    #[1126, 750, 100],       # RightLowLicensPlate      
    #[606, 819.79, 100],       # LeftHighLicensPlate
    [250, 1400, 550]       # RightHighLicensPlate
], dtype=np.float32)

faro = np.array([
    [137, 930.07, 300],      #LeftLight
    [1595, 930.07, 300]    #RightLight
], dtype=np.float32)

tetto = np.array([
    [386, 1400, 500],       # Left
    [1346, 1400, 500]       # Right
], dtype=np.float32)

specchio = np.array([
    [1845, 1100, 2293],       # Right
], dtype=np.float32)

specchioLeft = np.array([
    [20, 1100, 2293],       # Left
], dtype=np.float32)

# # Project 3D points to 2D image plane
# targa_pts = [project_point(pt, K, R, tvec) for pt in targa]
# print("Projected points:", targa_pts)

# for i in range(4):
#     pt1 = targa_pts[i]
#     pt2 = targa_pts[(i + 1) % 4]
#     cv2.line(frame, pt1, pt2, (0, 255, 0), 5)

# # Project 3D points to 2D image plane
# faro_pts = [project_point(pt, K, R, tvec) for pt in faro]
# print("Projected points:", faro_pts)

# for i in range(2):
#     pt1 = faro_pts[i]
#     pt2 = faro_pts[(i + 1) % 2]
#     cv2.line(frame, pt1, pt2, (0, 255, 0), 5)

# # Project 3D points to 2D image plane
# tetto_pts = [project_point(pt, K, R, tvec) for pt in tetto]
# print("Projected points:", tetto_pts)

# for i in range(2):
#     pt1 = tetto_pts[i]
#     pt2 = tetto_pts[(i + 1) % 2]
#     cv2.line(frame, pt1, pt2, (0, 255, 0), 5)

# Project 3D points to 2D image plane
# specchio_pts = [project_point(pt, K, R, tvec) for pt in specchio]

targa_pts, _ = cv2.projectPoints(targa, rvec, tvec, K, distCoeffs=distCoeffs)
print("Projected points specchio :", targa_pts)
ss = targa_pts[:, 0, 0]
print("specchio", ss[0])
ss1 = targa_pts[:, 0, 1]
print("specchio", ss1[0])
cv2.circle(frame, (int(ss[0]), int(ss1[0])), 10, (0, 0, 255), -1)


specchio_pts, _ = cv2.projectPoints(specchio, rvec, tvec, K, distCoeffs=distCoeffs)
print("Projected points specchio :", specchio_pts)
ss = specchio_pts[:, 0, 0]
print("specchio", ss[0])
ss1 = specchio_pts[:, 0, 1]
print("specchio", ss1[0])

cv2.circle(frame, (int(ss[0]), int(ss1[0])), 10, (0, 150, 255), -1)

specchioL_pts, _ = cv2.projectPoints(specchioLeft, rvec, tvec, K, distCoeffs=distCoeffs)
print("Projected points specchio :", specchioL_pts)
ss = specchioL_pts[:, 0, 0]
print("specchio", ss[0])
ss1 = specchioL_pts[:, 0, 1]
print("specchio", ss1[0])

cv2.circle(frame, (int(ss[0]), int(ss1[0])), 10, (200, 150, 255), -1)


# Bounding box points in 3D space (in millimeters)
points_3D = [
    [0, 0, 0],         # leftBottom
    [1732, 0, 0],      # rightBottom
    [1732, 1519, 0],   # rightTop
    [0, 1519, 0]       # leftTop
]

points_3Dfront = [
    [0, 0, 3997],         # leftBottom
    [1732, 0, 3997],      # rightBottom
    [1732, 1519, 3997],   # rightTop
    [0, 1519, 3997]       # leftTop
]

# # Project 3D points to 2D image plane
# projected_pts_front = [project_point(pt, K, R, tvec) for pt in points_3Dfront]
# print("Projected points:", projected_pts_front)



# # Project 3D points to 2D image plane
# projected_pts = [project_point(pt, K, R, tvec) for pt in points_3D]
# print("Projected points:", projected_pts)

# for i in range(4):
#     pt1 = projected_pts[i]
#     pt2 = projected_pts[(i + 1) % 4]
#     cv2.line(frame, pt1, pt2, (0, 255, 0), 2)


# Show the projected rectangle on the image
frame_resized = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4), interpolation=cv2.INTER_AREA)

cv2.imshow("Targa", frame_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


