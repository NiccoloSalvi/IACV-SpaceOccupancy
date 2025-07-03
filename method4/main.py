import sys
import numpy as np
import cv2
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'featureExtraction')))
from extractLights2 import process_frames
from getMirror import mirror_frame

def ang(u,v):
    return np.arccos(np.clip(u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0))

# load the image
img_path = os.path.join(os.getcwd(), "featureExtraction", "extractedFrames", "frame_08.png")
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError("frame non trovato")

# camera intrinsic parameters and distortion coefficients
K = np.array([
    [3.31565306e+03, 0.0, 1.91087753e+03],
    [0.0, 3.31989314e+03, 1.07238815e+03],
    [0.0, 0.0, 1.0]
], dtype=np.float64)
dist = np.array([[2.65104301e-01, -1.78436004e+00,  2.42978100e-03,  1.18030874e-04, 3.77074289e+00]], dtype=np.float64)

# undistort the image
img_ud = cv2.undistort(img, K, dist)

# ---------- 1. pixel (undistorti) dei 4 punti sullo stesso piano ------
pix = np.array([
    [TL[0], TL[1]],  # P0 = plate TL
    [TR[0], TR[1]],  # P1 = plate TR
    [L2[0], L2[1]],  # P2 = rear-light L
    [R2[0], R2[1]],  # P3 = rear-light R
], dtype=np.float64)

# undistort the points
uv = cv2.undistortPoints(pix.reshape(-1,1,2), K, dist, P=K).reshape(-1,2)
p0, p1, p2, p3 = [np.append(u,1.) for u in uv]

# line through top-left and top-right plate corners in homogeneous coordinates
l01 = np.cross(p0, p1)
# through left and right taillight points in homogeneous coordinates
l23 = np.cross(p2, p3) # retta L-R

# vanishing point in homogeneous coordinates
Vh = np.cross(l01, l23)
Vh /= Vh[2]

Kinv = np.linalg.inv(K)

# back-project vanishing point to 3d direction
v_dir = Kinv.dot(Vh)
v_dir /= np.linalg.norm(v_dir)

# back-project plate top-left corner to 3d viewing ray
r0_dir = Kinv.dot(p0)
r0_dir /= np.linalg.norm(r0_dir)

# back-project plate top-right corner to 3d viewing ray
r1_dir = Kinv.dot(p1)
r1_dir /= np.linalg.norm(r1_dir)

alpha = ang(r0_dir, v_dir)
beta = ang(r1_dir, v_dir)
gamma = ang(r0_dir, r1_dir)

PLATE_W = 0.520 # real-world width of the license plate in meters
PLATE_H = 0.130 # real-world height of the license plate
Z_FARO = 0.150 # height of rear lights above plate plane

# use law of sines to compute approximate distance from camera to each plate corner
d0 = PLATE_W * np.sin(beta) / np.sin(gamma) # depth of top-left corner of plate
d1 = PLATE_W * np.sin(alpha) / np.sin(gamma) # depth of top-right corner of plate

# compute 3d coordinates of the plate corners in the camera reference frame
P0 = d0 * r0_dir # 3d point corresponding to top-left corner
P1 = d1 * r1_dir # 3d point corresponding to top-right corner

# compute unit x-axis of the camera frame as the direction from left to right plate corner
x_cam = P1 - P0
x_cam /= np.linalg.norm(x_cam)

# compute unit z-axis as perpendicular to x_cam and the estimated vehicle forward direction
z_cam = np.cross(x_cam, v_dir)
z_cam /=np.linalg.norm(z_cam)

# compute unit y-axis to complete the right-handed coordinate system
y_cam = np.cross(z_cam, x_cam)

# assemble rotation matrix with x, y, z axes as columns
R0 = np.column_stack((x_cam, y_cam, z_cam))

# convert rotation matrix to rotation vector (rodrigues format)
rvec0, _ = cv2.Rodrigues(R0)

# set initial translation vector as the 3d position of the top-left plate corner
tvec0 = P0.reshape(3, 1)

obj_full = np.array([
    [0, 0, 0], # top-left corner of the plate
    [PLATE_W, 0, 0], # top-right corner of the plate
    [0, -PLATE_H, 0], # bottom-left corner of the plate
    [PLATE_W, -PLATE_H, 0], # bottom-right corner of the plate
    [-0.340, 0.100, Z_FARO], # left rear light (relative to TL)
    [PLATE_W + 0.340, 0.100, Z_FARO], # right rear light (relative to TL)
    [-0.7+1.958, 0.150, -2.050 + 0.3] # optional: right side mirror (relative to TL)
], dtype=np.float32)

pix_full = np.array([
    [TL[0], TL[1]],  # P0 = plate TL
    [TR[0], TR[1]],  # P1 = plate TR
    [BL[0], BL[1]],  # P2 = plate BL
    [BR[0], BR[1]],  # P3 = plate BR
    [L2[0], L2[1]],  # P4 = rear-light L
    [R2[0], R2[1]],  # P5 = rear-light R
    [mirror_point[0], mirror_point[1]] # P6 = side mirror R - uncomment it if you want to use just taillights and license plate
], dtype=np.float32)

# undistort pixel coordinates using the camera calibration parameters
uv_full = cv2.undistortPoints(pix_full.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)

# estimate camera pose using the 3d model points and their corresponding 2d image projections
# use pnp with epnp algorithm ignoring initial guess
# success, rvec, tvec = cv2.solvePnP(
#     obj_full, uv_full, K, None,
#     rvec=rvec0, tvec=tvec0,
#     useExtrinsicGuess=False,
#     flags=cv2.SOLVEPNP_EPNP
# )

# use pnp with iterative algorithm using initial guess
success, rvec, tvec = cv2.solvePnP(
    obj_full, uv_full, K, None,
    rvec=rvec0, tvec=tvec0,
    useExtrinsicGuess=True, 
    flags=cv2.SOLVEPNP_ITERATIVE
)

# refine the initial pose estimate using the levenberg-marquardt optimization
rvec, tvec = cv2.solvePnPRefineLM(obj_full, uv_full, K, None, rvec, tvec)

CAR_W, CAR_H, CAR_L = 1.732, 1.467, 3.997 # car width, height, length in meters

# define the 8 corners of the 3d bounding box in the car coordinate frame
# origin is at the top-left corner of the license plate
bbox_3d = np.array([
    [-0.606, -0.9, 0.3], # back bottom-left
    [-0.606+CAR_W, -0.9, 0.3], # back bottom-right
    [-0.606+CAR_W, 1.467-0.9, 0.3], # back top-right
    [-0.606, 1.467-0.9, 0.3], # back top-lef

    [-0.606, -0.9, -CAR_L+0.3], # front bottom-left
    [-0.606+1.958, -0.9, -CAR_L+0.3], # front bottom-right
    [-0.606+CAR_W, 1.467-0.9, -CAR_L+0.3], # front top-right
    [-0.606, 1.467-0.9, -CAR_L+0.3] # front top-left
], dtype=np.float32)

# project the 3d bounding box points into the image using the estimated pose
box2d, _ = cv2.projectPoints(bbox_3d, rvec, tvec, K, None)
box2d = box2d.reshape(-1, 2).astype(int)

# draw the undistorted 2d landmarks used for pnp
for u, v in uv_full:
    cv2.circle(img_ud, (int(u), int(v)), 6, (0, 0, 255), 10)

# draw the rear face of the bounding box in green
cv2.polylines(img_ud, [box2d[:4]], True, (0, 255, 0), 5)

# draw the front face of the bounding box in blue
cv2.polylines(img_ud, [box2d[4:]], True, (255, 0, 0), 5)

# connect corresponding corners between rear and front faces with red vertical lines
for i in range(4):
    cv2.line(img_ud, tuple(box2d[i]), tuple(box2d[i+4]), (0, 0, 255), 5)

cv2.imshow("box", cv2.resize(img_ud, None, fx=0.35, fy=0.35))
cv2.imwrite(os.path.join(os.getcwd(), "method4", "results", "bbox_5pts_iterative.jpg"), img_ud)
cv2.waitKey(0)
cv2.destroyAllWindows()
