import sys
import numpy as np
import cv2
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'featureExtraction')))
from extractLights2 import process_frames
from getMirror import mirror_frame

# ---------- 0. immagine e intrinseche ---------------------------------
img_path = os.path.join(os.getcwd(), "featureExtraction", "extractedFrames", "frame_11.png")
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError("frame non trovato")

K = np.array([
    [3.31565306e+03, 0.0, 1.91087753e+03],
    [0.0, 3.31989314e+03, 1.07238815e+03],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

dist = np.array([[2.65104301e-01, -1.78436004e+00,  2.42978100e-03,  1.18030874e-04, 3.77074289e+00]], dtype=np.float64)

img_ud = cv2.undistort(img, K, dist)
# ========= Inputs ========= #
# Input points (taillights)
L1, R1, L2, R2, TL, TR, BL, BR = process_frames("featureExtraction/extractedFrames/frame_02.png", "featureExtraction/extractedFrames/frame_11.png")
mirror_point = mirror_frame("featureExtraction/extractedFrames/frame_11.png")

print("Pixel rilevati:")
print("Frame 1 - Sinistra:", L1)
print("Frame 1 - Destra:", R1)
print("Frame 2 - Sinistra:", L2)
print("Frame 2 - Destra:", R2)

lights1 = [L1, R1]
lights2 = [L2, R2]
lights_top = [TL, TR]

# ---------- 1. pixel (undistorti) dei 4 punti sullo stesso piano ------
pix = np.array([
    [TL[0], TL[1]],  # P0 = plate TL
    [TR[0], TR[1]],  # P1 = plate TR
    [L2[0], L2[1]],  # P2 = rear-light L
    [R2[0], R2[1]],  # P3 = rear-light R
], dtype=np.float64)

uv = cv2.undistortPoints(pix.reshape(-1,1,2), K, dist, P=K).reshape(-1,2)

p0, p1, p2, p3 = [np.append(u,1.) for u in uv]

# ---------- 2. punto di fuga V delle due rette parallele --------------
l01 = np.cross(p0, p1) # retta TL-TR
l23 = np.cross(p2, p3) # retta L-R
Vh  = np.cross(l01, l23)
Vh /= Vh[2] # [vx, vy, 1]

# ---------- 3. direzioni 3-D ------------------------------------------
Kinv = np.linalg.inv(K)
v_dir = Kinv.dot(Vh)
v_dir /= np.linalg.norm(v_dir)
r0_dir = Kinv.dot(p0)
r0_dir /= np.linalg.norm(r0_dir)
r1_dir = Kinv.dot(p1)
r1_dir /= np.linalg.norm(r1_dir)

def ang(u,v):
    return np.arccos(np.clip(u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0,1.0))

alpha = ang(r0_dir, v_dir)
beta = ang(r1_dir, v_dir)
gamma = ang(r0_dir, r1_dir)

# ---------- 4. triangolazione (legge dei seni) ------------------------
PLATE_W = 0.520
PLATE_H = 0.130
Z_FARO = 0.150

d0 = PLATE_W * np.sin(beta) / np.sin(gamma) # distanza camera → TL
d1 = PLATE_W * np.sin(alpha) / np.sin(gamma) # distanza camera → TR
P0 = d0 * r0_dir # 3-D camera-frame di TL
P1 = d1 * r1_dir # 3-D camera-frame di TR

# ---------- 5. assi del piano e posa iniziale -------------------------
x_cam = P1 - P0
x_cam /= np.linalg.norm(x_cam)
z_cam = np.cross(x_cam, v_dir)
z_cam /=np.linalg.norm(z_cam)
y_cam = np.cross(z_cam, x_cam)

R0 = np.column_stack((x_cam, y_cam, z_cam))
rvec0, _ = cv2.Rodrigues(R0)
tvec0 = P0.reshape(3, 1) # origine = plate TL

# ---------- 7. PnP refinement con tutti i punti -----------------------
obj_full = np.array([
    [0, 0, 0], # TL
    [PLATE_W, 0, 0], # TR
    [0, -PLATE_H, 0], # BL
    [PLATE_W, -PLATE_H, 0], # BR
    [-0.340, 0.100, Z_FARO], # faro L
    [PLATE_W + 0.340, 0.100, Z_FARO], # faro R
    [-0.7+1.958, 0.150, -2.050 + 0.3] # side mirror R - uncomment it if you want to use just taillights and license plate
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

# pixel matching obj_full
uv_full = cv2.undistortPoints(pix_full.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)

# success, rvec, tvec = cv2.solvePnP(obj_full, uv_full, K, None, rvec=rvec0, tvec=tvec0, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
success, rvec, tvec = cv2.solvePnP(obj_full, uv_full, K, None, rvec=rvec0, tvec=tvec0, useExtrinsicGuess=False, flags=cv2.SOLVEPNP_EPNP)
rvec, tvec = cv2.solvePnPRefineLM(obj_full, uv_full, K, None, rvec, tvec)

# ---------- 8. bounding-box nel frame plate-TL ------------------------
CAR_W, CAR_H, CAR_L = 1.732, 1.467, 3.997 # m
bbox_3d = np.array([
    [-0.606, -0.9, 0.3], # BL
    [-0.606+CAR_W, -0.9, 0.3], # BR
    [-0.606+CAR_W, 1.467-0.9, 0.3], # TR
    [-0.606, 1.467-0.9, 0.3], # TL

    [-0.606, -0.9, -CAR_L+0.3], # front BL
    [-0.606+1.958, -0.9, -CAR_L+0.3], # front BR
    [-0.606+CAR_W, 1.467-0.9, -CAR_L+0.3], # front TR
    [-0.606, 1.467-0.9, -CAR_L+0.3] # front TL
], dtype=np.float32)

box2d, _ = cv2.projectPoints(bbox_3d, rvec, tvec, K, None)
box2d = box2d.reshape(-1, 2).astype(int)

# draw the points on the image
for u, v in uv_full:
    cv2.circle(img_ud, (int(u), int(v)), 6, (0, 0, 255), 10)


cv2.polylines(img_ud, [box2d[:4]], True, (0, 255, 0), 5)
cv2.polylines(img_ud, [box2d[4:]], True, (255, 0, 0), 5)
for i in range(4):
    cv2.line(img_ud, tuple(box2d[i]), tuple(box2d[i+4]), (0, 0, 255), 5)

cv2.imshow("box", cv2.resize(img_ud, None, fx=0.35, fy=0.35))
# cv2.imwrite(os.path.join(os.getcwd(), "method4", "results", "result.png"), img_ud)
cv2.waitKey(0)
cv2.destroyAllWindows()
