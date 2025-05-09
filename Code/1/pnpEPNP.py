import cv2
import numpy as np

# ---------- 0. immagine e intrinseche ---------------------------------
img = cv2.imread("sunny/frame_02.png")
if img is None:
    raise FileNotFoundError("frame non trovato")

K = np.array([
    [3.2014989e3, 0.0, 1.93982925e3],
    [0.0, 3.20637527e3, 1.06315413e3],
    [0.0, 0.0, 1]
], dtype=np.float64)

dist = np.array([[2.4377e-01, -1.5955e+00, -1.1528e-03, 4.1986e-04, 3.5668e+00]], dtype=np.float64)

img_ud = cv2.undistort(img, K, dist)

# ---------- 4. triangolazione (legge dei seni) ------------------------
PLATE_W = 0.520
PLATE_H = 0.130
Z_FARO = 0.150

# ---------- 7. PnP refinement con tutti i punti -----------------------
obj_full = np.array([
    [0, 0, 0], # TL
    [PLATE_W, 0, 0], # TR
    [0, -PLATE_H, 0], # BL
    [PLATE_W, -PLATE_H, 0], # BR
    [-0.340, 0.100, Z_FARO], # faro L
    [PLATE_W + 0.340, 0.100, Z_FARO], # faro R
    [-0.7, 0.150, -2.050 + 0.3], # speechietto L
    # [-0.220, 0.460, +0.350] # tetto S
], dtype=np.float64)

pix_full = np.array([
    [1020, 1804], # P0 = plate TL  (origine)
    [1324, 1780], # P1 = plate TR
    [1032, 1884], # P2 = plate BL
    [1328, 1852], # P3 = plate BR
    [792, 1768], # P4 = rear-light L 
    [1484, 1708], # P5 = rear-light R
    [336, 1644], # specchietto L
    # [740, 1500] # tetto S
], dtype=np.float64)

# pixel matching obj_full
uv_full = cv2.undistortPoints(pix_full.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)

success, rvec, tvec = cv2.solvePnP(obj_full, uv_full, K, None, useExtrinsicGuess=False, flags=cv2.SOLVEPNP_EPNP)
rvec, tvec = cv2.solvePnPRefineLM(obj_full, uv_full, K, None, rvec, tvec)

# ---------- 8. bounding-box nel frame plate-TL ------------------------
CAR_W, CAR_H, CAR_L = 1.732, 1.467, 3.997 # m
bbox_3d = np.array([
    [-0.606, -0.9, 0.3], # BL
    [-0.606+CAR_W, -0.9, 0.3], # BR
    # [-0.606+0.386+0.960, 1.467-0.9, 0.3], # TR
    # [-0.606+0.386, 1.467-0.9, 0.3], # TL
    [-0.606+CAR_W, 1.467-0.9, 0.3], # TR
    [-0.606, 1.467-0.9, 0.3], # TL

    [-0.606, -0.9, -CAR_L+0.3], # front BL
    [-0.606+1.958, -0.9, -CAR_L+0.3], # front BR
    [-0.606+CAR_W, 1.467-0.9, -CAR_L+0.3], # front TR
    [-0.606, 1.467-0.9, -CAR_L+0.3] # front TL
], dtype=np.float64)

box2d, _ = cv2.projectPoints(bbox_3d, rvec, tvec, K, None)
box2d = box2d.reshape(-1, 2).astype(int)

cv2.polylines(img_ud, [box2d[:4]], True, (0, 255, 0), 5)
cv2.polylines(img_ud, [box2d[4:]], True, (255, 0, 0), 5)
for i in range(4):
    cv2.line(img_ud, tuple(box2d[i]), tuple(box2d[i+4]), (0, 0, 255), 5)

cv2.imshow("box", cv2.resize(img_ud, None, fx=0.35, fy=0.35))
cv2.waitKey(0)
cv2.destroyAllWindows()
