import numpy as np
import cv2
import matplotlib.pyplot as plt

# draws bounding box around the car given the 2d points
def draw_box(ax, pts2d, color='lime'):
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical
    ]
    for i, j in connections:
        x = [pts2d[i][0], pts2d[j][0]]
        y = [pts2d[i][1], pts2d[j][1]]
        ax.plot(x, y, color=color, linewidth=2)

# transforms 2d points to homogeneous coordinates
def to_homogeneous(p):
    return np.array([p[0], p[1], 1.0])

def to_line(p1, p2):
    return np.cross(p1, p2)

# CAD measurements
plate_width = 0.533
car_width = 1.732
car_length = 3.997
car_height = 1.467
taillight_height = 0.930


# Camera intrinsics matrix
K = np.array([
    [3315.65306, 0.0, 1910.87753],
    [0.0, 3319.89314, 1072.38815],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# Coordinate immagine: targa TL, TR e fanali L, R
pix = np.array([
    [422, 752],  # P0 = plate TL  (origine)
    [549, 743],  # P1 = plate TR
    [328, 733],  # P2 = rear-light L
    [618, 710]  # P3 = rear-light R
], dtype=np.float64)


P1, P2, L1, R1 = pix
P1h, P2h = to_homogeneous(P1), to_homogeneous(P2)
L1h, R1h = to_homogeneous(L1), to_homogeneous(R1)
p_line = to_line(P1h, P2h)
r_line = to_line(L1h, R1h)

v = np.cross(p_line, r_line)
v = v / v[2]

# Direzione 3D del punto di fuga
dir_x = np.linalg.inv(K) @ v
dir_x = dir_x / np.linalg.norm(dir_x)

# -----------------------------
# 2. Raggi visivi verso P0, P1
# -----------------------------
r0 = np.linalg.inv(K) @ P1h
r1 = np.linalg.inv(K) @ P2h
r0 = r0 / np.linalg.norm(r0)
r1 = r1 / np.linalg.norm(r1)


# -------------------------------
# 3. Risolvi triangolo in 3D
# -------------------------------

# Calcola angolo tra r0 e r1

theta = np.arccos(np.clip(np.dot(r0, r1), -1, 1))
d = plate_width / (2 * np.sin(theta / 2))


# Ricostruisci i punti 3D nel piano della targa
r0_3D = d * r0
r1_3D = d * r1

plane_normal = np.cross(r0_3D - r1_3D, dir_x)
plane_point = r0_3D

# Define 3D bounding box in local car coords (you can adjust these)

x_dir = (r1_3D - r0_3D)
x_dir = x_dir / np.linalg.norm(x_dir)

y_dir = np.cross(x_dir, plane_normal)
y_dir = y_dir / np.linalg.norm(y_dir)

z_dir = np.cross(x_dir, y_dir)
z_dir = z_dir / np.linalg.norm(z_dir)

# find the center between the rear lights using the 3d points
plate_center = (r0_3D + r1_3D) / 2
plate_ground = plate_center - z_dir * taillight_height

# Compute all 8 corners of 3D bounding box
# useful measures
w, l, h = car_width / 2, car_length, car_height
# corners
rbl = plate_ground - w * x_dir # rear bottom left
rbr = plate_ground + w * x_dir # rear bottom right
fbl = rbl + l * y_dir # front bottom left
fbr = rbr + l * y_dir # front bottom right
rtl = rbl + h * z_dir # rear top left
rtr = rbr + h * z_dir # rear top right
ftl = fbl + h * z_dir # front top left
ftr = fbr + h * z_dir # front top right

box_3D_points = [rbl, rbr, fbr, fbl, rtl, rtr, ftr, ftl]


# ========= Project points and draw box ========= #
# Project 3D box corners to 2D
corners_2D = [K @ pt for pt in box_3D_points]
corners_2D = [(pt[0]/pt[2], pt[1]/pt[2]) for pt in corners_2D]

# Draw 3D bounding box
# prepare plot with frame1
img = cv2.imread("dayImg.jpeg")
fig, ax = plt.subplots()
ax.imshow(img)

# Draw box edges
draw_box(ax, corners_2D)

center_2d = (K @ plate_center)
center_2d = center_2d / center_2d[2]
ax.plot(center_2d[0], center_2d[1], 'ro', label='Plate Center')

ground_2d = (K @ plate_ground)
ground_2d = ground_2d / ground_2d[2]
ax.plot(ground_2d[0], ground_2d[1], 'ro')




colors = ['cyan', 'cyan', 'orange', 'orange']
for i, (pt, color) in enumerate(zip(pix, colors)):
    ax.plot(pt[0], pt[1], 'o', color=color, markersize=3)

ax.axis('off')
plt.tight_layout()
plt.show()
