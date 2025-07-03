import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'featureExtraction')))
from extractLights2 import process_frames


# ========= Useful functions ========= #
# returns line going through two points
def to_line(p1, p2):
    return np.cross(p1, p2)

# transforms 2d points to homogeneous coordinates
def to_homogeneous(p):
    return np.array([p[0], p[1], 1.0])

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



# Load image frames
frame1 = cv2.imread(os.path.join(os.getcwd(), "featureExtraction", "extractedFrames", "frame_02.png"))
frame2 = cv2.imread(os.path.join(os.getcwd(), "featureExtraction", "extractedFrames", "frame_11.png"))
# frame1 = cv2.imread("OutputFolder/frame_02.png")
# frame2 = cv2.imread("OutputFolder/frame_11.png")

# ========= Inputs ========= #
# Input points (taillights)
L1, R1, L2, R2, _, _, _, _ = process_frames("featureExtraction/extractedFrames/frame_02.png", "featureExtraction/extractedFrames/frame_11.png")

print("Pixel rilevati:")
print("Frame 1 - Sinistra:", L1)
print("Frame 1 - Destra:", R1)
print("Frame 2 - Sinistra:", L2)
print("Frame 2 - Destra:", R2)

lights1 = [L1, R1]
lights2 = [L2, R2]

# Camera intrinsics matrix
K = np.array([
    [3315.65306, 0.0, 1910.87753],
    [0.0, 3319.89314, 1072.38815],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# Car dimensions from CAD model (in meters)
car_width = 1.732
car_length = 3.997
car_height = 1.467
taillight_height = 0.812


# ========= Geometry computations ========= #
# Convert points to homogeneous coordinates
L1h, L2h = to_homogeneous(L1), to_homogeneous(L2)
R1h, R2h = to_homogeneous(R1), to_homogeneous(R2)

# Lines connecting Ls and Rs + vanishing point
L_line = to_line(L1h, L2h)
R_line = to_line(R1h, R2h)
Vy = np.cross(L_line, R_line)

# Lines connecting taillights for both frames + vanishing point
line1 = to_line(L1h, R1h)
line2 = to_line(L2h, R2h)
Vx = np.cross(line1, line2)

# Normalize vanishing points
Vy = Vy / Vy[2]
Vx = Vx / Vx[2]

# Get 3D direction rays using camera intrinsics
K_inv = np.linalg.inv(K)
dir_x = K_inv @ Vx
dir_y = K_inv @ Vy
dir_x /= np.linalg.norm(dir_x)
dir_y /= np.linalg.norm(dir_y)

# Check if motion is straight or turning
dot_product = np.dot(dir_x, dir_y)
is_translating = np.isclose(dot_product, 0, atol=1e-1)
print("Car is translating" if is_translating else "Car is steering")

# Compute vanishing line and normal vector to rear plane
vanishing_line = np.cross(Vx, Vy)
vanishing_line = vanishing_line / np.linalg.norm(vanishing_line[:2])
n = K.T @ vanishing_line
n = n / np.linalg.norm(n)

# Project taillights into 3D rays
L1_ray = K_inv @ to_homogeneous(L1)
R1_ray = K_inv @ to_homogeneous(R1)
L1_ray /= np.linalg.norm(L1_ray)
R1_ray /= np.linalg.norm(R1_ray)

# Angle between rays to get scale
# computes angle between the rays pointing to the taillights
# to then find the distance between the back of the car and the camera
# the distance is computed by exploiting trigonometric geometry
theta = np.arccos(np.clip(np.dot(L1_ray, R1_ray), -1, 1))
d = car_width / (2 * np.sin(theta / 2))

# 3D coordinates of taillights
L1_3D = d * L1_ray
R1_3D = d * R1_ray

# find center between the two lights and move it to the ground (given CAD measurements)
rear_center = (L1_3D + R1_3D) / 2
rear_ground = rear_center + n * taillight_height

# Define local car axes to draw the box
forward = dir_y

right = (R1_ray - L1_ray)
right /= np.linalg.norm(right)

up = n

# Compute all 8 corners of 3D bounding box
# useful measures
w, l, h = car_width / 2, car_length, car_height
# corners
rbl = rear_ground - w * right # rear bottom left
rbr = rear_ground + w * right # rear bottom right
fbl = rbl + l * forward # front bottom left
fbr = rbr + l * forward # front bottom right
rtl = rbl - h * up # rear top left
rtr = rbr - h * up # rear top right
ftl = fbl - h * up # front top left
ftr = fbr - h * up # front top right

box_3D_points = [rbl, rbr, fbr, fbl, rtl, rtr, ftr, ftl]


# ========= Draw geometric results ========= #
# draw segments, vanishing points and vanishing line
fig, ax = plt.subplots()
ax.imshow(frame1)

# Draw light points
for pt in lights1 + lights2:
    ax.add_patch(plt.Circle(pt, 10, color='green'))

# Draw light bars (blue), vertical light connections (green)
L1, R1 = lights1
L2, R2 = lights2
ax.plot([L1[0], R1[0]], [L1[1], R1[1]], color='blue', linewidth=2)
ax.plot([L2[0], R2[0]], [L2[1], R2[1]], color='blue', linewidth=2)
ax.plot([L1[0], L2[0]], [L1[1], L2[1]], color='green', linewidth=2)
ax.plot([R1[0], R2[0]], [R1[1], R2[1]], color='green', linewidth=2)

# Draw vanishing points and line
Vx_2d = np.array([Vx[0], Vx[1]]).astype(int)
Vy_2d = np.array([Vy[0], Vy[1]]).astype(int)
ax.add_patch(plt.Circle(Vx_2d, 10, color='green'))
ax.add_patch(plt.Circle(Vy_2d, 10, color='green'))
ax.plot([Vx_2d[0], Vy_2d[0]], [Vx_2d[1], Vy_2d[1]], color='red', linewidth=2)

ax.axis('off')
plt.tight_layout()
plt.show()


# ========= Project points and draw box ========= #
# Project 3D box corners to 2D
corners_2D = [K @ pt for pt in box_3D_points]
corners_2D = [(pt[0]/pt[2], pt[1]/pt[2]) for pt in corners_2D]

# Draw 3D bounding box
# prepare plot with frame1
fig, ax = plt.subplots()
ax.imshow(frame1)

# Draw box edges
draw_box(ax, corners_2D)
# save the image with bounding box
output_path = os.path.join(os.getcwd(), "method2", "results", "bbox.png")
plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
ax.axis('off')
plt.tight_layout()
plt.show()
