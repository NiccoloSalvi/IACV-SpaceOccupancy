import cv2
import numpy as np
import matplotlib.pyplot as plt

def to_line(p1, p2):
    return np.cross(p1, p2)

def to_homogeneous(p):
    return np.array([p[0], p[1], 1.0])

L1 = [318, 2212]
R1 = [1452, 2240]
L2 = [1211, 1464]
R2 = [1839, 1455]

lights1 = [L1, R1]
lights2 = [L2, R2]

frame1 = cv2.imread("OutputFolder/frame_02.png")
frame2 = cv2.imread("OutputFolder/frame_11.png")


# Camera intrinsics
K = np.array([
    [3298, 0, 1908],
    [0, 3302, 1071],
    [0, 0, 1]
], dtype=np.float64)


car_width = 1.732
car_length = 3.997
car_height = 1.467


L1 = to_homogeneous(L1)
L2 = to_homogeneous(L2)

R1 = to_homogeneous(R1)
R2 = to_homogeneous(R2)

L_line = to_line(L1, L2)

R_line = to_line(R1, R2)

# first vanishing point
Vy = np.cross(L_line, R_line)

line1 = to_line(L1, R1)
line2 = to_line(L2, R2)

# second vanishing point
Vx = np.cross(line1, line2)

Vy = Vy / Vy[2]
Vx = Vx / Vx[2]

# inverse of the camera calibration matrix
K_inv = np.linalg.inv(K)

# get 3D rays from camera center through vanishing points
dir_x = K_inv @ Vx
dir_y = K_inv @ Vy

dir_x = dir_x / np.linalg.norm(dir_x)
dir_y = dir_y / np.linalg.norm(dir_y)

# check orthogonality
print(np.dot(dir_x, dir_y))
is_translating = np.isclose(np.dot(dir_x, dir_y), 0, atol=1e-1)

if is_translating:
    print("Car is translating")
else:
    print("Car is steering")

# find vanishing line and normalize it
vanishing_line = np.cross(Vx, Vy)
vanishing_line = vanishing_line / np.linalg.norm(vanishing_line[:2])

# plane normal
# Plane is parallel to the backprojection of l: [K, 0]^T l.  However, we only need 'l' for the plane equation.
# The plane equation is l[0]*X + l[1]*Y + l[2]*Z + d = 0. The normal vector is (l[0], l[1], l[2]).
plane_normal = vanishing_line[:3]

# back-project the vanishing line using K.T
plane_normal_3D = K.T @ vanishing_line
plane_normal_3D = plane_normal_3D / np.linalg.norm(plane_normal_3D)


# Show image with matplotlib
fig, ax = plt.subplots()
ax.imshow(frame1)

# Plot light points
for pt in lights1:
    ax.add_patch(plt.Circle(pt, 10, color='green'))

for pt in lights2:
    ax.add_patch(plt.Circle(pt, 10, color='green'))

# Unpack light points
L1, R1 = lights1[0], lights1[1]
L2, R2 = lights2[0], lights2[1]

# Draw blue lines between L and R pairs
ax.plot([L1[0], R1[0]], [L1[1], R1[1]], color='blue', linewidth=2)
ax.plot([L2[0], R2[0]], [L2[1], R2[1]], color='blue', linewidth=2)

# Draw green lines between L1-L2 and R1-R2
ax.plot([L1[0], L2[0]], [L1[1], L2[1]], color='green', linewidth=2)
ax.plot([R1[0], R2[0]], [R1[1], R2[1]], color='green', linewidth=2)

# Draw vanishing points
Vx_2d = np.array([Vx[0], Vx[1]]).astype(int)
Vy_2d = np.array([Vy[0], Vy[1]]).astype(int)
ax.add_patch(plt.Circle(Vx_2d, 10, color='green'))
ax.add_patch(plt.Circle(Vy_2d, 10, color='green'))

# Draw vanishing line in red
ax.plot([Vx_2d[0], Vy_2d[0]], [Vx_2d[1], Vy_2d[1]], color='red', linewidth=2)

# Remove axes and show
ax.axis('off')
plt.tight_layout()
plt.show()

