import cv2
import numpy as np
import matplotlib.pyplot as plt

def to_line(p1, p2):
    return np.cross(p1, p2)

def to_homogeneous(p):
    return np.array([p[0], p[1], 1.0])

def project_point(pt3D, K):
    proj = K @ pt3D
    proj /= proj[2]
    return proj[:2].astype(int)

def draw_box(image, pts, color=(0, 255, 0)):
    # Bottom rectangle
    cv2.line(image, pts[0], pts[1], color, 2)
    cv2.line(image, pts[1], pts[2], color, 2)
    cv2.line(image, pts[2], pts[3], color, 2)
    cv2.line(image, pts[3], pts[0], color, 2)
    # Top rectangle
    cv2.line(image, pts[4], pts[5], color, 2)
    cv2.line(image, pts[5], pts[6], color, 2)
    cv2.line(image, pts[6], pts[7], color, 2)
    cv2.line(image, pts[7], pts[4], color, 2)
    # Vertical edges
    for i in range(4):
        cv2.line(image, pts[i], pts[i + 4], color, 2)


def draw_box_matplotlib(ax, pts2d, color='lime'):
    # Connect box corners using matplotlib lines
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
    ]
    for i, j in connections:
        x = [pts2d[i][0], pts2d[j][0]]
        y = [pts2d[i][1], pts2d[j][1]]
        ax.plot(x, y, color=color, linewidth=2)

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
taillight_height = 0.833


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

# normal plane
# n is the normal vector of the plane π (the plane of the rear of the car).
n = K.T @ vanishing_line
n = n / np.linalg.norm(n)

# 3D direction vectors from the camera through the light positions in frame1
L1_ray = K_inv @ to_homogeneous(L1)
R1_ray = K_inv @ to_homogeneous(R1)

L1_ray /= np.linalg.norm(L1_ray)
R1_ray /= np.linalg.norm(R1_ray)

# find the angle between these rays
cos_theta = np.dot(L1_ray, R1_ray)
theta = np.arccos(np.clip(cos_theta, -1, 1))

# given car width which is known from the CAD model
d = car_width / (2 * np.sin(theta / 2))

# find the 3D points of the light positions (frame1)
L1_3D = d * L1_ray
R1_3D = d * R1_ray

# Use midpoint as rear center, shift to full car center
mid_ray = (L1_ray + R1_ray) / 2
mid_ray /= np.linalg.norm(mid_ray)


rear_center = (L1_3D + R1_3D) / 2


forward = dir_y

# Define local axes
right = (R1_ray - L1_ray)
right /= np.linalg.norm(right)

up = n

# move that point to the ground plane
rear_ground = rear_center + up * taillight_height

# 3D box corners from center
w = car_width / 2
l = car_length
h = car_height

rbl = rear_ground - w*right              # rear bottom left
rbr = rear_ground + w*right              # rear bottom right
fbl = rear_ground - w*right + l*forward  # front bottom left
fbr = rear_ground + w*right + l*forward  # front bottom right

rtl = rbl - h*up   # rear top left
rtr = rbr - h*up   # rear top right
ftl = fbl - h*up   # front top left
ftr = fbr - h*up   # front top right

box_3D_points = [rbl, rbr, fbr, fbl, rtl, rtr, ftr, ftl]

# Project to 2D
corners_2D = [K @ pt for pt in box_3D_points]
corners_2D = [(pt[0]/pt[2], pt[1]/pt[2]) for pt in corners_2D]


# Plot the bounding box
fig, ax = plt.subplots()
ax.imshow(frame1)

# Draw the box lines
def draw_line(p1, p2, color='lime'):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2)

for corner in corners_2D:
    ax.add_patch(plt.Circle(corner, 10, color='red'))


# Draw bottom edges of the car
draw_line(corners_2D[0], corners_2D[1])
draw_line(corners_2D[1], corners_2D[2])
draw_line(corners_2D[2], corners_2D[3])
draw_line(corners_2D[3], corners_2D[0])

# Draw top edges of the car
draw_line(corners_2D[4], corners_2D[5])
draw_line(corners_2D[5], corners_2D[6])
draw_line(corners_2D[6], corners_2D[7])
draw_line(corners_2D[7], corners_2D[4])

# Draw vertical edges (connecting the top and bottom faces)
draw_line(corners_2D[0], corners_2D[4])
draw_line(corners_2D[1], corners_2D[5])
draw_line(corners_2D[2], corners_2D[6])
draw_line(corners_2D[3], corners_2D[7])

ax.axis('off')
plt.tight_layout()
plt.show()

# ======= DRAW EVERYTHING =======
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

# draw bounding box
# draw_box(frame1, box_2D_points, color=(0, 255, 0))

# display
# cv2.imshow("Bounding Box", frame1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Prepare for plotting
#fig, ax = plt.subplots()
#frame_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
#ax.imshow(frame_rgb)

# Draw light points
#ax.add_patch(plt.Circle(L1, 10, color='green'))
#ax.add_patch(plt.Circle(R1, 10, color='green'))

# Draw lines
#ax.plot([L1[0], R1[0]], [L1[1], R1[1]], color='blue', linewidth=2)

# Draw 3D box
#draw_box_matplotlib(ax, box_2D_points, color='lime')

# Format
#ax.set_title("3D Car Bounding Box in Perspective")
#ax.axis('off')
#plt.tight_layout()
#plt.show()