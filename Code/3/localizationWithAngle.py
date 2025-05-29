import cv2
import numpy as np
import matplotlib.pyplot as plt

# transforms 2d points to homogeneous coordinates
def to_homogeneous(p):
    return np.array([p[0], p[1], 1.0])

def project_point(K, point_3D):
    point_3D = point_3D.reshape(3, 1)  # Colonna
    point_2D_homogeneous = K @ point_3D  # Matrice intrinseca
    point_2D = point_2D_homogeneous[:2] / point_2D_homogeneous[2]  # Divide per z
    return point_2D

img = cv2.imread("OutputFolder/frame_02.png")

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
taillight_height = 0.950

# ========= Inputs ========= #
# Input points (taillights)
L1 = [318, 2212]
R1 = [1456, 2255]
Lint = [408, 2268]
Rint = [1256, 2304]

plateSxTop = [1336, 1504]
plateSxBottom = [1328, 1564]
plateDxTop = [1560, 1504]
plateDxBottom = [1560, 1556]

# Convert points to homogeneous coordinates
plateSxTop_h = to_homogeneous(plateSxTop)
plateSxBottom_h = to_homogeneous(plateSxBottom)
plateDxTop_h = to_homogeneous(plateDxTop)
plateDxBottom_h = to_homogeneous(plateDxBottom)

# convert to numpy array
plateSxBottom_h = np.array(plateSxBottom_h, dtype=np.float32)
plateSxTop_h = np.array(plateSxTop_h, dtype=np.float32)
plateDxBottom_h = np.array(plateDxBottom_h, dtype=np.float32)
plateDxTop_h = np.array(plateDxTop_h, dtype=np.float32)

H = np.cross(plateSxBottom_h, plateSxTop_h)
W = np.cross(plateDxBottom_h, plateDxTop_h)

Vz = np.cross(H, K)
Vz = Vz / Vz[2]

K_inv = np.linalg.inv(K)


# Project taillights into 3D rays
L1_ray = K_inv @ to_homogeneous(L1)
R1_ray = K_inv @ to_homogeneous(R1)
L1_ray /= np.linalg.norm(L1_ray)
R1_ray /= np.linalg.norm(R1_ray)
Lint_ray = K_inv @ to_homogeneous(Lint)
Rint_ray = K_inv @ to_homogeneous(Rint)
Lint_ray /= np.linalg.norm(Lint_ray)
Rint_ray /= np.linalg.norm(Rint_ray)

# Compute the angle between the rays
theta2 = np.arccos(np.clip(np.dot(Lint_ray, Rint_ray), -1, 1))
d2 = car_width / (2 * np.sin(theta2 / 2))

# Coordinate 3D dei taillights
B = d2 * Lint_ray
A = d2 * Rint_ray
# Angle between rays to get scale
# computes angle between the rays pointing to the taillights
# to then find the distance between the back of the car and the camera
# the distance is computed by exploiting trigonometric geometry
theta = np.arccos(np.clip(np.dot(L1_ray, R1_ray), -1, 1))
d = car_width / (2 * np.sin(theta / 2))

# Coordinate 3D dei taillights
F = d * L1_ray
E = d * R1_ray

camera_direction = -L1_ray  # Opposto al raggio verso L1

camera_center_world = F + camera_direction * d  # Distanza "d" è stimata
print("Camera center (world coordinates):", camera_center_world)
print("Z coordinate of camera center:", camera_center_world[2])

# Centro tra i taillights
center_back_car = (E + F) / 2
ray_points = {}
ray_points_front = {}
E[1] = 0  # Imposta l'altezza del taillight sinistro
F[1] = 0  # Imposta l'altezza del taillight destro
A[1] = 0  # Imposta l'altezza del taillight sinistro
B[1] = 0  # Imposta l'altezza del taillight destro
C = B.copy()
D = F.copy()
C[2] = A[2]  # Imposta la profondità del taillight sinistro
D[2] = E[2]  # Imposta la profondità del taillight sinistro
ray_points_front["C"] = C
ray_points_front["D"] = D
ray_points["E"] = E
ray_points["A"] = A
ray_points["B"] = B
ray_points["F"] = F

v1 = A - B
v2 = A - C

v1_flat = np.array([v1[0], v1[2]])  # (X, Z)
v2_flat = np.array([v2[0], v2[2]])

dot = np.dot(v1_flat, v2_flat)
norms = np.linalg.norm(v1_flat) * np.linalg.norm(v2_flat)
cos_theta = np.clip(dot / norms, -1.0, 1.0)
angle_rad = np.arccos(cos_theta)
angle_deg = np.degrees(angle_rad)
angle_deg = 14.5
print("Angle between L1 and R1 rays:", angle_deg)

# Angolo di rotazione (camera rispetto auto)
rotation_angle_rad = np.deg2rad(angle_deg)

# Matrice di rotazione attorno all'asse y (verticale)
R = np.array([
    [np.cos(rotation_angle_rad), 0, np.sin(rotation_angle_rad)],
    [0, 1, 0],
    [-np.sin(rotation_angle_rad), 0, np.cos(rotation_angle_rad)]
])

# ========== Step 2: Definisci punti locali sull'auto ========== #

# Punti 3D locali rispetto al centro dietro l'auto
# Es: targa centrata a 0.4 m sopra i taillights
points_local = {
    "plate_center": np.array([0.0, 0.0, 0.0]),
    "ground_center": np.array([0.0, taillight_height, 0.0]),
    "left_mirror": np.array([-car_width/2, -0.2, car_length]),
    "right_mirror": np.array([car_width/2, -0.2, car_length]),
}

bounding_box = {
    "rear_bottom_left": np.array([-car_width/2, taillight_height, 0]),
    "rear_bottom_right": np.array([car_width/2, taillight_height, 0]),
    "front_bottom_left": np.array([-car_width/2, taillight_height, car_length]),
    "front_bottom_right": np.array([car_width/2, taillight_height, car_length]),
    "rear_top_left": np.array([-car_width/2, -taillight_height, 0]),
    "rear_top_right": np.array([car_width/2, -taillight_height, 0]),
    "front_top_left": np.array([-car_width/2, -taillight_height, car_length]),
    "front_top_right": np.array([car_width/2, -taillight_height, car_length]),
}



# ========== Step 3: Trasforma in mondo 3D ========== #


# Proietta e disegna la bounding box
projected_ray_points = {}
for name, point_3D in ray_points.items():
    point_2D = project_point(K, point_3D)
    point_2D = np.round(point_2D).flatten()
    point_2D = tuple(int(x.item()) for x in point_2D)

    projected_ray_points[name] = point_2D
    
    # Disegna il punto sull'immagine
    cv2.circle(img, point_2D, 30, (0, 0, 255), -1)
    cv2.putText(img, name, (point_2D[0]+10, point_2D[1]), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3)

projected_ray_points_front = {}
for name, point_3D in ray_points_front.items():
    point_2D = project_point(K, point_3D)
    point_2D = np.round(point_2D).flatten()
    point_2D = tuple(int(x.item()) for x in point_2D)

    projected_ray_points[name] = point_2D
    
    # Disegna il punto sull'immagine
    cv2.circle(img, point_2D, 30, (0, 255, 0), -1)
    cv2.putText(img, name, (point_2D[0]+10, point_2D[1]), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3)


edges = [
    ("E", "F"),
    ("A", "B"),
]

edges2 = [
    ("A", "C"),
    ("C", "D"),
]

# Step 3: Draw lines between connected points
for start, end in edges:
    if start in projected_ray_points and end in projected_ray_points:
        cv2.line(img, projected_ray_points[start], projected_ray_points[end], (255, 0, 0), 10)

# Step 4: Draw lines between connected points for the second set of edges
for start, end in edges2:
    if start in projected_ray_points_front and end in projected_ray_points_front:
        cv2.line(img, projected_ray_points_front[start], projected_ray_points_front[end], (255, 0, 0), 10)

points_world = {}
for name, p_local in points_local.items():
    p_world = center_back_car + (R @ p_local)  # prima ruota, poi trasla
    points_world[name] = p_world

bounding_box_world = {}
for name, p_local in bounding_box.items():
    p_world = center_back_car + (R @ p_local)  # prima ruota, poi trasla
    bounding_box_world[name] = p_world


# Proietta e disegna la bounding box
projected_points = {}
for name, point_3D in bounding_box_world.items():
    point_2D = project_point(K, point_3D)
    point_2D = np.round(point_2D).flatten()
    point_2D = tuple(int(x.item()) for x in point_2D)

    projected_points[name] = point_2D
    
    # Disegna il punto sull'immagine
    cv2.circle(img, point_2D, 10, (0, 0, 255), -1)

# Step 2: Define edges between points for the 3D bounding box
edges = [
    ("rear_bottom_left", "rear_bottom_right"),
    ("rear_bottom_right", "front_bottom_right"),
    ("front_bottom_right", "front_bottom_left"),
    ("front_bottom_left", "rear_bottom_left"),

    ("rear_top_left", "rear_top_right"),
    ("rear_top_right", "front_top_right"),
    ("front_top_right", "front_top_left"),
    ("front_top_left", "rear_top_left"),

    ("rear_bottom_left", "rear_top_left"),
    ("rear_bottom_right", "rear_top_right"),
    ("front_bottom_right", "front_top_right"),
    ("front_bottom_left", "front_top_left")
]

# Step 3: Draw lines between connected points
for start, end in edges:
    if start in projected_points and end in projected_points:
        cv2.line(img, projected_points[start], projected_points[end], (0, 255, 0), 10)

# Mostra immagine
plt.figure(figsize=(12,8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Theta = 14.5°')
plt.axis('off')
plt.savefig("OutputFolder/3D_projection.png", bbox_inches='tight')
plt.show()