import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========= Funzioni Utili ========= #

def to_homogeneous(p):
    return np.array([p[0], p[1], 1.0])

def project_point(K, point_3D):
    point_3D = point_3D.reshape(3, 1)
    point_2D_homogeneous = K @ point_3D
    point_2D = point_2D_homogeneous[:2] / point_2D_homogeneous[2]
    return point_2D

def transform_and_project_points(CAD_points, center_back_car, R, K, color=(0, 0, 255), radius=10):
    projected_points = {}
    points_world = {}

    for name, p_local in CAD_points.items():
        p_world = center_back_car + (R @ p_local)
        points_world[name] = p_world

        point_2D = project_point(K, p_world)
        point_2D = np.round(point_2D).flatten()
        point_2D = tuple(int(x.item()) for x in point_2D)

        projected_points[name] = point_2D
        cv2.circle(img, point_2D, radius, color, -1)

    return projected_points, points_world

def draw_edges(img, projected_points, edges, color=(0, 255, 0), thickness=10):
    for start, end in edges:
        if start in projected_points and end in projected_points:
            cv2.line(img, projected_points[start], projected_points[end], color, thickness)

# ========= Input e Parametri ========= #

img = cv2.imread("OutputFolder/frame_02.png")

# K = np.array([
#     [3315.65306, 0.0, 1910.87753],
#     [0.0, 3319.89314, 1072.38815],
#     [0.0, 0.0, 1.0]
# ], dtype=np.float32)

K = np.array([
    [3.2014989e3, 0.0, 1.93982925e3],
    [0.0, 3.20637527e3, 1.06315413e3],
    [0.0, 0.0, 1]
], dtype=np.float64)

car_width = 1.732
car_length = 3.997
car_height = 1.467
taillight_height = 0.950

L1 = [318, 2212]
R1 = [1456, 2255]
Lint = [408, 2268]
Rint = [1256, 2304]

plateSxTop = [1336, 1504]
plateSxBottom = [1328, 1564]
plateDxTop = [1560, 1504]
plateDxBottom = [1560, 1556]

plateSxTop_h = to_homogeneous(plateSxTop)
plateSxBottom_h = to_homogeneous(plateSxBottom)
plateDxTop_h = to_homogeneous(plateDxTop)
plateDxBottom_h = to_homogeneous(plateDxBottom)

plateSxBottom_h = np.array(plateSxBottom_h, dtype=np.float32)
plateSxTop_h = np.array(plateSxTop_h, dtype=np.float32)
plateDxBottom_h = np.array(plateDxBottom_h, dtype=np.float32)
plateDxTop_h = np.array(plateDxTop_h, dtype=np.float32)

H = np.cross(plateSxBottom_h, plateSxTop_h)
W = np.cross(plateDxBottom_h, plateDxTop_h)

Vz = np.cross(H, W) 
Vz = Vz / Vz[2]

K_inv = np.linalg.inv(K)

L1_ray = K_inv @ to_homogeneous(L1)
R1_ray = K_inv @ to_homogeneous(R1)
L1_ray /= np.linalg.norm(L1_ray)
R1_ray /= np.linalg.norm(R1_ray)
Lint_ray = K_inv @ to_homogeneous(Lint)
Rint_ray = K_inv @ to_homogeneous(Rint)
Lint_ray /= np.linalg.norm(Lint_ray)
Rint_ray /= np.linalg.norm(Rint_ray)

phi2 = np.arccos(np.clip(np.dot(Lint_ray, Rint_ray), -1, 1))
d2 = car_width / (2 * np.sin(phi2 / 2))

B = d2 * Lint_ray
A = d2 * Rint_ray

phi = np.arccos(np.clip(np.dot(L1_ray, R1_ray), -1, 1))
d = car_width / (2 * np.sin(phi / 2))

F = d * L1_ray
E = d * R1_ray

camera_direction = -L1_ray
camera_center_world = F + camera_direction * d
print("Camera center (world coordinates):", camera_center_world)
print("Z coordinate of camera center:", camera_center_world[2])

center_back_car = (E + F) / 2
ray_points = {}
ray_points_front = {}
E[1] = 0
F[1] = 0
A[1] = 0
B[1] = 0
C = B.copy()
D = F.copy()
C[2] = A[2]
D[2] = E[2]
ray_points_front["C"] = C
ray_points_front["D"] = D
ray_points["E"] = E
ray_points["A"] = A
ray_points["B"] = B
ray_points["F"] = F

angle_deg = 14.5
rotation_angle_rad = np.deg2rad(angle_deg)
R = np.array([
    [np.cos(rotation_angle_rad), 0, np.sin(rotation_angle_rad)],
    [0, 1, 0],
    [-np.sin(rotation_angle_rad), 0, np.cos(rotation_angle_rad)]
])

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

# ========= Proiezione e Visualizzazione ========= #

# Proietta punti dei raggi posteriori e frontali
for name, point_3D in {**ray_points, **ray_points_front}.items():
    point_2D = project_point(K, point_3D)
    point_2D = np.round(point_2D).flatten()
    point_2D = tuple(int(x.item()) for x in point_2D)
    cv2.circle(img, point_2D, 30, (0, 255, 0) if name in ray_points_front else (0, 0, 255), -1)
    cv2.putText(img, name, (point_2D[0]+10, point_2D[1]), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)

# Proietta i punti dei raggi in 2D
projected_rays = {}
for name, p in {**ray_points, **ray_points_front}.items():
    pt_2D = project_point(K, p).flatten()
    pt_2D = tuple(map(int, np.round(pt_2D)))
    projected_rays[name] = pt_2D

# Disegna gli edge solo tra i punti proiettati
draw_edges(img, projected_rays, [
    ("E", "F"),
    ("A", "B"),
    ("A", "C"),
    ("C", "D"),
], color=(255, 0, 0), thickness=10)


# Proietta e disegna la bounding box
projected_points, bounding_box_world = transform_and_project_points(
    bounding_box, center_back_car, R, K, color=(0, 0, 255), radius=10
)

# Disegna edges della bounding box 3D
draw_edges(img, projected_points, [
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
])

# Mostra l'immagine risultante
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('3D Projection with Bounding Box')
plt.axis('off')
plt.savefig("OutputFolder/3D_projection.png", bbox_inches='tight')
plt.show()
