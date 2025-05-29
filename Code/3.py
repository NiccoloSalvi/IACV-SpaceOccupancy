import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os

def draw_points(img, A, B, E, F, C, D):
    cv.circle(img, tuple(A), 5, (0, 255, 0), 10)
    cv.putText(img, "A", (A[0] + 10, A[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
    cv.circle(img, tuple(B), 5, (0, 255, 0), 10)
    cv.putText(img, "B", (B[0] + 10, B[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
    cv.circle(img, tuple(E), 5, (0, 255, 0), 10)
    cv.putText(img, "E", (E[0] + 10, E[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
    cv.circle(img, tuple(F), 5, (0, 255, 0), 10)
    cv.putText(img, "F", (F[0] + 10, F[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
    cv.circle(img, tuple(C), 5, (255, 255, 0), 10)
    cv.putText(img, "C", (C[0] + 10, C[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3)
    cv.circle(img, tuple(D), 5, (255, 255, 0), 10)
    cv.putText(img, "D", (D[0] + 10, D[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3)

def triangulate_pts(A_h, B_h, AB_distance, u_AB):
    """
    Triangulate 3D points A and B from their image projections and known real-world separation.

    Args:
        K           : (3x3) camera intrinsic matrix
        A_h        : homogeneus coordinates of A' (left rear-light)
        B_h         : homogeneus coordinates of B' (right rear-light)
        AB_distance : real distance between A and B (scalar)
        u_AB        : unit 3D direction vector of AB (from Vx back-projection)

    Returns:
        A_3d, B_3d : 3D coordinates of points A and B in camera frame
    """
    # Back-project to normalized camera rays
    ray_A = K_inv @ A_h
    ray_B = K_inv @ B_h
    ray_A = ray_A / np.linalg.norm(ray_A)
    ray_B = ray_B / np.linalg.norm(ray_B)

    # Solve t*ray_A - s*ray_B = AB_distance * u_AB
    M = np.column_stack((ray_A, -ray_B))    # 3×2 system matrix
    b = AB_distance * u_AB                 # right-hand side

    # Solve in least-squares sense using pseudoinverse to avoid dimension mismatch
    ts = np.linalg.pinv(M) @ b            # shape (2,)
    t, s = ts

    # Compute 3D positions
    A_3d = t * ray_A
    B_3d = s * ray_B

    return A_3d, B_3d

def compute_camera_vertical(n_back, u_dir, phi_rad):
    """
    Compute the camera vertical direction vector from the back-plane normal and yaw axis u_dir.

    Args:
        n_back : array-like shape (3,) normal of the back plane in camera coords
        u_dir  : array-like shape (3,) unit direction vector of AB axis
        phi_rad: float, angle between back plane and vertical in radians

    Returns:
        v_cam  : array shape (3,) unit vector pointing upward in camera frame
    """
    # Ensure arrays
    n_b = np.asarray(n_back).reshape(3,)
    u = np.asarray(u_dir).reshape(3,)

    # Rodrigues rotation of n_b around axis u by angle phi
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)
    # Compute cross and dot
    cross_term = np.cross(u, n_b)
    dot_term = np.dot(u, n_b)

    v_cam = (cos_phi * n_b +
             sin_phi * cross_term +
             (1 - cos_phi) * dot_term * u)
    # Normalize to unit
    v_cam /= np.linalg.norm(v_cam)
    return v_cam

def project_to_horizon(point_img, Vz_h, l_inf):
    """
    Proietta il punto immagine sulla linea dell'orizzonte
    point_img: [x, y, 1] punto nell'immagine
    Vz_h: vanishing point verticale
    l_inf: linea dell'orizzonte
    """
    # Linea che connette Vz al punto (direzione verticale attraverso il punto)
    vertical_line = np.cross(Vz_h, point_img)
    
    # Intersezione con l'orizzonte
    point_on_horizon = np.cross(vertical_line, l_inf)
    
    # Normalizza
    if abs(point_on_horizon[2]) > 1e-6:
        point_on_horizon = point_on_horizon / point_on_horizon[2]
    
    return point_on_horizon

def compute_theta(A_h, B_h, Vz_h, l_inf):
    """
    Calcola l'imbardata theta dal punto immagine A' e B'.
    
    Args:
        A_h   : array-like shape (3,) = [u_A, v_A, 1]
        B_h   : array-like shape (3,) = [u_B, v_B, 1]
        Vz_h  : array-like shape (3,) = punto di fuga verticale in omogenee
        l_inf : array-like shape (3,) = linea all'infinito (horizon line) in omogenee

    Returns:
        theta_rad : imbardata in radianti
    """
    # Proietta A' e B' sull'orizzonte
    Ah2 = project_to_horizon(A_h, Vz_h, l_inf)
    Bh2 = project_to_horizon(B_h, Vz_h, l_inf)
    
    # Normalizza in pixel
    A2 = Ah2[:2] / Ah2[2]
    B2 = Bh2[:2] / Bh2[2]
    
    # Calcola delta
    dx = B2[0] - A2[0]
    dy = B2[1] - A2[1]
    
    # Yaw = atan2(dy, dx)
    theta_rad = np.arctan2(dy, dx)
    return theta_rad

def plot_image_with_horizon_and_projections(img, l_inf, Vz_h, A_h2, B_h2, color_line='y'):
    """
    Mostra immagine, linea all'infinito, punto di fuga verticale, proiezioni A'' e B''
    - img      : immagine BGR (OpenCV)
    - l_inf    : linea all'infinito in omogenee [a, b, c]
    - Vz_h     : vanishing point verticale in omogenee
    - A_h2/B_h2: proiezioni omogenee di A', B' su l_inf
    """
    h, w = img.shape[:2]
    a, b, c = l_inf

    # Estendi gli estremi x per disegnare la linea anche oltre l'immagine
    x_vals = np.linspace(-0.2 * w, 1.2 * w, 100)

    if abs(b) > 1e-6:
        y_vals = -(a * x_vals + c) / b
    else:
        x_vals = np.full(100, -c / a)
        y_vals = np.linspace(-0.2 * h, 1.2 * h, 100)

    # Convert image to RGB for matplotlib
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    
    # Draw line at infinity
    plt.plot(x_vals, y_vals, color_line, label='line at infinity (l_inf)')

    # Draw vanishing point Vz
    if abs(Vz_h[2]) > 1e-6:
        Vz_px = Vz_h[:2] / Vz_h[2]
        plt.scatter(*Vz_px, color='red', s=80, label='V_z (vanishing pt)')
        plt.text(Vz_px[0]+10, Vz_px[1]-10, 'V_z', color='red', fontsize=12)

    # Draw projected A'' and B''
    for pt_h, label, color in zip([A_h2, B_h2], ['A\'\'', 'B\'\''], ['limegreen', 'limegreen']):
        if abs(pt_h[2]) > 1e-6:
            pt = pt_h[:2] / pt_h[2]
            plt.scatter(*pt, color=color, s=60)
            plt.text(pt[0]+10, pt[1]-10, label, color=color, fontsize=12)

    # Axis control
    plt.xlim(-0.1 * w, 1.1 * w)
    plt.ylim(1.1 * h, -0.1 * h)  # flip y-axis AFTER plotting
    plt.title("Vanishing Geometry: V_z, l_inf, A'', B''")
    plt.legend()
    plt.grid(True)
    plt.show()

def draw_bb_3d(igm_undist, A_3d, B_3d, u_AB):
    C0 = 0.5*(A_3d + B_3d)   # rear‐axle midpoint
    w2 = 1.732  / 2.0
    l0 = 3.997
    h0 = 1.467

    # width‐axis = u_AB proiettato orizzontalmente
    width_axis = np.array([u_AB[0], 0, u_AB[2]])
    width_axis /= np.linalg.norm(width_axis)

    # length‐axis = rotazione di width_axis di +90° attorno a Y
    length_axis = np.array([ width_axis[2], 0, -width_axis[0] ])

    # 4 vertici sul ground
    rear_left  = C0 - w2*width_axis
    rear_right = C0 + w2*width_axis
    front_left  = C0 - w2*width_axis + l0*length_axis
    front_right = C0 + w2*width_axis + l0*length_axis

    # 4 vertici sopra di height
    rear_left_top   = rear_left  + np.array([0, -h0, 0])
    rear_right_top  = rear_right + np.array([0, -h0, 0])
    front_left_top  = front_left  + np.array([0, -h0, 0])
    front_right_top = front_right + np.array([0, -h0, 0])

    vertices_3d = np.vstack([
        rear_left, rear_right, front_right, front_left,
        rear_left_top, rear_right_top, front_right_top, front_left_top
    ])  # (8×3)

    # in omogenee
    verts_h = (K @ vertices_3d.T).T   # shape (8,3)
    # normalizza
    verts_px = verts_h[:,:2] / verts_h[:,2:3]  # shape (8,2), float
    verts_px = verts_px.astype(int)

    edges = [
        (0,1),(1,2),(2,3),(3,0),         # base
        (4,5),(5,6),(6,7),(7,4),         # top
        (0,4),(1,5),(2,6),(3,7)          # verticali
    ]

    for i,j in edges:
        pt1 = tuple(verts_px[i])
        pt2 = tuple(verts_px[j])
        cv.line(img_undist, pt1, pt2, (0,255,0), 2)

def draw_bb_3d_theta(img, width, length, height, theta):
    # box_local shape (8,3)
    C0 = 0.5*(A_3d + B_3d)   # rear‐axle midpoint
    w2 = width/2
    l0 = length
    h0 = height

    box_local = np.array([
        [-w2,  0,   0],     # rear_left
        [ w2,  0,   0],     # rear_right
        [ w2,  0, -l0],     # front_right
        [-w2,  0, -l0],     # front_left
        [-w2, -h0,  0],     # rear_left_top
        [ w2, -h0,  0],     # rear_right_top
        [ w2, -h0, -l0],    # front_right_top
        [-w2, -h0, -l0],    # front_left_top
    ])

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

    # ruota e trasla tutti i vertici
    vertices_world = (R @ box_local.T).T + C0   # shape (8,3)

    # proiezione omogenea
    verts_h = (K @ vertices_world.T).T      # (8,3)
    verts_px = (verts_h[:,:2] / verts_h[:,2:3]).astype(int)

    # disegna ariste
    edges = [(0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)]
    for i,j in edges:
        cv.line(img, tuple(verts_px[i]), tuple(verts_px[j]), (255,255,255), 10)

K = np.array([
    [3201.4989, 0.0, 1939.82925],
    [0.0, 3206.37527, 1063.15413],
    [0.0, 0.0, 1]
], dtype=np.float64)
K_inv = np.linalg.inv(K)

# Coefficienti di distorsione
dist = np.array([[0.24377, -1.5955, -0.0011528, 0.00041986, 3.5668]], dtype=np.float64)

phi = 8.53  # angolo tra piano posteriore e verticale in gradi

# Carica immagine
img = cv.imread(os.path.join(os.getcwd(), "Code", "OutputFolder2", "frame_02.png"))
if img is None:
    raise FileNotFoundError("Immagine non trovata")

# Undistorci l'immagine
img_undist = cv.undistort(img, K, dist)

# punti delle luci posteriori
luce_pts = [
    [408, 2268],  # sx interno
    [1256, 2304], # dx interno
    [318, 2212],  # sx esterno
    [1456, 2255]  # dx esterno
]
luce_pts_undist = []
for pt in luce_pts:
    pt_undist = cv.undistortPoints(np.array([[pt]], dtype=np.float32), K, dist, P=K)[0][0]
    pt_undist = np.round(pt_undist).astype(int)
    luce_pts_undist.append(pt_undist)
A, B, E, F = luce_pts_undist[0], luce_pts_undist[1], luce_pts_undist[2], luce_pts_undist[3]
A_h, B_h, E_h, F_h = np.array([A[0], A[1], 1]), np.array([B[0], B[1], 1]), np.array([E[0], E[1], 1]), np.array([F[0], F[1], 1])

# punti della targa
targa_pts = [
    [600, 2320],  # sx inferiore
    [1004, 2336]  # dx inferiore
]
targa_pts_undist = []
for pt in targa_pts:
    pt_undist = cv.undistortPoints(np.array([[pt]], dtype=np.float32), K, dist, P=K)[0][0]
    pt_undist = np.round(pt_undist).astype(int)
    targa_pts_undist.append(pt_undist)
C, D = targa_pts_undist[0], targa_pts_undist[1]
C_h, D_h = np.array([C[0], C[1], 1]), np.array([D[0], D[1], 1])

draw_points(img_undist, A, B, E, F, C, D)

# calcola la linea tra A e B
line_AB = np.cross(A_h, B_h)
# draw the line AB
cv.line(img_undist, (A[0], A[1]), (B[0], B[1]), (0, 255, 0), 4)

# calcola la linea tra C e D
line_CD = np.cross(C_h, D_h)
# draw the line CD
cv.line(img_undist, (C[0], C[1]), (D[0], D[1]), (255, 255, 0), 4)

Vx_h = np.cross(line_AB, line_CD)
Vx_px = (Vx_h[:2] / Vx_h[2]).astype(int)
# Draw the vanishing point Vx
cv.circle(img_undist, (Vx_px[0], Vx_px[1]), 5, (255, 0, 0), 10)

d_AB = K_inv @ Vx_h
u_AB = d_AB / np.linalg.norm(d_AB)

A_3d, B_3d = triangulate_pts(A_h, B_h, 0.52, u_AB)
C_3d, D_3d = triangulate_pts(C_h, D_h, 0.86, u_AB)

n_back = np.cross(B_3d - A_3d, D_3d - C_3d)
n_back /= np.linalg.norm(n_back)
# Verifica che la normale punti verso la camera (z > 0 tipicamente)
if n_back[2] < 0:
    n_back = -n_back

v_cam = compute_camera_vertical(n_back, u_AB, np.deg2rad(phi))
Vz_h = K @ v_cam
l_inf = np.linalg.inv(K).T @ v_cam
# draw the vertical vanishing point Vz
cv.circle(img_undist, (int(Vz_h[0]), int(Vz_h[1])), 5, (0, 0, 255), 10)
cv.putText(img_undist, "Vz", (int(Vz_h[0]) + 10, int(Vz_h[1]) - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
# Draw the horizon line l_inf
cv.line(img_undist, (0, int(-l_inf[2] / l_inf[1])), (img_undist.shape[1], int((-l_inf[2] - l_inf[0] * img_undist.shape[1]) / l_inf[1])), (0, 255, 255), 4)
cv.putText(img_undist, "l_inf", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)

theta = compute_theta(A_h, B_h, Vz_h, l_inf)
theta_deg = np.degrees(theta)

print(f"Imbardata (theta) calcolata: {theta_deg:.2f} gradi")

# draw_bb_3d(img_undist, A_3d, B_3d, u_AB)
draw_bb_3d_theta(img_undist, width=1.732, length=3.997, height=1.467, theta=theta_deg)

# A_pp = project_to_horizon(A_h, Vz_h, l_inf)
# B_pp = project_to_horizon(B_h, Vz_h, l_inf)
# E_pp = project_to_horizon(E_h, Vz_h, l_inf)
# F_pp = project_to_horizon(F_h, Vz_h, l_inf)

# # Draw projected points
# cv.circle(img_undist, (int(A_pp[0]), int(A_pp[1])), 5, (0, 0, 255), 10)
# cv.putText(img_undist, "A'", (int(A_pp[0]) + 10, int(A_pp[1]) - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
# cv.circle(img_undist, (int(B_pp[0]), int(B_pp[1])), 5, (0, 0, 255), 10)
# cv.putText(img_undist, "B'", (int(B_pp[0]) + 10, int(B_pp[1]) - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
# cv.circle(img_undist, (int(E_pp[0]), int(E_pp[1])), 5, (0, 0, 255), 10)
# cv.putText(img_undist, "E'", (int(E_pp[0]) + 10, int(E_pp[1]) - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
# cv.circle(img_undist, (int(F_pp[0]), int(F_pp[1])), 5, (0, 0, 255), 10)
# cv.putText(img_undist, "F'", (int(F_pp[0]) + 10, int(F_pp[1]) - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)

resized_img = cv.resize(img_undist, (0, 0), fx=0.35, fy=0.35)

# plot_image_with_horizon_and_projections(
#     img_undist,
#     l_inf=l_inf,         # [a, b, c]
#     Vz_h=Vz_h,           # homogeneous V_z
#     A_h2=A_pp,     # homogeneous A''
#     B_h2=B_pp      # homogeneous B''
# )

cv.imshow("Immagine Undistorta", resized_img)
cv.waitKey(0)
cv.destroyAllWindows()

# 1. PHI = 8.53
# 2. Vx
# 3. Vz
# # l_inf horiz
# trasla punti rispetto a Z camera
# A'' su l_inf
# calcoli theta
# update Vz, l_inf