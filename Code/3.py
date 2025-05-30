import numpy as np
import cv2 as cv
import os

def draw_points(img, A, B, E, F, C, D, color=(0, 255, 0)):
    A, B, E, F, C, D = map(lambda p: np.round(p).astype(int), [A, B, E, F, C, D])

    cv.circle(img, tuple(A), 5, color, 10)
    cv.putText(img, "A", (A[0] + 10, A[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
    cv.circle(img, tuple(B), 5, color, 10)
    cv.putText(img, "B", (B[0] + 10, B[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
    cv.circle(img, tuple(E), 5, color, 10)
    cv.putText(img, "E", (E[0] + 10, E[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
    cv.circle(img, tuple(F), 5, color, 10)
    cv.putText(img, "F", (F[0] + 10, F[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
    cv.circle(img, tuple(C), 5, color, 10)
    cv.putText(img, "C", (C[0] + 10, C[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3)
    cv.circle(img, tuple(D), 5, color, 10)
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

def draw_bb_3d_theta(img, width, length, height, theta, A_3d, B_3d, C_3d, D_3d, E_3d=None, F_3d=None, color=(0, 255, 0)):
    # box_local shape (8,3)
    # C0 = (A_3d + B_3d) / 2  # rear‐axle midpoint
    if E_3d is not None and F_3d is not None:
        print("Using E and F for 6-point average")
        C0 = (A_3d + B_3d + C_3d + D_3d + E_3d + F_3d) / 6
    else:
        print("Using A, B, C, D for 4-point average")
        C0 = (A_3d + B_3d + C_3d + D_3d) / 4
    w2 = width/2
    l0 = length
    h0 = height
    taillight_height = 0.8

    box_local = np.array([
        [-w2, -taillight_height, -0.3],     # rear_left
        [ w2, -taillight_height, -0.3],     # rear_right
        [ w2, -taillight_height, -l0 + 0.15],     # front_right
        [-w2, -taillight_height, -l0 + 0.15],     # front_left
        [-w2, h0 - taillight_height, -0.3],     # rear_left_top
        [ w2, h0 - taillight_height, -0.3],     # rear_right_top
        [ w2, h0 - taillight_height, -l0 + 0.15],    # front_right_top
        [-w2, h0 - taillight_height, -l0 + 0.15],    # front_left_top
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

    # draw circle for every vertex
    for v in verts_px:
        cv.circle(img, tuple(v), 10, (0, 0, 255), 10)

    # disegna ariste
    edges = [(0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)]
    for i,j in edges:
        cv.line(img, tuple(verts_px[i]), tuple(verts_px[j]), color, 10)

def refine_pose_iterative(points_img, AB_dist, CD_dist, phi_rad, theta0, vcam0, EF_dist=None, max_iter=10, tol=1e-3):
    """
    Iteratively refine vehicle yaw (theta), camera vertical (v_cam) and back-plane normal (n_back).

    Args:
        points_img : dict with keys 'A','B','C','D'; each a homogeneous [3,] image point
        AB_dist    : float, real distance between A and B
        CD_dist    : float, real distance between C and D
        EF_dist    : float, real distance between E and F
        phi_rad    : float, angle between back-plane and vertical (radians)
        theta0     : float, initial yaw estimate (radians)
        max_iter   : int, maximum number of iterations
        tol        : float, convergence threshold for change in theta

    Returns:
        theta      : float, refined yaw (rad)
        v_cam      : array (3,), camera vertical direction unit vector
        n_back     : array (3,), back-plane normal unit vector
        history    : list of theta values over iterations
    """
    theta = theta0
    history = [theta]

    v_cam = vcam0  # get initial camera vertical
    
    for k in range(max_iter):
        # 1) Guess direction from current theta
        c, s = np.cos(theta), np.sin(theta)
        u_guess = np.array([c, 0.0, s])  # local vehicle X–Z axis rotated by theta

        # 2) Project u_guess onto horizontal plane orthogonal to v_cam (skip on first iter)
        if k == 0:
            u_dir = u_guess
        else:
            # remove vertical component
            u_dir = u_guess - np.dot(u_guess, v_cam) * v_cam
            u_dir /= np.linalg.norm(u_dir)

        # 3) Triangulate A/B and C/D in 3D
        A3d, B3d = triangulate_pts(points_img['A'], points_img['B'], AB_dist, u_dir)
        C3d, D3d = triangulate_pts(points_img['C'], points_img['D'], CD_dist, u_dir)
        # # If E/F are provided, triangulate them too
        if 'E' in points_img and 'F' in points_img and EF_dist is not None:
            E3d, F3d = triangulate_pts(points_img['E'], points_img['F'], EF_dist, u_dir)

        """"
        # 4) Compute back-plane normal
        n_back = np.cross(B3d - A3d, D3d - C3d)
        n_back /= np.linalg.norm(n_back)
        if n_back[2] < 0:
            n_back = -n_back
        """
        
        # 4) Compute back-plane normal via SVD on all available rear‐side points
        pts = [A3d, B3d, C3d, D3d]
        if 'E' in points_img and 'F' in points_img and EF_dist is not None:
            pts += [E3d, F3d]

        pts = np.vstack(pts)           # shape (N,3), N=4 or 6
        centroid = pts.mean(axis=0)    # baricentro
        # PCA: l'ultima componente di V^T è la direzione di minore varianza → normale
        _, _, Vt = np.linalg.svd(pts - centroid)
        n_back = Vt[-1]                
        n_back /= np.linalg.norm(n_back)
        # facoltativo: assicura che punti verso la camera
        if n_back[2] < 0:
            n_back = -n_back

        # 5) Compute camera vertical via Rodrigues rotation
        v_cam = compute_camera_vertical(n_back, u_dir, phi_rad)

        # 6) Update vanishing point and horizon line
        Vz_h = K @ v_cam
        l_inf = K_inv.T @ v_cam

        # 7) Project A',B',C',D' to horizon
        Ah2 = project_to_horizon(points_img['A'], Vz_h, l_inf)
        Bh2 = project_to_horizon(points_img['B'], Vz_h, l_inf)
        Ch2 = project_to_horizon(points_img['C'], Vz_h, l_inf)
        Dh2 = project_to_horizon(points_img['D'], Vz_h, l_inf)
        if 'E' in points_img and 'F' in points_img and EF_dist is not None:
            Eh2 = project_to_horizon(points_img['E'], Vz_h, l_inf)
            Fh2 = project_to_horizon(points_img['F'], Vz_h, l_inf)

        # Normalize to pixel coords
        A2 = Ah2[:2] / Ah2[2]
        B2 = Bh2[:2] / Bh2[2]
        C2 = Ch2[:2] / Ch2[2]
        D2 = Dh2[:2] / Dh2[2]
        if 'E' in points_img and 'F' in points_img and EF_dist is not None:
            E2 = Eh2[:2] / Eh2[2]
            F2 = Fh2[:2] / Fh2[2]

        # 8) compute unit‐vectors for AB and CD
        th_AB = np.arctan2(B2[1]-A2[1], B2[0]-A2[0])
        th_CD = np.arctan2(D2[1]-C2[1], D2[0]-C2[0])
        if 'E' in points_img and 'F' in points_img and EF_dist is not None:
            th_EF = np.arctan2(F2[1]-E2[1], F2[0]-E2[0])
            angles = np.array([th_AB, th_CD, th_EF])
        else:
            angles = np.array([th_AB, th_CD])

        # 9) circular mean
        vecs = np.exp(1j * angles)
        mean_vec = vecs.mean()
        theta_mean = np.arctan2(mean_vec.imag, mean_vec.real)

        # 10) signed difference & damping
        delta = (theta_mean - theta + np.pi) % (2*np.pi) - np.pi

        # adaptive damping
        if k < 3:
            alpha = 0.2
        else:
            alpha = 0.8 if abs(delta) > 0.05 else 0.95

        theta += alpha * delta
        # wrap back into [-pi,pi]
        theta = (theta + np.pi) % (2*np.pi) - np.pi

        history.append(theta)
        
        print(f"Iter {k+1}: θ = {np.degrees(theta):.4f} deg, Δθ = {np.degrees(delta):.4f} deg")
        
        # 11) convergence on |delta|
        if abs(delta) < tol:
            break
        
    if EF_dist is not None:
        return theta, v_cam, n_back, history, A3d, B3d, C3d, D3d, E3d, F3d, u_dir
    return theta, v_cam, n_back, history, A3d, B3d, C3d, D3d, None, None, u_dir

K = np.array([
    [3201.4989, 0.0, 1939.82925],
    [0.0, 3206.37527, 1063.15413],
    [0.0, 0.0, 1]
], dtype=np.float64)
K_inv = np.linalg.inv(K)

# Coefficienti di distorsione
dist = np.array([[0.24377, -1.5955, -0.0011528, 0.00041986, 3.5668]], dtype=np.float64)

phi = 8.50 # angolo tra piano posteriore e verticale in gradi

distAB = 0.86 # distanza reale tra A e B in metri
distCD = 0.52 # distanza reale tra C e D in metri
distEF = 1.40 # distanza reale tra E e F in metri (se disponibili)
# distEF = None  # se non si usano le luci posteriori esterne, impostare a None

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
    targa_pts_undist.append(pt_undist)
C, D = targa_pts_undist[0], targa_pts_undist[1]
C_h, D_h = np.array([C[0], C[1], 1]), np.array([D[0], D[1], 1])

draw_points(img_undist, A, B, E, F, C, D)

# calcola la linea tra A e B
line_AB = np.cross(A_h, B_h)
# convert to pixel coordinates
A_pp = (A_h[:2] / A_h[2]).astype(int)
B_pp = (B_h[:2] / B_h[2]).astype(int)
# draw the line AB
cv.line(img_undist, (A_pp[0], A_pp[1]), (B_pp[0], B_pp[1]), (0, 255, 0), 10)

# calcola la linea tra C e D
line_CD = np.cross(C_h, D_h)
# convert to pixel coordinates
C_pp = (C_h[:2] / C_h[2]).astype(int)
D_pp = (D_h[:2] / D_h[2]).astype(int)
# draw the line CD
cv.line(img_undist, (C_pp[0], C_pp[1]), (D_pp[0], D_pp[1]), (255, 255, 0), 10)

Vx_h = np.cross(line_AB, line_CD)
Vx_px = (Vx_h[:2] / Vx_h[2]).astype(int)
# Draw the vanishing point Vx
cv.circle(img_undist, (Vx_px[0], Vx_px[1]), 5, (255, 0, 0), 10)

d_AB = K_inv @ Vx_h
u_AB = d_AB / np.linalg.norm(d_AB)

A_3d, B_3d = triangulate_pts(A_h, B_h, distAB, u_AB)
C_3d, D_3d = triangulate_pts(C_h, D_h, distCD, u_AB)

n_back = np.cross(B_3d - A_3d, D_3d - C_3d)
n_back /= np.linalg.norm(n_back)
# Verifica che la normale punti verso la camera (z > 0 tipicamente)
if n_back[2] < 0:
    n_back = -n_back

v_cam = compute_camera_vertical(n_back, u_AB, np.deg2rad(phi))
Vz_h = K @ v_cam
l_inf = np.linalg.inv(K).T @ v_cam

theta = compute_theta(A_h, B_h, Vz_h, l_inf)
theta_deg = np.degrees(theta)

print(f"θ iniziale: {theta_deg:.4f} deg")

theta_ref, v_cam_ref, n_back_ref, history, A3d_ref, B3d_ref, C3d_ref, D3d_ref, E3d_ref, F3d_ref, u_dir_ref = refine_pose_iterative(
    points_img={'A': A_h, 'B': B_h, 'C': C_h, 'D': D_h, 'E': E_h, 'F': F_h},
    AB_dist=distAB,
    CD_dist=distCD,
    EF_dist=distEF,
    phi_rad=np.deg2rad(phi),
    theta0=theta,
    vcam0=v_cam,
    max_iter=20,
    tol=1e-3
)

print("Iterazioni:", len(history))
print(f"θ finale: {np.degrees(theta_ref):.4f} deg")
print(f"Distanza A-B calcolata: {np.linalg.norm(B3d_ref - A3d_ref):.3f} (expected: {distAB:.2f})")
print(f"Distanza C-D calcolata: {np.linalg.norm(D3d_ref - C3d_ref):.3f} (expected: {distCD:.2f})")
if distEF is not None:
    print(f"Distanza E-F calcolata: {np.linalg.norm(F3d_ref - E3d_ref):.3f} (expected: {distEF:.2f})")
print(f"Normale piano: {n_back_ref}")
print(f"Verticale camera: {v_cam_ref}")

# width, length, height in meters (your vehicle dims)
width_m  = 1.732
length_m = 3.997
height_m = 1.467

draw_bb_3d_theta(img_undist, width_m, length_m, height_m, theta=theta_ref, A_3d=A3d_ref, B_3d=B3d_ref, C_3d=C3d_ref, D_3d=D3d_ref, E_3d=E3d_ref, F_3d=F3d_ref, color=(255, 255, 255))

resized_img = cv.resize(img_undist, (0, 0), fx=0.25, fy=0.25)
cv.imshow("Immagine Undistorta", resized_img)
cv.waitKey(0)
cv.destroyAllWindows()