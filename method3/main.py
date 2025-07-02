import numpy as np
import cv2 as cv
import os

def draw_points(img, A, B, E, F, C, D, color=(0, 255, 0)):
    # round and convert all points to integer pixel coords
    A, B, E, F, C, D = map(lambda p: np.round(p).astype(int), [A, B, E, F, C, D])

    # draw and label each point on the image
    cv.circle(img, tuple(A), 5, color, 10)
    cv.putText(img, "A", (A[0] + 10, A[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 3, color, 3)
    cv.circle(img, tuple(B), 5, color, 10)
    cv.putText(img, "B", (B[0] + 10, B[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 3, color, 3)
    cv.circle(img, tuple(E), 5, color, 10)
    cv.putText(img, "E", (E[0] + 10, E[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 3, color, 3)
    cv.circle(img, tuple(F), 5, color, 10)
    cv.putText(img, "F", (F[0] + 10, F[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 3, color, 3)
    # use different label color for C and D
    cv.circle(img, tuple(C), 5, color, 10)
    cv.putText(img, "C", (C[0] + 10, C[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3)
    cv.circle(img, tuple(D), 5, color, 10)
    cv.putText(img, "D", (D[0] + 10, D[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3)

def triangulate_pts(A_h, B_h, AB_distance, u_AB):
    # back-project homogeneous image pts to normalized camera rays
    ray_A = K_inv @ A_h
    ray_B = K_inv @ B_h
    ray_A /= np.linalg.norm(ray_A)
    ray_B /= np.linalg.norm(ray_B)

    # solve t*ray_A - s*ray_B = AB_distance * u_AB via least squares
    M = np.column_stack((ray_A, -ray_B))
    b = AB_distance * u_AB
    t, s = np.linalg.pinv(M) @ b

    # compute 3d positions along each ray
    A_3d = t * ray_A
    B_3d = s * ray_B
    return A_3d, B_3d

def compute_camera_vertical(n_back, u_dir, phi_rad):
    # ensure inputs are 3-element arrays
    n_b = np.asarray(n_back).reshape(3,)
    u = np.asarray(u_dir).reshape(3,)

    # rodrigues rotation of back-plane normal around axis u by phi
    cos_phi, sin_phi = np.cos(phi_rad), np.sin(phi_rad)
    cross_term = np.cross(u, n_b)
    dot_term = np.dot(u, n_b)
    v_cam = (cos_phi * n_b + sin_phi * cross_term + (1 - cos_phi) * dot_term * u)
    v_cam /= np.linalg.norm(v_cam)

    return v_cam

def project_to_horizon(point_img, Vz_h, l_inf):
    # vertical line through point_img toward vertical vanishing point
    vertical_line = np.cross(Vz_h, point_img)
    # intersect with horizon line
    p_horizon = np.cross(vertical_line, l_inf)
    if abs(p_horizon[2]) > 1e-6:
        p_horizon /= p_horizon[2]

    return p_horizon

def compute_theta(A_h, B_h, Vz_h, l_inf):
    # project A' and B' onto horizon to get their horizon coords
    Ah2 = project_to_horizon(A_h, Vz_h, l_inf)
    Bh2 = project_to_horizon(B_h, Vz_h, l_inf)
    A2 = Ah2[:2] / Ah2[2]
    B2 = Bh2[:2] / Bh2[2]

    # compute yaw angle between these horizon points
    dx, dy = B2[0] - A2[0], B2[1] - A2[1]
    theta_rad = np.arctan2(dy, dx)

    return theta_rad

def draw_bb_3d_theta(img, width, length, height, theta, A_3d, B_3d, C_3d, D_3d, E_3d=None, F_3d=None, color=(0, 255, 0)):
    # compute rear-axle midpoint, use 6-pt avg if E/F given
    if E_3d is not None and F_3d is not None:
        C0 = (A_3d + B_3d + C_3d + D_3d + E_3d + F_3d) / 6
    else:
        C0 = (A_3d + B_3d + C_3d + D_3d) / 4

    # define local 3d box vertices relative to rear axle
    w2 = width / 2
    l0, h0 = length, height
    taillight_h = 0.8
    box_local = np.array([
        [-w2, -taillight_h, -0.3], # rear lower left
        [ w2, -taillight_h, -0.3], # rear lower right
        [ w2, -taillight_h, -l0 + 0.15], # front lower right
        [-w2, -taillight_h, -l0 + 0.15], # front lower left
        [-w2, h0 - taillight_h, -0.3], # rear upper left
        [ w2, h0 - taillight_h, -0.3], # rear upper right
        [ w2, h0 - taillight_h, -l0 + 0.15], # front upper right
        [-w2, h0 - taillight_h, -l0 + 0.15], # front upper left
    ])

    # build yaw rotation matrix about camera y-axis
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]])

    # rotate and translate box vertices into world coords
    verts_world = (R @ box_local.T).T + C0

    # project to image plane
    verts_h = (K @ verts_world.T).T
    verts_px = (verts_h[:,:2] / verts_h[:,2:3]).astype(int)

    # draw each corner and each edge
    for v in verts_px:
        cv.circle(img, tuple(v), 10, (0, 0, 255), 10)
    edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]

    for i, j in edges:
        cv.line(img, tuple(verts_px[i]), tuple(verts_px[j]), color, 10)

def refine_pose_iterative(points_img, AB_dist, CD_dist, phi_rad, theta0, vcam0, EF_dist=None, max_iter=10, tol=1e-3):
    theta = theta0
    history = [theta]
    v_cam = vcam0

    for k in range(max_iter):
        # guess vehicle forward dir in camera frame
        c, s = np.cos(theta), np.sin(theta)
        u_guess = np.array([c, 0.0, s])

        # remove vertical comp. after first iter
        if k > 0:
            u_guess -= np.dot(u_guess, v_cam) * v_cam
            u_guess /= np.linalg.norm(u_guess)
        u_dir = u_guess

        # triangulate rear and plate pts in 3d
        A3d, B3d = triangulate_pts(points_img['A'], points_img['B'], AB_dist, u_dir)
        C3d, D3d = triangulate_pts(points_img['C'], points_img['D'], CD_dist, u_dir)
        if 'E' in points_img and 'F' in points_img and EF_dist:
            E3d, F3d = triangulate_pts(points_img['E'], points_img['F'], EF_dist, u_dir)

        # estimate back-plane normal via svd on rear-side pts
        pts = [A3d, B3d, C3d, D3d]
        if 'E' in locals(): pts += [E3d, F3d]
        pts = np.vstack(pts)
        centroid = pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts - centroid)
        n_back = Vt[-1]
        n_back /= np.linalg.norm(n_back)
        if n_back[2] < 0:
            n_back = -n_back

        # compute new camera vertical direction
        v_cam = compute_camera_vertical(n_back, u_dir, phi_rad)
        # horizon line and vertical vp for projection
        Vz_h = K @ v_cam
        l_inf = K_inv.T @ v_cam

        # project each rear-side point to horizon
        A2 = project_to_horizon(points_img['A'], Vz_h, l_inf)
        B2 = project_to_horizon(points_img['B'], Vz_h, l_inf)
        C2 = project_to_horizon(points_img['C'], Vz_h, l_inf)
        D2 = project_to_horizon(points_img['D'], Vz_h, l_inf)
        if 'E' in locals():
            E2 = project_to_horizon(points_img['E'], Vz_h, l_inf)
            F2 = project_to_horizon(points_img['F'], Vz_h, l_inf)

        # compute horizon angles for each segment
        angles = [np.arctan2(B2[1]-A2[1], B2[0]-A2[0]), np.arctan2(D2[1]-C2[1], D2[0]-C2[0])]
        if 'E2' in locals():
            angles.append(np.arctan2(F2[1]-E2[1], F2[0]-E2[0]))
        # circular mean of angles
        vecs = np.exp(1j * np.array(angles))
        theta_mean = np.arctan2(vecs.mean().imag, vecs.mean().real)

        # update with damping
        delta = (theta_mean - theta + np.pi) % (2*np.pi) - np.pi
        alpha = 0.2 if k < 3 else (0.8 if abs(delta) > 0.05 else 0.95)
        theta = (theta + alpha * delta + np.pi) % (2*np.pi) - np.pi
        history.append(theta)

        # stop if change small
        if abs(delta) < tol:
            break

    # return results, include E/F if used
    if EF_dist:
        return theta, v_cam, n_back, history, A3d, B3d, C3d, D3d, E3d, F3d, u_dir
    return theta, v_cam, n_back, history, A3d, B3d, C3d, D3d, None, None, u_dir

# camera intrinsics and distortion
K = np.array([
    [3201.4989, 0.0, 1939.82925],
    [0.0, 3206.37527, 1063.15413],
    [0.0, 0.0, 1]
], dtype=np.float64)
K_inv = np.linalg.inv(K)
dist = np.array([[0.24377, -1.5955, -0.0011528, 0.00041986, 3.5668]], dtype=np.float64)

# compute phi as tilt between headlight plane and vertical
phi_rad = np.arctan2(0.15, 0.90)
print(f"phi = {np.degrees(phi_rad):.2f}°")

# real distances between feature pairs
distAB, distCD, distEF = 0.86, 0.52, 1.40

# load and undistort image
img = cv.imread(os.path.join(os.getcwd(), "featureExtraction/extractedFrames/frame_02.png"))
if img is None:
    raise FileNotFoundError("image not found")
img_undist = cv.undistort(img, K, dist)

# undistort and prepare homogeneous image pts for lights and plate
luce_pts = [[408,2268],[1256,2304],[318,2212],[1456,2255]]
luce_pts_ud = [cv.undistortPoints(
    np.array([[pt]],np.float32),K,dist,P=K)[0][0]
    for pt in luce_pts]
A, B, E, F = luce_pts_ud
A_h = np.array([A[0],A[1],1]); B_h = np.array([B[0],B[1],1])
E_h = np.array([E[0],E[1],1]); F_h = np.array([F[0],F[1],1])

targa_pts = [[600,2320],[1004,2336]]
targa_pts_ud = [cv.undistortPoints(
    np.array([[pt]],np.float32),K,dist,P=K)[0][0]
    for pt in targa_pts]
C, D = targa_pts_ud
C_h = np.array([C[0],C[1],1]); D_h = np.array([D[0],D[1],1])

# draw feature points and lines, compute initial vanishing and rays
draw_points(img_undist, A, B, E, F, C, D)
line_AB = np.cross(A_h, B_h)
line_CD = np.cross(C_h, D_h)
A_pp, B_pp = (A_h[:2]/A_h[2]).astype(int), (B_h[:2]/B_h[2]).astype(int)
C_pp, D_pp = (C_h[:2]/C_h[2]).astype(int), (D_h[:2]/D_h[2]).astype(int)
cv.line(img_undist, tuple(A_pp), tuple(B_pp), (0,255,0),10)
cv.line(img_undist, tuple(C_pp), tuple(D_pp), (255,255,0),10)

Vx_h = np.cross(line_AB, line_CD)
Vx_px = (Vx_h[:2]/Vx_h[2]).astype(int)
cv.circle(img_undist, tuple(Vx_px),5,(255,0,0),10)

# back-project vanishing to get direction u_AB
d_AB = K_inv @ Vx_h
u_AB = d_AB / np.linalg.norm(d_AB)

# triangulate initial 3d points
A_3d, B_3d = triangulate_pts(A_h, B_h, distAB, u_AB)
C_3d, D_3d = triangulate_pts(C_h, D_h, distCD, u_AB)

# compute back-plane normal and camera vertical
n_back = np.cross(B_3d - A_3d, D_3d - C_3d)
n_back /= np.linalg.norm(n_back)
if n_back[2] < 0:
    n_back = -n_back

v_cam = compute_camera_vertical(n_back, u_AB, phi_rad)
Vz_h = K @ v_cam
l_inf = K_inv.T @ v_cam

# compute initial yaw from horizon projection
theta = compute_theta(A_h, B_h, Vz_h, l_inf)
print(f"initial θ: {np.degrees(theta):.4f} deg")

# refine pose iteratively
results = refine_pose_iterative(
    points_img={'A':A_h,'B':B_h,'C':C_h,'D':D_h,'E':E_h,'F':F_h},
    AB_dist=distAB, CD_dist=distCD, EF_dist=distEF,
    phi_rad=phi_rad, theta0=theta, vcam0=v_cam,
    max_iter=20, tol=1e-3
)
(theta_ref, v_cam_ref, n_back_ref, history, A3d_ref, B3d_ref, C3d_ref, D3d_ref, E3d_ref, F3d_ref, u_dir_ref) = results

print("iterations:", len(history))
print(f"final θ: {np.degrees(theta_ref):.4f} deg")
print(f"A-B dist: {np.linalg.norm(B3d_ref-A3d_ref):.3f} (exp {distAB})")
print(f"C-D dist: {np.linalg.norm(D3d_ref-C3d_ref):.3f} (exp {distCD})")
print(f"E-F dist: {np.linalg.norm(F3d_ref-E3d_ref):.3f} (exp {distEF})")
print(f"back-plane normal: {n_back_ref}")
print(f"camera vertical: {v_cam_ref}")

# draw final 3d bbox with refined yaw
draw_bb_3d_theta(
    img_undist,
    width=1.732, length=3.997, height=1.467,
    theta=theta_ref,
    A_3d=A3d_ref, B_3d=B3d_ref,
    C_3d=C3d_ref, D_3d=D3d_ref,
    E_3d=E3d_ref, F_3d=F3d_ref,
    color=(255,255,255)
)

# save and show result
resized = cv.resize(img_undist, None, fx=0.25, fy=0.25)
cv.imwrite(os.path.join(os.getcwd(), "method3/results/bbox.jpg"), img_undist)
cv.imshow("result", resized)
cv.waitKey(0)
cv.destroyAllWindows()