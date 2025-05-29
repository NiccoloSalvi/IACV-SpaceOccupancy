import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os


def draw_points(ax, A, B, E, F, C, D):
    # Draw points with labels using matplotlib
    points = {'A': A, 'B': B, 'E': E, 'F': F, 'C': C, 'D': D}
    colors = {'A': 'green', 'B': 'green', 'E': 'green', 'F': 'green', 'C': 'yellow', 'D': 'yellow'}
    for label, pt in points.items():
        ax.plot(pt[0], pt[1], 'o', color=colors[label], markersize=10)
        ax.text(pt[0] + 10, pt[1] - 10, label, fontsize=15, color=colors[label], weight='bold')


def triangulate_pts(A_h, B_h, AB_distance, u_AB, K_inv):
    """
    Triangulate 3D points A and B from their image projections and known real-world separation.
    """
    # Back-project to normalized camera rays
    ray_A = K_inv @ A_h
    ray_B = K_inv @ B_h
    ray_A = ray_A / np.linalg.norm(ray_A)
    ray_B = ray_B / np.linalg.norm(ray_B)

    # Solve t*ray_A - s*ray_B = AB_distance * u_AB
    M = np.column_stack((ray_A, -ray_B))
    b = AB_distance * u_AB

    # Solve in least-squares sense
    ts = np.linalg.pinv(M) @ b
    t, s = ts

    # Compute 3D positions
    A_3d = t * ray_A
    B_3d = s * ray_B

    return A_3d, B_3d


def compute_back_plane_normal(A_3d, B_3d, C_3d, D_3d):
    """
    Compute the normal vector of the back plane from 4 coplanar points.
    """
    # Use two vectors in the plane
    v1 = B_3d - A_3d  # horizontal direction
    v3 = C_3d - A_3d  # vertical direction in the plane

    # Cross product to get normal (use v1 and v3 for better numerical stability)
    n_back = np.cross(v1, v3)
    n_back = n_back / np.linalg.norm(n_back)

    # Ensure normal points toward camera (positive Z in camera coordinates)
    if n_back[2] < 0:
        n_back = -n_back

    return n_back


def compute_camera_vertical_from_phi(n_back, u_AB, phi_rad):
    """
    Compute camera vertical direction from back plane normal and inclination angle phi.
    """
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)

    cross_term = np.cross(u_AB, n_back)
    dot_term = np.dot(u_AB, n_back)

    v_cam = (cos_phi * n_back +
             sin_phi * cross_term +
             (1 - cos_phi) * dot_term * u_AB)

    v_cam = v_cam / np.linalg.norm(v_cam)
    return v_cam


def project_to_horizon(point_img, Vz_h, l_inf):
    """
    Project image point to horizon line through vertical vanishing point.
    """
    vertical_line = np.cross(Vz_h, point_img)
    point_on_horizon = np.cross(vertical_line, l_inf)

    if abs(point_on_horizon[2]) > 1e-6:
        point_on_horizon = point_on_horizon / point_on_horizon[2]

    return point_on_horizon


def compute_theta_from_horizon_projection(A_h, B_h, Vz_h, l_inf):
    """
    Compute yaw angle theta from horizon projections of A and B.
    """
    A_proj = project_to_horizon(A_h, Vz_h, l_inf)
    B_proj = project_to_horizon(B_h, Vz_h, l_inf)

    A2 = A_proj[:2] / A_proj[2] if abs(A_proj[2]) > 1e-6 else A_proj[:2]
    B2 = B_proj[:2] / B_proj[2] if abs(B_proj[2]) > 1e-6 else B_proj[:2]

    dx = B2[0] - A2[0]
    dy = B2[1] - A2[1]

    theta_rad = np.arctan2(dy, dx)
    return theta_rad


def update_direction_from_theta(theta_rad):
    """
    Update AB direction vector from yaw angle theta.
    """
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    u_AB = np.array([cos_theta, 0, sin_theta])
    u_AB = u_AB / np.linalg.norm(u_AB)

    return u_AB


def iterative_pose_estimation(A_h, B_h, C_h, D_h, K, K_inv, phi_rad,
                              AB_distance=0.86, CD_distance=0.52,
                              max_iterations=10, tolerance=1e-4):
    """
    Iterative pose estimation using Method 3.
    """
    line_AB = np.cross(A_h, B_h)
    line_CD = np.cross(C_h, D_h)
    Vx_h = np.cross(line_AB, line_CD)

    d_AB = K_inv @ Vx_h
    u_AB = d_AB / np.linalg.norm(d_AB)

    print("Starting iterative refinement...")

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}")

        A_3d, B_3d = triangulate_pts(A_h, B_h, AB_distance, u_AB, K_inv)
        C_3d, D_3d = triangulate_pts(C_h, D_h, CD_distance, u_AB, K_inv)

        n_back = compute_back_plane_normal(A_3d, B_3d, C_3d, D_3d)

        v_cam = compute_camera_vertical_from_phi(n_back, u_AB, phi_rad)

        Vz_h = K @ v_cam

        l_inf = np.linalg.inv(K).T @ v_cam
        l_inf = l_inf / np.linalg.norm(l_inf[:2])

        theta_new = compute_theta_from_horizon_projection(A_h, B_h, Vz_h, l_inf)

        u_AB_new = update_direction_from_theta(theta_new)

        direction_change = np.linalg.norm(u_AB_new - u_AB)
        print(f"  Direction change: {direction_change:.6f}")
        print(f"  Theta: {np.degrees(theta_new):.2f} degrees")

        if direction_change < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break

        u_AB = u_AB_new

    A_3d, B_3d = triangulate_pts(A_h, B_h, AB_distance, u_AB, K_inv)
    C_3d, D_3d = triangulate_pts(C_h, D_h, CD_distance, u_AB, K_inv)
    n_back = compute_back_plane_normal(A_3d, B_3d, C_3d, D_3d)
    v_cam = compute_camera_vertical_from_phi(n_back, u_AB, phi_rad)
    Vz_h = K @ v_cam
    l_inf = np.linalg.inv(K).T @ v_cam
    theta_final = compute_theta_from_horizon_projection(A_h, B_h, Vz_h, l_inf)

    return {
        'A_3d': A_3d,
        'B_3d': B_3d,
        'C_3d': C_3d,
        'D_3d': D_3d,
        'u_AB': u_AB,
        'n_back': n_back,
        'v_cam': v_cam,
        'Vz_h': Vz_h,
        'l_inf': l_inf,
        'theta': theta_final,
        'iterations': iteration + 1
    }


def draw_bounding_box_3d(ax, pose_result, K, car_width=1.732, car_length=3.997, car_height=1.467):
    """
    Draw 3D bounding box of the car using the estimated pose.
    """
    A_3d = pose_result['A_3d']
    B_3d = pose_result['B_3d']
    theta = pose_result['theta']

    taillight_height = 0.930

    C0 = 0.5 * (A_3d + B_3d)

    w2 = car_width / 2.0
    l0 = car_length
    h0 = car_height

    box_local = np.array([
        [-w2, -taillight_height, 0],  # rear_left
        [w2, -taillight_height, 0],  # rear_right
        [w2, -taillight_height, l0],  # front_right
        [-w2, -taillight_height, l0],  # front_left
        [-w2, +taillight_height, 0],  # rear_left_top
        [w2, +taillight_height, 0],  # rear_right_top
        [w2, +taillight_height, l0],  # front_right_top
        [-w2, +taillight_height, l0],  # front_left_top
    ])

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R_y = np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])

    vertices_world = (R_y @ box_local.T).T + C0

    vertices_h = (K @ vertices_world.T).T
    vertices_px = (vertices_h[:, :2] / vertices_h[:, 2:3])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7)  # vertical
    ]

    # Draw edges in cyan
    for i, j in edges:
        x = [vertices_px[i, 0], vertices_px[j, 0]]
        y = [vertices_px[i, 1], vertices_px[j, 1]]
        ax.plot(x, y, color='cyan', linewidth=3)

    # Rear face in red
    rear_edges = [(0, 1), (1, 5), (5, 4), (4, 0)]
    for i, j in rear_edges:
        x = [vertices_px[i, 0], vertices_px[j, 0]]
        y = [vertices_px[i, 1], vertices_px[j, 1]]
        ax.plot(x, y, color='red', linewidth=4)


def visualize_vanishing_geometry(ax, pose_result, img_shape):
    """
    Draw vanishing points, horizon line, and projected points using matplotlib.
    """
    Vz_h = pose_result['Vz_h']
    l_inf = pose_result['l_inf']

    if abs(Vz_h[2]) > 1e-6:
        Vz_px = Vz_h[:2] / Vz_h[2]
        ax.plot(Vz_px[0], Vz_px[1], 'o', color='red', markersize=10)
        ax.text(Vz_px[0] + 15, Vz_px[1] - 15, "Vz", color='red', fontsize=12, weight='bold')

    h, w = img_shape[:2]

    if abs(l_inf[1]) > 1e-6:
        x1, x2 = 0, w
        y1 = -(l_inf[0] * x1 + l_inf[2]) / l_inf[1]
        y2 = -(l_inf[0] * x2 + l_inf[2])


# Main execution
if __name__ == "__main__":
    # Camera parameters
    K = np.array([
        [3201.4989, 0.0, 1939.82925],
        [0.0, 3206.37527, 1063.15413],
        [0.0, 0.0, 1]
    ], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    # Distortion coefficients
    dist = np.array([[0.24377, -1.5955, -0.0011528, 0.00041986, 3.5668]], dtype=np.float64)

    # Car model parameters
    phi_deg = 8.53  # angle between back plane and vertical in degrees
    phi_rad = np.deg2rad(phi_deg)

    # Load and undistort image

    img_path = os.path.join(os.getcwd(), "OutputFolder", "frame_02.png")
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")

    #img_undist = cv.undistort(img, K, dist)
    img_undist = img

    # Define feature points (rear lights and license plate)
    luce_pts = [
        [408, 2268],  # A: left inner
        [1256, 2304],  # B: right inner
        [318, 2212],  # E: left outer
        [1456, 2255]  # F: right outer
    ]

    targa_pts = [
        [600, 2320],  # C: left license plate
        [1004, 2336]  # D: right license plate
    ]

    # Undistort feature points
    luce_pts_undist = luce_pts
    #for pt in luce_pts:
    #    pt_undist = cv.undistortPoints(np.array([[pt]], dtype=np.float32), K, dist, P=K)[0][0]
    #    pt_undist = np.round(pt_undist).astype(int)
    #    luce_pts_undist.append(pt_undist)

    targa_pts_undist = targa_pts
    #for pt in targa_pts:
        #pt_undist = cv.undistortPoints(np.array([[pt]], dtype=np.float32), K, dist, P=K)[0][0]
        #pt_undist = np.round(pt_undist).astype(int)
        #targa_pts_undist.append(pt_undist)

    # Extract points
    A, B, E, F = luce_pts_undist
    C, D = targa_pts_undist

    # Convert to homogeneous coordinates
    A_h = np.array([A[0], A[1], 1], dtype=np.float64)
    B_h = np.array([B[0], B[1], 1], dtype=np.float64)
    C_h = np.array([C[0], C[1], 1], dtype=np.float64)
    D_h = np.array([D[0], D[1], 1], dtype=np.float64)

    # Draw original points
    fig, ax = plt.subplots()
    #resized_img = cv.resize(img_undist, (0, 0), fx=0.35, fy=0.35)
    ax.imshow(img_undist)
    draw_points(ax, A, B, E, F, C, D)

    # Perform iterative pose estimation
    print("Performing iterative pose estimation...")
    pose_result = iterative_pose_estimation(
        A_h, B_h, C_h, D_h, K, K_inv, phi_rad,
        AB_distance=0.52, CD_distance=0.86,
        max_iterations=15, tolerance=1e-5
    )

    print(f"\nFinal Results:")
    print(f"Converged in {pose_result['iterations']} iterations")
    print(f"Final yaw angle (theta): {np.degrees(pose_result['theta']):.2f} degrees")
    print(f"Car position (rear axle center): {0.5 * (pose_result['A_3d'] + pose_result['B_3d'])}")

    # Visualize results
    visualize_vanishing_geometry(ax, pose_result, img_undist.shape)
    draw_bounding_box_3d(ax, pose_result, K)
    plt.show()
    # Display result
    #cv.imshow("Method 3 - Iterative Pose Estimation", resized_img)

    #cv.waitKey(0)
    #cv.destroyAllWindows()