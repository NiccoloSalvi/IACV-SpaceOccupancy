import numpy as np
import cv2
import os

def to_homogeneous(p):
    return np.array([p[0], p[1], 1.0], dtype=float)

def to_line(p1, p2):
    return np.cross(p1, p2)

def normalize(v):
    return v / np.linalg.norm(v)

def triangulate_plate_points(K, p0_img, p1_img, plate_width):
    ray0 = normalize(np.linalg.inv(K) @ to_homogeneous(p0_img))
    ray1 = normalize(np.linalg.inv(K) @ to_homogeneous(p1_img))
    cos_angle = np.clip(np.dot(ray0, ray1), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    d_approx = plate_width / (2 * np.sin(angle / 2))
    P0 = d_approx * ray0
    P1 = d_approx * ray1
    actual_dist = np.linalg.norm(P1 - P0)
    scale = plate_width / actual_dist
    return P0 * scale, P1 * scale

def compute_vanishing_direction(K, p0_img, p1_img, p2_img, p3_img):
    l1 = to_line(to_homogeneous(p0_img), to_homogeneous(p1_img))
    l2 = to_line(to_homogeneous(p2_img), to_homogeneous(p3_img))
    vp = normalize(np.cross(l1, l2))
    dir3d = normalize(np.linalg.inv(K) @ vp)
    return dir3d

def build_vehicle_frame(P0, P1, v_direction, plate_height):
    x_axis = normalize(P1 - P0)
    y_temp = v_direction - np.dot(v_direction, x_axis) * x_axis
    y_axis = normalize(y_temp)
    z_axis = normalize(np.cross(x_axis, y_axis))
    if z_axis[1] > 0:
        z_axis = -z_axis
        y_axis = -y_axis

    origin = (P0 + P1) / 2 + plate_height * z_axis
    R = np.column_stack((x_axis, y_axis, z_axis))
    return origin, R

def build_3d_box(origin, R, dims):
    w, l, h = dims['width'], dims['length'], dims['height']
    rear_offset = dims['rear_offset']
    local_pts = np.array([
        [-w/2, -rear_offset, 0], [w/2, -rear_offset, 0],
        [w/2, l - rear_offset, 0], [-w/2, l - rear_offset, 0],
        [-w/2, -rear_offset, -h], [w/2, -rear_offset, -h],
        [w/2, l - rear_offset, -h], [-w/2, l - rear_offset, -h]
    ])
    return [origin + R @ pt for pt in local_pts]

def project_box(K, pts_3d):
    pts_2d = []
    for pt in pts_3d:
        p = K @ pt
        pts_2d.append((int(p[0]/p[2]), int(p[1]/p[2])) if p[2] > 0 else None)
    return pts_2d

def draw_bbox(img, pts2d):
    edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
    labels = ['RBL','RBR','FBR','FBL','RTL','RTR','FTR','FTL']
    vis = img.copy()
    # draw edges
    for i,j in edges:
        if pts2d[i] and pts2d[j]:
            cv2.line(vis, pts2d[i], pts2d[j], (0,255,0), 2)
    # draw vertices & labels
    for idx, p in enumerate(pts2d):
        if p:
            cv2.circle(vis, p, 4, (0,0,255), -1)
            cv2.putText(vis, labels[idx], (p[0]+5, p[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,255), 3, cv2.LINE_AA)
    return vis

K = np.array([
    [3.31565306e+03, 0.0, 1.91087753e+03],
    [0.0, 3.31989314e+03, 1.07238815e+03],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

dist = np.array([[2.65104301e-01, -1.78436004e+00,  2.42978100e-03,  1.18030874e-04, 3.77074289e+00]], dtype=np.float64)

dims = {
    'width': 1.732, 'length': 3.997, 'height': 1.467,
    'plate_width': 0.520, 'plate_height': 0.9, 'rear_offset': 0.3
}

img_path = os.path.join(os.getcwd(), "featureExtraction", "extractedFrames", "frame_08.png")
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError("Image not found")

img_ud = cv2.undistort(img, K, dist)

pts_img = np.array([
    [1192, 1648], # P0 = plate TL (origine)
    [1448, 1656], # P1 = plate TR
    [1036, 1592], # P2 = rear-light L
    [1612, 1608] # P3 = rear-light R
], dtype=np.float64)

pts_ud = cv2.undistortPoints(pts_img.reshape(-1,1,2), K, dist, P=K).reshape(-1,2)

P0, P1 = triangulate_plate_points(K, pts_ud[0], pts_ud[1], dims['plate_width'])
v_dir = compute_vanishing_direction(K, *pts_ud)
print("P0 (3D):", P0)
print("P1 (3D):", P1)
print("Vanishing direction (3D):", v_dir)

origin, R = build_vehicle_frame(P0, P1, v_dir, dims['plate_height'])
box_3d = build_3d_box(origin, R, dims)
box_2d = project_box(K, box_3d)
img_final = draw_bbox(img_ud, box_2d)
print("origin (3D):", origin)
print("Rotation matrix (R):", R)

cv2.imshow("Combined 3D Box", cv2.resize(img_final, None, fx=0.35, fy=0.35))
cv2.imwrite(os.path.join(os.getcwd(), "method1", "results", "bbox.png"), img_final)
cv2.waitKey(0)
cv2.destroyAllWindows()