import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def to_homogeneous(p):
    return np.array([p[0], p[1], 1.0])

def to_line(p1, p2):
    return np.cross(p1, p2)

def triangulate_plate_points(K, p0_img, p1_img, plate_width):
    # Raggi visivi normalizzati
    ray0 = np.linalg.inv(K) @ to_homogeneous(p0_img)
    ray1 = np.linalg.inv(K) @ to_homogeneous(p1_img)
    ray0 = ray0 / np.linalg.norm(ray0)
    ray1 = ray1 / np.linalg.norm(ray1)
    
    # Metodo corretto per triangolazione con distanza nota
    # Risolviamo: ||t0*ray0 - t1*ray1|| = plate_width
    
    # Costruisci sistema lineare per trovare t0, t1
    # Minimizziamo ||A*t - b||^2 con vincolo di distanza
    A = np.column_stack([ray0, -ray1])
    
    # Usa SVD per soluzione robusta
    U, s, Vt = np.linalg.svd(A[:, :2])
    
    # Calcola l'angolo tra i raggi
    cos_angle = np.dot(ray0, ray1)
    sin_angle = np.sqrt(1 - cos_angle**2)
    
    # Distanza dal centro ottico ai punti (formula corretta)
    # Usando la legge dei seni nel triangolo camera-P0-P1
    baseline_angle = np.arccos(cos_angle)
    
    # Angoli nel triangolo
    # La targa sottende un angolo baseline_angle vista dalla camera
    # Usiamo la legge dei seni: a/sin(A) = b/sin(B) = c/sin(C)
    
    # Distanza media approssimativa (per inizializzazione)
    d_approx = plate_width / (2 * sin_angle)
    
    # Calcolo più preciso usando la geometria del triangolo
    # P0 e P1 sono equidistanti se la targa è frontale
    d0 = d_approx
    d1 = d_approx
    
    # Punti 3D
    P0_3d = d0 * ray0
    P1_3d = d1 * ray1
    
    # Verifica: la distanza dovrebbe essere plate_width
    actual_dist = np.linalg.norm(P1_3d - P0_3d)
    scale_correction = plate_width / actual_dist
    
    P0_3d *= scale_correction
    P1_3d *= scale_correction
    
    return P0_3d, P1_3d

def compute_vanishing_point(K, line1_pts, line2_pts):
    # Converti in linee omogenee
    l1 = to_line(to_homogeneous(line1_pts[0]), to_homogeneous(line1_pts[1]))
    l2 = to_line(to_homogeneous(line2_pts[0]), to_homogeneous(line2_pts[1]))
    
    # Punto di fuga
    v = np.cross(l1, l2)
    v = v / v[2]  # Normalizza
    
    # Direzione 3D
    v_3d = np.linalg.inv(K) @ v
    v_3d = v_3d / np.linalg.norm(v_3d)
    
    return v, v_3d

def build_car_coordinate_system(P0_3d, P1_3d, v_direction, car_dims):
    # Asse X: lungo la larghezza dell'auto (dalla targa)
    x_axis = P1_3d - P0_3d
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # v_direction è la direzione della profondità (lungo l'auto)
    # Ma dobbiamo assicurarci che sia ortogonale a x_axis
    y_temp = v_direction - np.dot(v_direction, x_axis) * x_axis
    y_axis = y_temp / np.linalg.norm(y_temp)
    
    # Asse Z: verso l'alto (prodotto vettoriale)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # Assicurati che z punti verso l'alto
    if z_axis[1] > 0:  # Se punta verso il basso
        z_axis = -z_axis
        y_axis = -y_axis  # Mantieni il sistema destrorso
    
    # Origine: centro della targa
    plate_center = (P0_3d + P1_3d) / 2
    
    # Proietta al suolo (assumendo che la targa sia a ~0.5m da terra)
    plate_height = car_dims['plate_height']
    origin = plate_center + plate_height * z_axis
    
    R = np.column_stack([x_axis, y_axis, z_axis])
    
    return origin, R

def create_3d_bounding_box(origin, R, car_dims):
    w = car_dims['width']
    l = car_dims['length'] 
    h = car_dims['height']
    
    # Offset della targa rispetto al centro dell'auto
    plate_offset_x = 0  # Targa centrata
    plate_offset_y = car_dims['rear_offset']  # Distanza targa-retro auto
    
    # Vertici nel sistema locale dell'auto
    # Origine = centro targa al suolo
    local_vertices = [
        # Base (z=0)
        [-w/2 - plate_offset_x, -plate_offset_y, 0],           # RBL
        [w/2 - plate_offset_x, -plate_offset_y, 0],            # RBR  
        [w/2 - plate_offset_x, l - plate_offset_y, 0],         # FBR
        [-w/2 - plate_offset_x, l - plate_offset_y, 0],        # FBL
        # Top (z=h)
        [-w/2 - plate_offset_x, -plate_offset_y, -h],          # RTL
        [w/2 - plate_offset_x, -plate_offset_y, -h],           # RTR
        [w/2 - plate_offset_x, l - plate_offset_y, -h],        # FTR  
        [-w/2 - plate_offset_x, l - plate_offset_y, -h],       # FTL
    ]
    
    # Trasforma nel sistema camera
    vertices_cam = []
    for v_local in local_vertices:
        v_cam = origin + R @ np.array(v_local)
        vertices_cam.append(v_cam)
    
    return vertices_cam

def draw_3d_box(img, K, vertices_3d, color=(0, 255, 0), thickness=2):
    """Disegna la bounding box 3D sull'immagine."""
    
    # Proietta i vertici
    vertices_2d = []
    for v3d in vertices_3d:
        v_proj = K @ v3d
        if v_proj[2] > 0:  # Davanti alla camera
            v2d = (int(v_proj[0]/v_proj[2]), int(v_proj[1]/v_proj[2]))
            vertices_2d.append(v2d)
        else:
            vertices_2d.append(None)
    
    # Definisci le connessioni
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Base
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top
        (0, 4), (1, 5), (2, 6), (3, 7)   # Verticali
    ]
    
    img_result = img.copy()
    
    # Disegna gli spigoli
    for i, j in edges:
        if vertices_2d[i] is not None and vertices_2d[j] is not None:
            cv2.line(img_result, vertices_2d[i], vertices_2d[j], color, thickness)
    
    # Disegna i vertici
    for i, v2d in enumerate(vertices_2d):
        if v2d is not None:
            cv2.circle(img_result, v2d, 5, (0, 0, 255), -1)
            cv2.putText(img_result, str(i), (v2d[0]+10, v2d[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_result

car_dimensions = {
    'width': 1.732,
    'length': 3.997,
    'height': 1.467,
    'plate_width': 0.520,
    'plate_height': 0.9, # Altezza targa da terra
    'rear_offset': 0.3, # Quanto la targa è avanti rispetto al retro
}

# Camera matrix (usa la tua)
K = np.array([
    [3201.4989, 0.0, 1939.82925],
    [0.0, 3206.37527, 1063.15413],
    [0.0, 0.0, 1]
], dtype=np.float64)

# Coefficienti di distorsione
dist = np.array([[0.24377, -1.5955, -0.0011528, 0.00041986, 3.5668]], dtype=np.float64)

# Carica immagine
img = cv2.imread(os.path.join(os.getcwd(), "Code", "1", "sunnyFrame.png"))
if img is None:
    raise FileNotFoundError("Immagine non trovata")

# Undistorci l'immagine
img_undist = cv2.undistort(img, K, dist)

# Punti nell'immagine (usa i tuoi punti corretti)
points_img = np.array([
    [1020, 1804],  # P0 = plate TL
    [1324, 1780],  # P1 = plate TR
    [792, 1768],   # P2 = rear-light L
    [1484, 1708]   # P3 = rear-light R
], dtype=np.float64)

# Undistorci i punti
points_undist = cv2.undistortPoints(points_img.reshape(-1,1,2), K, dist, P=K).reshape(-1,2)

# 1. Triangola i punti della targa
P0_3d, P1_3d = triangulate_plate_points(K, points_undist[0], points_undist[1], car_dimensions['plate_width'])

print(f"P0_3d: {P0_3d}")
print(f"P1_3d: {P1_3d}")
print(f"Distanza: {np.linalg.norm(P1_3d - P0_3d):.3f} m")

# 2. Calcola il punto di fuga dalle linee parallele
v_img, v_3d = compute_vanishing_point(
    K,
    [points_undist[0], points_undist[1]],  # Linea targa
    [points_undist[2], points_undist[3]]   # Linea fari
)

print(f"Vanishing direction: {v_3d}")

# 3. Costruisci il sistema di coordinate del veicolo
origin, R = build_car_coordinate_system(P0_3d, P1_3d, v_3d, car_dimensions)

print(f"Origin: {origin}")
print(f"Rotation matrix:\n{R}")

# 4. Crea la bounding box 3D
vertices_3d = create_3d_bounding_box(origin, R, car_dimensions)

# 5. Disegna il risultato
img_with_box = draw_3d_box(img_undist, K, vertices_3d, color=(0, 255, 0), thickness=3)

# Visualizza
scale_display = 0.35
h_display = int(img_with_box.shape[0] * scale_display)
w_display = int(img_with_box.shape[1] * scale_display)
img_display = cv2.resize(img_with_box, (w_display, h_display))

cv2.imshow("3D Bounding Box - Direct Method", img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()