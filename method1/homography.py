import numpy as np
import cv2
import os

def cad_to_plane_coords(points_cad, eps_parallel=1e-4):
    P0, P1, P2, P3 = np.asarray(points_cad, dtype=float)
    
    # Asse X lungo la targa
    vec_x = P1 - P0
    x1 = float(np.linalg.norm(vec_x))
    u = vec_x / x1
    
    # Proiezioni
    x2 = float(np.dot(P2 - P0, u))
    x3 = float(np.dot(P3 - P0, u))
    
    # Vettori ortogonali
    w2 = (P2 - P0) - x2 * u
    w3 = (P3 - P0) - x3 * u
    
    # Verifica parallelismo
    y2_abs = np.linalg.norm(w2)
    y3_abs = np.linalg.norm(w3)
    
    if abs(y2_abs - y3_abs) > eps_parallel:
        print(f"Warning: Non-parallel lights (diff={abs(y2_abs - y3_abs):.4f})")
    
    # Media delle distanze
    y_mag = 0.5 * (y2_abs + y3_abs)
    
    # Calcolo robusto del segno
    # Assumiamo che i punti luce siano "sopra" la targa nel senso del veicolo
    # Usiamo il punto medio tra P2 e P3
    mid_lights = 0.5 * (P2 + P3)
    mid_plate = 0.5 * (P0 + P1)
    
    # Vettore dalla targa ai fari
    plate_to_lights = mid_lights - mid_plate
    
    # Normale al piano (usando tutti e 4 i punti per robustezza)
    v1 = P1 - P0
    v2 = 0.5 * (P2 + P3) - P0
    n = np.cross(v1, v2)
    n = n / np.linalg.norm(n)
    
    # Il segno è positivo se w2 punta nella stessa "direzione" di plate_to_lights
    # rispetto al piano
    if y_mag > 0:
        w_avg = 0.5 * (w2 + w3)
        sign_y = 1.0 if np.dot(w_avg, plate_to_lights) > 0 else -1.0
    else:
        sign_y = 1.0
    
    y_common = sign_y * y_mag
    
    # Output
    plane_coords_2d = np.array([
        [0.0, 0.0],
        [x1, 0.0],
        [x2, y_common],
        [x3, y_common],
    ], dtype=np.float32)
    
    print(f"Plane coordinates:")
    print(f"  P0: (0, 0)")
    print(f"  P1: ({x1:.3f}, 0)")
    print(f"  P2: ({x2:.3f}, {y_common:.3f})")
    print(f"  P3: ({x3:.3f}, {y_common:.3f})")
    
    return plane_coords_2d

def calculate_homography(plane_coords_2d, image_coords_2d, method='DLT'):
    # Verifica input
    assert plane_coords_2d.shape == (4, 2), "Servono esattamente 4 punti piano"
    assert image_coords_2d.shape == (4, 2), "Servono esattamente 4 punti immagine"
    
    # Calcola omografia
    if method == 'DLT':
        # Con 4 punti usa il metodo diretto (più stabile)
        H, _ = cv2.findHomography(plane_coords_2d, image_coords_2d, method=0)
    else:
        # RANSAC ha senso solo con più punti
        H, mask = cv2.findHomography(plane_coords_2d, image_coords_2d, cv2.RANSAC, ransacReprojThreshold=5.0)
    
    if H is None:
        raise ValueError("Calcolo omografia fallito. Controlla i punti di input.")
    
    # Calcola metriche di qualità
    quality_metrics = validate_homography(H, plane_coords_2d, image_coords_2d)
    
    return H, quality_metrics

def validate_homography(H, plane_pts, img_pts):
    metrics = {}
    
    # 1. Determinante (non deve essere vicino a 0)
    det = np.linalg.det(H)
    metrics['determinant'] = det
    if abs(det) < 1e-10:
        print("WARNING: Determinante quasi zero!")
    
    # 2. Numero di condizione (misura stabilità numerica)
    cond = np.linalg.cond(H)
    metrics['condition_number'] = cond
    if cond > 1e6:
        print(f"WARNING: Numero di condizione alto: {cond:.2e}")
    
    # 3. Errore di reproiezione
    pts_h = np.hstack([plane_pts, np.ones((len(plane_pts), 1))])
    pts_reproj_h = (H @ pts_h.T).T
    pts_reproj = pts_reproj_h[:, :2] / pts_reproj_h[:, 2:3]
    
    errors = np.linalg.norm(pts_reproj - img_pts, axis=1)
    metrics['reprojection_errors'] = errors
    metrics['mean_error'] = np.mean(errors)
    metrics['max_error'] = np.max(errors)
    
    print(f"\nQualità omografia:")
    print(f"  Determinante: {det:.4f}")
    print(f"  Condition number: {cond:.2e}")
    print(f"  Errore medio reproiezione: {metrics['mean_error']:.2f} pixel")
    print(f"  Errore max reproiezione: {metrics['max_error']:.2f} pixel")
    
    return metrics

def _normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero vector cannot be normalised")
    return v / n

def _homog(p):
    return np.array([p[0], p[1], 1.0], dtype=float)

def _cross(a, b):
    return np.cross(a, b)

def estimate_scale_factor(K, H, p_frame, plate_width):
    """Compute scale factor triangulation method."""
    if p_frame.shape != (4, 2):
        raise ValueError("p_frame must be shape (4,2) - order: P0, P1, P2, P3")
    if K.shape != (3, 3):
        raise ValueError("K must be 3x3")

    K_inv = np.linalg.inv(K)

    # Build homogeneous rays
    rays = [_normalize(K_inv @ _homog(p)) for p in p_frame[:2]] # only P0, P1
    ray0, ray1 = rays

    # Vanishing point of the two parallel segments (targa & fari)
    l_plate = _cross(_homog(p_frame[0]), _homog(p_frame[1])) # P0 × P1
    l_lights = _cross(_homog(p_frame[2]), _homog(p_frame[3])) # P2 × P3
    V = _cross(l_plate, l_lights) # homogeneous vanishing point
    if np.allclose(V, 0):
        raise RuntimeError("Vanishing point at infinity or numerical issue - input points may not represent parallel segments")
    d_vp_cam = _normalize(K_inv @ V) # 3‑D direction of the common line

    print("d_vp_cam: ", d_vp_cam)

    # Angles needed for the law of sines triangulation
    cos_O = np.clip(ray0 @ ray1, -1.0, 1.0)
    alpha_O = np.arccos(cos_O)

    # Angles at P0 and P1 (between optical ray and plate direction)
    cos_P0 = np.clip((-ray0) @ d_vp_cam, -1.0, 1.0)
    alpha_P0 = np.arccos(cos_P0)
    cos_P1 = np.clip((-ray1) @ d_vp_cam, -1.0, 1.0)
    alpha_P1 = np.arccos(cos_P1)

    # Depths via law of sines
    depth_P0 = plate_width * np.sin(alpha_P1) / np.sin(alpha_O)
    depth_P1 = plate_width * np.sin(alpha_P0) / np.sin(alpha_O)

    # Triangulated 3‑D coordinates in camera frame
    P0_cam = depth_P0 * ray0 # since ray0 is unit‑norm.
    P1_cam = depth_P1 * ray1

    # Unscaled pose from inv(K)H  → extract o_pi_unscaled
    H_tilde = K_inv @ H
    o_pi_unscaled = H_tilde[:, 2] # third column (no scaling applied yet)

    # Ensure direction consistency (dot product > 0) to avoid negative scale
    sign = 1.0 if (P0_cam @ o_pi_unscaled) > 0 else -1.0

    if sign < 0:
        print("INFO: Applicato segno negativo (flip necessario)")

    # Scale factor such that scale * o_pi_unscaled  ≈  P0_cam
    scale = sign * (np.linalg.norm(P0_cam) / np.linalg.norm(o_pi_unscaled))

    return scale, P0_cam, P1_cam

def orthogonalize_rotation_preserving_r1(r1, r2):
    """Ortogonalizza preservando la direzione di r1."""
    
    # Normalizza r1 (preserva la direzione della targa)
    r1_norm = r1 / np.linalg.norm(r1)
    
    # Calcola r3 perpendicolare al piano
    r3_temp = np.cross(r1, r2)
    r3_norm = r3_temp / np.linalg.norm(r3_temp)
    
    # Ricalcola r2 per garantire ortogonalità
    r2_norm = np.cross(r3_norm, r1_norm)
    
    # Costruisci R ortogonale
    R = np.column_stack((r1_norm, r2_norm, r3_norm))
    
    # Verifica che sia una rotazione propria (det = +1)
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]  # Inverti r3 se necessario
    
    return R

def verify_scales_and_project(r1_scaled, r2_scaled, o_pi_scaled):
    # Calcola le scale
    scale_r1 = np.linalg.norm(r1_scaled)
    scale_r2 = np.linalg.norm(r2_scaled)
    
    print(f"\nScale dei vettori base:")
    print(f"  |r1_scaled| = {scale_r1:.4f}")
    print(f"  |r2_scaled| = {scale_r2:.4f}")
    print(f"  Rapporto = {scale_r1/scale_r2:.4f}")
    
    # Verifica che siano simili (dovrebbero esserlo per una camera calibrata)
    if abs(scale_r1/scale_r2 - 1.0) > 0.1:  # > 10% differenza
        print("WARNING: Scale molto diverse, possibile problema!")

def draw_vehicle_bounding_box(image, K, R_normalized, origin_cam, scale_r1, scale_r2, color=(0, 255, 0), thickness=2):
    # Estrai i vettori base normalizzati
    r1_n, r2_n, r3_n = R_normalized[:, 0], R_normalized[:, 1], R_normalized[:, 2]
   
    # Definisci gli 8 vertici della bounding box nel sistema veicolo
    # Sistema coordinate: origine = angolo sup sx targa
    # X: larghezza veicolo (destra), Y: altezza (su), Z: lunghezza (avanti)
    vertices_vehicle = {
        # Faccia posteriore (vicino alla targa)
        'RBL': np.array([-0.606, -0.9, 0.3],),
        'RBR': np.array([-0.606+CAR_W, -0.9, 0.3]),
        'RTL': np.array([-0.606, 1.467-0.9, 0.3]),
        'RTR': np.array([-0.606+CAR_W, CAR_H-0.9, 0.3]),
        
        # Faccia anteriore
        'FBL': np.array([-0.606, -0.9, -CAR_D+0.3]),
        'FBR': np.array([-0.606+1.958, -0.9, -CAR_D+0.3]),
        'FTL': np.array([-0.606, 1.467-0.9, -CAR_D+0.3]),
        'FTR': np.array([-0.606+CAR_W, 1.467-0.9, -CAR_D+0.3])
    }
    
    # Trasforma i vertici nel sistema camera
    vertices_cam = {}
    for name, v_vehicle in vertices_vehicle.items():
        # Applica scale e rotazione, poi trasla
        v_cam = origin_cam + (scale_r1 * v_vehicle[0] * r1_n + 
                             scale_r2 * v_vehicle[1] * r2_n + 
                             scale_r1 * v_vehicle[2] * r3_n)  # Usa scale_r1 per Z
        vertices_cam[name] = v_cam
    
    # Proietta i vertici nell'immagine
    vertices_2d = {}
    h_img, w_img = image.shape[:2]
    
    for name, v_cam in vertices_cam.items():
        # Proiezione prospettica
        p_hom = K @ v_cam
        
        if p_hom[2] <= 1e-6:  # Dietro la camera
            vertices_2d[name] = None
            print(f"Vertice {name} dietro la camera")
            continue
            
        # De-omogeneizzazione
        u = int(p_hom[0] / p_hom[2])
        v = int(p_hom[1] / p_hom[2])
        vertices_2d[name] = (u, v)
    
    # Copia l'immagine per disegnare
    img_result = image.copy()
    
    # Definisci gli spigoli da disegnare
    edges = [
        # Faccia posteriore
        ('RBL', 'RBR'), ('RBR', 'RTR'), ('RTR', 'RTL'), ('RTL', 'RBL'),
        # Faccia anteriore
        ('FBL', 'FBR'), ('FBR', 'FTR'), ('FTR', 'FTL'), ('FTL', 'FBL'),
        # Spigoli laterali
        ('RBL', 'FBL'), ('RBR', 'FBR'), ('RTL', 'FTL'), ('RTR', 'FTR')
    ]
    
    # Disegna gli spigoli
    for v1_name, v2_name in edges:
        v1 = vertices_2d.get(v1_name)
        v2 = vertices_2d.get(v2_name)
        
        if v1 is None or v2 is None:
            continue
            
        # Usa clipLine per gestire linee che escono dall'immagine
        clipped = cv2.clipLine((0, 0, w_img, h_img), v1, v2)
        if clipped[0]:
            cv2.line(img_result, clipped[1], clipped[2], color, thickness)
    
    # Disegna i vertici principali con colori diversi per orientamento
    vertex_colors = {
        'RBL': ((255, 0, 0), 'RBL'),
        'RBR': ((0, 0, 0), 'RBR'),
        'RTL': ((0, 255, 255), 'RTL'),
        'RTR': ((255, 0, 255), 'RTR'),
        'FBL': ((0, 0, 255), 'FBL'),
        'FBR': ((255, 255, 0), 'FBR'),
        'FTL': ((255, 128, 0), 'FTL'),
        'FTR': ((128, 0, 255), 'FTR')
    }
    
    for vertex_name, (vertex_color, label) in vertex_colors.items():
        pt = vertices_2d.get(vertex_name)
        if pt is not None and 0 <= pt[0] < w_img and 0 <= pt[1] < h_img:
            cv2.circle(img_result, pt, 6, vertex_color, -1)
            cv2.putText(img_result, label, (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, vertex_color, 3)
    
    # Opzionale: disegna gli assi coordinati
    origin = vertices_2d.get('RBL')
    if origin is not None:
        # Asse X (rosso)
        x_end = vertices_2d.get('RBR')
        if x_end is not None and all(p is not None for p in [origin, x_end]):
            cv2.arrowedLine(img_result, origin, (origin[0] + (x_end[0] - origin[0])//3, origin[1] + (x_end[1] - origin[1])//3), (0, 0, 255), 3, tipLength=0.2)
            
        # Asse Y (verde)
        y_end = vertices_2d.get('RTL')
        if y_end is not None and all(p is not None for p in [origin, y_end]):
            cv2.arrowedLine(img_result, origin, (origin[0] + (y_end[0] - origin[0])//3, origin[1] + (y_end[1] - origin[1])//3), (0, 255, 0), 3, tipLength=0.2)
            
        # Asse Z (blu)
        z_end = vertices_2d.get('FBL')
        if z_end is not None and all(p is not None for p in [origin, z_end]):
            cv2.arrowedLine(img_result, origin, (origin[0] + (z_end[0] - origin[0])//3, origin[1] + (z_end[1] - origin[1])//3), (255, 0, 0), 3, tipLength=0.2)
    
    return img_result

def visualize_extracted_points(img, points, labels):
    """Visualizza i punti estratti per verificare la correttezza."""
    img_vis = img.copy()
    
    colors = [(0,0,255), (0,0,0), (255,0,0), (255,255,0)]
    
    for i, (pt, label) in enumerate(zip(points, labels)):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img_vis, (x, y), 10, colors[i], -1)
        cv2.putText(img_vis, label, (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i], 2)
    
    # Disegna le linee di connessione
    cv2.line(img_vis, tuple(points[0].astype(int)), tuple(points[1].astype(int)), (0,255,0), 2)
    cv2.line(img_vis, tuple(points[2].astype(int)), tuple(points[3].astype(int)), (0,255,0), 2)
    
    return img_vis

K = np.array([
    [3.2014989e3, 0.0, 1.93982925e3],
    [0.0, 3.20637527e3, 1.06315413e3],
    [0.0, 0.0, 1]
], dtype=np.float64)

dist = np.array([[2.4377e-01, -1.5955e+00, -1.1528e-03, 4.1986e-04, 3.5668e+00]], dtype=np.float64)

PLATE_W = 0.520
PLATE_H = 0.130
Z_FARO = 0.150
CAR_W = 1.732
CAR_H = 1.467
CAR_D = 3.997

if __name__ == "__main__":
    frame = cv2.imread(os.path.join(os.getcwd(), "method1", "sunnyFrame.png"))
    if frame is None:
        raise FileNotFoundError("frame non trovato")
    
    print("Frame shape:", frame.shape)
    
    frame_ud = cv2.undistort(frame, K, dist)

    TTL = np.array([0.0, 0.0, 0.0]) # Target Top Left
    TTR = np.array([PLATE_W, 0.0, 0.0]) # Target Top Right
    LL = np.array([-0.340, 0.100, 0.0]) # Light Left
    LR = np.array([PLATE_W + 0.340, 0.100, 0.0]) # Light Right

    plane_pts = cad_to_plane_coords([TTL, TTR, LL, LR])
    print("Plane coordinates (homogeneous):\n", plane_pts)

    # punti hardcoded
    pix = np.array([
        [1020, 1804], # P0 = plate TL
        [1324, 1780], # P1 = plate TR
        [792, 1768], # P2 = rear-light L
        [1484, 1708] # P3 = rear-light R
    ], dtype=np.float64)

    pix = cv2.undistortPoints(pix.reshape(-1,1,2), K, dist, P=K).reshape(-1,2)

    # calcola l'omografia H
    H, mask = calculate_homography(plane_pts, pix)

    if H is not None:
        inv_K = np.linalg.inv(K) # K_matrix è la tua matrice K
        M_unscaled = inv_K @ H
        print("\nMatrice M_unscaled (inv(K) @ H):\n", M_unscaled)
    else:
        print("H non calcolata, impossibile procedere.")
        raise RuntimeError("Homography non calcolata, impossibile procedere.")

    r1_unscaled = M_unscaled[:, 0]
    r2_unscaled = M_unscaled[:, 1]
    o_pi_unscaled = M_unscaled[:, 2] # Questa è anche la traslazione t_unscaled

    print("\nr1_unscaled:\n", r1_unscaled)
    print("r2_unscaled:\n", r2_unscaled)
    print("o_pi_unscaled (t_unscaled):\n", o_pi_unscaled)

    scale, P0_tri, P1_tri = estimate_scale_factor(K, H, pix, PLATE_W)
    print(f"Scale factor ≈ {scale:.3f}\nP0_cam = {P0_tri}\nP1_cam = {P1_tri}")

    # 5. Calcola i punti scalati
    r1_scaled = scale * r1_unscaled
    r2_scaled = scale * r2_unscaled
    o_pi_scaled = scale * o_pi_unscaled
    print("\nr1_scaled:\n", r1_scaled)
    print("r2_scaled:\n", r2_scaled)
    print("o_pi_scaled (t_scaled):\n", o_pi_scaled)

    R = orthogonalize_rotation_preserving_r1(r1_scaled, r2_scaled)
    print("\nMatrice di rotazione R (ortogonalizzata):\n", R)
    r1_n, r2_n, r3_n = R[:,0], R[:,1], R[:,2]

    scale_r1 = np.linalg.norm(r1_scaled)
    scale_r2 = np.linalg.norm(r2_scaled)
    verify_scales_and_project(r1_scaled, r2_scaled, o_pi_scaled)

    frame_with_bb = draw_vehicle_bounding_box(frame_ud, K, R, o_pi_scaled, scale_r1, scale_r2, color=(0, 255, 0), thickness=3)

    # Usa questa funzione per verificare i punti
    labels = ['Targa TL', 'Targa TR', 'Faro L', 'Faro R']
    frame_with_bb = visualize_extracted_points(frame_with_bb, pix, labels)

    scale_display = 0.35
    h_display = int(frame_ud.shape[0] * scale_display)
    w_display = int(frame_ud.shape[1] * scale_display)
    frame_display = cv2.resize(frame_with_bb, (w_display, h_display))
    cv2.imshow("Bounding Box", frame_display)

    cv2.waitKey(0)
    cv2.destroyAllWindows()