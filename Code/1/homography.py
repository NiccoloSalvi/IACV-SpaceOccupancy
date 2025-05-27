"""
Changelog:
- v1.0 (Initial): Base script for homography calculation and bounding box projection.
- v1.1 (Task: Refactor draw_vehicle_bounding_box):
    - Modified `draw_vehicle_bounding_box` to use individual scales for each vehicle axis
      during vertex transformation.
    - `r3_scaled` (depth axis) is now calculated as `CAR_D * r3_n`.
    - `r1_scaled` and `r2_scaled` are used as vectors representing scaled x and y axes.
    - Corrected the 'FBR' (Front Bottom Right) vertex definition in `vertices_vehicle`
      to use `CAR_W` instead of a hardcoded value (1.958).
- v1.2 (Task: Correct sign_y calculation in cad_to_plane_coords):
    - Replaced the heuristic for `sign_y` in `cad_to_plane_coords` with a
      cross-product rule: `sign_y = np.sign(np.cross(u, w2)[2])`.
    - This provides a more robust geometric determination of the y-coordinate's sign.
- v1.3 (Task: Main block corrections - homography unpacking and scale ratio check):
    - Corrected the unpacking of `calculate_homography`'s return values in the
      `if __name__ == "__main__":` block to `H, quality_metrics = ...`.
    - Added a strict 1% scale ratio check for `|r1_scaled|` vs `|r2_scaled|`
      after their calculation in the main block, raising a ValueError if exceeded.
- v1.4 (Task: Robustness and Portability Enhancements):
    - Implemented conditional display in `if __name__ == "__main__":`:
        - Uses `cv2.imshow` if `DISPLAY` environment variable is set.
        - Saves the output image to `debug_bbox.png` (full resolution `frame_with_bb`)
          using `cv2.imwrite` and prints a message if `DISPLAY` is not set.
    - Enhanced error/warning messages for numerical degeneracies:
        - `validate_homography`: More informative messages for near-zero determinant
          and high condition number.
        - `estimate_scale_factor`: Added a warning if optical rays for P0 and P1
          are nearly collinear (small `alpha_O`).
        - `orthogonalize_rotation_preserving_r1`: Added ValueError exceptions for
          zero input vector `r1` or collinear `r1` and `r2` (unable to form a basis).
        - `draw_rear_bounding_box`: Updated error message for collinear `r1_vec`, `r2_vec`.
- v1.5 (Task: Finalization - Changelog and Comments Review):
    - Added this comprehensive changelog at the top of the file.
    - Reviewed and ensured sufficient inline comments for all significant modifications.
    - Confirmed `debug_bbox.png` saving mechanism for headless environments.
"""
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
    
    # Calcolo del segno di y_common basato sulla regola della mano destra. (Task v1.2)
    # u è il vettore unitario lungo l'asse x del piano della targa (P0->P1).
    # w2 è il vettore da P0 a P2, proiettato sul piano ortogonale a u.
    # Geometricamente, w2 rappresenta la componente y di P2 nel sistema di coordinate 2D locale del piano.
    # Il segno della componente z del prodotto vettoriale u x w2 determina se P2
    # è "sopra" (y positiva) o "sotto" (y negativa) l'asse x definito da P0-P1,
    # assumendo una convenzione destrorsa per il sistema di coordinate CAD (X, Y, Z).
    # Se P0, P1, P2 sono collineari, w2 sarà un vettore nullo ([0,0,0]),
    # quindi il prodotto vettoriale sarà [0,0,0], e np.sign(0) darà 0.
    # In questo caso, y_common = 0 * y_mag = 0, che è corretto.
    sign_y = np.sign(np.cross(u, w2)[2]) # Corrected sign_y calculation (Task v1.2)
    
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
        print(f"Warning: Homography determinant is close to zero (val={det:.2e}), results may be unstable.")
    
    # 2. Numero di condizione (misura stabilità numerica)
    cond = np.linalg.cond(H)
    metrics['condition_number'] = cond
    if cond > 1e6: # Typical threshold, can be application-dependent
        print(f"Warning: Homography condition number is high (val={cond:.2e}), indicating potential numerical instability.")
    
    # 3. Errore di reproiezione
    pts_h = np.hstack([plane_pts, np.ones((len(plane_pts), 1))])
    pts_reproj_h = (H @ pts_h.T).T
    pts_reproj = pts_reproj_h[:, :2] / pts_reproj_h[:, 2:3]
    
    errors = np.linalg.norm(pts_reproj - img_pts, axis=1)
    metrics['reprojection_errors'] = errors
    metrics['mean_error'] = np.mean(errors)
    metrics['max_error'] = np.max(errors)
    
    print(f"\nQualità omografia:")
    print(f"  Determinante: {det:.2e}") # Adjusted format for consistency
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
    eps_collinear_rad = 1e-3 # Threshold for detecting nearly collinear rays

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

    if abs(alpha_O) < eps_collinear_rad:
        print(f"Warning: Optical rays for P0 and P1 are nearly collinear (angle={alpha_O:.2f} rad). Depth estimation may be inaccurate.")

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
    eps = 1e-7 # Small epsilon for checking zero vectors

    # Normalizza r1 (preserva la direzione della targa)
    norm_r1 = np.linalg.norm(r1)
    if norm_r1 < eps:
        raise ValueError("Input r1 to orthogonalize_rotation_preserving_r1 is a zero vector.")
    r1_norm = r1 / norm_r1
    
    # Calcola r3 perpendicolare al piano
    r3_temp = np.cross(r1_norm, r2) # Use r1_norm for robustness if r2 is huge
    norm_r3_temp = np.linalg.norm(r3_temp)
    if norm_r3_temp < eps:
        # This happens if r1 (and thus r1_norm) and r2 are collinear
        raise ValueError("Vectors r1 and r2 are collinear, cannot form a basis in orthogonalize_rotation_preserving_r1.")
    r3_norm = r3_temp / norm_r3_temp
    
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

    # Definisci i vettori base scalati per la trasformazione dei vertici (Task v1.1)
    # scale_r1 e scale_r2 sono le magnitudini passate alla funzione (magnitudini di r1_scaled, r2_scaled from main)
    r1_scaled_vec = scale_r1 * r1_n # This is the scaled vehicle X-axis vector in camera coordinates
    r2_scaled_vec = scale_r2 * r2_n # This is the scaled vehicle Y-axis vector in camera coordinates
    # CAR_D è la dimensione fisica lungo l'asse z del veicolo (profondità)
    # r3_n è il vettore normalizzato dell'asse z del veicolo nel sistema camera
    # r3_scaled_vehicle_depth_vec is the scaled vehicle Z-axis (depth) vector in camera coordinates (Task v1.1)
    r3_scaled_vehicle_depth_vec = CAR_D * r3_n 
   
    # Definisci gli 8 vertici della bounding box nel sistema veicolo
    # Sistema coordinate: origine = un punto di riferimento sul veicolo (es. angolo della targa),
    # con X, Y, Z allineati agli assi del veicolo. Le coordinate qui sono relative a tale origine.
    # X: larghezza veicolo (destra), Y: altezza (su), Z: lunghezza (avanti, verso il fronte del veicolo)
    vertices_vehicle = {
        # Faccia posteriore (vicino alla targa)
        'RBL': np.array([-0.606, -0.9, 0.3],), # Rear-Bottom-Left
        'RBR': np.array([-0.606+CAR_W, -0.9, 0.3]), # Rear-Bottom-Right (CAR_W used)
        'RTL': np.array([-0.606, 1.467-0.9, 0.3]), # Rear-Top-Left
        'RTR': np.array([-0.606+CAR_W, CAR_H-0.9, 0.3]), # Rear-Top-Right (CAR_W, CAR_H used)
        
        # Faccia anteriore
        'FBL': np.array([-0.606, -0.9, -CAR_D+0.3]), # Front-Bottom-Left (CAR_D used for depth)
        'FBR': np.array([-0.606+CAR_W, -0.9, -CAR_D+0.3]), # Front-Bottom-Right (CAR_W, CAR_D used) - Corrected to use CAR_W (Task v1.1)
        'FTL': np.array([-0.606, 1.467-0.9, -CAR_D+0.3]), # Front-Top-Left (CAR_D, CAR_H used)
        'FTR': np.array([-0.606+CAR_W, 1.467-0.9, -CAR_D+0.3]) # Front-Top-Right (CAR_W, CAR_H, CAR_D used)
    }
    
    # Trasforma i vertici nel sistema camera
    vertices_cam = {}
    for name, v_vehicle in vertices_vehicle.items():
        # Applica scale e rotazione, poi trasla
        # v_vehicle[0] è lungo l'asse x del veicolo (r1_n direction)
        # v_vehicle[1] è lungo l'asse y del veicolo (r2_n direction)
        # v_vehicle[2] è lungo l'asse z del veicolo (r3_n direction)
        # Vertex transformation using individually scaled axes (Task v1.1)
        v_cam = origin_cam + (v_vehicle[0] * r1_scaled_vec +
                             v_vehicle[1] * r2_scaled_vec +
                             v_vehicle[2] * r3_scaled_vehicle_depth_vec)
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
        'RBL': ((255, 0, 0), 'RBL'),    # Blu - Rear Bottom Left
        'RBR': ((0, 255, 0), 'RBR'),    # Verde - Rear Bottom Right  
        'FBL': ((0, 0, 255), 'FBL'),    # Rosso - Front Bottom Left
        'FBR': ((255, 255, 0), 'FBR')   # Giallo - Front Bottom Right
    }
    
    for vertex_name, (vertex_color, label) in vertex_colors.items():
        pt = vertices_2d.get(vertex_name)
        if pt is not None and 0 <= pt[0] < w_img and 0 <= pt[1] < h_img:
            cv2.circle(img_result, pt, 6, vertex_color, -1)
            cv2.putText(img_result, label, (pt[0] + 10, pt[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, vertex_color, 2)
    
    # Opzionale: disegna gli assi coordinati
    origin = vertices_2d.get('RBL')
    if origin is not None:
        # Asse X (rosso)
        x_end = vertices_2d.get('RBR')
        if x_end is not None and all(p is not None for p in [origin, x_end]):
            cv2.arrowedLine(img_result, origin, 
                          (origin[0] + (x_end[0] - origin[0])//3, 
                           origin[1] + (x_end[1] - origin[1])//3),
                          (0, 0, 255), 3, tipLength=0.2)
            
        # Asse Y (verde)
        y_end = vertices_2d.get('RTL')
        if y_end is not None and all(p is not None for p in [origin, y_end]):
            cv2.arrowedLine(img_result, origin,
                          (origin[0] + (y_end[0] - origin[0])//3,
                           origin[1] + (y_end[1] - origin[1])//3),
                          (0, 255, 0), 3, tipLength=0.2)
            
        # Asse Z (blu)
        z_end = vertices_2d.get('FBL')
        if z_end is not None and all(p is not None for p in [origin, z_end]):
            cv2.arrowedLine(img_result, origin,
                          (origin[0] + (z_end[0] - origin[0])//3,
                           origin[1] + (z_end[1] - origin[1])//3),
                          (255, 0, 0), 3, tipLength=0.2)
    
    return img_result

def draw_rear_bounding_box(frame, K_matrix, r1_vec, r2_vec, o_pi_vec, width_bb, height_bb, depth_bb, color=(0,255,0), thickness=2):
    frame_with_bb = frame.copy()

    vec_width_cam = width_bb * r1_vec  # Questo assume che r1_vec sia per X_w=1
    vec_height_cam = height_bb * r2_vec # Questo assume che r2_vec sia per Y_w=1
    
    # Calcola la normale al piano per la direzione della profondità
    # Assicurati che r1_vec e r2_vec non siano collineari
    normal_vec_cam = np.cross(r1_vec, r2_vec)
    if np.linalg.norm(normal_vec_cam) < 1e-6: # Epsilon for collinearity check
        print("Error: r1_vec and r2_vec are collinear in `draw_rear_bounding_box`. Cannot define a unique normal vector for depth.")
        # Potresti usare una direzione di default o propagare l'errore
        # Per ora, creiamo una normale fittizia se questo accade, ma è un segnale di problemi precedenti.
        # Se r1 punta lungo X e r2 lungo Y, la normale dovrebbe essere lungo Z.
        # Tentativo di recupero (non ideale, indica problemi a monte):
        if np.allclose(r1_vec / np.linalg.norm(r1_vec), [1,0,0]) and \
           np.allclose(r2_vec / np.linalg.norm(r2_vec), [0,1,0]):
            normal_vec_cam = np.array([0,0,1.0])
        else: # Non si può fare molto altro senza più informazioni
            return frame_with_bb # Restituisce immagine originale

    normal_vec_cam = normal_vec_cam / np.linalg.norm(normal_vec_cam) # Rendi unitario
    vec_depth_cam = depth_bb * normal_vec_cam # Vettore per la profondità

    # 2. Calcolare gli 8 Vertici della Bounding Box in coordinate camera
    # L'origine della BB è o_pi_vec (che corrisponde a P0_plane, es. angolo inf sx targa)
    # Vertici del piano "posteriore" (Z=0 locale della BB)
    v000 = o_pi_vec
    v100 = o_pi_vec + vec_width_cam
    v010 = o_pi_vec + vec_height_cam
    v110 = o_pi_vec + vec_width_cam + vec_height_cam
    
    # Vertici del piano "anteriore" (Z=depth_bb locale della BB)
    # Questi puntano "fuori" dal piano targa/luci se depth_bb > 0 e normal_vec_cam punta fuori
    v001 = o_pi_vec + vec_depth_cam
    v101 = o_pi_vec + vec_width_cam + vec_depth_cam
    v011 = o_pi_vec + vec_height_cam + vec_depth_cam
    v111 = o_pi_vec + vec_width_cam + vec_height_cam + vec_depth_cam

    vertices_3d_cam = np.array([
        v000, v100, v010, v110,
        v001, v101, v011, v111
    ])

    # 3. Proiettare i Vertici 3D sull'Immagine 2D
    # K @ P_cam, dove P_cam è (3,N) e K è (3,3)
    # vertices_3d_cam è (8,3), quindi trasponilo
    projected_hom = K_matrix @ vertices_3d_cam.T # Risultato (3,8)
    
    # De-omogeneizza e gestisci punti dietro la camera
    vertices_2d_image = []
    valid_projection = True
    for i in range(projected_hom.shape[1]):
        s = projected_hom[2, i]
        if s <= 1e-3: # Punto dietro o troppo vicino al piano immagine
            # print(f"Warning: Vertice {i} della BB è dietro o sul piano della camera (s={s}). Non verrà disegnato.")
            valid_projection = False # O gestisci disegnando solo le parti visibili
            # Per semplicità, se un punto è dietro, potremmo non disegnare l'intera BB
            # o sostituire il punto con None e gestire il disegno di conseguenza.
            # Qui, se un punto è problematico, non disegniamo la BB.
            # return frame_with_bb # Opzione drastica
            vertices_2d_image.append(None) # Segna come non valido
            continue

        u = int(projected_hom[0, i] / s)
        v = int(projected_hom[1, i] / s)
        vertices_2d_image.append((u, v))
    
    # if not valid_projection and not any(pt is None for pt in vertices_2d_image):
        # return frame_with_bb # Se abbiamo deciso di non disegnare se un punto è dietro

    # 4. Disegnare le Areste della Bounding Box sull'Immagine
    # Definisci le connessioni tra i vertici (indici 0-7)
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Faccia posteriore
        (4, 5), (5, 7), (7, 6), (6, 4),  # Faccia anteriore
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connessioni tra facce
    ]

    for pt1_idx, pt2_idx in edges:
        pt1 = vertices_2d_image[pt1_idx]
        pt2 = vertices_2d_image[pt2_idx]
        
        # Disegna la linea solo se entrambi i punti sono validi (proiettati correttamente)
        if pt1 is not None and pt2 is not None:
            # Verifica se i punti sono all'interno dei limiti dell'immagine (opzionale ma buona pratica)
            h_frame, w_frame = frame_with_bb.shape[:2]
            if (0 <= pt1[0] < w_frame and 0 <= pt1[1] < h_frame and 0 <= pt2[0] < w_frame and 0 <= pt2[1] < h_frame):
                cv2.line(frame_with_bb, pt1, pt2, color, thickness)
            else:
                # Puoi usare cv2.clipLine per disegnare solo la parte visibile
                clipped = cv2.clipLine((0, 0, w_frame, h_frame), pt1, pt2)
                if clipped[0]:
                    cv2.line(frame_with_bb, clipped[1], clipped[2], color, thickness)

    return frame_with_bb

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
    frame = cv2.imread(os.path.join(os.getcwd(), "Code", "1", "sunnyFrame.png"))
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
    H, quality_metrics = calculate_homography(plane_pts, pix) # Corrected unpacking (Task v1.3)

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

    # SOLUZIONE: mantieni la scala originale
    scale_r1 = np.linalg.norm(r1_scaled) # Magnitude of the scaled vehicle X-axis vector
    scale_r2 = np.linalg.norm(r2_scaled) # Magnitude of the scaled vehicle Y-axis vector

    # Scale Ratio Check (Task v1.3)
    # Enforces that the derived scales for X and Y axes are similar,
    # which is expected for a reasonably calibrated camera and good point correspondences.
    if abs(scale_r1 / scale_r2 - 1.0) > 0.01: # 1% threshold
        raise ValueError(f"Error: Scale ratio |r1|/|r2| = {scale_r1/scale_r2:.4f} differs by more than 1%. Check camera calibration or point correspondences.")

    # R = np.column_stack((r1_scaled, r2_scaled, np.cross(r1_scaled, r2_scaled)))
    # U, _, Vt = np.linalg.svd(R)
    # R = U @ Vt
    # r1_n, r2_n, r3_n = R[:,0], R[:,1], R[:,2]

    """
    # wheel_offset = np.array([-0.606+0.386, 1.467-0.9, 0.3]) # front-TL
    # wheel_offset = np.array([-0.606, -0.9, 0.3]) # WBL
    # wheel_offset = np.array([-0.606+CAR_W, -0.9, 0.3]) # WBR
    # wheel_offset = np.array([0.52, 0, 0]) # TTR
    # wheel_offset = np.array([-0.606, -0.9, +CAR_D-0.3]) # WTL
    P_wheel_cam = (o_pi_scaled +
                   wheel_offset[0] * scale_r1 * r1_n +
                   wheel_offset[1] * scale_r2 * r2_n +
                   wheel_offset[2] * scale_r1 * r3_n)

    # proiezione in pixel:
    p_hom = K @ P_wheel_cam
    if p_hom[2] <= 1e-6:
        raise ValueError("Il punto è dietro il piano immagine")

    u = p_hom[0] / p_hom[2]
    v = p_hom[1] / p_hom[2]
    print(f"Punto proiettato: (u,v) = ({u:.1f}, {v:.1f})")
    h_frame, w_frame = frame_ud.shape[:2]
    if 0 <= u < w_frame and 0 <= v < h_frame:
        cv2.circle(frame_ud, (int(round(u)), int(round(v))), 8, (0,0,255), -1)
    else:
        print("La proiezione cade fuori immagine")
    """

    verify_scales_and_project(r1_scaled, r2_scaled, o_pi_scaled)

    frame_with_bb = draw_vehicle_bounding_box(frame_ud, K, R, o_pi_scaled, scale_r1, scale_r2, color=(0, 255, 0), thickness=3)

    scale_display = 0.35 # Rescale for display if shown
    h_display = int(frame_ud.shape[0] * scale_display)
    w_display = int(frame_ud.shape[1] * scale_display)
    frame_display_resized = cv2.resize(frame_with_bb, (w_display, h_display)) # Resized version for display

    # Conditional display or save based on DISPLAY environment variable (Task v1.4)
    if os.environ.get("DISPLAY"):
        cv2.imshow("Bounding Box", frame_display_resized) # Show resized image
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        output_path = "debug_bbox.png"
        cv2.imwrite(output_path, frame_with_bb) # Save the full-resolution image with bounding box
        print(f"DISPLAY environment variable not set. Image saved to {output_path}")