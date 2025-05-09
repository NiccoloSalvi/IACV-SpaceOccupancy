import cv2
import numpy as np

def cad_to_plane_coords(points_cad, eps_parallel=1e-4):
    P0, P1, P2, P3 = np.asarray(points_cad, dtype=float)

    # 1. Axis X  (plate direction)
    vec_x = P1 - P0
    x1 = float(np.linalg.norm(vec_x))  # width of the plate (metres)
    if x1 == 0: # Corrected from if x1 < 1e-9 for clarity, though 1e-9 is safer for float comparisons
        raise ValueError("P0 and P1 must be distinct.")
    u = vec_x / x1                     # unit vector along X

    # 2. Projections of the light points onto X to get x2, x3
    x2 = float(np.dot(P2 - P0, u))
    x3 = float(np.dot(P3 - P0, u))

    # 3. Residual vectors orthogonal to u –> distance between the two lines
    w2 = (P2 - P0) - x2 * u
    w3 = (P3 - P0) - x3 * u
    y2_abs = np.linalg.norm(w2) # Renamed from y2 to y2_abs for clarity
    y3_abs = np.linalg.norm(w3) # Renamed from y3 to y3_abs for clarity

    print(f"y2_abs = {y2_abs:.4f} m, y3_abs = {y3_abs:.4f} m") # Corrected print f-string

    if abs(y2_abs - y3_abs) > eps_parallel: # Used y2_abs, y3_abs
        # Consider making this a warning instead of an error, or making eps_parallel an argument
        # raise ValueError( # Or print a warning
        print(
            f"Warning: Light points might not be on a line parallel to the plate within "
            f"±{eps_parallel} m (|y2_abs - y3_abs| = {abs(y2_abs - y3_abs):.4f}). Using average."
        )

    n = np.cross(u, w2)  # normal vector of the plane
    n = n / np.linalg.norm(n)  # unit normal vector

    # 1) compute the magnitude of the offset (always ≥0)
    y_mag = 0.5 * (y2_abs + y3_abs)

    # 2) compute sign_y  (as in the corrected version)
    #    n is any vector normal to the CAD plane,
    #    for instance n = np.cross(u, P3-P0) normalized.
    n = _normalize(np.cross(u, (P3 - P0)))
    sign_y = np.sign(np.dot(np.cross(u, w2), n))
    if sign_y == 0:
        sign_y = 1.0

    # 3) form the signed coordinate
    y_common = sign_y * y_mag

    # 4. Assemble 2‑D coordinates (for findHomography)
    # findHomography expects (N, 2) or (N, 1, 2) of float32 typically
    plane_coords_2d = np.array([
        [0.0, 0.0],          # P0
        [x1, 0.0],           # P1
        [x2, y_common],      # P2 (using y_common instead of y)
        [x3, y_common],      # P3 (using y_common instead of y)
    ], dtype=np.float32) # Changed to float32 and (N,2) output

    # The original code returned homogeneous coordinates.
    # If you want to stick to the professor's (0,0), (x1,0), (x2,y), (x3,y) for H input,
    # this (N,2) format is what cv2.findHomography takes for srcPoints.
    # If you need homogeneous (N,3) for other reasons, you can hstack a column of ones later.
    # For now, returning (N,2) is more direct for findHomography.
    
    # If the task was to return the homogeneous points as per the original code:
    # plane_hom = np.array([
    #     [0.0, 0.0, 1.0],
    #     [x1, 0.0, 1.0],
    #     [x2, y_common, 1.0], # using y_common
    #     [x3, y_common, 1.0], # using y_common
    # ])
    # return plane_hom
    return plane_coords_2d

def calculate_homography(plane_coords_2d, image_coords_2d):
    H, mask = cv2.findHomography(plane_coords_2d, image_coords_2d, cv2.RANSAC, ransacReprojThreshold=5.0)
    # H, mask = cv2.findHomography(plane_coords_2d, image_coords_2d, 0) # Per DLT semplice

    if H is None:
        print("Error: Homography calculation failed.")
    
    return H, mask

def _normalize(v: np.ndarray) -> np.ndarray:
    """Return v/‖v‖ with shape preserved."""
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero vector cannot be normalised")
    return v / n

def _homog(p: np.ndarray) -> np.ndarray:
    """Append 1 to a 2‑D pixel point to build a 3‑D homogeneous vector."""
    return np.array([p[0], p[1], 1.0], dtype=float)

def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortcut for np.cross with float64 output."""
    return np.cross(a, b)

def estimate_scale_factor(K, H, p_img, plate_width):
    """Compute scale factor via professor's triangulation method."""
    # ---- sanity checks -----------------------------------------------------
    if p_img.shape != (4, 2):
        raise ValueError("p_img must be shape (4,2) – order: P0, P1, P2, P3")
    if K.shape != (3, 3):
        raise ValueError("K must be 3×3")

    # ----------------------------------------------------------------------
    # Step 0 · Prepare frequently used matrices / functions
    # ----------------------------------------------------------------------
    K_inv = np.linalg.inv(K)

    # Build homogeneous rays   r_i = K⁻¹ p̃_i  (unnormalised)
    rays = [_normalize(K_inv @ _homog(p)) for p in p_img[:2]]  # only P0, P1
    ray0, ray1 = rays

    # ----------------------------------------------------------------------
    # Step 1 · Vanishing point of the two parallel segments (targa & fari)
    # ----------------------------------------------------------------------
    #   line = p_i × p_j  (in homogeneous image coords)
    l_plate = _cross(_homog(p_img[0]), _homog(p_img[1]))  # P0 × P1
    l_lights = _cross(_homog(p_img[2]), _homog(p_img[3]))  # P2 × P3
    V = _cross(l_plate, l_lights)  # homogeneous vanishing point
    if np.allclose(V, 0):
        raise RuntimeError("Vanishing point at infinity or numerical issue – "
                           "input points may not represent parallel segments")
    d_vp_cam = _normalize(K_inv @ V)  # 3‑D direction of the common line

    # ----------------------------------------------------------------------
    # Step 2 · Angles needed for the law of sines triangulation
    # ----------------------------------------------------------------------
    # ∠O  (between the two viewing rays)
    cos_O = np.clip(ray0 @ ray1, -1.0, 1.0)
    alpha_O = np.arccos(cos_O)

    # Angles at P0 and P1 (between optical ray and plate direction)
    cos_P0 = np.clip((-ray0) @ d_vp_cam, -1.0, 1.0)
    alpha_P0 = np.arccos(cos_P0)
    cos_P1 = np.clip((-ray1) @ d_vp_cam, -1.0, 1.0)
    alpha_P1 = np.arccos(cos_P1)

    # ----------------------------------------------------------------------
    # Step 3 · Depths via law of sines
    # ----------------------------------------------------------------------
    # depth_P0 / sin(alpha_P1) = plate_width / sin(alpha_O)
    depth_P0 = plate_width * np.sin(alpha_P1) / np.sin(alpha_O)
    depth_P1 = plate_width * np.sin(alpha_P0) / np.sin(alpha_O)

    # Triangulated 3‑D coordinates in camera frame
    P0_cam = depth_P0 * ray0  # since ray0 is unit‑norm.
    P1_cam = depth_P1 * ray1

    # ----------------------------------------------------------------------
    # Step 4 · Unscaled pose from inv(K)H  → extract o_pi_unscaled
    # ----------------------------------------------------------------------
    H_tilde = K_inv @ H
    o_pi_unscaled = H_tilde[:, 2]  # third column (no scaling applied yet)

    # Ensure direction consistency (dot product > 0) to avoid negative scale
    sign = 1.0 if (P0_cam @ o_pi_unscaled) > 0 else -1.0

    # Scale factor such that scale * o_pi_unscaled  ≈  P0_cam
    scale = sign * (np.linalg.norm(P0_cam) / np.linalg.norm(o_pi_unscaled))

    return scale, P0_cam, P1_cam

def draw_rear_bounding_box(image: np.ndarray, 
                           K_matrix: np.ndarray, 
                           r1_vec: np.ndarray, 
                           r2_vec: np.ndarray, 
                           o_pi_vec: np.ndarray,
                           width_bb: float,    # Larghezza della BB (es. W_targa)
                           height_bb: float,   # Altezza della BB (es. y_common dei fari)
                           depth_bb: float,    # Profondità della BB (es. 0.05 per 5cm, o 0 per superficie)
                           color: tuple = (0, 255, 0), # Verde BGR
                           thickness: int = 2) -> np.ndarray:
    """
    Disegna una bounding box 3D sul retro del veicolo sull'immagine.

    Args:
        image: L'immagine su cui disegnare (copia modificata verrà restituita).
        K_matrix: Matrice di calibrazione intrinseca 3x3.
        r1_vec: Vettore base r1 (scalato) nel sistema camera, definisce l'asse X del piano mondo.
        r2_vec: Vettore base r2 (scalato) nel sistema camera, definisce l'asse Y del piano mondo.
        o_pi_vec: Origine o_pi (scalata) del piano mondo nel sistema camera.
        width_bb: Larghezza desiderata della bounding box lungo la direzione r1.
        height_bb: Altezza desiderata della bounding box lungo la direzione r2.
        depth_bb: Profondità desiderata della bounding box lungo la normale al piano (r1,r2).
        color: Colore BGR per le linee.
        thickness: Spessore delle linee.

    Returns:
        Immagine con la bounding box disegnata.
    """
    img_with_bb = image.copy()

    # 1. Definire gli assi della Bounding Box nel sistema camera
    # r1_vec e r2_vec definiscono le direzioni e le scale per le unità X e Y del piano mondo.
    # L'asse Z della BB sarà perpendicolare al piano definito da r1 e r2.
    
    # Vettore per la larghezza (lungo r1)
    # La lunghezza di r1_vec è la scala per "1 unità X del piano mondo".
    # Quindi, se width_bb è in quelle stesse unità (es. W_targa), moltiplichiamo per width_bb.
    # Se r1_vec è già il vettore che va da (0,0) a (width_bb,0) allora non serve moltiplicare.
    # Dalla definizione [r1 r2 o_pi] = inv(K)H, r1 è l'immagine di (1,0,1) del piano mondo - o_pi.
    # Quindi P_cam = Xw*r1 + Yw*r2 + o_pi.
    # Se P_targa_dx_plane = (width_bb, 0), allora P_targa_dx_cam = width_bb * r1_vec + o_pi_vec.
    # Il vettore che definisce la larghezza è width_bb * r1_vec.
    
    vec_width_cam = width_bb * r1_vec  # Questo assume che r1_vec sia per X_w=1
    vec_height_cam = height_bb * r2_vec # Questo assume che r2_vec sia per Y_w=1
    
    # Calcola la normale al piano per la direzione della profondità
    # Assicurati che r1_vec e r2_vec non siano collineari
    normal_vec_cam = np.cross(r1_vec, r2_vec)
    if np.linalg.norm(normal_vec_cam) < 1e-6:
        print("Error: r1_vec and r2_vec sono collineari, impossibile definire la normale.")
        # Potresti usare una direzione di default o propagare l'errore
        # Per ora, creiamo una normale fittizia se questo accade, ma è un segnale di problemi precedenti.
        # Se r1 punta lungo X e r2 lungo Y, la normale dovrebbe essere lungo Z.
        # Tentativo di recupero (non ideale, indica problemi a monte):
        if np.allclose(r1_vec / np.linalg.norm(r1_vec), [1,0,0]) and \
           np.allclose(r2_vec / np.linalg.norm(r2_vec), [0,1,0]):
            normal_vec_cam = np.array([0,0,1.0])
        else: # Non si può fare molto altro senza più informazioni
            return img_with_bb # Restituisce immagine originale

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
            # return img_with_bb # Opzione drastica
            vertices_2d_image.append(None) # Segna come non valido
            continue

        u = int(projected_hom[0, i] / s)
        v = int(projected_hom[1, i] / s)
        vertices_2d_image.append((u, v))
    
    # if not valid_projection and not any(pt is None for pt in vertices_2d_image):
        # return img_with_bb # Se abbiamo deciso di non disegnare se un punto è dietro

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
            h_img, w_img = img_with_bb.shape[:2]
            if (0 <= pt1[0] < w_img and 0 <= pt1[1] < h_img and 0 <= pt2[0] < w_img and 0 <= pt2[1] < h_img):
                cv2.line(img_with_bb, pt1, pt2, color, thickness)
            else:
                # Puoi usare cv2.clipLine per disegnare solo la parte visibile
                clipped = cv2.clipLine((0, 0, w_img, h_img), pt1, pt2)
                if clipped[0]:
                    cv2.line(img_with_bb, clipped[1], clipped[2], color, thickness)


    return img_with_bb

def verification(K, r1, r2, o_pi, P0_tri, P1_tri, img):
    # check if o_pi_scaled and P0_tri are similar
    print("\n--- Verification 1: o_pi_scaled vs P0_tri ---")
    diff_o_pi_vs_P0_tri = np.linalg.norm(o_pi_scaled - P0_tri)
    print(f"o_pi_scaled: {o_pi_scaled}")
    print(f"P0_tri:      {P0_tri}")
    print(f"Euclidean distance between o_pi_scaled and P0_tri: {diff_o_pi_vs_P0_tri:.6e} meters")
    if diff_o_pi_vs_P0_tri < 1e-3: # Threshold for "close enough", e.g., 1mm
        print("SUCCESS: o_pi_scaled is very close to P0_tri.")
    else:
        print("WARNING: o_pi_scaled differs significantly from P0_tri.")
    
    # --- Verification Step 2: Compare (o_pi_scaled + PLATE_W * r1_scaled) with P1_tri ---
    print("\n--- Verification 2: Reconstructed P1 vs P1_tri ---")
    # P1_plane in world coordinates was (PLATE_W, 0)
    # So, its 3D position in camera coordinates using the scaled pose parameters is:
    # P1_reconstructed_from_H = o_pi_scaled + Xw * r1_scaled + Yw * r2_scaled
    # Since Yw = 0 for P1_plane:
    P1_reconstructed_from_H = o_pi_scaled + PLATE_W * r1_scaled
    
    diff_P1_reconstructed_vs_P1_tri = np.linalg.norm(P1_reconstructed_from_H - P1_tri)
    print(f"P1_reconstructed_from_H (o_pi_s + PLATE_W*r1_s): {P1_reconstructed_from_H}")
    print(f"P1_tri:                                       {P1_tri}")
    print(f"Euclidean distance between P1_reconstructed_from_H and P1_tri: {diff_P1_reconstructed_vs_P1_tri:.6e} meters")

    if diff_P1_reconstructed_vs_P1_tri < 1e-3: # Threshold, e.g., 1mm
        print("SUCCESS: P1 reconstructed from H components is very close to P1_tri.")
    else:
        print("WARNING: P1 reconstructed from H components differs significantly from P1_tri.")

K = np.array([
    [3.2014989e3, 0.0, 1.93982925e3],
    [0.0, 3.20637527e3, 1.06315413e3],
    [0.0, 0.0, 1]
], dtype=np.float64)

dist = np.array([[2.4377e-01, -1.5955e+00, -1.1528e-03, 4.1986e-04, 3.5668e+00]], dtype=np.float64)

PLATE_W = 0.520
PLATE_H = 0.130
Z_FARO = 0.150
CAR_H = 1.467
CAR_L = 3.997
CAR_W = 1.732

if __name__ == "__main__":
    # ---------- 0. immagine e intrinseche ---------------------------------
    img = cv2.imread("sunny/frame_02.png")
    if img is None:
        raise FileNotFoundError("frame non trovato")
    
    # img_ud = cv2.undistort(img, K, dist)
    img_ud = img.copy()  # For testing without undistortion

    # Minimal example with synthetic data (plate 1 m wide, lights 0.3 m above)
    TTL = np.array([0.0, 0.0, 0.0])
    TTR = np.array([PLATE_W, 0.0, 0.0])
    FL = np.array([-0.340, 0.100, 0.0])
    FR = np.array([PLATE_W + 0.340, 0.100, 0.0])

    plane_pts = cad_to_plane_coords([TTL, TTR, FL, FR])
    print("Plane coordinates (homogeneous):\n", plane_pts)

    # ---------- 1. pixel (undistorti) dei 4 punti sullo stesso piano ------
    pix = np.array([
        [1020, 1804], # P0 = plate TL  (origine)
        [1324, 1780], # P1 = plate TR
        [792, 1768], # P2 = rear-light L
        [1484, 1708] # P3 = rear-light R
    ], dtype=np.float64)

    pix = cv2.undistortPoints(pix.reshape(-1,1,2), K, dist, P=K).reshape(-1,2)

    # display the points
    for p in pix:
        cv2.circle(img_ud, (int(p[0]), int(p[1])), 8, (0, 255, 0), 10)
    
    # 4. Calcola l'omografia H
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

    r3_vec = np.cross(r1_scaled, r2_scaled)
    r3_vec /= np.linalg.norm(r3_vec)

    wheel_offset = np.array([-0.606+0.386, 1.467-0.9, 0.3]) # TOP LEFT
    wheel_offset = np.array([-0.606, -0.9, 0.3]) # BOTTOM LEFT    w
    wheel_offset = np.array([-0.606+CAR_W, -0.9, 0.3]) # BOTTOM Xright    
    # wheel_offset = np.array([0.52, 0, 0])   # (Xw, Yw, Zw) in metri
    # wheel_offset = np.array([-0.606, -0.9, -CAR_L-0.3])   # (Xw, Yw, Zw) in metri
    P_wheel_cam = (o_pi_scaled +
                wheel_offset[0] * r1_scaled +
                wheel_offset[1] * r2_scaled +
                wheel_offset[2] * r3_vec)
    p_hom = K @ P_wheel_cam        # (3,)
    u = p_hom[0] / p_hom[2]
    v = p_hom[1] / p_hom[2]
    print(f"Ruota anteriore sinistra in pixel: u={u:.1f}, v={v:.1f}")

    cv2.circle(img_ud, (int(u), int(v)), 8, (0, 0, 255), 10)  # r

    verification(K, r1_scaled, r2_scaled, o_pi_scaled, P0_tri, P1_tri, img_ud)

    bb = draw_rear_bounding_box(
        img_ud, K, r1_scaled, r2_scaled, o_pi_scaled,
        width_bb=-1.5, height_bb=-1, depth_bb=-1.5,
        color=(0, 255, 0), thickness=5
    )

    # bb1 = draw_rear_bounding_box(
    #     bb, K, r1_scaled, r2_scaled, o_pi_scaled,
    #     width_bb=1, height_bb=0.75, depth_bb=1,
    #     color=(255, 255, 0), thickness=5
    # )

    # rescale l'immagine per visualizzarla correttamente
    # scale_factor = 0.35  # Riduci la dimensione dell'immagine per la visualizzazione
    # new_width = int(bb.shape[1] * scale_factor)
    # new_height = int(bb.shape[0] * scale_factor)
    # bb = cv2.resize(bb, (new_width, new_height))
    # cv2.imshow("Bounding Box", bb)

    scale_factor = 0.35  # Riduci la dimensione dell'immagine per la visualizzazione
    new_width = int(img_ud.shape[1] * scale_factor)
    new_height = int(img_ud.shape[0] * scale_factor)
    bb = cv2.resize(img_ud, (new_width, new_height))
    cv2.imshow("Bounding Box", bb)
    
    
    # bb1 = cv2.resize(bb1, (new_width, new_height))
    # cv2.imshow("Bounding Box with Wheel", bb1)

    # bb_wheel = cv2.resize(bb_wheel, (new_width, new_height))
    # cv2.imshow("Bounding Box with Wheel", bb_wheel)

    cv2.waitKey(0)
    cv2.destroyAllWindows()