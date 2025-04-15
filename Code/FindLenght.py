import cv2
import numpy as np
import math

# Lista dei punti selezionati
points = []

# Zoom e Pan
zoom = 1.0
offset = np.array([0, 0], dtype=np.float32)
drag_start = None
is_dragging = False

def calculate_Y_distance(p1, p2):
    return abs(p1[1] - p2[1])

def calculate_X_distance(p1, p2):
    return abs(p1[0] - p2[0])


def mouse_callback(event, x, y, flags, param):
    global drag_start, is_dragging, offset, zoom

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            # Converti le coordinate in base allo zoom e offset
            ix = int((x + offset[0]) / zoom)
            iy = int((y + offset[1]) / zoom)
            points.append((ix, iy))
            print(f"Punto {len(points)} selezionato: ({ix}, {iy})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        is_dragging = True
        drag_start = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_dragging:
            dx = drag_start[0] - x
            dy = drag_start[1] - y
            offset[0] = min(max(offset[0] + dx, 0), max(param.shape[1]*zoom - param.shape[1], 0))
            offset[1] = min(max(offset[1] + dy, 0), max(param.shape[0]*zoom - param.shape[0], 0))
            drag_start = (x, y)

    elif event == cv2.EVENT_RBUTTONUP:
        is_dragging = False

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            zoom *= 1.1
        else:
            zoom /= 1.1
        zoom = max(0.1, min(zoom, 10.0))

def draw_overlay(display_img):
    for p in points:
        px = int(p[0] * zoom - offset[0])
        py = int(p[1] * zoom - offset[1])
        cv2.circle(display_img, (px, py), 5, (0, 0, 255), -1)

    if len(points) >= 2:
        cv2.line(display_img,
                 (int(points[0][0]*zoom - offset[0]), int(points[0][1]*zoom - offset[1])),
                 (int(points[1][0]*zoom - offset[0]), int(points[1][1]*zoom - offset[1])),
                 (255, 0, 0), 2)

    if len(points) >= 4:
        cv2.line(display_img,
                 (int(points[2][0]*zoom - offset[0]), int(points[2][1]*zoom - offset[1])),
                 (int(points[3][0]*zoom - offset[0]), int(points[3][1]*zoom - offset[1])),
                 (0, 255, 0), 2)

def main():
    global offset, zoom

    image_path = "CAD\Fabia_CAD.png"
    img = cv2.imread(image_path)

    if img is None:
        print("Errore nel caricamento dell'immagine.")
        return

    cv2.namedWindow("Zoom e Pan")
    cv2.setMouseCallback("Zoom e Pan", mouse_callback, img)

    while True:
        h, w = img.shape[:2]
        resized = cv2.resize(img, (int(w * zoom), int(h * zoom)), interpolation=cv2.INTER_LINEAR)

        # Gestione bordi pan per evitare crash
        max_x = max(resized.shape[1] - w, 0)
        max_y = max(resized.shape[0] - h, 0)
        offset[0] = min(offset[0], max_x)
        offset[1] = min(offset[1], max_y)

        x1, y1 = int(offset[0]), int(offset[1])
        x2, y2 = x1 + w, y1 + h

        # Crop l'immagine zoomata
        display = resized[y1:y2, x1:x2].copy()

        draw_overlay(display)

        cv2.imshow("Zoom e Pan", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break


    if len(points) == 4:
        d1_y = calculate_Y_distance(points[0], points[1])
        d2_y = calculate_Y_distance(points[2], points[3])
        prop_y = d1_y / d2_y if d2_y != 0 else float('inf')

        d1_x = calculate_X_distance(points[0], points[1])
        d2_x = calculate_X_distance(points[2], points[3])
        prop_x = d1_x / d2_x if d2_x != 0 else float('inf')

        print(f"\n--- DISTANZE CALCOLATE ---")
        print(f"Distanza verticale linea 1: {d1_y:.2f}")
        print(f"Distanza verticale linea 2: {d2_y:.2f}")
        print(f"Proporzione Y (linea1 / linea2): {prop_y:.4f}")

        print(f"\nDistanza orizzontale linea 1: {d1_x:.2f}")
        print(f"Distanza orizzontale linea 2: {d2_x:.2f}")
        print(f"Proporzione X (linea1 / linea2): {prop_x:.4f}")

if __name__ == "__main__":
    main()
