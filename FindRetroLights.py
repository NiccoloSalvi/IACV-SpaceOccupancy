import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_c_shaped_headlights(image_path):
    # Carica l'immagine
    image = cv2.imread(image_path)
    
    # Converte l'immagine in scala di grigi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Applica una soglia per ottenere i pixel bianchi
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    
    # Applica operazioni morfologiche per rimuovere rumore
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Trova i contorni
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtra i contorni per dimensione (elimina piccoli contorni di rumore)
    min_area = 500
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    headlight_contours = filtered_contours
    
    # Identifica i fari sinistro e destro
    left_headlight = None
    right_headlight = None
    
    if len(headlight_contours) >= 2:
        # Ordina i contorni per la coordinata x
        headlight_contours.sort(key=lambda cnt: cv2.boundingRect(cnt)[0])
        
        # Prendi il contorno più a sinistra e quello più a destra
        left_headlight = headlight_contours[0]
        right_headlight = headlight_contours[-1]
    
    # Crea un'immagine di output per visualizzare i risultati
    result_image = image.copy()
    
    # Trova i pixel più interni dei fari
    left_inner_point = None
    right_inner_point = None
    
    if left_headlight is not None and right_headlight is not None:
        # Per il faro sinistro (C normale), il punto più interno è il più a destra
        left_x, left_y, left_w, left_h = cv2.boundingRect(left_headlight)
        left_inner_point = (left_x + left_w, left_y + left_h)
        
        # Per il faro destro (C ribaltata), il punto più interno è il più a sinistra
        right_x, right_y, right_w, right_h = cv2.boundingRect(right_headlight)
        right_inner_point = (right_x, right_y + right_h)
        
        # Disegna i punti interni trovati
        cv2.circle(result_image, left_inner_point, 10, (255, 0, 0), -1)
        cv2.circle(result_image, right_inner_point, 10, (255, 0, 0), -1)
        
        # Disegna i contorni dei fari
        cv2.drawContours(result_image, [left_headlight], -1, (0, 255, 0), 2)
        cv2.drawContours(result_image, [right_headlight], -1, (0, 255, 0), 2)
    
    # Mostra i risultati
    plt.figure(figsize=(12, 8))
    plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Immagine originale')
    plt.subplot(132), plt.imshow(thresh, cmap='gray'), plt.title('Threshold')
    plt.subplot(133), plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)), plt.title('Risultato')
    plt.tight_layout()
    plt.show()
    
    return left_inner_point, right_inner_point


# Carica l'immagine
image_path = "image_path"
left_inner, right_inner = find_c_shaped_headlights(image_path)

print(f"Punto interno del faro sinistro: {left_inner}")
print(f"Punto interno del faro destro: {right_inner}")





