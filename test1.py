import cv2
import numpy as np
from scipy.ndimage import label


def segment_on_dt(a, img):
    border = cv2.dilate(img, None, iterations=5)
    border = border - cv2.erode(border, None)

    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
    _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
    lbl, ncc = label(dt)
    lbl = lbl * (255 / (ncc + 1))
    # Completing the markers now.
    lbl[border == 255] = 255

    lbl = lbl.astype(np.int32)
    cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    return 255 - lbl

# Función para contar los contornos y dibujarlos en la imagen original
def count_and_draw_contours(img_contornos, img_contornos_gris):
    # Encontrar los contornos en la imagen
    contours, _ = cv2.findContours(img_contornos_gris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos en la imagen original
    cv2.drawContours(img_contornos, contours, -1, (0, 255, 0), 2)

    # Contar el número de contornos encontrados
    numero_contornos = len(contours)
    print("Número de contornos:", numero_contornos)

    # Mostrar la imagen con los contornos dibujados y contarlos
    cv2.imshow("Contornos encontrados", img_contornos)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Cargar la imagen en escala de grises
image = cv2.imread('imagenes/A1/1.1.bmp', cv2.IMREAD_GRAYSCALE)

# Aplicar umbralización adaptativa para resaltar las semillas
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Encontrar contornos en la imagen umbralizada
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar contornos en la imagen original y contar las semillas llenas y vacías
for contour in contours:
    # Calcular el área del contorno
    area = cv2.contourArea(contour)

    # Si el área es menor que un cierto umbral, consideramos la semilla como vacía
    if area < 100:
        cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)  # Dibujar contorno en rojo
    else:
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)  # Dibujar contorno en verde

# Mostrar la imagen con los contornos identificados
cv2.imshow('Semillas Detectadas', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
