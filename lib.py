from scipy.ndimage import label
import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt


# Carga la imagen en OpenCV
def leer_imagen(ruta):
    imagen = cv2.imread(ruta)
    return imagen


def mostrar_imagen(titulo, imagen):
    # Mostrar imagen Original
    cv2.imshow(titulo, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convertir_grises(imagen):
    # Convertir Imagen a escala de grises
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)


def create_histograma(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist[0] = 0
    plt.plot(hist, color='red')
    plt.scatter(10, hist[10], color='blue')
    plt.show()
    return hist


def suavizante_histograma(histograma):
    datos = []
    for i in range(0, 255):
        datos.append(np.mean(histograma[i:i + 10]))
    for i in range(1, 254):  # Ajuste para evitar índices fuera de rango
        if datos[i - 1] > datos[i] and datos[i + 1] > datos[i]:
            print(i)
    plt.plot(datos, color='red')
    plt.scatter(10, datos[10], color='blue')  # Solo graficamos un punto, asegúrate de que 10 sea un índice válido
    plt.show()


def barrido_valores(gris):
    # barrido para encontrar las que necesitamos
    h, w = gris.shape
    img = np.zeros((h, w))
    for x in range(w):
        for y in range(h):
            p = gris.item(y, x)
            if p >= 70 and p < 100:
                img.itemset((y, x), 255)
            else:
                img.itemset((y, x), 0)
    cv2.imshow("Imagen barrido", gris)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


def generate_contornos(gris, img):
    cv2.imwrite('copia.png', gris)
    im2 = cv2.imread("copia.png")
    im2G = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im2G = cv2.medianBlur(im2G, 3)
    (contornos, _) = cv2.findContours(im2G, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contornos, -1, (0, 255, 0), 4)
    cv2.imshow("imagen4", img)
    print("cantidad de contornos", len(contornos))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def segment_on_dt(a, img):
    border = cv2.dilate(img, None, iterations=5)
    border = border - cv2.erode(border, None)

    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
    _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
    lbl, ncc = label(dt)
    lbl = lbl * (255 / (ncc + 1))
    # Completing the markers now.
    lbl[border == 255] = 255

    lbl = lbl.astype(numpy.int32)
    cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(numpy.uint8)
    return 255 - lbl
