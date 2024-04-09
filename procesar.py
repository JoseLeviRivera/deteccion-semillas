import numpy as np
from matplotlib import pyplot as plt
import os
import cv2


def counter_contorts_beans(path):
    image = cv2.imread(path)
    grisses = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(grisses, 20, 255)
    ctns, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Contorts number: ", len(ctns))
    cv2.imshow("Bordes: ", bordes)
    cv2.imshow("Image: ", image)
    # Esperar a que se presione una tecla
    cv2.waitKey(0)
    # Cerrar todas las ventanas
    cv2.destroyAllWindows()

def procesar_imagen(imagen):
    #Carga imagen
    img = cv2.imread(imagen)
    # Convertir a escala de grises
    imGris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro de Gaussiana para eliminar ruido
    imGauss = cv2.GaussianBlur(imGris, (5, 5), 0)

    # Umbralizar la imagen para obtener una imagen binaria
    _, imBinaria = cv2.threshold(imGauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Encontrar contornos en la imagen binaria
    contornos, _ = cv2.findContours(imBinaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_semillas = len(contornos)


    print("Contorts number: ", total_semillas)
    cv2.imshow("Bordes: ", contornos)
    cv2.imshow("Image: ", img)
    # Esperar a que se presione una tecla
    cv2.waitKey(0)
    # Cerrar todas las ventanas
    cv2.destroyAllWindows()

# Ruta de la imagen
ruta_imagen = "/home/levi/PycharmProjects/ReconocimientoImagenesP1/imagenes/A1/recortado_1.1.bmp"

# Verificar si la imagen existe
if not os.path.exists(ruta_imagen):
    print("La imagen no se encuentra en la ruta especificada.")
    exit()

# counter_contorts_beans(ruta_imagen)
procesar_imagen(ruta_imagen)
# Cargar la imagen en escala de grises
# imagen_gris = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

# Mostrar la imagen en una ventana
# cv2.imshow("Imagen en escala de grises", imagen_gris)

# Umbral automático con método de Otsu
# _, imagen_umbral = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Mostrar la imagen umbralizada
# cv2.imshow("Imagen umbralizada", imagen_umbral)

# Calcular histograma
# hist = cv2.calcHist([imagen_gris], [0], None, [256], [0, 256])
# plt.plot(hist, color='red')

# Encontrar contornos
# (contornos,_) = cv2.findContours(imagen_umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(imagen_gris, contornos, -1, (0,255,0), 4)
# print("Contornos:", len(contornos))

# Mostrar la imagen con contornos
#cv2.imshow("Imagen con contornos", imagen_gris)

"""" 
imagen_path = "imagenes/A1/1.1.bmp"
img = cv2.imread(imagen_path)

# Pre-processing.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img_bin = cv2.threshold(img_gray, 0, 255,
                           cv2.THRESH_OTSU)
img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
                           numpy.ones((3, 3), dtype=int))

result = lib.segment_on_dt(img, img_bin)
cv2.imwrite("img1.png", result)

result[result != 255] = 0
result = cv2.dilate(result, None)
img[result == 255] = (0, 0, 255)
# Leer la imagen con los contornos dibujados
# Leer la imagen con los contornos dibujados
img_contornos = cv2.imread("img2.png")
img_contornos_gris = cv2.cvtColor(img_contornos, cv2.COLOR_BGR2GRAY)

# Encontrar los contornos en la imagen
contours, _ = cv2.findContours(img_contornos_gris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos para visualización
cv2.drawContours(img_contornos, contours, -1, (0, 255, 0), 2)

# Contar el número de contornos encontrados
numero_contornos = len(contours)
print("Número de contornos:", numero_contornos)

# Mostrar la imagen con los contornos dibujados y contarlos
cv2.imshow("Contornos encontrados", img_contornos)
cv2.waitKey(0)
cv2.destroyAllWindows()



lib.mostrar_imagen("Imagen Original", imagen)

gris = lib.convertir_grises(imagen)

lib.mostrar_imagen("Imagen Escala de grises", gris)

hist = lib.create_histograma(gris)

lib.suavizante_histograma(hist)

img = lib.barrido_valores(gris)

lib.generate_contornos(gris, img)

"""
# Mode: cv2.RETR_LIST, RETR_EXTERNAL, RETR_CCOMP, RETR_TREE
#Method: CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE

import cv2

original = cv2.imread( "imagenes/A1/1.1.bmp")
cv2.imshow("Original", original)
cv2.waitKey(0)


grises = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grises", grises)
cv2.waitKey(0)


hist = cv2.calcHist([grises], [0], None, [256], [0, 256])
hist[0] = 0
plt.plot(hist, color='red')
plt.scatter(10, hist[10], color='blue')
plt.show()


datos = []
for i in range(0, 255):
    datos.append(np.mean(hist[i:i + 10]))
for i in range(1, 254):  # Ajuste para evitar índices fuera de rango
    if datos[i - 1] > datos[i] and datos[i + 1] > datos[i]:
        print(i)
plt.plot(datos, color='red')
plt.scatter(10, datos[10], color='blue')  # Solo graficamos un punto, asegúrate de que 10 sea un índice válido
plt.show()

h, w = grises.shape
img = np.zeros((h, w))
for x in range(w):
    for y in range(h):
        p = grises.item(y, x)
        if p >= 100 and p < 200:
            grises.itemset((y, x), 255)
        else:
            grises.itemset((y, x), 0)
cv2.imshow("Imagen barrido", img)
cv2.waitKey(0)

gauss = cv2.GaussianBlur(grises, (15, 15), 0)
cv2.imshow("Suavizado", gauss)
cv2.waitKey(0)

canny = cv2.Canny(gauss, 50, 150)
cv2.imshow("canny", canny)
cv2.waitKey(0)

contours, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(canny.copy(), contours, -1, (0, 255, 0), 2)
print("Contornos", len(contours))






"""" 
imagen_path = "imagenes/A1/1.1.bmp"

img = cv2.imread(imagen_path)

gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(gris, 100, 255, cv2.THRESH_BINARY)

# Encuentra los contornos en la imagen binarizada
contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Dibujar los contornos para visualización
image = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 2)

# Muestra las imágenes
cv2.imshow("th", th)
cv2.imshow("img", img)
cv2.imshow("image", image)
cv2.waitKey(0)

"""
cv2.destroyAllWindows()


# 17.291 Total de horas que debo de poner
