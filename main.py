import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt

from lib import leer_imagen, mostrar_imagen, convertir_grises, create_histograma

imagen = leer_imagen("imagenes/A1/recortado_1.1.bmp")

mostrar_imagen("Imagen Original", imagen)

gris = convertir_grises(imagen)

mostrar_imagen("Imagen Escala de grises", gris)

kernel = np.ones((5, 5), np.uint8)

# Aplicar la erosión a la imagen en escala de grises
erosion = cv2.erode(imagen, kernel, iterations=1)

# Aplicar filtro de mediana para reducir el ruido
imagen_filtrada = cv2.medianBlur(imagen, 5)  # El segundo parámetro es el tamaño del kernel (debe ser impar)
mostrar_imagen("Imagen Ruido Reducido", imagen_filtrada)

# Aplicar la dilatación a la imagen
dilation = cv2.dilate(imagen_filtrada, kernel, iterations=1)

# Aplicar la apertura a la imagen
opening = cv2.morphologyEx(imagen_filtrada, cv2.MORPH_OPEN, kernel)

h, w, _ = imagen_filtrada.shape
img = np.zeros((h, w))
for x in range(w):
    for y in range(h):
        p = gris.item(y, x)
        if p >= 70 and p < 100:
            img.itemset((y, x), 255)
        else:
            img.itemset((y, x), p)

mostrar_imagen("Imagen Filtrado", imagen_filtrada)

hist = create_histograma(imagen_filtrada)

datos = []
for i in range(0, 255):
    datos.append(hist[i:i + 10].mean())
for i in range(1, len(datos) - 1):
    if datos[i - 1] > datos[i] and datos[i + 1] > datos[i]:
        print(i)
plt.plot(datos, color='red')
plt.scatter(10, datos[10], color='blue')  # Solo graficamos un punto, asegúrate de que 10 sea un índice válido
plt.show()

h, w = imagen_filtrada.shape[:2]  # Ignoramos la tercera dimensión si existe
gris = np.zeros((h, w), dtype=np.uint8)

for x in range(w):
    for y in range(h):
        p = imagen_filtrada[y, x]
        if np.any(p >= 0) and np.any(p < 129):  # Corrección aquí
            gris[y, x] = 255
        else:
            gris[y, x] = 0

mostrar_imagen("Imagen Gris", gris)

cv2.imwrite('copia.png', gris)
im2 = cv2.imread("copia.png")
im2G = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
im2G = cv2.GaussianBlur(im2G, (0, 0), 5, 9);
(contornos, _) = cv2.findContours(im2G, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imagen, contornos, -1, (0, 255, 0), 2)
mostrar_imagen("Imagen Contornos", imagen)


# Invertir la imagen umbralizada
im2G_invertida = cv2.bitwise_not(im2G)

# Encuentra los contornos en la imagen invertida
(contornos_vacios, _) = cv2.findContours(im2G_invertida, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Dibuja los contornos de las semillas vacías en la imagen original
cv2.drawContours(imagen, contornos_vacios, -1, (0, 0, 255), 2)

# Muestra la imagen con los contornos dibujados (contornos llenos en verde, contornos vacíos en rojo)
mostrar_imagen("Imagen Dibujo Contornos", imagen)


numero_llenas = len(contornos)
numero_vacias = len(contornos_vacios)

print("Número de semillas llenas:", numero_llenas)
print("Número de semillas vacías:", numero_vacias)