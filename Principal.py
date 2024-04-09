import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def procesar_imagen(imagen):
    # Convertir a escala de grises
    imGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro de Gaussiana para eliminar ruido
    imGauss = cv2.GaussianBlur(imGris, (5, 5), 0)

    # Umbralizar la imagen para obtener una imagen binaria
    _, imBinaria = cv2.threshold(imGauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Encontrar contornos en la imagen binaria
    contornos, _ = cv2.findContours(imBinaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inicializar contadores
    total_semillas = len(contornos)
    semillas_llenas = 0
    semillas_vacias = 0

    # Clasificar semillas llenas y vacías
    for contorno in contornos:
        # Calcular el área del contorno
        area = cv2.contourArea(contorno)

        # Si el área es menor que un umbral, considerarlo como vacío
        if area < 100:  # Ajustar este umbral según sea necesario
            semillas_vacias += 1
        else:
            semillas_llenas += 1

    return total_semillas, semillas_llenas, semillas_vacias


def procesar_carpeta(carpeta):
    # Obtener la lista de archivos en la carpeta
    archivos = os.listdir(carpeta)

    resultados = []

    # Procesar cada imagen en la carpeta
    for archivo in archivos:
        ruta_imagen = os.path.join(carpeta, archivo)
        imagen = cv2.imread(ruta_imagen)

        if imagen is not None:
            # Procesar la imagen y obtener los resultados
            total, llenas, vacias = procesar_imagen(imagen)

            # Guardar los resultados en una lista
            resultados.append((archivo, total, llenas, vacias))

    return resultados


def main():
    # Carpeta que contiene las imágenes
    carpeta_imagenes_entrenamiento = "imagenes/A1"
    carpeta_imagenes_prueba = "imagenes/A1"

    # Procesar las imágenes de entrenamiento
    resultados_entrenamiento = procesar_carpeta(carpeta_imagenes_entrenamiento)

    # Mostrar los resultados de las imágenes de entrenamiento
    print("Resultados de imágenes de entrenamiento:")
    print("Nombre del archivo | Total de semillas | Semillas llenas | Semillas vacías")
    for resultado in resultados_entrenamiento:
        print(f"{resultado[0]} | {resultado[1]} | {resultado[2]} | {resultado[3]}")

    # Procesar las imágenes de prueba
    resultados_prueba = procesar_carpeta(carpeta_imagenes_prueba)

    # Mostrar los resultados de las imágenes de prueba
    print("\nResultados de imágenes de prueba:")
    print("Nombre del archivo | Total de semillas | Semillas llenas | Semillas vacías")
    for resultado in resultados_prueba:
        print(f"{resultado[0]} | {resultado[1]} | {resultado[2]} | {resultado[3]}")


if __name__ == "__main__":
    # main()
    imagen = cv2.imread("imagenes/A1/1.1.bmp")
    cv2.imshow("Original", imagen)
    cv2.waitKey(0)

    # Convertir a escala de grises
    imGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gris", imGris)
    cv2.waitKey(0)

    # Aplicar filtro de Gaussiana para eliminar ruido
    imGauss = cv2.GaussianBlur(imGris, (5, 5), 0)
    cv2.imshow("Gauss", imGauss)
    cv2.waitKey(0)

    # Umbralizar la imagen para obtener una imagen binaria
    _, imBinaria = cv2.threshold(imGauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Encontrar contornos en la imagen binaria
    contornos, _ = cv2.findContours(imBinaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imBinaria.copy(), contornos, -1, (0, 255, 0), 2)
    cv2.waitKey(0)

    # Inicializar contadores
    total_semillas = len(contornos)
    semillas_llenas = 0
    semillas_vacias = 0

    # Clasificar semillas llenas y vacías
    for contorno in contornos:
        # Calcular el área del contorno
        area = cv2.contourArea(contorno)

        # Si el área es menor que un umbral, considerarlo como vacío
        if area < 100:  # Ajustar este umbral según sea necesario
            semillas_vacias += 1
        else:
            semillas_llenas += 1
    print(semillas_vacias)
    print(semillas_vacias)