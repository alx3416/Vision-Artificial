import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret, img = cap.read()
imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 30)
fontScale = 1
fontColor = (228, 200, 50)
lineType = 2

while (True):
    # Capture frame-by-frame
    ret, img = cap.read()
    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    circles = cv2.HoughCircles(imgG, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=50, minRadius=50, maxRadius=100)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Dibujamos circulo con 255 en mascara de cada circulo detectado
            mascara = np.zeros(imgG.shape)
            cv2.circle(mascara, (i[0], i[1]), i[2], 255, -1)
            # Tomamos canales a y b de CIElab
            a = imgLAB[:, :, 1]  # Tomamos canal a (verdes negativo (0-127) y rojos positivo (128-255))
            b = imgLAB[:, :, 2]  # tomamos canal b (positivos amarillo, negativos azules)
            a = np.double(a)
            b = np.double(b)
            a = a - 128
            b = b - 128
            # Quitamos todos los colores fuera del circulo
            a[mascara != 255] = 0
            b[mascara != 255] = 0
            # creamos mascara de circulo azul
            z = np.bitwise_and(b < 0, np.abs(b) > np.abs(a))  # Detección de azules
            # z son los pixeles que detectó dentro de un círculo y son azules
            if np.sum(z) > 1000:  # cantidad minima de pixeles azules por circulo
                # dibujar circunferencia
                radii = i[2]
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # dibujar centro circulo
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
                cv2.putText(img, str(2 * (radii - 50)),
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)

    cv2.imshow('Circulos azules', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
