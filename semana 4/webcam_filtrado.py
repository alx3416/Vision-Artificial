import numpy as np
import cv2

cap = cv2.VideoCapture(1)
ret, frame = cap.read()
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
Size = np.shape(img)
img_out = np.zeros((Size[0], Size[1]))

sigma_r = 10
sigma_d = 75
direccion = 1

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 30)
fontScale = 1
fontColor = (28, 20, 250)
lineType = 2

while (True):
    # Captura frame
    ret, frame = cap.read()

    # Operaciones

    img_out = cv2.bilateralFilter(frame, 11, sigma_r, sigma_d)

    if direccion == 1:
        sigma_r = sigma_r + 1

    if direccion == -1:
        sigma_r = sigma_r - 1

    if sigma_r == 150:
        direccion = -1

    if sigma_r == 10:
        direccion = 1

    cv2.putText(img_out, str(sigma_r),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    # Resultado
    cv2.imshow('frame', img_out)
    k = cv2.waitKey(1)
    if k & 0xFF == ord('q'):
        print(k)
        break

# Al finalizar, apagar webcam
cap.release()
cv2.destroyAllWindows()
