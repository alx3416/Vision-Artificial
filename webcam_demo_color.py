import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    Im1lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    a = Im1lab[:, :, 1]  # Tomamos canal S (verdes negativo (0-127) y rojos positivo (128-255))
    b = Im1lab[:, :, 2]  # tomamos canal b (positivos amarillo, negativos azules)
    a = np.double(a)
    b = np.double(b)
    a = a - 128
    b = b - 128
    K = 0  # Ajuste para determinar tonos de rojo detectados
    z = (np.bitwise_and(b < 0, np.abs(b) > np.abs(a)))
    # z=(S>0)
    new_a = a
    new_b = b
    new_a[z == 1] = np.abs(b[z == 1])
    new_b[z == 1] = a[z == 1]
    new_a[z == 1] = new_a[z == 1]
    Im1lab[:, :, 1] = new_a - 128
    Im1lab[:, :, 2] = new_b - 128
    Im1lab = np.uint8(Im1lab)
    Im2 = cv2.cvtColor(Im1lab, cv2.COLOR_LAB2BGR)

    # Display the resulting frame
    cv2.imshow('frame', Im2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
