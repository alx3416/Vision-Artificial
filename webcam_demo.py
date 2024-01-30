import numpy as np
import cv2


cap = cv2.VideoCapture(0)
ret, frame = cap.read()
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
Size = np.shape(img)
img_out = np.zeros((Size[0], Size[1]))

while(True):
    ret, frame = cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame original', frame)
    cv2.imshow('frame procesado', img)
    k = cv2.waitKey(1) 
    if k & 0xFF == ord('q'):
        print(k)
        break

cap.release()
cv2.destroyAllWindows()
