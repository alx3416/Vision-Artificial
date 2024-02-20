import numpy as np
import cv2
from LBP import LBP

cap = cv2.VideoCapture(1)
ret, frame = cap.read()
cimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Our operations on the frame come here
    img_out = LBP(img, 3)

    # Display the resulting frame
    cv2.imshow('Census', img_out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
