import numpy as np
import cv2
import pywt
import pywt.data

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

ret, frame = cap1.read()
img_ref = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
Size=np.shape(img_ref)
img_prev = np.zeros((Size[0],Size[1]))
img_out = np.zeros((Size[0],Size[1]))


coeffs2 = pywt.dwt2(img_ref, 'db9')
LL1, (LH1, HL1, HH1) = coeffs2
Size=np.shape(LL1)

k=0
while(True):
    # Capturas frame
    ret, frame = cap1.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     Para fusión de 2 videos 
    # ret, frame2 = cap2.read()
    # img_ref = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # fusión con 1 frame de video
    
    # coeffs2 = pywt.dwt2(img, 'db9')
    # LL1, (LH1, HL1, HH1) = coeffs2
    # coeffs2_ref = pywt.dwt2(img_ref, 'db9')
    # LL2, (LH2, HL2, HH2) = coeffs2_ref
    # # promediado de bajas bajas
    # LL3=np.true_divide(np.add(LL1,LL2), 2)
    # # Selección de máximo por par de filtrados
    # LH3=np.maximum(LH1,LH2)
    # HL3=np.maximum(HL1,HL2)
    # HH3=np.maximum(HH1,HH2)
    # # wavelet inversa
    # coeffs3 = LL3, (LH3, HL3, HH3)
    # img_out=np.uint8(pywt.idwt2(coeffs3, 'db9'))
    
    # fusion con décimo frame anterior
    
    coeffs2 = pywt.dwt2(img, 'db9')
    LL1, (LH1, HL1, HH1) = coeffs2
    coeffs2_ref = pywt.dwt2(img_prev, 'db9')
    LL2, (LH2, HL2, HH2) = coeffs2_ref
    # promediado de bajas bajas
    LL3=np.true_divide(np.add(LL1,LL2), 2)
    # Selección de máximo por par de filtrados
    LH3=np.maximum(LH1,LH2)
    HL3=np.maximum(HL1,HL2)
    HH3=np.maximum(HH1,HH2)
    # wavelet inversa
    coeffs3 = LL3, (LH3, HL3, HH3)
    img_out=np.uint8(pywt.idwt2(coeffs3, 'db9'))
    k=k+1
    if k==10:    
        img_prev = img
        k=0
    
    cv2.imshow('original',img)
    cv2.imshow('fusion',img_out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap1.release()
cv2.destroyAllWindows()