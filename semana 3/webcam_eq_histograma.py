import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
ret, frame = cap.read()


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Opcion 1 blanco y negro
    
    # img_out = cv2.equalizeHist(img)
    # hist,bins = np.histogram(img.flatten(),256,[0,256])
    # cdf = hist.cumsum()
    # cdf_normalized = cdf * hist.max()/ cdf.max()
    # cdf_m = np.ma.masked_equal(cdf,0)
    # cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    # cdf = np.ma.filled(cdf_m,0).astype('uint8')
    # img2 = cdf[img]
    
#    Opcion 2 color    
#         # Color RGB
    # b,g,r=cv2.split(frame)
    # b = cv2.equalizeHist(b)
    # g = cv2.equalizeHist(g)
    # r = cv2.equalizeHist(r)
    # img_out = cv2.merge((b,g,r))
    
    # Color CIELab
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(img)
    l = cv2.equalizeHist(l)
    a = cv2.equalizeHist(a)
    b = cv2.equalizeHist(b)
    img_out = cv2.merge((l,a,b))
    img_out = cv2.cvtColor(img_out, cv2.COLOR_LAB2BGR) 
    
    cv2.imshow('frame',frame)
    cv2.imshow('frame_eq',img_out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()