import numpy as np
import cv2 as cv
import time
from keras.models import load_model

model = load_model('my_model.h5')
# something
def getH():
    return cap.get(cv.CAP_PROP_FRAME_HEIGHT)
def getW():
    return cap.get(cv.CAP_PROP_FRAME_WIDTH)
def setH(val):
    cap.set(cv.CAP_PROP_FRAME_HEIGHT,val)
    return getH()
def setW(val):
    cap.set(cv.CAP_PROP_FRAME_WIDTH,val)
    return getW()

def ka(interval=None):
    while(True):
        # Capture frame-by-frame, shape of (480, 640, 3)
        ret, frame = cap.read()
        # ROI
        frame = frame[:480,140:500]
        # resize
        frame = cv.resize(frame,(240,320))
        # predict
        y = model.predict(np.expand_dims(frame, axis=0))
        img = cv.putText(frame, str(float(y)), (30,150), cv.FONT_HERSHEY_SIMPLEX, .8, (0,255,255),2)
        # Display the resulting frame
        cv.imshow('frame',img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if interval:
            time.sleep(interval)
    # When everything done, release the capture
    cv.destroyAllWindows()

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    ka()