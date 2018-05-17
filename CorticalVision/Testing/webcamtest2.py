# Image Thresholding

import cv2

# init webcam
cam = cv2.VideoCapture(0)

#define the region of interest
x,y,w,h=400,400,100,100

#display webcam stream
while cam.isOpened():
    # read frame from cam
    ret,frame=cam.read()
    #convert from to HSV color scheme
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #draw rectangle in frame
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),thickness=1)
    #print color values to screen
    cv2.putText(frame,"HSV:{0}".format(frame[y+1, x+1]),(x,600),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),thickness=2)
    #show frame
    cv2.imshow("frame",frame)
    #wait for key press
    key=cv2.waitKey(1) & 0xff
    #if ESC, exit
    if key == 27:
        break
    
