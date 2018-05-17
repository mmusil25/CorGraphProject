import cv2

#init webcam
cam = cv2.VideoCapture(0)
#define color ranges
lower_yellow=(18,100,210)
upper_yellow=(40,160,245)

#show webcam stream
while cam.isOpened():
    #read frame from webcam
    ret,frame=cam.read()
    #convert frame to HSV
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #filter image for color ranges
    mask = cv2.inRange(frame,lower_yellow,upper_yellow)
    #show mask
    cv2.imshow("threshold",mask)
    #wait for key
    key=cv2.waitKey(1) & 0xff
    #if ESC, exit
    if key==27:
        break
