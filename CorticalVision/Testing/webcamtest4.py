import cv2

# init webcam

cam = cv2.VideoCapture(0)

# define color ranges

lower_yellow = (18, 100, 210)

upper_yellow = (40, 160, 245)

# show webcam stream

while cam.isOpened():

    # read frame from cam

    ret, frame = cam.read()

    # convert frame to HSV

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # filter for color ranges

    mask = cv2.inRange(frame, lower_yellow, upper_yellow)

    # find contours on mask of "tennis-ball" pixels

    k = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    countours = k[1]
        
    # now find the largest contour, this is most likely the tennis balls

    # for this we use the area of the contour

    if len(contours) > 0:

        tennis_ball = max(contours, key=cv2.contourArea)

        # draw bounding box around tennis ball

        x, y, w, h = cv2.boundingRect(tennis_ball)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=3)

    # show frame

    cv2.imshow("frame", frame)

    # wait for key

    key = cv2.waitKey(1) & 0xff

    # if ESC, exit

    if key == 27:

        break
