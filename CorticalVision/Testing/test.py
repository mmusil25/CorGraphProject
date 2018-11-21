# Testing Out some features in OpenCv
import cv2 as cv

# Read image from HDD
image = cv.imread("Ronaldo.jpg")

# Get color at x,y = 200,500
x,y=400,400
(b,g,r)=image[y,x]

#print color values
print(b,g,r)

#set pixel color to RED in BGR color scheme
image[y, x] = (0,0,255)

#choose region of interest at (x,y) with dimension 50x50 pixel
region_of_interest = image[y:y+100, x:x+100]

#show image on screen
cv.imshow("Bild",image)

#show ROI in a seperate window
cv.imshow("ROI",region_of_interest)

#set all ROI pixels to green
region_of_interest[:,:]=(0,255,0)

#now show modified image, note that ROU is a pointer to the original image
cv.imshow("Modified Image", image)

#Wait for a keystroke
cv.waitKey(0)
