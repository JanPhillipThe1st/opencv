import cv2 as cv 
import numpy as np
#get the scanned copy of an image
original_image = cv.imread("./images/photo_tilted.jpg")
if original_image.shape[0] > 900:
        #resize the image
        original_image = cv.resize(original_image,(int(original_image.shape[1]/2),int(original_image.shape[0]/2)))
#show the image until a key is pressed
threshold = 130

#image to black and white
gray = cv.cvtColor(original_image,cv.COLOR_BGR2GRAY)
#create a threshed image
ret, threshed = cv.threshold(gray,threshold%255,255,cv.THRESH_BINARY)
#display threshold value along image

def detectImageCircles():
    #show the threshed image
    #detect circles in this image
    circles = cv.HoughCircles(threshed,cv.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=2,maxRadius=60)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(original_image,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(original_image,(i[0],i[1]),2,(0,0,255),3)
        
        
def detectRect():
    #detect rectangles in the image
    #load the threshed image
    contours, hierarchy = cv.findContours(threshed,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[4]
    cv.drawContours(original_image, [cnt], 0, (0,255,0), 3)

detectRect()
while True:
    detectImageCircles()
    cv.imshow("original",original_image)
    #Exit if 'q' is pressed
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
    elif(cv.waitKey(25) & 0xFF == ord('i')):
        threshold += 1
    elif(cv.waitKey(25) & 0xFF == ord('o')):
        thresh_maxval += 1