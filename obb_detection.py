import cv2 as cv 

#get the scanned copy of an image
image_mat = cv.imread("./images/scanned.jpg")

#show the image until a key is pressed
while True:
    cv.imshow("scanned",image_mat)
    
    #Exit if 'q' is pressed
    if cv.waitKey(25) & 0xFF == ord('q'):
        break