import cv2
import numpy as np
import util

pathImage = "<image_path>.jpg"
heightImg = 640
widthImg  = 480

"""

get image 

resize the image 

convert to gray scale ( to get the edges )

apply blur to remove sharp drops (without blur, we get edges of even normal drops)

apply canny edge to get exact drop picture

after getting exact drop , dilate it to get the complete good image of the drop 

"""

count=0

img = cv2.imread(pathImage)

img = cv2.resize(img, (widthImg, heightImg)) # resize image
imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # blank image with 3 channels

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray scale
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # add gaussian blurs

cv2.imwrite(img=imgBlur,filename="blur.png")

imgEdged = cv2.Canny(imgBlur,255,220) # apply canny blur
kernel = np.ones((4, 4))

imgDial = cv2.dilate(imgEdged, kernel, iterations=2) #  dilation
imgEdged = imgDial

# change dilated one to original image

imgContours = img.copy()
imgBigContour = img.copy()

contours, hierarchy = cv2.findContours(imgEdged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find all contours
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 5) # draw detected contours

# find biggest contour
biggest, maxArea = util.biggestContour(contours)
if biggest.size != 0:
    biggest=util.makeBox(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # mark biggest contour
    imgBigContour = util.drawRectangle(imgBigContour,biggest,2)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
    imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))
    imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
    imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
    imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)

    imageArray = ([img,imgGray,imgEdged,imgContours],
                    [imgBigContour,imgWarpColored, imgWarpGray,imgAdaptiveThre])

else:
    imageArray = ([img,imgGray,imgEdged,imgContours],
                    [imgBlank, imgBlank, imgBlank, imgBlank])

labels = [["Original","Gray","Threshold","Contours"],
            ["Biggest Contour","Warp Prespective","Warp Gray","Adaptive Threshold"]]

stackedImage = util.stackImages(imageArray,0.75,labels)
cv2.imshow("Result",stackedImage)

if cv2.waitKey() & 0xFF == ord('s'):
    print("s pressed")
    cv2.imwrite("Scanned/myImage"+str(count)+".jpg",imgWarpColored)
    cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                    (1100, 350), (0, 255, 0), cv2.FILLED)
    cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
    cv2.imshow('Result', stackedImage)
    cv2.waitKey(300)
    count += 1