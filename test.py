import cv2
import numpy as np

img = cv2.imread("1.jpg")
imgBitwiseColor = cv2.bitwise_not(img)
imgGray = cv2.cvtColor(src=img,code=cv2.COLOR_RGB2GRAY)
imgBitwiseGray = cv2.bitwise_not(imgGray)
imgGaussianBlur = cv2.GaussianBlur(src=img,ksize=(5,5),sigmaX=1,sigmaY=1)
imgGaussianGray = cv2.cvtColor(src=imgGaussianBlur,code=cv2.COLOR_BGR2GRAY)
imgEdge = cv2.Canny(image=imgGaussianGray,threshold1=220,threshold2=255)
imgDilated = cv2.dilate(src=imgEdge,kernel=(5,5,))
imgErr  = cv2.erode(src=imgEdge,kernel=(5,5))

imgGaussianBlurCopy = imgGaussianBlur.copy()
contours , heirarachy = cv2.findContours(image=imgEdge, mode= cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
imgWithMarkedContour = cv2.drawContours(image=imgGaussianBlur,contours=contours,contourIdx=-1,thickness=5,color=(0,100,200))

maxArea = 0
orderedMax = np.array([])


for contour in contours:
   
    area = cv2.contourArea(contour=contour)

    if area > maxArea and area > 5000:
        peri = cv2.arcLength(curve=contour,closed=True)
        approx = cv2.approxPolyDP(curve=contour,epsilon=0.02 * peri, closed=True)
        if len(approx) == 4:
            orderedMax = approx
            maxArea  = area


imgWithBiggestContour = cv2.drawContours(image=imgGaussianBlurCopy,contours=orderedMax,contourIdx=-1,color=(0,100,255),thickness=5)
# imgAdaptive = None


cv2.imwrite(img=imgBitwiseColor,filename="imgBitwiseColor.png")
cv2.imwrite(img=imgGray,filename="imgGray.png")
cv2.imwrite(img=imgBitwiseGray,filename="imgBitwiseGray.png")
cv2.imwrite(img=imgGaussianBlur,filename="imgGaussianBlur.png")
cv2.imwrite(img=imgEdge,filename="imgEdge.png")
cv2.imwrite(img=imgDilated,filename="imgDilated.png")
cv2.imwrite(img=imgErr,filename="imgErr.png")
cv2.imwrite(img=imgWithMarkedContour,filename="imgWithMarkedContour.png")
cv2.imwrite(img=imgGaussianBlurCopy,filename="imgWithBiggestContour.png")

cv2.waitKey()