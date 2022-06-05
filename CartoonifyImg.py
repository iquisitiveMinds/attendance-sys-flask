import cv2
from cv2 import blur
import numpy as np
 
# reading image 
img = cv2.imread("C:/Users/owner/Desktop/MS_Engage'22/trainimg/Bill_Gates.jpg")
   
def edge_detection(img,line_wdt,blur):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grayBlur = cv2.medianBlur(gray,blur)
    edges = cv2.adaptiveThreshold(grayBlur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,line_wdt,blur)
    return edges

def color_quantization(img,k):
    data = np.float32(img).reshape((-1,3)) 
    criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,0.001)
    ret, label, center = cv2.kmeans(data,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

line_wdt = 9
blur_value = 7
totalColors = 5

edgeImg = edge_detection(img,line_wdt,blur_value)
img = color_quantization(img,totalColors)
blurred = cv2.bilateralFilter(img,d=7,sigmaColor=200,sigmaSpace=200)
cartoon = cv2.bitwise_and(blurred,blurred,mask=edgeImg)
cv2.imwrite('cartoon.jpg',cartoon)

cv2.imshow("Image", img)
cv2.imshow("edges", edgeImg)
cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()
