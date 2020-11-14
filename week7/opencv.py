import cv2
import numpy
import matplotlib
from pytesseract import *

img = cv2.imread('images/image1.png',cv2.IMREAD_COLOR)
text = pytesseract.image_to_string(img,config='--psm 6')
cv2.imshow('image1', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
