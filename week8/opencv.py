# https://github.com/UB-Mannheim/tesseract/wiki
import cv2
import numpy
import matplotlib
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
img = cv2.imread('images/image1.png',cv2.IMREAD_COLOR)
text = pytesseract.image_to_string(img,config='digits')