import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image, Video, Live Video
# def rescaleFrame(frame, scale=0.75):
#   width = int(frame.shape[1] * scale)
#   height = int(frame.shape[0] * scale)
#   dimensions = (width, height)
#   return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Live Video Only
# def changeResolution(width, height):
#   capture.set(3, width)
#   capture.set(4, height)

# ------------------------------------------------ #

# Read Image
# img = cv2.imread('Photos/cat_large.jpg')
# cv2.imshow('Cat', rescaleFrame(img))

# ------------------------------------------------ #

# Read Video
# capture = cv2.VideoCapture('Videos/dog.mp4')
# while True:
#   isTrue, frame = capture.read()
#   cv2.imshow('Video', rescaleFrame(frame, 0.5))
#   if cv2.waitKey(20) & 0xFF==ord('d'):
#     break

# ------------------------------------------------ #

# Draw
# blank = np.zeros((500, 500, 3), dtype='uint8')

# Rectangle
# cv2.rectangle(blank, (0,100), (250,250), (0,250,0), thickness=2)

# Circle
# cv2.circle(blank, (250,250), 40, (250,250,0), thickness=2)

# Line
# cv2.line(blank, (0,0), (500,500), (0,0,255), thickness=2)
# cv2.line(blank, (500,0), (0,500), (0,0,255), thickness=2)

# Text
# cv2.putText(blank, 'Hello Jeff', (250,250), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255,255,255), thickness=2)

# cv2.imshow('Blank', blank)

# ------------------------------------------------ #

# Converting Image
# img = cv2.imread('Photos/park.jpg')

# grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', grayImg)

# blurImg = cv2.GaussianBlur(img, (7,7), cv2.BORDER_DEFAULT)
# cv2.imshow('blur', blurImg)

# cannyImg = cv2.Canny(blurImg, 125, 175)
# cv2.imshow('canny', cannyImg)

# dilatedImg = cv2.dilate(cannyImg, (7,7), iterations=3)
# cv2.imshow('dilate', dilatedImg)

# erodedImg = cv2.erode(dilatedImg, (7,7), iterations=3)
# cv2.imshow('erode', erodedImg)

# resizedImg = cv2.resize(img, (500,500), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('resize', resizedImg)

# croppedImg = img[50:200, 200:400]
# cv2.imshow('crop', croppedImg)

# ------------------------------------------------ #

# Translate Image
# def translate(img, x, y):
  # -x: Left
  # -y: Up
  # x: Right
  # y: Down
  # transMat = np.float32([[1,0,x],[0,1,y]])
  # dimensions = (img.shape[1], img.shape[0])
  # return cv2.warpAffine(img, transMat, dimensions)

# def rotate(img, angle, rotPoint=None):
#   (height, width) = (img.shape[1], img.shape[0])
#   if rotPoint is None:
#     rotPoint = (width//2, height//2)
#   rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)
#   dimensions = (width, height)
#   return cv2.warpAffine(img, rotMat, dimensions)

# img = cv2.imread('Photos/park.jpg')

# translatedImg = translate(img, -100, -100)
# cv2.imshow('translate', translatedImg)

# rotatedImg = rotate(img, -45)
# cv2.imshow('rotate', rotatedImg)

# flippedImg = cv2.flip(img, 0)
# cv2.imshow('flip', flippedImg)

# ------------------------------------------------ #

# img = cv2.imread('Photos/cats.jpg')
# blank = np.zeros((img.shape[:3]), dtype='uint8')

# grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', grayImg)

# blurImg = cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT)
# cv2.imshow('blur', blurImg)

# cannyImg = cv2.Canny(blurImg, 125, 175)
# cv2.imshow('canny', cannyImg)

# ret, thresh = cv2.threshold(grayImg, 125, 255, cv2.THRESH_BINARY)
# cv2.imshow('thresh', thresh)

# contours, hierarchies = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))
# print(len(hierarchies))

# cv2.drawContours(blank, contours, -1, (0,255,0), 1)
# cv2.imshow('coutours drawn', blank)

# ------------------------------------------------ #

# img = cv2.imread('Photos/cats.jpg')
# cv2.imshow('img', img)
# plt.imshow(img)
# plt.show()

# grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', grayImg)

# hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow('hsv', hsvImg)

# labImg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# cv2.imshow('lab', labImg)

# rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(rgbImg)
# plt.show()

# ------------------------------------------------ #

# img = cv2.imread('Photos/cats.jpg')
# blank = np.zeros((img.shape[:2]), dtype='uint8')

# b,g,r = cv2.split(img)
# cv2.imshow('b', b)
# cv2.imshow('g', g)
# cv2.imshow('r', r)

# mergedImg = cv2.merge([b,g,r])
# cv2.imshow('merge', mergedImg)

# blue = cv2.merge([b, blank, blank])
# green = cv2.merge([blank, g, blank])
# red = cv2.merge([blank, blank, r])
# cv2.imshow('blue', blue)
# cv2.imshow('green', green)
# cv2.imshow('red', red)

# ------------------------------------------------ #

# img = cv2.imread('Photos/cats.jpg')

# average = cv2.blur(img, (7,7))
# cv2.imshow('average', average)

# gaussian = cv2.GaussianBlur(img, (7,7), 0)
# cv2.imshow('gaussian', gaussian)

# median = cv2.medianBlur(img, 7)
# cv2.imshow('median', median)

# bilateral = cv2.bilateralFilter(img, 15, 35, 35)
# cv2.imshow('bilateral', bilateral)

# ------------------------------------------------ #

# blank = np.zeros((400,400), dtype='uint8')

# rectangle = cv2.rectangle(blank.copy(), (30,30), (370,370), 255, -1)
# circle = cv2.circle(blank.copy(), (200,200), 200, 255, -1)

# cv2.imshow('rectangle', rectangle)
# cv2.imshow('circle', circle)

# bit_and = cv2.bitwise_and(rectangle, circle)
# cv2.imshow('bit_and', bit_and)

# bit_or = cv2.bitwise_or(rectangle, circle)
# cv2.imshow('bit_or', bit_or)

# bit_xor = cv2.bitwise_xor(rectangle, circle)
# cv2.imshow('bit_xor', bit_xor)

# ------------------------------------------------ #

# img = cv2.imread('Photos/cats.jpg')
# blank = np.zeros(img.shape[:2], dtype='uint8')

# mask = cv2.circle(blank, (img.shape[1]//2, img.shape[0]//2), 200, 255, -1)
# cv2.imshow('mask', mask)

# maskedImg = cv2.bitwise_and(img, img, mask=mask)
# cv2.imshow('maskedImg', maskedImg)

# ------------------------------------------------ #

# img = cv2.imread('Photos/cats.jpg')
# blank = np.zeros(img.shape[:2], dtype='uint8')

# grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', grayImg)

# mask = cv2.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
# cv2.imshow('mask', mask)

# maskedImg = cv2.bitwise_and(grayImg, grayImg, mask=mask)

# grayHist = cv2.calcHist([grayImg], [0], mask, [256], [0,256])
# plt.figure()
# plt.title('grayscale histogram')
# plt.plot(grayHist)
# plt.show()

# colors = ('b', 'g', 'r')

# mask = cv2.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
# cv2.imshow('mask', mask)

# maskedImg = cv2.bitwise_and(img, img, mask=mask)
# cv2.imshow('maskedImg', maskedImg)

# plt.figure()
# plt.title('color histogram')
# for i, color in enumerate(colors):
#   colorHist = cv2.calcHist([img], [i], mask, [256], [0,256])
#   plt.plot(colorHist, color=color)
#   plt.xlim([0,256])
# plt.show()

# ------------------------------------------------ #

# img = cv2.imread('Photos/cats.jpg')

# grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', grayImg)

# ret, simpleThresh = cv2.threshold(grayImg, 128, 255, cv2.THRESH_BINARY)
# cv2.imshow('simpleThresh', simpleThresh)

# adaptiveThresh = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
# cv2.imshow('adaptiveThresh', adaptiveThresh)

# ------------------------------------------------ #

# img = cv2.imread('Photos/cats.jpg')

# grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', grayImg)

# laplacian = cv2.Laplacian(grayImg, cv2.CV_64F)
# laplacian = np.uint8(np.absolute(laplacian))
# cv2.imshow('laplacian', laplacian)

# sobelx = cv2.Sobel(grayImg, cv2.CV_64F, 1, 0)
# cv2.imshow('sobelx', sobelx)

# sobely = cv2.Sobel(grayImg, cv2.CV_64F, 0, 1)
# cv2.imshow('sobely', sobely)

# combinedSobel = cv2.bitwise_or(sobelx, sobely)
# cv2.imshow('combinedSobel', combinedSobel)

# cannyImg = cv2.Canny(grayImg, 125, 175)
# cv2.imshow('canny', cannyImg)

# cv2.waitKey(0)
cv2.destroyAllWindows()