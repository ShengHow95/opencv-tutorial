import os
import cv2
import numpy as np

# ------------------------------------------------ #

# img = cv2.imread('Photos/group 1.jpg')
# grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# haarCascade = cv2.CascadeClassifier('haar_face.xml')

# facesRect = haarCascade.detectMultiScale(
#     grayImg, scaleFactor=1.1, minNeighbors=1)

# for x, y, w, h in facesRect:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=3)

# cv2.imshow('img', img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ------------------------------------------------ #

people = ['Ben Afflek', 'Elton John',
          'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
directory = 'Faces/train'

haarCascade = cv2.CascadeClassifier('haar_face.xml')

features = []
labels = []


def train():
    for person in people:
        path = os.path.join(directory, person)
        label = people.index(person)

        for img in os.listdir(path):
            imgPath = os.path.join(path, img)

            imgArray = cv2.imread(imgPath)
            grayImg = cv2.cvtColor(imgArray, cv2.COLOR_BGR2GRAY)

            facesRect = haarCascade.detectMultiScale(
                grayImg, scaleFactor=1.1, minNeighbors=4)
            for x, y, w, h in facesRect:
                facesROI = grayImg[y:y+h, x:x+w]
                features.append(facesROI)
                labels.append(label)


train()

features = np.array(features)
labels = np.array(labels)

np.save('featuers.npy', features)
np.save('labels.npy', labels)
# features = np.load('featuers.npy')
# labels = np.load('labels.npy')

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.train(features, labels)

faceRecognizer.save('faceTrained.yml')
# faceRecognizer.load('faceTrained.yml')

testImg = cv2.imread('Faces/val/madonna/3.jpg')
grayTestImg = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)

facesRect = haarCascade.detectMultiScale(
    grayTestImg, scaleFactor=1.1, minNeighbors=1)
for x, y, w, h in facesRect:
    facesROI = grayTestImg[y:y+h, x:x+w]

    label, confidence = faceRecognizer.predict(facesROI)
    print(people[label], confidence)

    cv2.rectangle(testImg, (x, y), (x+w, y+h), (0, 255, 0), thickness=3)
    cv2.putText(testImg, str(people[label]), (0, 0),
                cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)

cv2.imshow('testImg', testImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
