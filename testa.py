# you need to install opencv 
import cv2
from cv2 import imread
from cv2 import imshow
from cv2 import imwrite
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
import os


def emptyFolder(path):
    # using listdir() method to list the files of the folder
    faces = os.listdir((path))

    for images in faces:
        if images.endswith(".jpg"):
            os.remove(os.path.join(path, images))

classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
def getFaces(pixels):
    faces = []

    for box in classifier.detectMultiScale(pixels, 1.35, 3):
        # extract
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # draw a rectangle over the pixels
        rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1) # remove after done testing
        imshow('Face Detection', pixels) # this too
        faces.append(pixels[y:y2, x:x2])

    return faces

# first, file cleaner to remove old pictures from last time
# specify path - can change later, I just put local /faces directory
emptyFolder('faces')
images = getFaces(imread("test3.jpg"))

num = 1 # or 0
for img in images:
    filename = "faces/face" + str(num) + ".jpg"
    imwrite(filename, img)
    num += 1

waitKey(0) #how to automate this?
destroyAllWindows() 