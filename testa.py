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

# first, file cleaner to remove old pictures from last time
# specify path - can change later, I just put local /faces directory
path = ('faces')

#using listdir() method to list the files of the folder
faces = os.listdir(path)

for images in faces:
    if images.endswith(".jpg"):
        os.remove(os.path.join(path, images))


# here is where detection begins 
# load photograph
pixels = imread('test1.jpg')

# pre-trained model
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

# perform detection
bboxes = classifier.detectMultiScale(pixels, 1.35, 3) #configure accuracy - default high settings: 1.6 & 4

facex = []
facey = []
facex2 = []
facey2 = []

# print bounding box for each detected face
for box in bboxes:
	# extract
	x, y, width, height = box
	x2, y2 = x + width, y + height
	# draw a rectangle over the pixels
	rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
	
	facex.append(x)
	facey.append(y)
	facex2.append(x2)
	facey2.append(y2)

# show images
imshow('Face Detection', pixels)

i = 0
num = 1

for coord in facex: #count no. of facex coords collected, aka number of faces

	crop_image = pixels[facey[i]:facey2[i], facex[i]:facex2[i]]

	# concat different file names for each image
	filename = "faces/face" + str(num) + ".jpg"
	
	imwrite(filename, crop_image)

	i += 1
	num += 1

waitKey(0) #how to automate this?
destroyAllWindows()
