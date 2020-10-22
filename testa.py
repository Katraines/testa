# you need to install opencv 
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle

# load photograph
pixels = imread('test3.png')

# pre-trained model
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

# perform detection
bboxes = classifier.detectMultiScale(pixels)

# print bounding box for each detected face
for box in bboxes:
	# extract
	x, y, width, height = box
	x2, y2 = x + width, y + height
	# draw a rectangle over the pixels
	rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
	crop_image = pixels[y:y2, x:x2]

# show images
imshow('Face Detection', pixels)
imshow("Cropped", crop_image)

waitKey(0)
destroyAllWindows()