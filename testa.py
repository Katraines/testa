# you need to install opencv 
from cv2 import imread
from cv2 import imshow
from cv2 import imwrite
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle

# load photograph
pixels = imread('test3.jpg')

# pre-trained model
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

# perform detection
bboxes = classifier.detectMultiScale(pixels, 1.6, 4) #configure accuracy - default settings: 1.6 & 4

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
	#imshow("Cropped image(s)", crop_image)
	print (1)

	filename = "faces/face" + str(num) + ".jpg"
	print (filename)
	
	imwrite(filename, crop_image)

	i += 1
	num += 1

	waitKey(0) #how to automate this?

destroyAllWindows()
