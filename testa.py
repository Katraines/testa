from cv2 import imdecode, imshow, waitKey, destroyAllWindows, CascadeClassifier, rectangle
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, request, make_response
from flask_httpauth import HTTPBasicAuth
from flask_limiter import Limiter
from cv2 import imread, imwrite # temp
from numpy import fromstring
from io import BytesIO
import os

app = Flask('app')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

auth = HTTPBasicAuth()

limiter = Limiter(
    app, key_func=  auth.username, default_limits=["100 per second"])


@auth.verify_password
def verify_password(username, password):
    return password == "password" # just cuz i dont want to make real passwords

@app.route('/uploadImage', methods=['POST'])
@auth.login_required
def img():
    if 'photo' in request.files:
        photo = request.files['photo']
        in_memory_file = BytesIO()
        photo.save(in_memory_file)
        data = fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        img = cv2.imdecode(data, color_image_flag)
        output = AIMagic(img)
        return make_response(output, 200)
    else: return make_response("Need an input file" ,400)


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