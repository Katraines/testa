from werkzeug.security import generate_password_hash, check_password_hash
from cv2 import imdecode, CascadeClassifier, dnn, rectangle, imshow
from flask import Flask, request, make_response
from flask_httpauth import HTTPBasicAuth
from numpy import fromstring, uint8
from flask_limiter import Limiter
from io import BytesIO
from json import dumps

app = Flask('app')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

auth = HTTPBasicAuth()

limiter = Limiter(
    app, key_func=  auth.username, default_limits=["1 per minute"])


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
        data = fromstring(in_memory_file.getvalue(), dtype=uint8)
        color_image_flag = 1
        img = imdecode(data, color_image_flag)
        return make_response(imgProcessing(img), 200)
    else: return make_response("Need an input file" ,400)

def emptyFolder(path):
    # using listdir() method to list the files of the folder
    faces = os.listdir((path))

    for images in faces:
        if images.endswith(".jpg"):
            os.remove(os.path.join(path, images))

classifier = CascadeClassifier('models/haarcascade_frontalface_default.xml')

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

genderProto = "models/deploy_gender.prototxt"
genderModel = "models/gender_net.caffemodel"
genderNet = dnn.readNet(genderModel, genderProto)

genderList = ['Male', 'Female']

def getGender(face):
    blob = dnn.blobFromImage(face)
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]
    return gender

def getAge(face):
    return 35

def getLandmarks(face):
    # more img processing code
    return [(5,3,5), (0, 5, 64), (255, 54, 63)]

def imgProcessing(pixels):
    out = []
    for face in getFaces(pixels):
        age = getGender(face)
        landmarks = getLandmarks(face)
        out.append((age, landmarks))
    return dumps(out)