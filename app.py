from werkzeug.security import generate_password_hash, check_password_hash
# pylint: disable=fixme, no-name-in-module
from cv2 import imdecode, CascadeClassifier, dnn, rectangle, imshow
from flask import Flask, request, make_response
from flask_httpauth import HTTPBasicAuth
from numpy import fromstring, uint8
from flask_limiter import Limiter
from flask_cors import CORS
from io import BytesIO
from json import dumps
import random
import os


app = Flask('app')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

auth = HTTPBasicAuth()

limiter = Limiter(
    app, key_func=auth.username, default_limits=["10 per second"])


@auth.verify_password
def verify_password(username, password):
    # just cuz i dont want to make real passwords
    return check_password_hash('pbkdf2:sha256:150000$yfpwL5TV$8534b192e4691cab12683cc3e0b1bb5b3d23313b02b667165cb2df9f0ee4823c', password)


@app.route('/uploadImage', methods=['POST'])
def img():
    if 'photo' in request.files:
        photo = request.files['photo']
        in_memory_file = BytesIO()
        photo.save(in_memory_file)
        data = fromstring(in_memory_file.getvalue(), dtype=uint8)
        color_image_flag = 1
        img = imdecode(data, color_image_flag)
        return make_response(imgProcessing(img), 200)
    else:
        return make_response("Need an input file", 400)


@app.route('/')
def home():
    return make_response("""
        <form method="POST" enctype="multipart/form-data" action="http://gad:grapefukt@127.0.0.1/uploadImage">
            <input type="file" name="photo" />
            <input type="submit" />
        </form>
    """, 200)


classifier = CascadeClassifier('models/haarcascade_frontalface_default.xml')


def getFaces(pixels):
    faces = []

    for box in classifier.detectMultiScale(pixels, 1.35, 3):
        # extract
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # draw a rectangle over the pixels
        # remove after done testing
        rectangle(pixels, (x, y), (x2, y2), (0, 0, 255), 1)
        imshow('Face Detection', pixels)  # this too
        faces.append(pixels[y:y2, x:x2])

    return faces


def getGender(face):
    return random.choice(["Male", "Female"])  # Dummy Data


AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
               "(38-43)", "(48-53)", "(60-100)"]

prototxtPath = os.path.sep.join(["models", "age_deploy.prototxt"])
weightsPath = os.path.sep.join(["models", "age_net.caffemodel"])

# ageNet = dnn.readNet(prototxtPath, weightsPath)
def getAge(face):
    return random.randint(8, 85)  # Dummy Data


def getLandmarks(face):
    # more img processing code
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(random.randint(10, 50))]


def imgProcessing(pixels):
    out = []
    for face in getFaces(pixels):
        age = getAge(face)
        gender = getGender(face)
        landmarks = getLandmarks(face)

        out.append({
            "age": age,
            "gender": gender,
            "landmarks": landmarks
        })

    return dumps(out)
