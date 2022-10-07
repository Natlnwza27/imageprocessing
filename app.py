from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2

app = Flask(__name__)

IMAGE_SIZE = (224, 224)
UPLOAD_FOLDER = 'static\\uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('tuned_resnet50v2.h5')

def image_preprocessor(path):
    currImg_BGR = cv2.imread(path)
    b, g, r = cv2.split(currImg_BGR)
    currImg_RGB = cv2.merge([r, g, b])
    currImg = cv2.resize(currImg_RGB, IMAGE_SIZE)
    currImg = currImg/255.0
    currImg = np.reshape(currImg, (1, 224, 224, 3))
    return currImg

def model_pred(image):
    prediction = model.predict(image)[0]
    return (prediction)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def homepage():
    return render_template('Homepage.html')

@app.route('/analysis', methods = ['GET', 'POST'])
def analysis():
    return render_template('Analysis.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'imageFile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['imageFile']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            imgPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(imgPath)
            print(f"Image saved at {imgPath}")
            image = image_preprocessor(imgPath)
            pred = model_pred(image)
            if pred[0] > 0.50:
                num = "{:.2f}".format((pred[0]*100))
                return render_template('Upload.html', name = filename, result = pred, confidence = num)
            else:
                num = "{:.2f}".format((1-pred[0])*100)
                return render_template('Upload.html', name = filename, result = pred, confidence = num)
    return redirect(url_for('analysis'))

@app.route('/guide')
def guide():
    return render_template('Guide.html')

@app.route('/normal')
def normal():
    return render_template('Normal.html')

@app.route('/pneumonia')
def pneumonia():
    return render_template('Pneumonia.html')

if __name__ == "__main__":
    app.run(debug = True)