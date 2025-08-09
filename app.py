from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import shutil
import time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
brain_model = load_model('models/VGG_model.h5', compile=False)
ecg_model = load_model('models/ecg_model_255x255.h5', compile=False)
ecg_classes = ['Abnormal', 'Myocardial Infarction', 'History of MI', 'Normal']


def clear_upload_folder():
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER)


def predict_brain_tumor(filepath):
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    start = time.time()
    prediction = brain_model.predict(img_array)[0][0]
    end = time.time()
    label = "🔴 Tumor Detected" if prediction >= 0.5 else "🟢 No Tumor Detected"
    return label, f"{prediction * 100:.2f}%", f"{end - start:.3f} sec"


def predict_ecg(filepath):
    img = Image.open(filepath).convert('L').resize((255, 255))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    start = time.time()
    prediction = ecg_model.predict(img_array)
    end = time.time()
    index = np.argmax(prediction)
    label = ecg_classes[index]
    confidence = f"{np.max(prediction) * 100:.2f}%"
    return label, confidence, f"{end - start:.3f} sec"


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/brain', methods=['GET', 'POST'])
def brain():
    if request.method == 'POST':
        clear_upload_folder()
        files = request.files.getlist("images")
        results = []
        total_time = 0
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            label, confidence, elapsed = predict_brain_tumor(filepath)
            total_time += float(elapsed.split()[0])
            results.append({'filename': filename, 'label': label, 'confidence': confidence, 'elapsed': elapsed})
        return render_template("brain_tumor.html", results=results, total_time=f"{total_time:.2f}", count=len(results))
    return render_template("brain_tumor.html", results=None)


@app.route('/heart', methods=['GET', 'POST'])
def heart():
    if request.method == 'POST':
        clear_upload_folder()
        files = request.files.getlist("images")
        results = []
        total_time = 0
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            label, confidence, elapsed = predict_ecg(filepath)
            total_time += float(elapsed.split()[0])
            results.append({'filename': filename, 'label': label, 'confidence': confidence, 'elapsed': elapsed})
        return render_template("heart_ecg.html", results=results, total_time=f"{total_time:.2f}", count=len(results))
    return render_template("heart_ecg.html", results=None)


if __name__ == '__main__':
    app.run(debug=True)
