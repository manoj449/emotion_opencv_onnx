import os
import numpy as np
import cv2
import onnxruntime
from flask import Flask, render_template, request

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load ONNX model
session = onnxruntime.InferenceSession("model/facial_expression_recognition_mobilenetv1.onnx")



# Labels for prediction
labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def predict_emotion(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face detected"

    # Assume the first face
    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]

    # Resize to model input size
    resized = cv2.resize(face, (112, 112))

    img_float = resized.astype(np.float32) / 255.0
    img_rgb = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)
    img_transposed = np.transpose(img_rgb, (2, 0, 1))
    blob = np.expand_dims(img_transposed, axis=0)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    probs = outputs[0][0]
    top_index = np.argmax(probs)
    return labels[top_index]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    emotion = predict_emotion(filepath)
    return render_template('result.html', emotion=emotion, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
