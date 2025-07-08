import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'


model = tf.keras.models.load_model('model/densenet_model (1).keras')


class_names = ['Peach___healthy', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Grape___healthy', 'Blueberry___healthy', 'Apple___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___Common_rust_', 'Potato___Early_blight', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Grape___Black_rot', 'Apple___Apple_scab', 'Soybean___healthy', 'Pepper,_bell___Bacterial_spot', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Tomato___Late_blight', 'Strawberry___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Cherry_(including_sour)___healthy', 'Tomato___Bacterial_spot', 'Grape___Esca_(Black_Measles)', 'Tomato___Early_blight', 'Orange___Haunglongbing_(Citrus_greening)', 'Corn_(maize)___healthy', 'Pepper,_bell___healthy', 'Tomato___Leaf_Mold', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Tomato___Septoria_leaf_spot', 'Squash___Powdery_mildew', 'Peach___Bacterial_spot', 'Potato___healthy', 'Tomato___Target_Spot', 'Cherry_(including_sour)___Powdery_mildew']


@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return 'No file uploaded!', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return render_template('index.html', prediction=predicted_class, image_path=filepath)
    
if __name__ == '__main__':
    app.run(debug=True)
