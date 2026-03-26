import os
from keras import utils
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, url_for
from keras.src.legacy.preprocessing import image


app = Flask(__name__, template_folder='templates', static_folder='static')

# Load your trained Keras model
MODEL_PATH = "Plant_disease_model_3.keras"
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        print(f"Received message from {name} ({email}): {message}")
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('prediction.html', error="No file part")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('prediction.html', error="No selected file")

        # Create the temp path if it doesn't exist
        temp_path = os.path.join('static', 'temp')
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        file_path = os.path.join(temp_path, file.filename)
        file.save(file_path)

        # Image Preprocessing
        # Replace line 50 and 54 in app_2.py
        try:
            # Use utils instead of image
            img = utils.load_img(file_path, target_size=(224, 224)) 
        except Exception as e:
            return f"Error processing image: {str(e)}", 500

        img_array = utils.img_to_array(img) # Use utils here too
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Run Prediction
        preds = model.predict(img_array)
        prediction_index = np.argmax(preds, axis=1)[0]

        # SYNC: Must match train_3.py classes_list
        class_labels = {
            0: "Healthy",
            1: "Bacterial Wilt Disease",
            2: "Manganese Toxicity"
        }
        result = class_labels.get(prediction_index, "Unknown")

        # Pass both prediction and the relative path to the image
        return render_template('prediction.html', 
                               prediction=result, 
                               img_path=file.filename) # Just the name

    return render_template('prediction.html')

if __name__ == "__main__":
    app.run(debug=True)
