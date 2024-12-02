import os
import numpy as np
from flask import Flask, request, jsonify, render_template, Blueprint
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore

# Initialize the Flask application
app = Flask(__name__)

# Register Blueprints for each model
from cnn_app.app import cnn_app
from genai_app.app import genai_app
from nlp_app.app import nlp_app
from volunteer_app.app import volunteer_app

# Register Blueprints with appropriate URL prefixes
app.register_blueprint(cnn_app, url_prefix='/cnn')
app.register_blueprint(genai_app, url_prefix='/genai')
app.register_blueprint(nlp_app, url_prefix='/nlp')
app.register_blueprint(volunteer_app, url_prefix='/volunteer')

# Dynamically construct the absolute path to the CNN model file
current_dir = os.path.dirname(os.path.abspath(__file__))
cnn_model_path = os.path.join(current_dir, 'datasets', 'cnn_dataset', 'best_model.keras')

# Load the CNN model
try:
    cnn_model = load_model(cnn_model_path)
    print("CNN model loaded successfully.")
except Exception as e:
    print(f"Error loading CNN model: {e}")
    cnn_model = None

# Define label names for the CNN model
cnn_label_names = {0: 'cyclone', 1: 'earthquake', 2: 'flood', 3: 'wildfire'}

# Function for predicting disaster type using CNN model
def cnn_predict_disaster(img_path):
    try:
        # Load and process the image for CNN prediction
        img = image.load_img(img_path, target_size=(180, 180))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the image

        # Make prediction using CNN model
        prediction = cnn_model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)
        predicted_disaster_type = cnn_label_names[predicted_class_index[0]]
        return predicted_disaster_type

    except Exception as e:
        print(f"Error during CNN prediction: {e}")
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# Flask route for CNN model prediction
@app.route('/cnn/predict', methods=['POST'])
def cnn_upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        # Ensure the uploads directory exists
        uploads_dir = 'uploads'
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)

        # Save the uploaded file to a temporary path
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)

        # Predict the disaster type using CNN model
        predicted_type = cnn_predict_disaster(file_path)

        # Handle prediction error
        if predicted_type is None:
            return jsonify({'error': 'Prediction failed'})

        # Remove the file after prediction
        os.remove(file_path)

        return jsonify({'predicted_disaster_type': predicted_type})

    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"})


if __name__ == '__main__':
    app.run(debug=True)
