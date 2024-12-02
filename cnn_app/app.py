import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, request, jsonify, render_template, Blueprint # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
import numpy as np # type: ignore

# Initialize the Flask blueprint
cnn_app = Blueprint('cnn_app', __name__, template_folder='templates', static_folder='static')

# Dynamically construct the absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'datasets', 'cnn_dataset', 'best_model.keras')

# Load the model using the absolute path
model = load_model(model_path)
label_names = {0: 'Cyclone', 1: 'Earthquake', 2: 'Flood', 3: 'Wildfire'}

# Define a function to process and predict the image
def predict_disaster(img_path):
    try:
        img = image.load_img(img_path, target_size=(180, 180))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Debugging: Print the shape of the image
        print(f"Image array shape: {img_array.shape}")

        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)
        predicted_disaster_type = label_names[predicted_class_index[0]]
        return predicted_disaster_type
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None

# Flask route for home page
@cnn_app.route('/')
def home():
    return render_template('index_cnn.html')

# Flask route for image prediction
@cnn_app.route('/predict', methods=['POST'])
def upload_image():
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

        # Predict the disaster type
        predicted_type = predict_disaster(file_path)

        # Handle prediction error
        if predicted_type is None:
            return jsonify({'error': 'Prediction failed'})

        # Remove the file after prediction
        os.remove(file_path)

        return jsonify({'predicted_disaster_type': predicted_type})

    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"})

# Initialize Flask application
if __name__ == '__main__':
    app = Flask(__name__)
    app.register_blueprint(cnn_app)
    app.run(debug=True)
