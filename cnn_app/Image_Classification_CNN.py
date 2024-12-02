import os
import numpy as np
from datetime import datetime
import json
from keras import layers, models
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import random
import cv2 as cv

# Use absolute paths for dataset locations to avoid path-related issues
base_dir = 'C:/Users/Sagar/Downloads/Capstone-Final-Project/Capstone-Final-Project/cnn_app/datasets/cnn_dataset/DisasterModel'
model_dir = base_dir  # Define model directory

# Corrected paths for cyclone, earthquake, flood, and wildfire directories
cyclone_dir = os.path.join(base_dir, 'Cyclone_Wildfire_Flood_Earthquake_Dataset', 'Cyclone')
earthquake_dir = os.path.join(base_dir, 'Cyclone_Wildfire_Flood_Earthquake_Dataset', 'earthquake')
flood_dir = os.path.join(base_dir, 'Cyclone_Wildfire_Flood_Earthquake_Dataset', 'flood')
wildfire_dir = os.path.join(base_dir, 'Cyclone_Wildfire_Flood_Earthquake_Dataset', 'wildfire')

# Correct directory structure for train, validation, and test sets
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_dir_c = os.path.join(train_dir, 'cyclone')
train_dir_e = os.path.join(train_dir, 'earthquake')
train_dir_f = os.path.join(train_dir, 'flood')
train_dir_w = os.path.join(train_dir, 'wildfire')

validation_dir_c = os.path.join(validation_dir, 'cyclone')
validation_dir_e = os.path.join(validation_dir, 'earthquake')
validation_dir_f = os.path.join(validation_dir, 'flood')
validation_dir_w = os.path.join(validation_dir, 'wildfire')

test_dir_c = os.path.join(test_dir, 'cyclone')
test_dir_e = os.path.join(test_dir, 'earthquake')
test_dir_f = os.path.join(test_dir, 'flood')
test_dir_w = os.path.join(test_dir, 'wildfire')

# Check if directories exist
print('Check if samples are of same size:\n\nTrain:')
print(len(os.listdir(train_dir_c)), len(os.listdir(train_dir_e)), len(os.listdir(train_dir_f)), len(os.listdir(train_dir_w)))
print('Validation:')
print(len(os.listdir(validation_dir_c)), len(os.listdir(validation_dir_e)), len(os.listdir(validation_dir_f)), len(os.listdir(validation_dir_w)))
print('Test:')
print(len(os.listdir(test_dir_c)), len(os.listdir(test_dir_e)), len(os.listdir(test_dir_f)), len(os.listdir(test_dir_w)))

# Ensure model loading logic works based on existence of model
model = None
model_path = 'cnn_app/Datasets/cnn_dataset/model.keras'
model_history_path = 'cnn_app/Datasets/cnn_dataset/history.json'

model_exists = os.path.exists(model_path)
history_exists = os.path.exists(model_history_path)

if model_exists:
    model = load_model(model_path)
else:
    # Build model if not pre-trained
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

model.summary()

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=.2,
                                   height_shift_range=.2,
                                   shear_range=.2,
                                   zoom_range=.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(180, 180),
                                                    batch_size=32,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(180, 180),
                                                        batch_size=32,
                                                        class_mode='categorical')

print('\n\nCheck a size of data/label batches:')
for data_batch, label_batch in train_generator:
    print('data batch size:', data_batch.shape)
    print('label batch size:', label_batch.shape)
    break

# Set callbacks for early stopping and model checkpoint
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=6)
mc = ModelCheckpoint(model_path, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = None

if history_exists:
    with open(model_history_path, 'r') as json_file:
        history = json.load(json_file)
else:
    history = model.fit(
        train_generator,
        steps_per_epoch=150,
        epochs=1,  # Adjust the number of epochs
        validation_data=validation_generator,
        validation_steps=100,
        callbacks=[es, mc]
    )

    # Save the history of training
    with open(model_history_path, 'w') as json_file:
        json.dump(history.history, json_file)

# Save the trained model in .keras format
model.save(model_path)

# Load best model for evaluation
model = load_model(model_path)

# Evaluate on test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(180, 180), batch_size=1)

print('Test dataset size:', len(test_generator))

loss_test, acc_test = model.evaluate(test_generator)

print('\nThe accuracy of the model is:', str(np.round(acc_test, 2)),'% for loss value',str(np.round(loss_test, 2)),'%.')

# Image prediction function
def predict_disaster(image_path):
    img = image.load_img(image_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)
    label_names = {0: 'cyclone', 1: 'earthquake', 2: 'flood', 3: 'wildfire'}
    predicted_disaster_type = label_names[predicted_class_index[0]]
    return predicted_disaster_type
    
sample_image_path = 'C:/Users/Sagar/Downloads/image classification/image classification/cyclone.jpg'
predicted_type = predict_disaster(sample_image_path)
print(f'The predicted disaster type for the image is: {predicted_type}')
