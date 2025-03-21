import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

# TODO: 
# 1. Load the dataset images
# 2. Preprocess the images (resize, normalize)
# 3. Create the generative model
# 4. Train the generative model and save the model
# 5. Evaluate the generative model
# 6. Load the test images
# 7. Preprocess the test images
# 8. Generate images using the trained model
# 9. Compare generated images with the test images
# 10. Classify the generated images using the model trained in the previous task
# 11. Evaluate the classification accuracy

# Load the dataset images that are in the form of numpy arrays
dataset_path = "transistor/train/good"

# Dimensioni target per il ridimensionamento
img_height, img_width = 1024, 1024

# Caricare e preprocessare le immagini
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):  # Controlla che sia un JPG
            img_path = os.path.join(folder, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width), color_mode='rgb')
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img_array)
    return np.array(images)

# Load dataset
X_train = load_images_from_folder(dataset_path)
print(f"Dataset shape: {X_train.shape}")

# Create the generative model
