import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Imposta il livello di log su ERROR
import random

# TODO: 
# 1. Load the dataset images
# 2. Preprocess the images (resize, normalize)
# 3. Create the generative model
# 4. Create the discriminative model
# 5. Train the GAN
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
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width), color_mode='grayscale')
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img_array)
    return np.array(images)

# Load dataset
X_train = load_images_from_folder(dataset_path)
print(f"Dataset shape: {X_train.shape}")

im = random.randint(0, X_train.shape[0])

# Visualize 1024x1024 image and 128x128 image together
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(X_train[im], cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(tf.image.resize(X_train[im], [128, 128]), cmap='gray')
plt.show()

# Resize the images
X_train = tf.image.resize(X_train, [128, 128])
print(f"Dataset shape: {X_train.shape}")

