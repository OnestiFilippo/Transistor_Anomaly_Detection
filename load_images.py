import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Imposta il livello di log su ERROR
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Evita conflitti di librerie
os.environ['PYTHONWARNINGS'] = 'ignore'  # Ignora i warning di Python


# ---------------------- 01. Load the dataset images ----------------------
def load_images():
  # Load the images that are in the form of numpy arrays
  dataset_path = "transistor/train/good"

  # Images dimensions
  img_height, img_width = 1024, 1024

  # Loading images from a folder in grayscale and normalizing them
  def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
      if filename.endswith(".png"):  # Controlla che sia un JPG
        img_path = os.path.join(folder, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width), color_mode='grayscale')
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
    return np.array(images)

  # Load dataset
  X_train = load_images_from_folder(dataset_path)
  X_train_len = len(X_train)
  print(f"Original dataset shape: {X_train.shape}")

  # Visualize a random image
  im = random.randint(0, X_train.shape[0])
  # Visualize 1024x1024 image and 128x128 image together
  plt.figure(figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.imshow(X_train[im], cmap='gray')
  plt.subplot(1, 2, 2)
  plt.imshow(tf.image.resize(X_train[im], [128, 128]), cmap='gray')
  plt.show()

  # ---------------------- 02. Preprocess the images ----------------------

  # Resize the images
  X_train = tf.image.resize(X_train, [128, 128])

  # Normalize the images
  X_train = X_train / 255.0

  print(f"Processed dataset shape: {X_train.shape}")

  # Turn into tf.data.Dataset
  X_train_original = X_train
  X_train = tf.data.Dataset.from_tensor_slices(X_train).shuffle(60000).batch(256)

  return X_train, X_train_original