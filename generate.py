import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Imposta il livello di log su ERROR
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Evita conflitti di librerie
os.environ['PYTHONWARNINGS'] = 'ignore'  # Ignora i warning di Python
from gan import generate_and_save_images

# Load the model and generate images
def generate(X_train_original, generator_file, discriminator_file):
    generator = tf.keras.models.load_model(generator_file)
    discriminator = tf.keras.models.load_model(discriminator_file)

    noise_dim = 100 # Dimension of the noise vector
    num_examples_to_generate = 16 # Number of examples to generate

    # Seed for the generator
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # Generate and save the images
    predictions = generate_and_save_images(generator, 0, seed, save=False)

    # Visualize the generated images
    plt.figure(figsize=(10, 5))
    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
    plt.show()

    # Get the generated image that is the most similar to the original images using SSIM
    min_diff = -1  # SSIM ranges from -1 to 1, initialize with the lowest possible value
    min_diff_img = None
    rand_index = random.randint(0, X_train_original.shape[0] - 1)
    for i in range(predictions.shape[0]):
      # Compute SSIM between the two images
      score = tf.image.ssim(X_train_original[rand_index], predictions[i], max_val=1.0).numpy()
      if score > min_diff:  # Higher SSIM indicates more similarity
        min_diff = score
        min_diff_img = predictions[i]

    min_diff = round(min_diff * 100, 4)  # Convert to percentage
    print(f"Most similar image SSIM: "+ str(min_diff) + "%")

    # Save the original image in a file using matplotlib
    plt.imsave('generated/original.png', X_train_original[rand_index, :, :, 0] * 127.5 + 127.5, cmap='gray')
    # Save the generated image in a file using matplotlib
    plt.imsave('generated/generated.png', min_diff_img[:, :, 0] * 127.5 + 127.5, cmap='gray')

    # Visualize the most similar image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(X_train_original[rand_index], cmap='gray')
    plt.title("Original image")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(min_diff_img[:, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.title("Generated image with similarity: " + str(min_diff) + "%")
    plt.axis('off')
    plt.show()