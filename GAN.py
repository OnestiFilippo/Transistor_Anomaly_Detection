import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Imposta il livello di log su ERROR
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Evita conflitti di librerie
os.environ['PYTHONWARNINGS'] = 'ignore'  # Ignora i warning di Python


# TODO: 
# 01. Load the dataset images (DONE)
# 02. Preprocess the images (resize, normalize) (DONE)
# 03. Create the generative model (DONE)
# 04. Create the discriminative model (DONE)
# 05. Train the GAN to obtain 16 images (DONE)
# 06. Load the test images
# 07. Preprocess the test images
# 08. Generate images using the trained model
# 09. Compare generated images with the test images
# 10. Classify the generated images using the model trained in the previous task
# 11. Evaluate the classification accuracy

# Select the mode
# train, generate
mode = 'train'

# ---------------------- 01. Load the dataset images ----------------------

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

# ---------------------- 03. Create the generative model ----------------------

def make_generator_model():
  model = tf.keras.Sequential()

  model.add(layers.Input(shape=(100,)))
  model.add(layers.Dense(8 * 8 * 512, use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Reshape((8, 8, 512)))

  model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same", use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same", use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding="same", use_bias=False, activation="tanh"))

  return model

generator = make_generator_model()

print("GENERATOR MODEL:")
print(generator.summary())

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

# ---------------------- 04. Create the discriminative model ----------------------

def make_discriminator_model():
  model = tf.keras.Sequential()

  model.add(layers.Input(shape=[128, 128, 1]))
  model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU(negative_slope=0.2))

  model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(negative_slope=0.2))

  model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(negative_slope=0.2))

  model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(negative_slope=0.2))

  model.add(layers.Flatten())
  model.add(layers.Dense(1, activation="sigmoid"))

  return model

discriminator = make_discriminator_model()
print("DISCRIMINATOR MODEL:")
print(discriminator.summary())
decision = discriminator(generated_image)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# Discriminator loss with label smoothing
def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output) + 0.1, fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

# ---------------------- 05. Train the GAN ----------------------

MAX_EPOCHS = 3000
noise_dim = 100 # Dimension of the noise vector
num_examples_to_generate = 16 # Number of examples to generate
BATCH_SIZE = X_train_len
# Batch size for the training is the same as the batch size for the dataset

# List to store the difference between the real and fake images
diff_list = []
variance_list = []

# Seed for the generator
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Define the training step
@tf.function
def train_step(images):
  noise = tf.random.normal([BATCH_SIZE, noise_dim])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

  # Calculate the difference between the real and fake images
  diff = tf.reduce_mean(tf.abs(images - generated_images))
  return diff

# Train the GAN
def train(dataset, epochs):
  for epoch in range(epochs):
    try:
      start = time.time()

      for image_batch in dataset:
        diff = train_step(image_batch)

        # Save an image every 100 epochs
        if epoch % 100 == 0:
          generate_and_save_images(generator, epoch, seed)

        print ('Epoch {} - {} s - Diff: {}'.format(epoch, str(round(time.time()-start,2)), str(round(diff.numpy(), 4))))
        print(f"Variance: {np.var(diff_list[-100:])}")
      
      # Append the difference to the list
      diff_list.append(diff)
      if epoch > 100:
        variance_list.append(np.var(diff_list[-100:]))

      # Stop training if the difference variance remain the same
      if len(diff_list) > 200 and np.var(diff_list[-100:]) < 1.5e-07:
        break
      
      # Save the model every 100 epochs
      if (epoch + 1) % 100 == 0:
        generator.save('models/generator'+(epoch+1)+'.keras')
        discriminator.save('models/discriminator'+(epoch+1)+'.keras')
  
    # Stop training if KeyboardInterrupt
    except KeyboardInterrupt:
      break

  # Generate after the final epoch
  generate_and_save_images(generator, epochs, seed)

  # Save the model
  generator.save('models/generatorF.keras')
  discriminator.save('models/discriminatorF.keras')

# Generate and save the images
def generate_and_save_images(model, epoch, test_input, save=True):
  # training is set to False so all layers run in inference mode.
  predictions = model(test_input, training=False)

  if save == True:
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

    plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

  return predictions

if mode == 'train':
  print("START TRAINING ".center(50, "="))

  # Train the GAN on the dataset
  train(X_train, MAX_EPOCHS)

  # Load the model and generate images
  generator = tf.keras.models.load_model('models/generator.keras')
  discriminator = tf.keras.models.load_model('models/discriminator.keras')

  # Generate and save the images
  generate_and_save_images(generator, 0, seed)

  print("END TRAINING ".center(100, "="))

  # Plot the difference between the real and fake images and the variance
  plt.figure(figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.plot(diff_list)
  plt.title("Difference between real and fake images")
  plt.xlabel("Epochs")
  plt.ylabel("Difference")
  plt.subplot(1, 2, 2)
  plt.plot(variance_list)
  plt.title("Variance of the difference")
  plt.xlabel("Epochs")
  plt.ylabel("Variance")
  plt.show()

elif mode == 'generate':
  # Load the model and generate images
  generator = tf.keras.models.load_model('models/generator3k.keras')
  discriminator = tf.keras.models.load_model('models/discriminator3k.keras')

  # Generate and save the images
  predictions = generate_and_save_images(generator, 0, seed, save=False)

  # Visualize the generated images
  plt.figure(figsize=(10, 5))
  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')
  plt.show()

  # Get the generated image that is the most similar to the original images
  min_diff = 1000
  min_diff_img = None
  rand_index = random.randint(0, X_train_original.shape[0])
  for i in range(predictions.shape[0]):
    diff = tf.reduce_mean(tf.abs(X_train_original[rand_index] - predictions[i]))
    if diff < min_diff:
      min_diff = diff
      min_diff_img = predictions[i]

  # Visualize the most similar image
  plt.figure(figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.imshow(X_train_original[rand_index], cmap='gray')
  plt.title("Original image")
  plt.axis('off')
  plt.subplot(1, 2, 2)
  plt.imshow(min_diff_img[:, :, 0] * 127.5 + 127.5, cmap='gray')
  plt.title("Generated image with diff: " + str(round(min_diff.numpy(), 4)))
  plt.axis('off')
  plt.show()

