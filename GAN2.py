import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Imposta il livello di log su ERROR
import random
import time

# TODO: 
# 01. Load the dataset images (DONE)
# 02. Preprocess the images (resize, normalize) (DONE)
# 03. Create the generative model (DONE)
# 04. Create the discriminative model (DONE)
# 05. Train the GAN (DONE)
# 06. Load the test images
# 07. Preprocess the test images
# 08. Generate images using the trained model
# 09. Compare generated images with the test images
# 10. Classify the generated images using the model trained in the previous task
# 11. Evaluate the classification accuracy

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
X_train = tf.data.Dataset.from_tensor_slices(X_train).shuffle(60000).batch(256)

# ---------------------- 03. Create the generative model ----------------------

def make_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))

    model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding="same", use_bias=False, activation="tanh"))

    return model

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

# ---------------------- 04. Create the discriminative model ----------------------

def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=[128, 128, 1]))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))  # Output probabilità (Sigmoid invece di Logits)

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Discriminator loss with label smoothing
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output) + 0.1, fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999)
discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

# ---------------------- 05. Train the GAN ----------------------

EPOCHS = 2000
noise_dim = 100 # Dimension of the noise vector
num_examples_to_generate = 16 # Number of examples to generate
BATCH_SIZE = 256 # Batch size

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

gen_losses = []
disc_losses = []

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
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

    return gen_loss, disc_loss

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      gen_loss, disc_loss = train_step(image_batch)

      # Save an image every 100 epochs
      if epoch % 100 == 0:
        generate_and_save_images(generator, epoch, seed)

      if epoch % 50 == 0:
        # Save losses
        gen_losses.append(gen_loss)
        disc_losses.append(disc_loss)
  
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    print(f"Generator loss: {gen_loss}, Discriminator loss: {disc_loss}")
    # If discriminator loss is too low, stop training
    if disc_loss < 0.1:
      break

  generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()

# Train the GAN on the dataset
train(X_train, EPOCHS)

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(gen_losses, label="Generator loss")
plt.plot(disc_losses, label="Discriminator loss")
plt.legend()
plt.show()