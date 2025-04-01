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

# Function to create the generator model
def make_generator_model():
  model = tf.keras.Sequential()

  model.add(layers.Input(shape=(100,)))
  model.add(layers.Dense(8 * 8 * 512, use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())

  model.add(layers.Reshape((8, 8, 512)))

  model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same", use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same", use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding="same", use_bias=False, activation="tanh"))

  return model

# Function to create the discriminator model
def make_discriminator_model():
  model = tf.keras.Sequential()

  model.add(layers.Input(shape=[128, 128, 1]))
  model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.3))
  model.add(layers.LeakyReLU(negative_slope=0.2))

  model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.3))
  model.add(layers.LeakyReLU(negative_slope=0.2))

  model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.3))
  model.add(layers.LeakyReLU(negative_slope=0.2))

  model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.3))
  model.add(layers.LeakyReLU(negative_slope=0.2))

  model.add(layers.Flatten())
  model.add(layers.Dense(1, activation="sigmoid"))

  return model

# Discriminator loss with label smoothing
def discriminator_loss(real_output, fake_output):
  # This method returns a helper function to compute cross entropy loss
  cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

  real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output) + 0.1, fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

# Generator loss
def generator_loss(fake_output):
  # This method returns a helper function to compute cross entropy loss
  cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

  return cross_entropy(tf.ones_like(fake_output), fake_output)

# Define the training step
@tf.function
def train_step(BATCH_SIZE, noise_dim, generator, discriminator, generator_optimizer, discriminator_optimizer, images):
  noise = tf.random.normal([BATCH_SIZE, noise_dim])

  # Gradient tape is used to record the gradients of the loss with respect to the model parameters
  # The gradients are then used to update the model parameters
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
  
  # Calculate the difference between the real and fake images with SSIM
  images_scaled = tf.cast(images * 255, tf.uint8)
  generated_images_scaled = tf.cast(generated_images * 255, tf.uint8)
  ssim_value = tf.reduce_mean(tf.image.ssim(images_scaled, generated_images_scaled, max_val=255))
  
  return diff, ssim_value

# Train the GAN
def train(dataset, epochs, generator, discriminator, seed, BATCH_SIZE, noise_dim, generator_optimizer, discriminator_optimizer):
  # List to store the difference between the real and fake images
  diff_list = []
  variance_list = []
  ssim_list = []

  plt.ion()  # Modalità interattiva
  fig, ax = plt.subplots()
  ax.set_title("Difference between real and fake images")
  ax.set_xlabel("Epochs")
  ax.set_ylabel("Difference")
  ax.set_xlim(0, epochs)
  ax.set_ylim(0, 1)
  line1, = ax.plot([], [], 'r-')  # Linea rossa
  line2, = ax.plot([], [], 'b-')  # Linea blu

  for epoch in range(epochs):
    try:
      start = time.time()

      for image_batch in dataset:
        diff, ssim_value = train_step(BATCH_SIZE, noise_dim, generator, discriminator, generator_optimizer, discriminator_optimizer, image_batch)

        # Save an image every 100 epochs
        if epoch % 100 == 0:
          generate_and_save_images(generator, epoch, seed)

        print('Epoch {} - {} s'.format(epoch, str(round(time.time()-start,2))))
        print(f"Diff: {str(round(diff.numpy(), 4))}")
        print(f"SSIM: {str(round(ssim_value.numpy() * 100, 4))}%")
        print(f"Variance: {np.var(diff_list[-100:])}")
      
      # Append the difference to the list
      diff_list.append(diff)
      ssim_list.append(ssim_value)
      if epoch > 100:
        variance_list.append(np.var(diff_list[-100:]))

      # Stop training if the difference variance remain the same
      if len(diff_list) > 200 and np.var(diff_list[-100:]) < 1.5e-07:
        print("Early stopping: Variance threshold reached.")
        break
      
      # Save the model every 100 epochs
      if (epoch + 1) % 100 == 0:
        generator.save('models/generator'+str(epoch+1)+'.keras')
        discriminator.save('models/discriminator'+str(epoch+1)+'.keras')

      # Stop training if the model reaches ssim threshold
      if ssim_value > 80:
        print("Early stopping: SSIM threshold reached.")
        break

      # Real time plotting
      line1.set_xdata(np.arange(len(diff_list)))
      line1.set_ydata(diff_list)
      line2.set_xdata(np.arange(len(ssim_list)))
      line2.set_ydata(ssim_list)
      ax.relim()  # Aggiorna i limiti degli assi
      ax.autoscale_view()  # Ridimensiona gli assi
      # Aggiorna il grafico
      plt.draw()  # Disegna il nuovo grafico
      plt.pause(0.05)  # Attendi un po' per simulare dati in tempo reale
    
    # Stop training if KeyboardInterrupt
    except KeyboardInterrupt:
      break

  plt.ioff()  # Disattiva la modalità interattiva
  # Mostra il grafico finale
  plt.show()

  # Generate after the final epoch
  generate_and_save_images(generator, epochs, seed)

  # Save the model
  generator.save('models/generatorF.keras')
  discriminator.save('models/discriminatorF.keras')

  return diff_list, variance_list, ssim_list

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

# Function to create the models and train the GAN
def train_gan(X_train, max_epochs, batch_size, generator, discriminator):
    noise_dim = 100 # Dimension of the noise vector
    num_examples_to_generate = 16 # Number of examples to generate

    # Seed for the generator
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

    """
    # Resume training from the last checkpoint
    generator = tf.keras.models.load_model('models/generator2000.keras')
    discriminator = tf.keras.models.load_model('models/discriminator2000.keras')
    generated_image = generator(noise, training=False)
    decision = discriminator(generated_image)
    print("Generator and discriminator loaded from checkpoint.")
    print("Resuming training from the last checkpoint...")
    """
    
    print("START TRAINING ".center(50, "="))

    # Train the GAN on the dataset
    diff_list, variance_list, ssim_list = train(X_train, max_epochs, generator, discriminator, seed, batch_size, noise_dim, generator_optimizer, discriminator_optimizer)

    print("END TRAINING ".center(50, "="))

    # Load the model and generate images
    generator = tf.keras.models.load_model('models/generatorF.keras')
    discriminator = tf.keras.models.load_model('models/discriminatorF.keras')

    # Generate and save the images
    generate_and_save_images(generator, 0, seed)

    # Plot the difference between the real and fake images and the variance
    # Plot in real time the difference between the real and fake images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(diff_list, label='Difference', color='blue')
    plt.plot(ssim_list, label='SSIM', color='red')   
    plt.title("Difference between real and fake images")
    plt.xlabel("Epochs")
    plt.ylabel("Difference")
    plt.subplot(1, 2, 2)
    plt.plot(variance_list)
    plt.title("Variance of the difference")
    plt.xlabel("Epochs")
    plt.ylabel("Variance")
    plt.show()
