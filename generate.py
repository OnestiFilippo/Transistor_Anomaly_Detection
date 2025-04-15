import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Imposta il livello di log su ERROR
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Evita conflitti di librerie
os.environ['PYTHONWARNINGS'] = 'ignore'  # Ignora i warning di Python

# Generate and save the images
def generate_images(model, test_input):
  # training is set to False so all layers run in inference mode.
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')
  plt.close()

  return predictions

# Visualize the generated images
def visualize_generated_images():
    plt.figure(figsize=(10, 10))
    for i in range(16):
      image = plt.imread('generated/generated_' + str(i) + '.png')
      plt.subplot(4, 4, i+1)
      plt.imshow(image, cmap='gray')
      plt.axis('off')
    plt.show()

# Load the model and generate images
def generate(generator_file, num_images_to_generate):
    # Empty the generated directory
    for file in os.listdir('generated'):
        os.remove(os.path.join('generated', file))

    generator = tf.keras.models.load_model(generator_file)

    for i in range(int(num_images_to_generate / 16)):
      noise_dim = 100 # Dimension of the noise vector
      num_examples_to_generate = 16 # Number of examples to generate

      # Seed for the generator
      seed = tf.random.normal([num_examples_to_generate, noise_dim])

      # Generate and save the images
      predictions = generate_images(generator, seed)

      # Save the generated images
      for j in range(predictions.shape[0]):
        plt.imsave('generated/generated_' + str(j+i*16) + '.png', predictions[j, :, :, 0] * 127.5 + 127.5, cmap='gray')
    
    # Visualize the generated images
    visualize_generated_images()
