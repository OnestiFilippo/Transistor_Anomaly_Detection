from train import train_gan, make_discriminator_model, make_generator_model
from generate import generate
from load_images import load_images
from differences2 import differences

# TODO: 
# 01. Load and preprocess the dataset images (DONE)
# 02. Create the generative and discriminative model (DONE)
# 03. Train the GAN (DONE)
# 04. Generate images using the trained model (DONE)
# 05. Load the test images (DONE)
# 06. Preprocess the test images (DONE)
# 07. Compare generated images with the test images (DONE)
# 08. Classify the generated images using the model trained in the previous task
# 09. Evaluate the classification accuracy

mode = 'generate' # 'train', 'generate' or 'differences'

if __name__ == "__main__":

  if mode == 'train':
    # ---------------------- 01. Load and preprocess the dataset the images ----------------------

    X_train, X_train_original = load_images()
    X_train_len = X_train_original.shape[0]

    # ---------------------- 02. Create the generative and discriminative models ----------------------

    # Create the generator model
    generator = make_generator_model()
    print("GENERATOR MODEL:")
    print(generator.summary())

    # Create the discriminator model
    discriminator = make_discriminator_model()
    print("DISCRIMINATOR MODEL:")
    print(discriminator.summary())

    # ---------------------- 03. Train the GAN ----------------------

    train_gan(X_train, 10000, X_train_len, generator, discriminator)

  elif mode == 'generate':
    # ---------------------- 04. Generate images using the trained model ----------------------
    generate('models/generator5000.keras', 96)

  elif mode == 'differences':
    # ---------------------- 07. Compare generated images with the test images ----------------------
  
    differences()

