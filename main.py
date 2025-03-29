import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Imposta il livello di log su ERROR
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Evita conflitti di librerie
os.environ['PYTHONWARNINGS'] = 'ignore'  # Ignora i warning di Python
from gan import train_gan
from generate import generate
from load_images import load_images

mode = 'generate' # 'train' or 'generate'

if __name__ == "__main__":

  X_train, X_train_original = load_images()

  X_train_len = X_train_original.shape[0]

  # ---------------------- 03. Create the models and train the GAN ----------------------
  if mode == 'train':
    train_gan(X_train, max_epochs=5000, batch_size=X_train_len)

  elif mode == 'generate':
    generate(X_train_original)
