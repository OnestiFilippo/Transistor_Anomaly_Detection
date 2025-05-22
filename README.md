# Anomaly Detection on Transistor Images
## GAN-Based Image Generation and Classification

This project implements a **Generative Adversarial Network (GAN)** in Python to generate synthetic images from a training dataset. The generated images are then used to classify a separate set of test images through a similarity-based comparison with ground truth images. The classification is performed in both **binary** and **multi-class** modes.

---

## Network Architecture

> *(Insert an image showing the GAN architecture here)*

The GAN consists of two main components:

- **Generator**: Learns to produce realistic images that resemble those in the training dataset.
> *(Insert an image showing the generator architecture here)*
- **Discriminator**: Attempts to distinguish between real and generated images, guiding the generator to improve.
> *(Insert an image showing the discriminator architecture here)*

---

## Dataset

- **Training Set**: Set of 1024x1024 colored images of good transistors used to train the GAN.
- **Test Set**: Set of 100 1024x1024 colored images to be classified.
- **Ground Truth**: Labeled 1024x1024 binary masks used for evaluation of similarity with the generated content.

---

## Classification Process

> *(Insert an image showing the classification process here)*

1. Each test image is compared with the generated images to obtain the most similar one.
2. A **binary mask** is created based on visual similarity.
3. This mask is compared with the corresponding ground truth image using the following metrics:
   - **SSIM (Structural Similarity Index)** – *Weight: 10%*
   - **IoU (Intersection over Union)** – *Weight: 40%*
   - **Dice Coefficient** – *Weight: 40%*
   - **Pixel Accuracy** – *Weight: 10%*
4. The weighted combination of these metrics determines the final class of the test image.

---

## Results

- **Binary Classification Accuracy**:  `83%`

- **Multi-class Classification Accuracy**:  `74%`

- **Binary Classification Metrics Graph**:  
  > *(Insert a plot or image file showing SSIM, IoU, Dice, Pixel Accuracy over the test set)*

---

## Execution

To run the project, execute the `main.py` file. 
At the beginning of the file, set the `mode` variable to specify the desired action:

```python
mode = "train"       # Train the GAN model
mode = "generate"    # Generate images using the trained model
mode = "differences" # Classify test images using the generated images