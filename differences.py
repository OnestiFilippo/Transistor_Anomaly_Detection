import cv2
import numpy as np
import os
#from tensorflow import image as tf
from skimage.metrics import structural_similarity as ssim

def get_final_mask(test):
    # Initialize a final mask with all pixels set to 255 (white)
    final_mask = np.ones(test.shape, dtype='uint8') * 255

    best_score = 0
    best_image = None

    # Get every generated image
    for file in os.listdir('generated/'):
        if file.endswith('.png'):
            # Load the generated image
            generated = cv2.imread('generated/' + file)
            # Convert the generated image to grayscale
            generated = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)

            # Compute SSIM between the two images
            # Tensorflow SSIM
            # score = tf.ssim(test, generated, max_val=1.0).numpy()
            # Skimage SSIM
            (score, diff) = ssim(test, generated, full=True)

            #print("Image Similarity: {:.4f}%".format(score * 100))

            # If the score is better than the best score, update the best score
            if score > best_score:
                best_score = score
                best_image = generated
            
            # The diff image contains the actual image differences between the two images
            # and is represented as a floating point data type in the range [0,1] 
            # so we must convert the array to 8-bit unsigned integers in the range
            # [0,255] so we can use it with OpenCV
            diff = (diff * 255).astype("uint8")


            thresh = cv2.threshold(diff, 200 , 255, cv2.THRESH_BINARY_INV)[1]
            # Find contours to obtain the regions of the two input images that differ
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # contours = contours[0] if len(contours) == 2 else contours[1]

            mask = np.zeros(test.shape, dtype='uint8')
            for c in contours:
                area = cv2.contourArea(c)
                if area > 2000:
                        cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)

            # Combine the current mask with the final mask using bitwise AND
            final_mask = cv2.bitwise_and(final_mask, mask)

    return best_image, final_mask
    
def differences():
    path = 'transistor/test/'

    # Create a list with all the test images
    test_images = []
    for dir in os.listdir(path):
        if os.path.isdir(path + dir):
            for im in os.listdir(path + dir):
                if im.endswith('.png'):
                    test_images.append(path+dir+'/'+im)
    
    # Shuffle the test images
    np.random.shuffle(test_images)
 
    print('Number of test images:', len(test_images))
    #print(test_images)

    for test_im_file in test_images:
        # Load the test image
        test_im = cv2.imread(test_im_file)
        test_im = cv2.resize(test_im, (128, 128))
        test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2GRAY)   
        # Get the final mask
        best_image, final_mask = get_final_mask(test_im)

        # Compare the final mask with the ground truth mask
        # Load the ground truth mask from the folder transistor/ground_truth/
        best_score = 0
        best_mask = None
        for subdir in os.listdir('transistor/ground_truth/'):
            if os.path.isdir('transistor/ground_truth/'+subdir):
                for truth_file in os.listdir('transistor/ground_truth/'+subdir):
                    if truth_file.endswith('.png'):
                        ground_truth_mask = cv2.imread('transistor/ground_truth/' + subdir+'/' + truth_file)
                        ground_truth_mask = cv2.resize(ground_truth_mask, (128, 128))
                        ground_truth_mask = cv2.cvtColor(ground_truth_mask, cv2.COLOR_BGR2GRAY)

                        # Compare the final mask with the ground truth mask
                        # Compute SSIM between the two images

                        # Tensorflow SSIM
                        #score = tf.ssim(ground_truth_mask, final_mask, max_val=1.0).numpy()

                        # Skimage SSIM
                        (score, _) = ssim(ground_truth_mask, final_mask, full=True)

                        #print("Image Similarity: {:.4f}%".format(score * 100))
                        # If the score is better than the best score, update the best score
                        if score > best_score:
                            best_score = score
                            best_mask = ground_truth_mask
                            best_subdir = subdir

        # Show the final mask
        print('Class:' + best_subdir + ' - Score: {:.4f}%'.format(best_score * 100))
        cv2.imshow('Test Image', test_im)
        cv2.imshow('Best Generated Image', best_image)
        cv2.imshow('Mask', final_mask)
        cv2.imshow('Ground Truth', best_mask)
        # if ESC is pressed, exit the loop
        if cv2.waitKey(0) & 0xFF == 27:
            break