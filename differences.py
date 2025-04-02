from skimage.metrics import structural_similarity
import cv2
import numpy as np
import os

# Load images
test = cv2.imread('transistor/test/good/000.png')
# scale the image to 128x128
test = cv2.resize(test, (128, 128))
test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

def get_final_mask(test):
    # Initialize a final mask with all pixels set to 255 (white)
    final_mask = np.ones(test.shape, dtype='uint8') * 255

    # Test every generated image
    for i in range(16):
        # Load the generated image
        generated = cv2.imread('generated/generated_' + str(i) + '.png')

        # Convert images to grayscale
        generated = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)

        # Compute SSIM between the two images
        (score, diff) = structural_similarity(test, generated, full=True)
        #print("Image Similarity: {:.4f}%".format(score * 100))

        # The diff image contains the actual image differences between the two images
        # and is represented as a floating point data type in the range [0,1] 
        # so we must convert the array to 8-bit unsigned integers in the range
        # [0,255] so we can use it with OpenCV
        diff = (diff * 255).astype("uint8")

        # Manually set a threshold for the sensitivity
        manual_threshold = 50  # Adjust this value as needed
        thresh = cv2.threshold(diff, manual_threshold, 255, cv2.THRESH_BINARY_INV)[1]
        # Find contours to obtain the regions of the two input images that differ
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        mask = np.zeros(test.shape, dtype='uint8')
        for c in contours:
            area = cv2.contourArea(c)
            if area > 100:
                x, y, w, h = cv2.boundingRect(c)
                cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)

        # Combine the current mask with the final mask using bitwise AND
        final_mask = cv2.bitwise_and(final_mask, mask)

        # Show the final mask
        #cv2.imshow('final_mask', final_mask)
        #cv2.waitKey()

        return final_mask
    
if __name__ == "__main__":
    for test_im_file in os.listdir('transistor/test/damaged_case/'):
        # Load the test image
        test_im = cv2.imread('transistor/test/damaged_case/' + test_im_file)
        test_im = cv2.resize(test_im, (128, 128))
        test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2GRAY)   
        # Get the final mask
        final_mask = get_final_mask(test_im)
        # Show the final mask
        cv2.imshow('final_mask', final_mask)
        cv2.waitKey()

"""
# Show the original image
cv2.imshow('original', test)
# Show the generated image
cv2.imshow('generated', generated)
cv2.imshow('final_mask', final_mask)
cv2.waitKey()


# Test every generated image
for i in range(16):
    # Load the generated image
    generated = cv2.imread('generated/generated_' + str(i) + '.png')

    # Convert images to grayscale
    generated = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(test, generated, full=True)
    print("Image Similarity: {:.4f}%".format(score * 100))

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] test we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

    # Manually set a threshold for the sensitivity
    manual_threshold = 100  # Adjust this value as needed
    thresh = cv2.threshold(diff, manual_threshold, 255, cv2.THRESH_BINARY_INV)[1]
    # Find contours to obtain the regions of the two input images that differ
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(test.shape, dtype='uint8')
    for c in contours:
        area = cv2.contourArea(c)
        if area > 100:
            x,y,w,h = cv2.boundingRect(c)
            cv2.drawContours(mask, [c], 0, (255,255,255), -1)

    # Show the images
    cv2.imshow('mask', mask)
    # Show the original image
    cv2.imshow('original', test)
    # Show the generated image
    cv2.imshow('generated', generated)
    cv2.waitKey()
    """