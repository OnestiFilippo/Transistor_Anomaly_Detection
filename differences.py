import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2
import numpy as np

# Function to get the final mask
def get_final_mask(test):
    # Initialize a final mask with all pixels set to 255 (white)
    final_mask = np.ones(test.shape, dtype='uint8') * 255

    best_score = 0
    best_image = None

    # Iterate through all generated images
    for file in os.listdir('generated/'):
        if file.endswith('.png'):
            # Load the generated image
            generated = cv2.imread('generated/' + file)
            # Convert the generated image to grayscale
            generated = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)

            # Compute SSIM between the two images
            (score, diff) = ssim(test, generated, full=True)

            # SSIM difference ranges from -1 to 1, convert it to the range 0-1
            diff = (diff * -1 + 1) / 2
            
            # Convert to 8-bit for visualization
            diff_8bit = (diff * 255).astype("uint8")
            
            # Compute the absolute difference between the images
            abs_diff = cv2.absdiff(test, generated)
            
            # Combine the differences (you can modify this part to emphasize certain types of differences)
            combined_diff = cv2.addWeighted(diff_8bit, 0.5, abs_diff, 0.5, 0)
            
            # Apply a threshold to obtain a binary mask
            _, thresh = cv2.threshold(combined_diff, 30, 255, cv2.THRESH_BINARY)
            
            # Optional enhancement: clean the mask using morphological operations
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # If the score is better than the best score, update the best score
            if score > best_score:
                best_score = score
                best_image = generated
            
            # The final mask is obtained with bitwise AND
            final_mask = cv2.bitwise_and(final_mask, mask)

    return best_image, final_mask

# Intersection over Union
def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union > 0:
        return (intersection / union)
    else:
        return 0.0

# Dice coefficient
def dice_coefficient(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    if (mask1.sum() + mask2.sum()) == 0:
        return 1.0
    else:
        return (2 * intersection / (mask1.sum() + mask2.sum())) * 100

# Pixel accuracy
def pixel_accuracy(mask1, mask2):
    return np.mean(mask1 == mask2)

# Function to get the final class based on weighted scores
def get_final_class(results_dict, weights):
    # Collect all unique classes
    all_classes = set()
    for metric in results_dict:
        all_classes.add(results_dict[metric]['class'])
    
    # Dictionary to keep track of weighted scores for each class
    class_scores = defaultdict(float)
    
    # For each metric
    for metric, weight in weights.items():
        if metric in results_dict:
            class_name = results_dict[metric]['class']
            score = results_dict[metric]['score']
            
            # Add the weighted score to the class
            class_scores[class_name] += score * weight
    
    # Find the class with the highest score
    best_class = max(class_scores, key=class_scores.get)
    best_score = class_scores[best_class]
    
    return best_class, best_score, dict(class_scores)

# Function to visualize the results
def visualize_results(test_im, best_image, final_mask, 
                      best_mask, best_subdir, best_score,
                      best_mask_iou, best_subdir_iou, best_iou,
                      best_mask_dice, best_subdir_dice, best_dice,
                      best_mask_pixel_acc, best_subdir_pixel_acc, best_pixel_acc,
                      real_class):
    
    # Create a figure with 3 rows and 3 columns
    plt.figure(figsize=(16, 8))
    
    # Define colors for the titles
    green_color = 'green'
    normal_color = 'red'
    
    # First row: original images and final mask
    plt.subplot(3, 3, 1)
    plt.title(f'Test Image', color='black')
    # Convert from BGR to RGB if necessary
    if len(test_im.shape) == 3 and test_im.shape[2] == 3:
        test_im_rgb = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
    else:
        test_im_rgb = test_im
    plt.imshow(test_im_rgb, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.title('Best Generated Image')
    if len(best_image.shape) == 3 and best_image.shape[2] == 3:
        best_image_rgb = cv2.cvtColor(best_image, cv2.COLOR_BGR2RGB)
    else:
        best_image_rgb = best_image
    plt.imshow(best_image_rgb, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    plt.title('Mask')
    plt.imshow(final_mask, cmap='gray')
    plt.axis('off')
    
    # Second row: SSIM and IoU
    plt.subplot(3, 3, 4)
    # Color the title green if the class matches the real class
    ssim_color = green_color if best_subdir == real_class else normal_color
    plt.title(f'Ground Truth SSIM\nClass: {best_subdir}\nScore: {best_score*100:.4f}%', color=ssim_color)
    plt.imshow(best_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    # Color the title green if the class matches the real class
    iou_color = green_color if best_subdir_iou == real_class else normal_color
    plt.title(f'Ground Truth IoU\nClass: {best_subdir_iou}\nScore: {best_iou*100:.4f}%', color=iou_color)
    plt.imshow(best_mask_iou, cmap='gray')
    plt.axis('off')
    
    # Third row: Dice coefficient and Pixel accuracy
    plt.subplot(3, 3, 7)
    # Color the title green if the class matches the real class
    dice_color = green_color if best_subdir_dice == real_class else normal_color
    plt.title(f'Ground Truth Dice coefficient\nClass: {best_subdir_dice}\nScore: {best_dice*100:.4f}%', color=dice_color)
    plt.imshow(best_mask_dice, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    # Color the title green if the class matches the real class
    pixel_acc_color = green_color if best_subdir_pixel_acc == real_class else normal_color
    plt.title(f'Ground Truth Pixel accuracy\nClass: {best_subdir_pixel_acc}\nScore: {best_pixel_acc*100:.4f}%', color=pixel_acc_color)
    plt.imshow(best_mask_pixel_acc, cmap='gray')
    plt.axis('off')
    
    # Add a comparison visualization (overlay)
    plt.subplot(3, 3, 6)
    plt.title('Mask vs Ground Truth Images')
    # Create a colored overlay between the predicted mask and the best-scoring ground truth
    # Green: true positives, Red: false positives, Blue: false negatives
    overlay = np.zeros((*final_mask.shape, 3), dtype=np.uint8)
    
    # Combine the best masks into a single overlay
    best_gt = np.zeros_like(final_mask, dtype=np.uint8)
    masks = [best_mask, best_mask_iou, best_mask_dice, best_mask_pixel_acc]
    
    for mask in masks:
        if mask is not None:
            best_gt = cv2.bitwise_or(best_gt, mask)
    
    # Ensure the masks are binary
    if final_mask.max() > 1:
        mask_bin = final_mask > 127
    else:
        mask_bin = final_mask > 0
        
    if best_gt.max() > 1:
        gt_bin = best_gt > 127
    else:
        gt_bin = best_gt > 0
    
    overlay[..., 0] = np.logical_and(mask_bin, np.logical_not(gt_bin)) * 255  # False positives (red)
    overlay[..., 1] = np.logical_and(mask_bin, gt_bin) * 255  # True positives (green)
    overlay[..., 2] = np.logical_and(np.logical_not(mask_bin), gt_bin) * 255  # False negatives (blue)
    
    plt.imshow(overlay)
    plt.axis('off')
    
    # Calculate the weighted average and final class
    weights = {
        'ssim': 0.1,
        'iou': 0.4,
        'dice': 0.4,
        'pixel_acc': 0.1
    }
    
    results = {
        'ssim': {'class': best_subdir, 'score': best_score},
        'iou': {'class': best_subdir_iou, 'score': best_iou},
        'dice': {'class': best_subdir_dice, 'score': best_dice},
        'pixel_acc': {'class': best_subdir_pixel_acc, 'score': best_pixel_acc}
    }
    
    # Calculate the weighted score for each class
    class_scores = {}
    for metric, weight in weights.items():
        class_name = results[metric]['class']
        score = results[metric]['score']
        
        if class_name not in class_scores:
            class_scores[class_name] = 0
        class_scores[class_name] += score * weight
    
    # Find the class with the highest score
    final_class = max(class_scores, key=class_scores.get)
    final_score = class_scores[final_class]
    
    # Color for the final class
    final_class_color = green_color if final_class == real_class else normal_color
    
    # Visualization of metrics and final class in a subplot
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    # Display the metrics
    plt.text(0.1, 0.9, f"SSIM: {best_score*100:.4f}%", fontsize=12, 
             color=ssim_color if best_subdir == real_class else normal_color)
    plt.text(0.1, 0.8, f"IoU: {best_iou*100:.4f}%", fontsize=12, 
             color=iou_color if best_subdir_iou == real_class else normal_color)
    plt.text(0.1, 0.7, f"Dice: {best_dice*100:.4f}%", fontsize=12, 
             color=dice_color if best_subdir_dice == real_class else normal_color)
    plt.text(0.1, 0.6, f"Pixel Acc: {best_pixel_acc*100:.4f}%", fontsize=12, 
             color=pixel_acc_color if best_subdir_pixel_acc == real_class else normal_color)
    
    # Display the real class
    plt.text(0.1, 0.4, f"Real Class: {real_class}", fontsize=14, weight='bold')
    
    # Display the final class with the weighted average
    plt.text(0.1, 0.3, f"Final Class: {final_class}", fontsize=14, weight='bold', 
             color=final_class_color)
    plt.text(0.1, 0.2, f"Final Score: {final_score*100:.4f}%", fontsize=12)
    
    plt.tight_layout()
    plt.show()

def differences(view=False):
    path = 'transistor/test/'

    # Create a list with all the test images
    test_images = []
    real_classes = []
    for dir in os.listdir(path):
        if os.path.isdir(path + dir):
            for im in os.listdir(path + dir):
                if im.endswith('.png'):
                    test_images.append(path+dir+'/'+im)
                    real_classes.append(dir)
    
    # Shuffle the test images
    #np.random.shuffle(test_images)
 
    print('Number of test images:', len(test_images))
    total_correct = 0

    # For each test image
    for test_im_file in test_images:
        im_index = test_images.index(test_im_file)
        real_class = real_classes[im_index]
        total_images = len(test_images)
        
        # Load the test image
        test_im = cv2.imread(test_im_file)
        test_im = cv2.resize(test_im, (128, 128))
        test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2GRAY)
        # Get the final mask
        best_image, final_mask = get_final_mask(test_im)

        # Compare the final mask with the ground truth mask
        best_score = 0
        best_mask = None

        ssim_list = []
        iou_list = []
        best_iou = 0
        best_mask_iou = None
        best_subdir_iou = None

        dice_list = []
        best_dice = 0
        best_mask_dice = None
        best_subdir_dice = None

        pixel_acc_list = []
        best_pixel_acc = 0
        best_mask_pixel_acc = None
        best_subdir_pixel_acc = None

        for subdir in os.listdir('transistor/ground_truth/'):
            if os.path.isdir('transistor/ground_truth/'+subdir):
                for truth_file in os.listdir('transistor/ground_truth/'+subdir):
                    if truth_file.endswith('.png'):
                        ground_truth_mask = cv2.imread('transistor/ground_truth/' + subdir + '/' + truth_file)
                        #print('Ground truth mask:', 'transistor/ground_truth/' + subdir + '/' + truth_file)
                        ground_truth_mask = cv2.resize(ground_truth_mask, (128, 128))
                        ground_truth_mask = cv2.cvtColor(ground_truth_mask, cv2.COLOR_BGR2GRAY)

                        # Compare the final mask with the ground truth mask
                        # Compute SSIM between the two images
                        (score, _) = ssim(ground_truth_mask, final_mask, full=True)
                        #print('SSIM: {:.4f}%'.format(score * 100))
                        ssim_list.append(score)

                        # If the score is better than the best score, update the best score
                        if score > best_score:
                            best_score = score
                            best_mask = ground_truth_mask
                            best_subdir = subdir

                        # Compute IoU between the two images
                        iou_score = iou(ground_truth_mask, final_mask)
                        #print('IoU: {:.4f}%'.format(iou_score * 100))
                        iou_list.append(iou_score)

                        if iou_score >= best_iou:
                            best_iou = iou_score
                            best_mask_iou = ground_truth_mask
                            best_subdir_iou = subdir

                        # Compute the difference between the two images
                        diff = cv2.absdiff(ground_truth_mask, final_mask)
                        diff = (diff * 255).astype("uint8")
                        #cv2.imshow('Difference', diff)

                        # Compute the Dice coefficient between the two images
                        dice = dice_coefficient(ground_truth_mask, final_mask)
                        #print('Dice coefficient: {:.4f}%'.format(dice * 100))
                        dice_list.append(dice)

                        if dice > best_dice:
                            best_dice = dice
                            best_mask_dice = ground_truth_mask
                            best_subdir_dice = subdir

                        # Compute the pixel accuracy between the two images
                        pixel_acc = pixel_accuracy(ground_truth_mask, final_mask)
                        #print('Pixel accuracy: {:.4f}%'.format(pixel_acc * 100))
                        pixel_acc_list.append(pixel_acc)

                        if pixel_acc > best_pixel_acc:
                            best_pixel_acc = pixel_acc
                            best_mask_pixel_acc = ground_truth_mask
                            best_subdir_pixel_acc = subdir
        
        
        # Results

        results = {
        'ssim': {'class': best_subdir, 'score': best_score},
        'iou': {'class': best_subdir_iou, 'score': best_iou},
        'dice': {'class': best_subdir_dice, 'score': best_dice},
        'pixel_acc': {'class': best_subdir_pixel_acc, 'score': best_pixel_acc}
        }
        
        # Apply the specified weights
        weights = {
            'ssim': 0.1,
            'iou': 0.4,
            'dice': 0.4,
            'pixel_acc': 0.1
        }
        
        final_class, final_score, class_scores = get_final_class(results, weights)
        
        if final_class == real_class:
            total_correct += 1
            
        print("SSIM:", best_score)
        best_score = best_score * weights['ssim']
        print("IoU:", best_iou)
        best_iou = best_iou * weights['iou']
        print("Dice coefficient:", best_dice)
        best_dice = best_dice * weights['dice']
        print("Pixel accuracy:", best_pixel_acc)
        best_pixel_acc = best_pixel_acc * weights['pixel_acc']

        """# Pie chart for the class scores
        plt.figure(figsize=(8, 8))
        plt.pie(class_scores.values(), labels=class_scores.keys(), autopct='%1.1f%%', startangle=140)
        plt.title('Class Scores')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()"""

        print()
        print(f"Final Class: {final_class} with score: {final_score:.4f}")
        print("Class Scores:")
        for cls, score in sorted(class_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {score:.4f}")
        print("Real Class:", real_class)
        print('\n----------------------------------------\n')

       
        if view == True:
            # Visualize the results
            visualize_results(test_im, best_image, final_mask, 
                best_mask, best_subdir, best_score,
                best_mask_iou, best_subdir_iou, best_iou,
                best_mask_dice, best_subdir_dice, best_dice,
                best_mask_pixel_acc, best_subdir_pixel_acc, best_pixel_acc,
                real_class)
        
    accuracy = total_correct / total_images
    return accuracy



