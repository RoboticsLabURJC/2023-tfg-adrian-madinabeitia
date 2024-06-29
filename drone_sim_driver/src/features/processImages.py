import os
import cv2
import argparse
import numpy as np



def color_filter(image, color):

        # Apply a red filter to the image
    if color == "Red":
        lower = np.array([0, 0, 70])
        upper = np.array([50, 50, 255])
        mask = cv2.inRange(image, lower, upper)
        
    else:
        lower = np.array([60, 60, 60])
        upper = np.array([120, 120, 120])
        mask = cv2.inRange(image, lower, upper)
    
    return mask

def image_aperture(mask):
    erosion_kernel = np.ones((3, 3), np.uint8)
    dilate_kernel = np.ones((2, 2), np.uint8)
    n_erosion = 1
    n_dilatation = 1

    # Perform aperture
    eroded_mask = cv2.erode(mask, erosion_kernel, iterations=n_erosion)
    dilated_mask = cv2.dilate(eroded_mask, dilate_kernel, iterations=n_dilatation)


    return dilated_mask 

def apply_filters(image):

    
    red_mask = color_filter(image, "G")
    aperture = image_aperture(red_mask)
    
    return aperture

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Read the image
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)
            
            # Apply filters
            processed_image = apply_filters(image)
            
            # Save the processed image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_image)
            print(f"Processed and saved: {filename}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process ROS bag data')
    parser.add_argument('--folder', type=str, help='Path to images', required=True)
    args = parser.parse_args()

    process_images(args.folder + "/frontal_images", args.folder + "/proccesedImages")

if __name__ == "__main__":
    main()