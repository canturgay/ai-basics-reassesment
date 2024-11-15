import cv2
import imgaug.augmenters as iaa

IMAGE_DIMENSIONS = (256, 256)  # Standard dimensions for resizing images

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(rgb_image, IMAGE_DIMENSIONS)
    
    # Image augmentation
    seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0, 1.0)),  # Apply Gaussian blur
        iaa.Fliplr(0.5),  # Horizontally flip 50% of the images
        iaa.Affine(rotate=(-20, 20))  # Rotate images
    ])
    
    augmented_images = seq(images=[resized_image])
    
    # Return the original and augmented images
    return [resized_image] + augmented_images