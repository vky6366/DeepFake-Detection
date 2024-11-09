import shutil
import os
import glob

def move_images(source_folder, target_folder, num_images=100, file_extension='*.mp4'):
    """
    Move a specific number of images from the source folder to the target folder.
    
    :param source_folder: Path to the source folder where images are stored.
    :param target_folder: Path to the target folder where images will be moved.
    :param num_images: Number of images to move.
    :param file_extension: The file extension of the images to move.
    """
    # Create the target folder if it doesn't already exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Get a list of images in the source folder
    images = glob.glob(os.path.join(source_folder, file_extension))
    
    # Move the first 'num_images' images to the target folder
    for image in images[:num_images]:
        # Construct the full path to the destination where the image will be moved
        dest = os.path.join(target_folder, os.path.basename(image))
        # Move the image
        shutil.move(image, dest)
        print(f"Moved {image} to {dest}")

# Example usage:
source = r"D:\DeepFake Detection System\Test_Model\Dataset\Fake\Fake10"
target = r"D:\DeepFake Detection System\Test_Model\Dataset\Fake\Fake8"
move_images(source, target, num_images=100)  # Adjust the number of images as needed
