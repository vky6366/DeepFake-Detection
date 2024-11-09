import os
import cv2
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Initialize OpenCV Haar Cascade face detector
face_cascade = None

def init_face_detector():
    global face_cascade
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

def zoom_and_resize_image(img_path, output_path, size=(360, 360), zoom_factor=2.4):
    """
    Detect the face in the image, zoom in, and then resize the image.
    """
    global face_cascade

    # Load image and convert to grayscale
    img = Image.open(img_path)
    img = img.convert("RGB")  # Ensure image is in RGB
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Take the first detected face
        (x, y, w, h) = faces[0]

        # Calculate zoomed-in bounding box with padding
        padding_w = int(w * (zoom_factor - 1) / 2)
        padding_h = int(h * (zoom_factor - 1) / 2)
        
        crop_box = (
            max(0, x - padding_w),             # left
            max(0, y - padding_h),             # upper
            min(img.width, x + w + padding_w), # right
            min(img.height, y + h + padding_h) # lower
        )
        
        # Crop and resize the image
        cropped_img = img.crop(crop_box)
        resized_img = cropped_img.resize(size, Image.Resampling.LANCZOS)
        resized_img.save(output_path)
        return True
    else:
        print(f"No face detected in {img_path}. Skipping this image.")
        return False

def process_single_image(args):
    img_path, output_path, size, zoom_factor = args
    try:
        processed = zoom_and_resize_image(img_path, output_path, size, zoom_factor)
        if not processed:
            print(f"Discarded {img_path} due to no face detection.")
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def process_images(source_dir, target_dir, size=(360, 360), zoom_factor=2.4, max_workers=9):
    """
    Process images in parallel using multiprocessing.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    subdirs = ['Real', 'Fake']  # Subdirectory names
    all_tasks = []

    for subdir in subdirs:
        input_subdir = os.path.join(source_dir, subdir)
        output_subdir = os.path.join(target_dir, subdir)

        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        files = os.listdir(input_subdir)
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_subdir, file)
                output_path = os.path.join(output_subdir, file)
                
                all_tasks.append((img_path, output_path, size, zoom_factor))

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_face_detector) as executor:
        executor.map(process_single_image, all_tasks)

if __name__ == "__main__":
    # Specify the directories
    train_source_dir = r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\Train4"
    validation_source_dir = r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\Validation"
    test_source_dir = r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\Test"

    train_target_dir = r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\Resize_Train"
    validation_target_dir = r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\Resize_Validation"
    test_target_dir = r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\ResizeZ_Test"
    
    # Process images in parallel using available CPU cores
    process_images(train_source_dir, train_target_dir, size=(360, 360), zoom_factor=2.5, max_workers=11)

    print("Zoom and resizing complete.")
