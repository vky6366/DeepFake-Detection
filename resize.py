'''from PIL import Image
import os

def resize_images(source_dir, target_dir, size=(360, 360)):
    """
    Resize images in the specified directory to the given size and save them to the target directory.
    
    :param source_dir: Directory containing subdirectories of images.
    :param target_dir: Directory where resized images will be saved.
    :param size: Desired size of the images as a tuple (width, height).
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    subdirs = ['Real', 'Fake']  # Subdirectory names

    for subdir in subdirs:
        input_subdir = os.path.join(source_dir, subdir)
        output_subdir = os.path.join(target_dir, subdir)

        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        files = os.listdir(input_subdir)
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_subdir, file)
                img = Image.open(img_path)
                img = img.resize(size, Image.Resampling.LANCZOS)  # Resize the image using high-quality downsampling filter
                img.save(os.path.join(output_subdir, file))  # Save the resized image

# Specify the directories
train_source_dir = r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\Priliminary\Train" # Change this to your 'Train' directory path
validation_source_dir = r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\Priliminary\Validation"  # Change this to your 'Validation' directory path

train_target_dir = r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\Priliminary\Resize_Train"  # Change this to where you want to store resized train images
validation_target_dir = r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\Priliminary\Resize_Validation"  # Change this to where you want to store resized validation images

# Resize images
resize_images(train_source_dir, train_target_dir)
resize_images(validation_source_dir, validation_target_dir)

print("Resizing complete.")'''


'''import os
import dlib
from PIL import Image, ImageDraw
import numpy as np

detector = None
predictor = None

def init_detector_predictor():
    global detector
    global predictor
    predictor_path = r"D:\DeepFake Detection System\Test_Model\shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

def zoom_and_resize_image(img_path, output_path, size=(360, 360), zoom_factor=2.4):
    """
    Detect the face in the image, zoom in, and then resize the image.
    """
    global detector
    global predictor

    img = Image.open(img_path)
    img = img.convert("RGB")  # Convert to RGB if needed

    # Convert to numpy array for Dlib processing
    img_np = np.array(img)
    
    # Detect faces
    detected_faces = detector(img_np, 1)
    
    if len(detected_faces) > 0:
        # Take the first detected face
        face = detected_faces[0]
        
        # Get the landmarks
        landmarks = predictor(img_np, face)
        
        # Get the bounding box around the face landmarks
        x_coords = [landmarks.part(n).x for n in range(68)]
        y_coords = [landmarks.part(n).y for n in range(68)]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Calculate zoomed-in bounding box with padding
        width = max_x - min_x
        height = max_y - min_y
        padding_w = int(width * (zoom_factor - 1) / 2)
        padding_h = int(height * (zoom_factor - 1) / 2)
        
        crop_box = (
            max(0, min_x - padding_w),  # left
            max(0, min_y - padding_h),  # upper
            min(img.width, max_x + padding_w),  # right
            min(img.height, max_y + padding_h)  # lower
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

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_detector_predictor) as executor:
        executor.map(process_single_image, all_tasks)

if __name__ == "__main__":
    # Specify the directories
    train_source_dir = r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\Train4"
    validation_source_dir = r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\Validation"
    test_source_dir = r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\Test"

    train_target_dir = r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\Resize_Train"
    validation_target_dir = r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\Resize_Validation"
    test_target_dir = r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\ResizeZ_Test"
    
    # Process images in parallel using all available CPU cores
    #process_images(validation_source_dir, validation_target_dir, size=(360, 360), zoom_factor=2.5, max_workers=10)
    #process_images(test_source_dir, test_target_dir, size=(360, 360), zoom_factor=2.5, max_workers=9)
    process_images(train_source_dir, train_target_dir, size=(360,360), zoom_factor=2.5, max_workers=11)

    print("Zoom and resizing complete.")'''

import cv2
import os
from mtcnn import MTCNN
from multiprocessing import Pool
import tensorflow as tf

# Function to process a single image
def process_image(img_path):
    output_folder = r"D:\DeepFake Detection System\Test_Model\Frames\Zoom_Real"
    os.makedirs(output_folder, exist_ok=True)
    detector = MTCNN()
    img = cv2.imread(img_path)
    detections = detector.detect_faces(img)

    for i, detection in enumerate(detections):
        x, y, width, height = detection['box']
        x, y = max(0, x), max(0, y)
        face = img[y:y+height, x:x+width]
        zoom_factor = 1.8  # Adjust zoom factor as needed
        center_x, center_y = x + width // 2, y + height // 2
        new_width, new_height = int(width * zoom_factor), int(height * zoom_factor)
        new_x = max(0, center_x - new_width // 2)
        new_y = max(0, center_y - new_height // 2)
        zoomed_face = img[new_y:new_y+new_height, new_x:new_x+new_width]
        resized_face = cv2.resize(zoomed_face, (360, 360))  # Resize to target dimensions
        face_filename = os.path.basename(img_path).replace('.jpg', f'_face{i}.jpg')
        cv2.imwrite(os.path.join(output_folder, face_filename), resized_face)

    print(f"Processed {img_path}")

# Wrapper function to handle directory processing
def process_directory(directory):
    images = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.jpg')]
    with Pool(processes=12) as pool:  # Utilize all CPU cores
        pool.map(process_image, images)

# Usage example
if __name__ == '__main__':
    # Paths to your directories
    image_dir = r"D:\DeepFake Detection System\Test_Model\Frames\Real_Frames"
    process_directory(image_dir)

'''import os
import cv2
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Set up TensorFlow to use DirectML
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        print("Using GPU:", physical_devices[0])
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU.")

def process_video(video_info):
    video_path, output_dir, zoom_factor = video_info
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    os.makedirs(output_dir, exist_ok=True)

    # Initialize MTCNN inside the function for multiprocessing
    detector = MTCNN()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = detector.detect_faces(rgb_frame)

        if results:
            for i, result in enumerate(results):
                x1, y1, w, h = result['box']
                x1, y1 = max(0, x1), max(0, y1)

                # Calculate zoomed coordinates
                center_x = x1 + w / 2
                center_y = y1 + h / 2
                new_w = w * zoom_factor
                new_h = h * zoom_factor
                new_x1 = int(center_x - new_w / 2)
                new_y1 = int(center_y - new_h / 2)
                new_x2 = int(center_x + new_w / 2)
                new_y2 = int(center_y + new_h / 2)

                # Ensure coordinates are within frame bounds
                new_x1 = max(0, new_x1)
                new_y1 = max(0, new_y1)
                new_x2 = min(frame.shape[1], new_x2)
                new_y2 = min(frame.shape[0], new_y2)

                face_img = frame[new_y1:new_y2, new_x1:new_x2]

                # Save the zoomed face image
                output_path = os.path.join(output_dir, f'{video_name}_frame{frame_idx}_face{i}.jpg')
                cv2.imwrite(output_path, face_img)

        frame_idx += 1

    cap.release()

def process_videos_in_parallel(video_paths, output_base_dir, zoom_factor=2.0, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 0)  # Reserve one CPU core

    video_info_list = []
    for video_path in video_paths:
        output_dir = os.path.join(output_base_dir, os.path.splitext(os.path.basename(video_path))[0])
        video_info_list.append((video_path, output_dir, zoom_factor))

    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_video, video_info_list), total=len(video_info_list)))

if __name__ == '__main__':
    dataset_dir = r"D:\DeepFake Detection System\Test_Model\Dataset\Real1"        # Replace with your dataset directory
    output_base_dir = r"D:\DeepFake Detection System\Test_Model\Frames\Real"    # Replace with desired output directory
    zoom_factor = 1.6                            # As per your requirement
    num_workers = None                           # Set to None to use max available CPUs

    # Get list of video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(video_extensions)]

    print(f'Total videos to process: {len(video_paths)}')
    process_videos_in_parallel(video_paths, output_base_dir, zoom_factor, num_workers)
'''







