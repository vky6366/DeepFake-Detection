import dlib
import cv2
import numpy as np
import os
import glob
from multiprocessing import Pool

def initialize_predictor():
    global detector, predictor
    # Initialize dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"D:\DeepFake Detection System\Test_Model\shape_predictor_68_face_landmarks.dat")

def process_image(args):
    image_path, landmark_file = args

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = detector(gray)
    if len(faces) == 0:
        print(f"No faces found in image {image_path}")
        return None
    
    # Assume the first face detected is the primary face
    face = faces[0]
    
    # Predict facial landmarks
    landmarks = predictor(gray, face)
    
    # Convert landmarks to NumPy array
    landmark_coords = np.array([[p.x, p.y] for p in landmarks.parts()])
    
    # Save landmarks as a .npy file
    np.save(landmark_file, landmark_coords)
    print(f"Processed {image_path}")
    return landmark_file

def process_and_save_landmarks_parallel(image_dir, landmark_dir):
    # Create the landmark directory if it doesn't exist
    os.makedirs(landmark_dir, exist_ok=True)
    
    # Get all image file paths
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))  # Adjust the extension if necessary

    # Prepare a list of tuples containing image paths and corresponding landmark file paths
    tasks = []
    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        landmark_file = os.path.join(landmark_dir, base_name.replace('.jpg', '_landmarks.npy'))
        tasks.append((image_path, landmark_file))

    # Use multiprocessing Pool to process images in parallel
    with Pool(initializer=initialize_predictor) as pool:
        pool.map(process_image, tasks)

if __name__ == '__main__':
    # Directories
    data_dir = r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\Resize_Validation"
    landmark_output_dir = r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\Validation_Landmark"

    # Process real images
    process_and_save_landmarks_parallel(
        os.path.join(data_dir, 'real'),
        os.path.join(landmark_output_dir, 'real')
    )

    # Process fake images
    process_and_save_landmarks_parallel(
        os.path.join(data_dir, 'fake'),
        os.path.join(landmark_output_dir, 'fake')
    )
