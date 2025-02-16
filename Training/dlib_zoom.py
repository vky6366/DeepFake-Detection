import cv2
import dlib
import os
import numpy as np
import multiprocessing
from tqdm import tqdm

# ✅ Load dlib models (Face Detector + Landmark Predictor)
detector = dlib.get_frontal_face_detector()  # Face detection
predictor = dlib.shape_predictor(r"D:\Deepfake\shape_predictor_68_face_landmarks.dat")  # Load landmarks model

def zoom_into_face(image_path, output_folder, zoom_factor=1.3, target_size=(224, 224)):
    """Detects face, finds landmarks, adjusts zoom, and saves cropped face."""
    image = cv2.imread(image_path)
    if image is None:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = detector(gray)  # Detect faces

    if len(faces) == 0:
        return False  # No face detected

    # ✅ Get the first detected face & landmarks
    face = faces[0]
    landmarks = predictor(gray, face)

    # ✅ Get key points: Eyes, Nose, Chin (for better cropping)
    x_min = min([landmarks.part(n).x for n in range(0, 68)])
    y_min = min([landmarks.part(n).y for n in range(0, 68)])
    x_max = max([landmarks.part(n).x for n in range(0, 68)])
    y_max = max([landmarks.part(n).y for n in range(0, 68)])

    # ✅ Expand bounding box with zoom factor
    w, h = x_max - x_min, y_max - y_min
    x_min = max(0, x_min - int(w * (zoom_factor - 1) / 2))
    y_min = max(0, y_min - int(h * (zoom_factor - 1) / 2))
    x_max = min(image.shape[1], x_max + int(w * (zoom_factor - 1) / 2))
    y_max = min(image.shape[0], y_max + int(h * (zoom_factor - 1) / 2))

    # ✅ Crop the adjusted face region
    cropped_face = image[y_min:y_max, x_min:x_max]

    # Resize to 224x224
    resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_CUBIC)

    # Save the zoomed-in face
    os.makedirs(output_folder, exist_ok=True)
    face_filename = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(face_filename, resized_face)
    
    return True

def process_image(args):
    """Helper function to unpack arguments for multiprocessing."""
    return zoom_into_face(*args)

def process_folder(input_folder, output_folder):
    """Processes all images in a folder using multiprocessing."""
    os.makedirs(output_folder, exist_ok=True)
    images = [os.path.join(input_folder, img) for img in os.listdir(input_folder) if img.lower().endswith(('png', 'jpg', 'jpeg'))]

    # ✅ Use multiprocessing for faster execution
    num_workers = 10  # Get available CPU cores
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_image, [(img, output_folder) for img in images]), 
                            total=len(images), desc="🔄 Processing Images", unit="image"))
    
    processed = sum(results)
    print(f"✅ Processed {processed} faces out of {len(images)} images.")

# Example Usage
if __name__ == "__main__":
    real_input_folder = r"D:\videos\second_10p\Real_frames"
    real_output_folder = r"D:\videos\Test\Real"
    
    fake_input_folder = r"D:\videos\second_10p\Fake_frames"
    fake_output_folder = r"D:\videos\Test\Fake"
    print("--------------------Starting Real----------------------")
    process_folder(real_input_folder, real_output_folder)
    print("--------------------Starting Fake----------------------")
    process_folder(fake_input_folder, fake_output_folder)
    print("✅ Face zooming with dlib landmarks completed!")
