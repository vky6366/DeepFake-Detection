import cv2
import os
import dlib
from pathlib import Path

detector = dlib.get_frontal_face_detector()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define path relative to the base directory
predictor_path = os.path.join(BASE_DIR, 'shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(predictor_path)


def extract_frames(video_path, output_folder, frame_interval=3):
    """Extracts frames from video."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    video_name = Path(video_path).stem

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = f"{output_folder}/{video_name}_frame_{frame_count}.jpg"
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()

def zoom_into_face(image_path, output_folder, zoom_factor=1.3):
    """Zooms into detected face."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    face = faces[0]
    landmarks = predictor(gray, face)
    x_min = min([landmarks.part(n).x for n in range(68)])
    y_min = min([landmarks.part(n).y for n in range(68)])
    x_max = max([landmarks.part(n).x for n in range(68)])
    y_max = max([landmarks.part(n).y for n in range(68)])

    w, h = x_max - x_min, y_max - y_min
    x_min = max(0, x_min - int(w * (zoom_factor - 1) / 2))
    y_min = max(0, y_min - int(h * (zoom_factor - 1) / 2))
    x_max = min(image.shape[1], x_max + int(w * (zoom_factor - 1) / 2))
    y_max = min(image.shape[0], y_max + int(h * (zoom_factor - 1) / 2))

    cropped_face = image[y_min:y_max, x_min:x_max]
    resized_face = cv2.resize(cropped_face, (224, 224))

    face_filename = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(face_filename, resized_face)
    return face_filename
