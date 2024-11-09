import os
import cv2
import dlib
import logging
from multiprocessing import Pool

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO)

# Initialize Dlib's face detector (HOG-based) and the facial landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def extract_and_process_frames(video_path, output_folder, max_frames_per_second):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        fps = int(video.get(cv2.CAP_PROP_FPS))
        frame_interval = max(1, fps // max_frames_per_second)
        count = 0
        success, image = video.read()

        while success:
            if count % frame_interval == 0:
                # Convert the BGR image to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Perform face detection
                faces = face_detector(gray_image, 1)
                for face in faces:
                    # Get the landmarks/parts for the face in box d.
                    landmarks = landmark_predictor(gray_image, face)
                    # Draw landmarks on the face
                    for n in range(0, 68):  # There are 68 landmark points
                        x = landmarks.part(n).x
                        y = landmarks.part(n).y
                        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

                # Save the processed frame with landmarks
                frame_filename = f"{video_name}_frame{count}_faces.jpg"
                cv2.imwrite(os.path.join(output_folder, frame_filename), image)

            success, image = video.read()
            count += 1
    finally:
        video.release()

def process_video(video_data):
    video_path, output_folder, frames_per_second = video_data
    extract_and_process_frames(video_path, output_folder, frames_per_second)

def process_videos(video_folder, output_folder, frames_per_second):
    os.makedirs(output_folder, exist_ok=True)
    videos = [(os.path.join(video_folder, video), output_folder, frames_per_second)
              for video in os.listdir(video_folder) if video.lower().endswith(('.mp4', '.avi'))]
    pool = Pool(processes=10)
    try:
        pool.map(process_video, videos)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, terminating processes.")
        pool.terminate()
    finally:
        pool.close()
        pool.join()

if __name__ == '__main__':
    try:
        process_videos(r"D:\DeepFake Detection System\Test_Model\Dataset\Real", r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\Dlib_Real", 15)
        process_videos(r"D:\DeepFake Detection System\Test_Model\Dataset\Fake", r"D:\DeepFake Detection System\Test_Model\Dlib_Dataset\Dlib_Fake", 15)
    except KeyboardInterrupt:
        print("Terminated by user")
