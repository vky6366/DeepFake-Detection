'''import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import dlib
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load your model (replace with your model loading)
model = load_model(r"D:\DeepFake Detection System\Test_Model\best_model.keras")

# Predictor path for dlib (ensure this is correct for your system)
predictor_path = r"D:\DeepFake Detection System\Test_Model\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Define function to extract frames and predict
def extract_and_predict(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return {"error": "Failed to open video"}

    frame_results = []
    detector = MTCNN()
    success, image = video.read()
    count = 0

    while success:
        faces = detector.detect_faces(image)
        for face in faces:
            x, y, width, height = face['box']
            face_rect = dlib.rectangle(x, y, x + width, y + height)
            landmarks = predictor(image, face_rect)
            # Process landmarks and feed into model
            face_crop = image[y:y+height, x:x+width]  # Example face crop
            face_crop = cv2.resize(face_crop, (360, 360))  # Resize for model
            face_crop = np.expand_dims(face_crop, axis=0)  # Add batch dimension

            # Predict (using your deep learning model)
            prediction = model.predict(face_crop)
            frame_results.append({"frame": count, "prediction": float(prediction[0][0])})

        success, image = video.read()
        count += 1

    video.release()
    return {"predictions": frame_results}

# Route to upload video and process it
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files['video']
    video_path = os.path.join("uploads", video_file.filename)
    video_file.save(video_path)

    # Extract frames and run prediction
    result = extract_and_predict(video_path)

    return jsonify(result)

if __name__ == '__main__':
    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
'''

'''import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import dlib
import numpy as np
from mtcnn import MTCNN

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Predictor path for dlib (ensure this is correct for your system)
predictor_path = r"D:\DeepFake Detection System\Test_Model\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Define function to extract frames
def extract_frames(video_path, output_folder):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return {"error": "Failed to open video"}

    frame_results = []
    detector = MTCNN()
    count = 0
    success, image = video.read()

    while success:
        # Detect faces in the current frame
        faces = detector.detect_faces(image)
        for idx, face in enumerate(faces):
            x, y, width, height = face['box']
            # Save the face crop as an image file
            face_crop = image[y:y+height, x:x+width]
            frame_filename = f"{count}_face{idx}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, face_crop)
            frame_results.append({"frame": count, "face_id": idx, "file": frame_filename})

        success, image = video.read()
        count += 1

    video.release()
    return {"frames": frame_results}

# Route to upload video and process it
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files['video']
    video_path = os.path.join("uploads", video_file.filename)
    video_file.save(video_path)

    # Create a directory for the frames
    frames_output_folder = os.path.join("uploads", "frames", os.path.splitext(video_file.filename)[0])
    os.makedirs(frames_output_folder, exist_ok=True)

    # Extract frames from the video
    result = extract_frames(video_path, frames_output_folder)

    return jsonify(result)

if __name__ == '__main__':
    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
'''

import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import dlib
import numpy as np
from mtcnn import MTCNN

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Predictor path for dlib (ensure this is correct for your system)
predictor_path = r"D:\DeepFake Detection System\Test_Model\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

def download_video(video_url, download_folder):
    try:
        # Log the URL being requested
        print(f"Attempting to download video from URL: {video_url}")
        
        # Make a request to the video URL
        response = requests.get(video_url, stream=True)
        print(f"HTTP status code received: {response.status_code}")

        if response.status_code == 200:
            video_filename = os.path.join(download_folder, "downloaded_video.mp4")
            with open(video_filename, 'wb') as video_file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        video_file.write(chunk)

            # Check if the file is a valid video
            if os.path.getsize(video_filename) < 1024:  # Arbitrary size check
                print("Downloaded file is too small, possibly not a valid video.")
                return None

            print(f"Video successfully downloaded to: {video_filename}")
            return video_filename
        else:
            print(f"Failed to download video. HTTP status code: {response.status_code}")
            print(f"Response content: {response.text}")  # Log the response content for troubleshooting
            return None
    except Exception as e:
        print(f"Error occurred while downloading video: {e}")
        return None


@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video_url' in request.form:
            video_url = request.form['video_url']
            download_folder = "uploads"
            os.makedirs(download_folder, exist_ok=True)
            
            # Attempt to download the video from the provided URL
            video_path = download_video(video_url, download_folder)
            if not video_path:
                return jsonify({"error": "Failed to download video from URL"}), 400

            # Create a directory for the frames
            frames_output_folder = os.path.join(download_folder, "frames")
            os.makedirs(frames_output_folder, exist_ok=True)

            # Extract frames from the video (not shown here for brevity)
            result = extract_frames(video_path, frames_output_folder)
        else:
            return jsonify({"error": "No video URL or file uploaded"}), 400

        # Return frame extraction results
        return jsonify(result), 200

    except Exception as e:
        print(f"Exception occurred: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(host='0.0.0.0', port=5000)




