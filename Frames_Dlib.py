import cv2
import numpy as np
import os
from openvino.runtime import Core
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Parameters
ZOOM_FACTOR = 1.6  # Adjust between 1.6 and 1.7 as needed
TARGET_FPS = 15    # Desired frame extraction rate in fps
VIDEO_FOLDER = r"D:\DeepFake Detection System\Test_Model\Dataset\Real1"  # Folder containing videos
OUTPUT_FOLDER = r"D:\DeepFake Detection System\Test_Model\Frames\Real"  # Output folder for frames
MODEL_PATH = r"D:\DeepFake Detection System\Test_Model\intel\face-detection-0200\FP16"  # Path to OpenVINO model files (.xml and .bin)

# Load OpenVINO Model
core = Core()
model = core.read_model(model=f"{MODEL_PATH}/face-detection-0200.xml")
compiled_model = core.compile_model(model=model, device_name="GPU")  # Use GPU if supported, fallback to CPU otherwise

# Get the model's input and output layers
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

def zoom_face(frame, box, zoom_factor):
    """Zoom into the face area within the frame."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    center_x, center_y = x1 + width // 2, y1 + height // 2

    # Calculate new width and height based on the zoom factor
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)

    # Calculate new top-left corner to keep the zoomed area centered
    x1 = max(center_x - new_width // 2, 0)
    y1 = max(center_y - new_height // 2, 0)
    x2 = min(center_x + new_width // 2, frame.shape[1])
    y2 = min(center_y + new_height // 2, frame.shape[0])

    # Crop the zoomed face area
    zoomed_face = frame[y1:y2, x1:x2]
    return zoomed_face

def detect_faces_openvino(frame):
    """Detect faces using OpenVINO."""
    h, w = frame.shape[:2]

    # Preprocess the frame to match the model's input requirements
    resized_frame = cv2.resize(frame, (256, 256))
    input_image = np.transpose(resized_frame, (2, 0, 1)).astype(np.float32)  # HWC to CHW
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

    # Run inference
    results = compiled_model([input_image])[output_layer]

    boxes = []
    for detection in results[0][0]:
        confidence = detection[2]
        if confidence > 0.5:  # Confidence threshold
            x_min = int(detection[3] * w)
            y_min = int(detection[4] * h)
            x_max = int(detection[5] * w)
            y_max = int(detection[6] * h)
            boxes.append((x_min, y_min, x_max, y_max))

    return boxes

def process_video(video_path):
    """Extract frames at 15fps, detect faces, zoom in, and save results using OpenVINO."""
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / TARGET_FPS)

    # Create output subfolder for the video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(OUTPUT_FOLDER, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    frame_count = 0

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval)
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces using OpenVINO
        boxes = detect_faces_openvino(frame)

        for i, box in enumerate(boxes):
            # Apply zoom to the detected face area
            zoomed_face = zoom_face(frame, box, ZOOM_FACTOR)

            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1

            # Resize the zoomed face to match the bounding box dimensions
            resized_zoomed_face = cv2.resize(zoomed_face, (width, height))

            # Place the resized zoomed face back onto the original frame
            frame[y1:y2, x1:x2] = resized_zoomed_face

            # Save frame with zoomed face
            frame_filename = f"{video_output_folder}/frame_{frame_count}_face_{i}.jpg"
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()


def process_all_videos(video_folder, output_folder):
    """Process all videos in a folder using multiprocessing and OpenVINO for face detection."""
    os.makedirs(output_folder, exist_ok=True)
    video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder)
                   if f.endswith(('.mp4', '.avi', '.mov'))]

    # Use multiprocessing Pool to process multiple videos at once
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(process_video, video_files), total=len(video_files), desc="Processing videos"))

# Main entry point
if __name__ == "__main__":
    # Run the batch processing with multiprocessing and OpenVINO for face detection
    process_all_videos(VIDEO_FOLDER, OUTPUT_FOLDER)
