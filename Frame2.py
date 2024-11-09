import cv2
import os
import numpy as np
from cv2 import dnn

# Define your functions at the top level

def extract_and_focus_faces(video_path, output_folder, max_frames_per_second, zoom_factor=2):
    try:
        # Initialize the DNN model inside the function
        net = dnn.readNetFromCaffe(
            r"D:\DeepFake Detection System\Test_Model\deploy.prototxt",
            r"D:\DeepFake Detection System\Test_Model\res10_300x300_ssd_iter_140000_fp16.caffemodel"
        )
        
        video = cv2.VideoCapture(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        if not video.isOpened():
            print(f"Failed to open video: {video_path}")
            return
        fps = int(video.get(cv2.CAP_PROP_FPS))
        frames_per_second = min(fps, max_frames_per_second)
        frame_interval = max(1, fps // frames_per_second)
        count = 0
        success, image = video.read()
        
        while success:
            if count % frame_interval == 0:
                h, w = image.shape[:2]
                blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                faces = net.forward()
                for i in range(faces.shape[2]):
                    confidence = faces[0, 0, i, 2]
                    if confidence > 0.5:
                        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x, y, x1, y1) = box.astype("int")
                        
                        # Adjust box dimensions based on zoom factor
                        center_x, center_y = (x + x1) // 2, (y + y1) // 2
                        width = int((x1 - x) * zoom_factor / 2)
                        height = int((y1 - y) * zoom_factor / 2)
                        new_x, new_x1 = max(0, center_x - width), min(w, center_x + width)
                        new_y, new_y1 = max(0, center_y - height), min(h, center_y + height)
    
                        face = image[new_y:new_y1, new_x:new_x1]
                        resized_face = cv2.resize(face, (720, 720))
                        frame_filename = f"{video_name}_frame{count}_face{i}.jpg"
                        cv2.imwrite(os.path.join(output_folder, frame_filename), resized_face)
            success, image = video.read()
            count += 1
        video.release()
    except Exception as e:
        print(f"Error processing {video_path}: {e}")

def process_videos_sequential(video_folder, output_folder, frames_per_second):
    os.makedirs(output_folder, exist_ok=True)
    videos = [
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.lower().endswith(('.mp4', '.avi'))
    ]
    for v in videos:
        extract_and_focus_faces(v, output_folder, frames_per_second)

# Add your paths
real_videos_path = r"D:\DeepFake Detection System\Test_Model\Dataset\Real\Real0"
fake_videos_path = r"D:\DeepFake Detection System\Test_Model\Dataset\Fake"

if __name__ == "__main__":
    # Process real videos
    process_videos_sequential(
        real_videos_path,
        r"D:\DeepFake Detection System\Test_Model\Frames\Real",
        15
    )

    # Process fake videos
    #process_videos_sequential(
    #    fake_videos_path,
    #    r"D:\DeepFake Detection System\Test_Model\Frames\Fake",
    #    15
    #)
