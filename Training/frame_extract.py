import cv2
import os
from pathlib import Path

# Set paths
video_folder_fake = r"D:\videos\New_Dataset\Train\Fake"
video_folder_real = r"D:\videos\New_Dataset\Train\Real"
output_folder_fake = r"D:\videos\New_Dataset\Train\Fake_Frames"
output_folder_real = r"D:\videos\New_Dataset\Train\Real_Frames"

os.makedirs(output_folder_fake, exist_ok=True)
os.makedirs(output_folder_real, exist_ok=True)

def extract_frames(video_path, output_folder, frame_interval=3):  # Extract every 3rd frame (~10 FPS)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    video_name = Path(video_path).stem  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends

        if frame_count % frame_interval == 0:
            frame_filename = f"{output_folder}/{video_name}_frame_{frame_count}.jpg"
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    print(f"---Extracted {frame_count//frame_interval} frames from {video_name}---")

print("✅ Starting Fake")

# Process Fake videos
for video in os.listdir(video_folder_fake):
    if video.endswith(('.mp4', '.avi', '.mov')):
        extract_frames(os.path.join(video_folder_fake, video), output_folder_fake)

print("✅ Starting Real")

#Process Real videos
for video in os.listdir(video_folder_real):
    if video.endswith(('.mp4', '.avi', '.mov')):
        extract_frames(os.path.join(video_folder_real, video), output_folder_real)

print("✅ Frame extraction complete!")
