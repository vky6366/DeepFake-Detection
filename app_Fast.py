import os
import cv2
import torch
import numpy as np
import dlib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from werkzeug.utils import secure_filename
from torchvision import transforms
from tqdm import tqdm
from Model.model import DeepfakeDetector, DeepfakeDetectorb5
from Utils.preprocess import extract_frames, zoom_into_face  
from PIL import Image
import io
from Utils.gradient import GradCAM
from Utils.face_regions import FacialRegionAnalyzer
from typing import List, Optional
import socket

app = FastAPI(title="Deepfake Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRAME_FOLDER = os.path.join(BASE_DIR, 'processed_frames')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
WEBSITE_FOLDER = os.path.join(BASE_DIR, 'Website')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=WEBSITE_FOLDER), name="static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_b4 = DeepfakeDetector().to(device)
model_b5 = DeepfakeDetectorb5().to(device)

gradcam = GradCAM(model_b4, model_b4.base_model.features[-1])

try:
    b4_ckpt = torch.load(os.path.join(BASE_DIR, 'Model', 'best_b4_model_epoch6.pth'), map_location=device, weights_only=False)
    model_b4.load_state_dict(b4_ckpt["model_state_dict"])
    model_b4.eval()

    b5_ckpt = torch.load(os.path.join(BASE_DIR, 'Model', 'best_b5_model_epoch4.pth'), map_location=device, weights_only=False)
    model_b5.load_state_dict(b5_ckpt["model_state_dict"])
    model_b5.eval()

    print("✅ Both models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

transform = get_inference_transforms()

# Global variables
latest_3rd_frame: Optional[str] = None
facial_analysis_data: Optional[dict] = None

def get_ip_address():
    try:
        # Get all network interfaces
        hostname = socket.gethostname()
        ip_addresses = socket.gethostbyname_ex(hostname)[2]
        # Filter out localhost and return the first valid IP
        return [ip for ip in ip_addresses if not ip.startswith("127.")][0]
    except Exception as e:
        print(f"Error getting IP address: {e}")
        return "0.0.0.0"

app = FastAPI(title="Deepfake Detection API")

app.mount("/static", StaticFiles(directory=WEBSITE_FOLDER), name="static")
@app.get("/")
async def read_root():
    return FileResponse(os.path.join(WEBSITE_FOLDER, "index.html"))

@app.post("/upload")
async def upload_and_process(video: UploadFile = File(...)):
    """Handles video upload, extracts frames, stores the 3rd frame, and predicts deepfake probability."""
    global latest_3rd_frame

    if not video:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename = secure_filename(video.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    # Save uploaded file
    with open(filepath, "wb") as buffer:
        content = await video.read()
        buffer.write(content)

    video_name = os.path.splitext(filename)[0]
    frame_output_folder = os.path.join(FRAME_FOLDER, video_name)

    if os.path.exists(frame_output_folder):
        for file in os.listdir(frame_output_folder):
            file_path = os.path.join(frame_output_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(frame_output_folder, exist_ok=True)

    extract_frames(filepath, frame_output_folder, frame_interval=3)

    frame_list = sorted(os.listdir(frame_output_folder))
    if len(frame_list) >= 3:
        latest_3rd_frame = os.path.join(frame_output_folder, frame_list[2])

    processed_images = []
    frame_files = os.listdir(frame_output_folder)
    for img_file in tqdm(frame_files, desc="Processing Faces", unit="frame"):
        img_path = os.path.join(frame_output_folder, img_file)
        processed_img = zoom_into_face(img_path, frame_output_folder)
        if processed_img:
            processed_images.append(processed_img)

    if not processed_images:
        raise HTTPException(status_code=400, detail="No faces detected in the video")

    predictions = []
    with torch.no_grad():
        for img_file in tqdm(os.listdir(frame_output_folder), desc="Predicting Deepfakes"):
            img_path = os.path.join(frame_output_folder, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image_tensor = transform(image).unsqueeze(0).to(device)

            out_x = model_b4(image_tensor).squeeze().item()
            out_b5 = model_b5(image_tensor).squeeze().item()

            prob_x = torch.sigmoid(torch.tensor(out_x)).item()
            prob_b5 = torch.sigmoid(torch.tensor(out_b5)).item()
            final_prob = (prob_x + prob_b5) / 2

            predictions.append(final_prob)

    if not predictions:
        raise HTTPException(status_code=400, detail="No valid frames were processed for prediction.")

    avg_probability = np.mean(predictions)
    is_fake = "FAKE" if avg_probability > 0.6 else "REAL"

    print(f"Prediction: {is_fake}, Score: {avg_probability:.4f}")

    return {
        "message": "Prediction Complete!",
        "prediction": is_fake,
        "score": float(avg_probability)
    }

@app.get("/get_3rd_frame")
async def get_3rd_frame():
    """Returns the 3rd frame as a raw byte array in JSON response."""
    global latest_3rd_frame

    if not latest_3rd_frame or not os.path.exists(latest_3rd_frame):
        raise HTTPException(status_code=400, detail="3rd frame not found or not available")

    try:
        with open(latest_3rd_frame, "rb") as img_file:
            img_bytes = img_file.read()

        latest_3rd_frame = None

        return {
            "message": "3rd Frame Retrieved",
            "image_bytes": list(img_bytes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read frame: {str(e)}")

@app.get("/gradcam")
async def get_gradcam():
    """Returns Grad-CAM heatmap of the 3rd frame with facial region analysis."""
    global latest_3rd_frame, facial_analysis_data

    if not latest_3rd_frame or not os.path.exists(latest_3rd_frame):
        raise HTTPException(status_code=400, detail="3rd frame not found")

    image = cv2.imread(latest_3rd_frame)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    heatmap = gradcam.generate_heatmap(input_tensor)
    overlay = gradcam.apply_heatmap(np.array(pil_image), heatmap)

    region_scores, focused_regions = gradcam.analyze_facial_regions(np.array(pil_image), heatmap)
    facial_analysis_data = {
        "focused_regions": focused_regions
    }

    gradcam_path = os.path.join(FRAME_FOLDER, "gradcam_heatmap.jpg")
    cv2.imwrite(gradcam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    try:
        with open(gradcam_path, "rb") as img_file:
            img_bytes = img_file.read()

        os.remove(gradcam_path)

        return {
            "message": "Grad-CAM heatmap generated!",
            "heatmap_bytes": list(img_bytes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read heatmap: {str(e)}")

@app.get("/facial_analysis")
async def get_facial_analysis():
    """Returns the facial region analysis from the latest Grad-CAM computation."""
    global facial_analysis_data, latest_3rd_frame

    if facial_analysis_data is None:
        raise HTTPException(status_code=400, detail="No facial analysis data available")

    response_data = facial_analysis_data
    facial_analysis_data = None

    if latest_3rd_frame:
        video_name = os.path.splitext(os.path.basename(latest_3rd_frame))[0]
        frame_output_folder = os.path.join(FRAME_FOLDER, video_name)
        try:
            if os.path.exists(frame_output_folder):
                for img_file in os.listdir(frame_output_folder):
                    os.remove(os.path.join(frame_output_folder, img_file))
                os.rmdir(frame_output_folder)
        except FileNotFoundError:
            print(f"Directory {frame_output_folder} not found for deletion.")

    for file in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, file))

    return response_data

if __name__ == "__main__":
    import uvicorn
    
    # Get the IP address
    host_ip = get_ip_address()
    port = 5000
    
    print("\n" + "="*50)
    print(f"Server is running on:")
    print(f"Local URL:     http://localhost:{port}")
    print(f"Network URL:   http://{host_ip}:{port}")
    print(f"API Docs URL:  http://{host_ip}:{port}/docs")
    print("="*50 + "\n")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=port)