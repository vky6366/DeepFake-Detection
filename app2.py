import os
import cv2
import torch
import numpy as np
import dlib
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from torchvision import transforms
from tqdm import tqdm
from Model.model import DeepfakeDetector  # Import trained model
from Utils.preprocess import extract_frames, zoom_into_face  # Preprocessing functions
from PIL import Image
import io
from Utils.gradient import GradCAM
from Utils.face_regions import FacialRegionAnalyzer
from flask import Flask, request, jsonify, send_from_directory

# ✅ Initialize Flask App
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the base directory
MODEL_PATH = os.path.join(BASE_DIR, 'Model', 'best_b3_model_epoch6.pth')
FRAME_FOLDER = os.path.join(BASE_DIR, 'processed_frames')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
WEBSITE_FOLDER = os.path.join(BASE_DIR, 'Website')

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

# ✅ Load Deepfake Detection Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeDetector().to(device)
print(device)
gradcam = GradCAM(model, model.base_model.features[-1])  # Last convolutional layer

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("✅ Model successfully loaded!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ✅ Define Image Preprocessing for Inference
def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Ensure consistency
    ])

transform = get_inference_transforms()

# ✅ Global Variables to Store Data
latest_3rd_frame = None
facial_analysis_data = None

@app.route("/")
def home():
    return send_from_directory(WEBSITE_FOLDER, "index.html")


@app.route("/upload", methods=["POST"])
def upload_and_process():
    """Handles video upload, extracts frames, stores the 3rd frame, and predicts deepfake probability."""
    
    global latest_3rd_frame  # Use global variable for storing 3rd frame

    # ✅ Step 1: File Handling
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # ✅ Step 2: Extract Frames from Video
    video_name = os.path.splitext(filename)[0]
    frame_output_folder = os.path.join(FRAME_FOLDER, video_name)
    
    # 🚀 New Step: Clear old images before storing new frames
    if os.path.exists(frame_output_folder):
        for file in os.listdir(frame_output_folder):
            file_path = os.path.join(frame_output_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(frame_output_folder, exist_ok=True)

    extract_frames(filepath, frame_output_folder, frame_interval=3)

    # ✅ Step 3: Store 3rd Frame (If Exists)
    frame_list = sorted(os.listdir(frame_output_folder))
    if len(frame_list) >= 3:
        latest_3rd_frame = os.path.join(frame_output_folder, frame_list[2])  # Store 3rd frame path

    # ✅ Step 4: Process Faces
    processed_images = []
    for img_file in os.listdir(frame_output_folder):
        img_path = os.path.join(frame_output_folder, img_file)
        processed_img = zoom_into_face(img_path, frame_output_folder)
        if processed_img:
            processed_images.append(processed_img)

    if not processed_images:
        return jsonify({"error": "No faces detected in the video"}), 400

    # ✅ Step 5: Predict Deepfake Probability
    predictions = []
    with torch.no_grad():
        for img_file in tqdm(os.listdir(frame_output_folder), desc="🔍 Predicting Deepfakes"):
            img_path = os.path.join(frame_output_folder, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Convert to RGB & PIL Image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            # Apply Preprocessing
            image = transform(image).unsqueeze(0).to(device)
            print("🧐 Inference Image Mean:", image.mean().item())
            print("🧐 Inference Image Std:", image.std().item())

            # Model Inference
            output = model(image).squeeze().item()
            probability = float(torch.sigmoid(torch.tensor(output)).item())  # Apply Sigmoid Activation
            predictions.append(probability)

    # ✅ Step 6: Final Decision
    if not predictions:
        return jsonify({"error": "No valid frames were processed for prediction."}), 400

    avg_probability = np.mean(predictions)  # Compute Mean Probability
    is_fake = "FAKE" if avg_probability > 0.6 else "REAL"

    print(f"🎯 Prediction: {is_fake}, Score: {avg_probability:.4f}")

    return jsonify({
        "message": "✅ Prediction Complete!",
        "prediction": is_fake,
        "score": avg_probability
    })


@app.route("/get_3rd_frame", methods=["GET"])
def get_3rd_frame():
    """Returns the 3rd frame as a raw byte array in JSON response."""

    global latest_3rd_frame  # Use global variable for 3rd frame

    if not latest_3rd_frame or not os.path.exists(latest_3rd_frame):
        return jsonify({"error": "3rd frame not found or not available"}), 400

    try:
        # ✅ Read Image as Bytes
        with open(latest_3rd_frame, "rb") as img_file:
            img_bytes = img_file.read()

        # Clear the stored frame after sending to prevent interference
        latest_3rd_frame = None

        return jsonify({
            "message": "✅ 3rd Frame Retrieved",
            "image_bytes": list(img_bytes)  # Convert bytes to a list for JSON serialization
        })

    except Exception as e:
        return jsonify({"error": f"Failed to read frame: {str(e)}"}), 500

@app.route("/gradcam", methods=["GET"])
def get_gradcam():
    """Returns Grad-CAM heatmap of the 3rd frame with facial region analysis."""

    global latest_3rd_frame, facial_analysis_data
    if not latest_3rd_frame or not os.path.exists(latest_3rd_frame):
        return jsonify({"error": "3rd frame not found"}), 400

    # Load Image
    image = cv2.imread(latest_3rd_frame)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for PIL
    pil_image = Image.fromarray(image)

    # Apply Model Transformations
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Compute Grad-CAM
    heatmap = gradcam.generate_heatmap(input_tensor)
    overlay = gradcam.apply_heatmap(np.array(pil_image), heatmap)

    # Analyze facial regions
    region_scores, focused_regions = gradcam.analyze_facial_regions(np.array(pil_image), heatmap)
    facial_analysis_data = {
        "focused_regions": focused_regions
    }

    # Save Heatmap Image
    gradcam_path = os.path.join(FRAME_FOLDER, "gradcam_heatmap.jpg")
    cv2.imwrite(gradcam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    try:
        # ✅ Read Heatmap Image as Bytes
        with open(gradcam_path, "rb") as img_file:
            img_bytes = img_file.read()

        # Delete the heatmap file after sending the response
        os.remove(gradcam_path)

        return jsonify({
            "message": "✅ Grad-CAM heatmap generated!",
            "heatmap_bytes": list(img_bytes)
        })

    except Exception as e:
        return jsonify({"error": f"Failed to read heatmap: {str(e)}"}), 500

@app.route("/facial_analysis", methods=["GET"])
def get_facial_analysis():
    """Returns the facial region analysis from the latest Grad-CAM computation."""

    global facial_analysis_data, latest_3rd_frame

    if facial_analysis_data is None:
        return jsonify({"error": "No facial analysis data available"}), 400

    response = jsonify(facial_analysis_data)
    facial_analysis_data = None  # Clear the stored data after sending the response

    # Clean up uploaded video and processed frames
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

    return response

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)