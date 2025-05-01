# Real-Time Deepfake Detection via Frame-Level EfficientNet  Ensemble and Client-Server Deployment

This project presents a real-time deepfake detection system that leverages frame-level analysis using an ensemble of **EfficientNet-B4 and B5 models**. The models are trained on large-scale datasets (Celeb-DF, FaceForensics++, and DeeperForensics) and achieve **AUC** of **0.9958** and **F1-score** of **0.9726**, rivaling state-of-the-art methods.


## Key Features :
**Frame-based Detection:** Avoids costly video-level models while maintaining high detection accuracy through dense frame sampling and face alignment.

**Model Ensemble:** Combines EfficientNet-B4 (coarse artifact detection) and B5 (fine-grained forgery detection with SE attention) for robust classification.

**Real-time Deployment:** Delivered through a scalable FastAPI backend with both web (React + TailwindCSS) and mobile (Android MVVM + Ktor) frontends.

**Explainable AI:** Grad-CAM heatmaps provide visual interpretability for predictions.

**Client-Server Architecture:** Optimized for low-latency inference and cross-platform usability.

**📱 Cross-Platform Interfaces:**

- Android App (Ktor + MVVM + Clean Architecture)

- Web App (React.js + TailwindCSS)

🔍**Grad-CAM Visualizations:** Enhances interpretability by highlighting manipulated regions in fake videos.

🚀 **Real-Time Performance:** Optimized for inference under 40ms per frame (25 FPS) on Tesla T4 GPUs.



## 🌐 Client-Server Architecture

**Frontend Clients:** Android and Web apps allow users to upload videos and view predictions.

**Backend Server:** Hosts EfficientNet models and handles preprocessing, inference, and Grad-CAM generation.

**API Communication:** FastAPI backend interacts with clients via JSON over REST endpoints.

**Data Format:** Multipart uploads and byte-array responses for efficient video and image processing

## 🚀 Project Structure

- [📱 Android App](./MainApp) - Mobile application for deepfake detection
- [🌐 Website](./Web/deepfake) - Website for deepfake detection
- [🧠 Training](./Training) - Model training code and documentation
- [🔍 Model](./Model) - Trained models and inference code
- [ Utils](./Utils) - All essential codes

## 📁Directory Structure :
```
Base Directory:
|- .gitignore
|- Abstract.txt
|- app.py
|- app_Fast.py
|- byte.py
|- LICENSE
|- tree.txt
|
|-📁MainApp
|                           
|-📁Model
|           
|-📁Training
|      
|-📁Utils
|            
|-📁Web
```
## 📦 Requirements
Install all dependencies using:
``` python
pip install -r requirements.txt
```

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
