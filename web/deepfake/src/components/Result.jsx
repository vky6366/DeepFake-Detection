import React, { useState } from "react";
import axios from "axios";


const BASE_URL = "http://127.0.0.1:5000"; // Fix typo in 127.0.0:5000

function Display() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [frameImg, setFrameImg] = useState(null);
  const [gradcamImg, setGradcamImg] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type.startsWith("video/")) {
      setFile(selectedFile);
    } else {
      alert("Please select a valid video file.");
      setFile(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert("Please select a video file.");
      return;
    }

    const formData = new FormData();
    formData.append("video", file);

    try {
      setLoading(true);
      const response = await axios.post(`${BASE_URL}/upload`, formData);
      const data = response.data;

      setPrediction(data);
      fetch3rdFrame();
      fetchGradCAM();
    } catch (error) {
      console.error("Upload error:", error);
      setPrediction({ message: "❌ Error occurred during prediction." });
    } finally {
      setLoading(false);
    }
  };

  // Fetch frame image from backend
  const fetch3rdFrame = async () => {
    try {
      const response = await fetch(`${BASE_URL}/get_3rd_frame`);
      const data = await response.json();

      const blob = new Blob([new Uint8Array(data.image_bytes)], { type: "image/jpeg" });
      setFrameImg(URL.createObjectURL(blob));
    } catch (error) {
      console.error("Failed to load 3rd frame:", error);
    }
  };

  // Fetch Grad-CAM image from backend
  const fetchGradCAM = async () => {
    try {
      const response = await fetch(`${BASE_URL}/gradcam?timestamp=${new Date().getTime()}`);
      const data = await response.json();

      const blob = new Blob([new Uint8Array(data.heatmap_bytes)], { type: "image/jpeg" });
      setGradcamImg(URL.createObjectURL(blob));
    } catch (error) {
      console.error("Failed to load Grad-CAM:", error);
    }
  };

  return (
    <div className="p-4">
      <Header />
      <Upload onChange={handleFileChange} onSubmit={handleSubmit} />
      {loading && <Loading />}
      {prediction && <Result response={prediction} />}
      <VisualAnalysis frameImg={frameImg} gradcamImg={gradcamImg} />
    </div>
  );
}

export default Display;
