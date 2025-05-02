import axios from "axios";
import { useState } from "react";

const Upload = ({ setLoading, setPrediction, fetchVisualAnalysis }) => {
  const [file, setFile] = useState(null);
  const BASE_URL = "http://localhost:5000"; // no space at start!

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
    formData.append("file", file); // ✅ match FastAPI parameter

    try {
      setLoading(true);
      const response = await axios.post(`${BASE_URL}/upload`, formData);
      console.log("Prediction response:", response.data);
      setPrediction(response.data);
      await fetchVisualAnalysis();
    } catch (error) {
      console.error("Upload error:", error);
      setPrediction({ message: "❌ Error occurred during prediction." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4 bg-white p-6 rounded-xl shadow-md">
      <h2 className="text-xl font-bold text-gray-800 text-center">🎬 Deepfake Video Detector</h2>

      <label className="block w-full cursor-pointer">
        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          className="hidden"
        />
        <div className="w-full text-center py-3 px-4 bg-indigo-100 text-indigo-800 rounded-lg hover:bg-indigo-200 transition">
          {file ? file.name : "📁 Click to upload a video file"}
        </div>
      </label>

      <button
        type="submit"
        className="w-full bg-indigo-600 text-white py-2 rounded-lg hover:bg-indigo-700 transition"
      >
        🚀 Submit Video
      </button>
    </form>
  );
};

export default Upload;
