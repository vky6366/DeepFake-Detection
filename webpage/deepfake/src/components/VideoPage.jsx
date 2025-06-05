
import React, { useRef, useState } from "react";

export default function VideoPage() {
  const [fileName, setFileName] = useState("Drop your video here or click to browse");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const fileInputRef = useRef();

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setFileName(file ? file.name : "Drop your video here or click to browse");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const file = fileInputRef.current.files[0];
    if (!file) return alert("Please select a video before uploading.");
    setLoading(true);
    setResult(null);
    setHeatmapUrl(null);
    try {
      const formData = new FormData();
      formData.append("video", file);
      const response = await fetch("http://localhost:5000/upload", { method: "POST", body: formData });
      if (!response.ok) throw new Error("Video upload failed");
      const data = await response.json();
      setResult(data.prediction || "--");
      // Fetch heatmap/frame
      const endpoint = (data.prediction?.toLowerCase() === "fake") ? "/gradcam" : "/frame";
      const visualResp = await fetch(`${endpoint}?timestamp=${Date.now()}`);
      if (visualResp.ok) {
        const visualData = await visualResp.json();
        const bytes = visualData.heatmap_bytes || visualData.frame_bytes;
        if (bytes && bytes.length > 0) {
          const blob = new Blob([new Uint8Array(bytes)], { type: "image/jpeg" });
          setHeatmapUrl(URL.createObjectURL(blob));
        }
      }
    } catch {
      alert("Error processing the video. Please try again.");
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col items-center py-10 w-full">
      <section className="card p-8 w-full max-w-xl bg-blue-300">
        <h2 className="text-3xl font-semibold text-center text-[#0D47A1] mb-4">Upload a Video</h2>
        <form className="flex flex-col gap-6 items-center" onSubmit={handleSubmit}>
          <label className="cursor-pointer upload-box flex flex-col items-center gap-4 p-6 rounded-lg bg-white w-full">
            <input type="file" accept="video/*" required className="hidden" ref={fileInputRef} onChange={handleFileChange} />
            <span className="text-gray-700 font-medium text-center">{fileName}</span>
          </label>
          <button type="submit" className="w-full bg-[#0D47A1] text-white font-semibold px-8 py-3 rounded-lg hover:bg-[#08337a] transition-colors duration-300" disabled={loading}>
            Analyze Video
          </button>
        </form>
        {loading && (
          <div className="mt-6 text-center">
            <div className="flex items-center justify-center gap-3">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#0D47A1]"></div>
              <span className="text-[#0D47A1] font-medium">Processing video... Please wait.</span>
            </div>
          </div>
        )}
        {result && (
          <section className="card p-8 mt-6">
            <h2 className="text-3xl font-semibold text-center text-[#0D47A1]">Analysis Results</h2>
            <div className="text-center p-4 bg-[#E3F2FD] rounded-lg mt-4">
              <p className="text-2xl text-[#0D47A1]">
                <strong>Prediction:</strong> <span className="ml-2">{result}</span>
              </p>
            </div>
          </section>
        )}
        {heatmapUrl && (
          <section className="card p-8 mt-6">
            <div className="space-y-4 text-center">
              <h3 className="font-semibold text-[#0D47A1]">Heatmap Analysis</h3>
              <div className="relative aspect-video bg-[#E3F2FD] rounded-lg overflow-hidden">
                <img src={heatmapUrl} className="w-full h-full object-contain" alt="Grad-CAM Heatmap" />
              </div>
            </div>
          </section>
        )}
      </section>
    </div>
  );
}