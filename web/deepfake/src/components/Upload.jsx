import React, { useState } from "react";
import axios from "axios";

// uploading the video file 
const Uploadform = ({ setLoading, setPrediction, fetchVisualAnalysis }) => {
  const [file, setFile] = useState(null);

  // selecting the video file
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type.startsWith("video/")) {
      setFile(selectedFile);
    } else {
      alert("Please select a video file");
      setFile(null);
    }
  };
};

// handling the form submission
const handleSubmit = async (e) => {
  e.preventDefault();

  if (!file) {
    alert("Please select a video file");
    return;
  }

  const formData = new FormData();
  formData.append("video", file);

  // set loading  for processing the video file
  try {
    setLoading(true);
    const response = await axios.post('/upload', formData)
    setPrediction(response.data);
    await fetchVisualAnalysis();
    alert("File uploaded successfully");
  } catch (error) {
    console.error("Error on  uploading file:", error);
    alert("Error on  uploading file. Please try again.");

  } finally {
    setLoading(false);
  }
  // methode
};

return (
  // form for thr input of file 
  <form onSunmit={handleSubmit} className="flex flex-col gap-6 items-center mt-6">
    <label
      htmlFor="videoInput"
      className="cursor-pointer upload-box flex flex-col items-center gap-4 p-6 rounded-lg transition-all duration-300 bg-white border-2 border-dashed border-[#0D47A1]"
    >
      <input
        type="file"
        id="videoInput"
        name="video"
        accept="video/*"
        required
        cladsName="hidden"
        onChange={handleFileChange}
      // handle the file uploading
      />

      {/* if file is npt selected correctly */}
      <span className="text-gray-700 font-medium">
        {file ? file.name : "No file is selected"}
      </span>
    </label>

    {/* submission button  */}
    <button
      type="submit"
      type="submit"
      className="bg-[#0D47A1] text-white font-semibold px-8 py-3 rounded-lg hover:bg-[#08337a] transition-colors duration-300"
    >  analyisis the video</button>
  </form>
);

export default Uploadform;
