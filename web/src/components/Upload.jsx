import React, { useState } from 'react';

const Upload = ({ onUpload }) => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('Drop your video here or click to browse');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type.startsWith('video/')) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
    } else {
      alert("Please select a valid video file.");
      e.target.value = '';
      setFile(null);
      setFileName('Drop your video here or click to browse');
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (file) {
      onUpload(file);
    }
  };

  return (
    <div className="card p-8">
      <h2 className="text-3xl font-semibold text-center text-[#0D47A1]">Upload a Video</h2>
      <form onSubmit={handleSubmit} className="flex flex-col gap-6 items-center mt-6">
        <label htmlFor="video" className="cursor-pointer upload-box flex flex-col items-center gap-4 p-6 rounded-lg transition-all duration-300 bg-white">
          <input
            type="file"
            id="video"
            name="video"
            accept="video/*"
            required
            className="hidden"
            onChange={handleFileChange}
          />
          <span className="text-gray-700 font-medium">{fileName}</span>
        </label>
        <button type="submit" className="bg-[#0D47A1] text-white font-semibold px-8 py-3 rounded-lg hover:bg-[#08337a] transition-colors duration-300">
          Predict
        </button>
      </form>
    </div>
  );
};

export default Upload;
