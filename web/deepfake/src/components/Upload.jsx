
import React, { useState } from "react";
import axios from "../server/api";

// uploading the video file
const Upload = ({ setLoading, setPrediction, fetchVisualAnalysis }) => {
    const [file, setFile] = useState(null);
    const [result, setResult] = useState(null); // State to store the result from the backend

    //   selecting the video file
    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile && selectedFile.type.startsWith("video/")) {
            setFile(selectedFile);
        } else {
            alert("Please select a valid video file");
            setFile(null);
        }
    };

    //   handling the form submission
    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!file) {
            alert("Please select a video file");
            return;
        }

        const formData = new FormData();
        formData.append("video", file);

        // set loading while processing the video file
        try {
            setLoading(true);
            const response = await axios.post("/upload", formData);
            const { message, prediction, score } = response.data;

            setPrediction(response.data);
            setResult({ message, prediction, score }); 
            await fetchVisualAnalysis();
        } catch (error) {
            console.error("Error uploading file:", error);
            alert("Error uploading file. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="upload-container">
            {/* form for the input of the file */}
            <form onSubmit={handleSubmit} className="flex flex-col gap-6 items-center mt-6">
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
                        className="hidden"
                        onChange={handleFileChange}
                    />

                    {/* if the file is not selected correctly */}
                    <span className="text-gray-700 font-medium">
                        {file ? file.name : " Drag and drop your video here"}
                    </span>
                </label>
                {/* submission button */}
                <button
                    type="submit"
                    className="bg-[#0D47A1] text-white font-semibold px-8 py-3 rounded-lg hover:bg-[#08337a] transition-colors duration-300"
                >
                    Analyze the video
                </button>
            </form>

            {/* Display the result if available */}
            {result && (
                <div className="result-box mt-6 p-4 border rounded-lg bg-gray-100">
                    <p className="font-bold text-gray-800">{result.message}</p>
                    <p className="text-gray-700">Prediction: {result.prediction ? "Fake" : "Real"}</p>
                    <p className="text-gray-700">Score: {result.score.toFixed(2)}</p>
                </div>
            )}
        </div>
    );
};

export default Upload;