
import React, { useRef, useState } from 'react';

export default function ImagePage() {
    const [fileName, setFileName] = useState("Drop your image here or click to browse");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const fileInputRef = useRef();

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        setFileName(file ? file.name : "Drop your image here or click to browse");
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const file = fileInputRef.current.files[0];
        if (!file) {
            alert("Please select an image before uploading.");
            return;
        }
        setLoading(true);
        setResult(null);
        try {
            const formData = new FormData();
            formData.append('image', file);
            const response = await fetch('http://localhost:5000/upload_image', { method: 'POST', body: formData });
            if (!response.ok) throw new Error("Image upload failed");
            const data = await response.json();
            setResult({
                result: data.result || '--',
                confidence: typeof data.confidence === 'number'
                    ? `Confidence: ${(data.confidence * 100).toFixed(1)}%` : '',
                message: data.message || ''
            });
        } catch (err) {
            alert("Error processing the image. Please try again.");
        }
        setLoading(false);
    };

    return (
        <div className="flex flex-col items-center py-10 w-full">
            <section className="card p-8 w-full max-w-xl bg-pink-300">
                <h2 className="text-3xl font-semibold text-center text-[#0D47A1] mb-4">Upload an Image</h2>
                <form className="flex flex-col gap-6 items-center" onSubmit={handleSubmit}>
                    <label className="cursor-pointer upload-box flex flex-col items-center gap-4 p-6 rounded-lg bg-white w-full">
                        <input
                            type="file"
                            accept="image/*"
                            required
                            className="hidden"
                            ref={fileInputRef}
                            onChange={handleFileChange}
                        />
                        <span className="text-gray-700 font-medium text-center">{fileName}</span>
                    </label>
                    <button type="submit"
                        className="w-full bg-[#0D47A1] text-white font-semibold px-8 py-3 rounded-lg hover:bg-[#08337a] transition-colors duration-300"
                        disabled={loading}
                    >
                        Analyze Image
                    </button>
                </form>
                {loading && (
                    <div className="mt-6 text-center">
                        <div className="flex items-center justify-center gap-3">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#0D47A1]"></div>
                            <span className="text-[#0D47A1] font-medium">Processing image... Please wait.</span>
                        </div>
                    </div>
                )}
                {result && (
                    <section className="card p-8 mt-6">
                        <h2 className="text-3xl font-semibold text-center text-[#0D47A1]">Analysis Results</h2>
                        <div className="text-center p-4 bg-[#E3F2FD] rounded-lg mt-4">
                            <p className="text-2xl text-[#0D47A1]">
                                <strong>Prediction:</strong> <span className="ml-2">{result.result}</span>
                            </p>
                            <div className="mt-2 text-lg text-[#0D47A1]">{result.confidence}</div>
                            <div className="mt-2 text-base text-[#0D47A1]">{result.message}</div>
                        </div>
                    </section>
                )}
            </section>
        </div>
    );
}
