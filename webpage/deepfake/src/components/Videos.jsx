// the video route where user can upload videos/mp4 files for detection

// importing required modules
import { useEffect, useRef, useState } from "react";
import { uploadFile } from "../api";

// function call for uploading
export default function Video() {

    // the browser except file form only
    const [fileName, setFileName] = useState("Drop your video file here or click to browse");

    //   loading bar till the execution get over and json returns
    const [loading, setLoading] = useState(false);

    // returns message, prediction, score  of the detection
    const [result, setResult] = useState(null);

    // ObjectURL for heatmap/frame
    const [visualUrl, setVisualUrl] = useState(null);
    const fileRef = useRef();

    //   react hooks is used for rendering
    //   revoking the ObjectURL
    useEffect(() => {
        return () => {
            if (visualUrl) URL.revokeObjectURL(visualUrl);
        };
    }, [visualUrl]);

    //   handling the file selection from the device or browser
    // only files of type mp4 are allowed 
    const handleFileChange = (e) => {
        const f = e.target.files[0];
        setFileName(f ? f.name : "Drop your video file here or click to browse");
    };

    //   display the prediction result on the webpage
    const fetchVisual = async (prediction) => {

        // Backend: /gradcam returns { heatmap_bytes: [..] }
        //          /3rd_frame returns { image_bytes: [..] }

        // if prediction is fake, fetch gradcam, else fetch 3rd_frame
        const endpoint = prediction?.toLowerCase() === "fake" ? "/gradcam" : "/3rd_frame";

        // redering the result through api from backend model 
        try {
            const res = await fetch(`http://localhost:5000${endpoint}?t=${Date.now()}`);

            //   checking the response
            if (!res.ok) return;
            const data = await resp.json();
            //   data in return in the form of bytes array
            const bytes = data.heatmap_bytes || data.image_bytes || data.frame_bytes;

            //   the lenght if bytes array is greater than 0 or not
            if (bytes && Array.isArray(bytes) && bytes.length > 0) {

                // creating the ObjectURL
                // type of image is jpeg
                const blob = new Blob([new Uint8Array(bytes)], { type: "image/jpeg" });
                const url = URL.createObjectURL(blob);

                setVisualUrl((prev) => {
                    if (prev) URL.revokeObjectURL(prev);
                    return url;
                });
            }
            //   catching the error
        } catch (err) {
            console.error("Failed to fetch visual:", err);
        }
    };

    //  handling the  video uploading  and submission  of the video 
    // alert if no video is selected
    const handleSubmit = async (e) => {
        e.preventDefault();
        const file = fileRef.current?.files?.[0];
        if (!file) return alert("Please select a video before uploading.");

        // resetting the state
        // loading till the json return, result is null
        setLoading(true);
        setResult(null);
        setVisualUrl((prev) => {
            if (prev) URL.revokeObjectURL(prev);
            return null;
        });

        // the formfor uploading the video to the backend
        try {
            const formData = new FormData();
            formData.append("video", file);

            //   calling the api
            const res = await uploadFile("/upload", formData);

            // result returns   message, prediction, score 
            setResult(res.data);
            await fetchVisual(res.data?.prediction);

            //   catching the error
        } catch (err) {
            console.error("Error uploading video:", err);
            alert("Error processing the video. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    //   rendering the webpage,displaying the result on the browser
    return (
        <div className="flex flex-col items-center py-10 w-full">

            {/* the section for uploading video */}
            <section className="card p-8 w-full max-w-xl bg-blue-300">

                <h2 className="text-3xl font-semibold text-center text-[#0D47A1] mb-4">Upload a Video</h2>

                {/* form for the file */}
                <form className="flex flex-col gap-6 items-center" onSubmit={handleSubmit}>

                    <label className="cursor-pointer upload-box flex flex-col items-center gap-4 p-6 rounded-lg bg-white w-full">
                        <input type="file" accept="video/*" required className="hidden" ref={fileRef} onChange={handleFileChange} />
                        <span className="text-gray-700 font-medium text-center">{fileName}</span>
                    </label>

                    {/* submission button for file submission */}
                    <button
                        type="submit"
                        className="w-full bg-[#0D47A1] text-white font-semibold px-8 py-3 rounded-lg hover:bg-[#08337a] transition-colors duration-300"
                        disabled={loading}
                    >
                        Analyze Video
                    </button>

                </form>

                {/* loading bar till the json returns/result  */}
                {loading && (
                    <div className="mt-6 text-center">
                        <div className="flex items-center justify-center gap-3">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#0D47A1]"></div>
                            <span className="text-[#0D47A1] font-medium">Processing video... Please wait.</span>
                        </div>
                    </div>
                )}

                {/* the section for displaying the result */}
                {/* if the result is not null */}
                {/* the result is the message, prediction, score */}
                {result && (
                    <section className="card p-8 mt-6">

                        <h2 className="text-3xl font-semibold text-center text-[#0D47A1]">Analysis Results</h2>

                        <div className="text-center p-4 bg-[#E3F2FD] rounded-lg mt-4">

                            <p className="text-2xl text-[#0D47A1]">
                                <strong>Prediction:</strong> <span className="ml-2">{result.prediction || "--"}</span>
                            </p>

                            {typeof result.score === "number" && (
                                <p className="text-lg text-[#0D47A1] mt-2">
                                    <strong>Score:</strong> {(result.score * 100).toFixed(1)}%
                                </p>
                            )}

                        </div>
                    </section>
                )}

                {/* the section for displaying the visualization */}
                {/* if the visualUrl is not null */}
                {/* the visualUrl is the grad-cam heatmap */}
                {visualUrl && (
                    <section className="card p-8 mt-6">

                        <div className="space-y-4 text-center">

                            <h3 className="font-semibold text-[#0D47A1]">
                                {result?.prediction?.toLowerCase() === "fake" ? "Grad-CAM Heatmap" : "Extracted Frame"}
                            </h3>
                            <div className="relative aspect-video bg-[#E3F2FD] rounded-lg overflow-hidden">
                                <img src={visualUrl} className="w-full h-full object-contain" alt="Visualization" />
                            </div>

                        </div>
                    </section>

                )}

            </section>
        </div>
    );
}


