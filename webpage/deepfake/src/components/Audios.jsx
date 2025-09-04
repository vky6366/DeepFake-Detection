// the audio route where user can upload audios/mp3 files for detection

// importing required modules
import { useRef, useState } from "react";
import { uploadFile } from "../api";

// function call for uploading
export default function Audio() {

    // the browser except file form only
    const [fileName, setFileName] = useState("Drop your Audio file here or click to browse");

    //   loading bar till the execution get over and json returns
    const [loading, setLoading] = useState(false);

    // returns message, prediction of the detection
    const [prediction, setPrediction] = useState(null);
    const fileRef = useRef();

    //   handling the file selection from the device or browser
    // only files of type mp3 are allowed 
    const handleFileChange = (e) => {
        const f = e.target.files[0];
        setFileName(f ? f.name : "Drop your Audio file here or click to browse");
    };

    //  handling the  audio uploading  and submission  of the audio
    // alert if no audio is selected
    const handleSubmit = async (e) => {
        e.preventDefault();
        const file = fileRef.current?.files?.[0];
        if (!file) return alert("Please select an audio file before uploading.");

        // resetting the state
        // loading till the json return, result is null
        setLoading(true);
        setPrediction(null);

        // the form for uploading the video to the backend
        try {
            // backend expects 'file'
            const formData = new FormData();
            formData.append("file", file);

            //   calling the api
            const res = await uploadFile("/upload_audio", formData);
            setPrediction(res.data?.prediction || "--");

            //   catching the error
        } catch (err) {
            console.error("Error uploading audio:", err);
            alert("Error processing the audio. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    //   rendering the webpage,displaying the result on the browser
    return (
        <div className="flex flex-col items-center py-10 w-full">

            {/* the section for uploading audios */}
            <section className="card p-8 w-full max-w-xl bg-blue-300">

                <h2 className="text-3xl font-semibold text-center text-[#0D47A1] mb-4">Upload an Audio</h2>

                {/* form for the file */}
                <form className="flex flex-col gap-6 items-center" onSubmit={handleSubmit}>

                    <label className="cursor-pointer upload-box flex flex-col items-center gap-4 p-6 rounded-lg bg-white w-full">
                        <input type="file" accept="audio/*" required className="hidden" ref={fileRef} onChange={handleFileChange} />
                        <span className="text-gray-700 font-medium text-center">{fileName}</span>
                    </label>

                    {/* submission button for file submission */}
                    <button
                        type="submit"
                        className="w-full bg-[#0D47A1] text-white font-semibold px-8 py-3 rounded-lg hover:bg-[#08337a] transition-colors duration-300"
                        disabled={loading}
                    >
                        Analyze Audio
                    </button>

                </form>

                {/* loading bar till the json returns/result  */}
                {loading && (
                    <div className="mt-6 text-center">
                        <div className="flex items-center justify-center gap-3">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#0D47A1]"></div>
                            <span className="text-[#0D47A1] font-medium">Processing audio... Please wait.</span>
                        </div>
                    </div>
                )}

                {/* the section for displaying the result */}
                {/* if the result is not null */}
                {/* the result is the message, prediction, score */}
                {prediction && (
                    <section className="card p-8 mt-6">
                        <h2 className="text-3xl font-semibold text-center text-[#0D47A1]">Analysis Results</h2>
                        <div className="text-center p-4 bg-[#E3F2FD] rounded-lg mt-4">

                            <p className="text-2xl text-[#0D47A1]">
                                <strong>Prediction:</strong> <span className="ml-2">{prediction}</span>
                            </p>

                        </div>
                    </section>
                )}

            </section>
        </div>
    );
}


