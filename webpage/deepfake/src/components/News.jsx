// the news route where user can upload news/text files for detection

// importing required modules
import { useState } from "react";
import api from "../api";

// function call for uploading
export default function News() {
    // the browser except text form only
    const [claim, setClaim] = useState("");

    //   loading bar till the execution get over and json returns
    const [loading, setLoading] = useState(false);

    // result returns claim, result, similarity_score, sources[] 
    const [result, setResult] = useState(null);


    //   handling the file selection from the device or browser
    // only files of type text are allowed 
    const handleCheck = async (e) => {
        e.preventDefault();
        if (!claim.trim()) return;

        // resetting the state
        // loading till the json return, result is null
        setLoading(true);
        setResult(null);

        // the form for uploading the image to the backend
        try {
            // return the claim after checking
            // claim is a string

            //   calling the api
            const res = await api.get("/fact-check", { params: { claim } });
            setResult(res.data);

            //   catching the error
        } catch (err) {
            console.error("Error checking news:", err);
            alert("Error while checking the claim. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    //   rendering the webpage,displaying the result on the browser
    return (
        <div className="flex flex-col items-center py-10 w-full">

            {/* the section for uploading news */}
            <section className="card p-8 w-full max-w-xl bg-blue-300">
                <h2 className="text-3xl font-semibold text-center text-[#0D47A1] mb-4">Fact-Check a Claim</h2>

                {/* form for the file */}
                <form className="flex flex-col gap-6 items-center w-full" onSubmit={handleCheck}>

                    <textarea
                        rows={2}
                        placeholder="Enter a claim to fact-check"
                        value={claim}
                        onChange={(e) => setClaim(e.target.value)}
                        className="w-full p-4 rounded-lg border-2 border-[#0D47A1] text-gray-700 font-medium resize-y upload-box"
                    />

                    {/* submission button for file submission */}
                    <button
                        type="submit"
                        className="w-full bg-[#0D47A1] text-white font-semibold px-8 py-3 rounded-lg hover:bg-[#08337a] transition-colors duration-300"
                        disabled={loading}
                    >
                        {loading ? "Checking..." : "Check"}
                    </button>

                </form>

                {/* loading bar till the json returns/result  */}
                {loading && (
                    <div className="mt-6 text-center">
                        <div className="flex items-center justify-center gap-3">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#0D47A1]"></div>
                            <span className="text-[#0D47A1] font-medium">Checking claim... Please wait.</span>
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

                            {/* the prediction */}
                            <p className="text-2xl text-[#0D47A1]">
                                <strong>Prediction:</strong> <span className="ml-2">{result.result || "--"}</span>
                            </p>

                            {/* the similarity score */}
                            {typeof result.similarity_score === "number" && (
                                <div className="mt-2 text-lg text-[#0D47A1]">
                                    Similarity Score: {(result.similarity_score * 100).toFixed(1)}%
                                </div>
                            )}

                            {/* the claim true or false */}
                            {result.claim && (
                                <div className="mt-2 text-base text-[#0D47A1]">Claim: {result.claim}</div>
                            )}

                        </div>

                        {/* the sources from where the claim is taken wheather the claim is true or false */}
                        {/* it ia type of recommendation where it returns the sources */}
                        {/* if the sources is an array */}
                        {Array.isArray(result.sources) && result.sources.length > 0 && (
                            <div className="card p-4 sm:p-8">
                                <ul className="mt-4 list-disc list-inside space-y-2">
                                    {result.sources.map((src, idx) => (
                                        <li key={idx}>
                                            <span className="font-medium">{src.title} </span>
                                            <a
                                                href={src.url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="text-blue-800 underline hover:text-blue-600 break-all"
                                            >
                                                {src.url}
                                            </a>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}

                    </section>
                )}

            </section>
        </div>
    );
}
