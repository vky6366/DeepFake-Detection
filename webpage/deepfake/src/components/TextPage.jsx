

import React, { useRef, useState } from 'react';

async function analyzeTextAPI(text) {
    // This should call your backend but here is a mock for demonstration
    return new Promise((resolve) =>
        setTimeout(() =>
            resolve({
                "claim": "The nasa has found alien",
                "result": "Fake",
                "similarity_score": 0.61,
                "sources": [
                    {
                        "title": "Scientists find promising hints of life on distant planet K2-18b",
                        "url": "https://www.bbc.com/news/articles/c39jj9vkr34o"
                    }
                ]
            }), 1000)
    );
}

export default function TextPage() {
    const textInputRef = useRef();
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setResult(null);
        const text = textInputRef.current.value.trim();
        try {
            const analysis = await analyzeTextAPI(text);
            setResult(analysis);
        } catch (err) {
            alert("An error occurred while analyzing the text. Please try again.");
        }
        setLoading(false);
        textInputRef.current.value = '';
    };

    return (
        <div className="flex flex-col items-center py-10 w-full">
            <section className="card p-8 w-full max-w-xl bg-yellow-300">
                <h2 className="text-3xl font-semibold text-center text-blue-900 mb-4">Enter a Text</h2>
                <form className="flex flex-col gap-6 items-center" onSubmit={handleSubmit}>
                    <label className="w-full">
                        <textarea
                            ref={textInputRef}
                            name="text"
                            rows={2}
                            placeholder="Write your text or Paste your text here"
                            required
                            className="w-full p-4 rounded-lg border-2 border-[#0D47A1] text-gray-700 font-medium resize-y upload-box"
                        />
                    </label>
                    <button
                        type="submit"
                        className="w-full bg-[#0D47A1] text-white font-semibold px-8 py-3 rounded-lg hover:bg-blue-800 transition-colors duration-300"
                        disabled={loading}
                    >
                        Analyze Text
                    </button>
                </form>
                {loading && (
                    <div className="mt-6 text-center">
                        <div className="flex items-center justify-center gap-3">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#0D47A1]"></div>
                            <span className="text-[#0D47A1] font-medium">Processing text... Please wait.</span>
                        </div>
                    </div>
                )}
                {result && (
                    <section className="card p-8 mt-6 bg-white rounded-lg shadow">
                        <h2 className="text-3xl font-semibold text-center text-blue-900">Analysis Results</h2>
                        <div className="text-center p-4 bg-blue-100 rounded-lg mt-4">
                            <p className="text-2xl text-blue-900">
                                <strong>Prediction:</strong> <span className="ml-2">{result.result || '--'}</span>
                            </p>
                            <div className="mt-2 text-lg text-blue-900">
                                {typeof result.similarity_score === "number" ?
                                    `Similarity Score: ${(result.similarity_score * 100).toFixed(1)}%` : null}
                            </div>
                            <div className="mt-2 text-base text-blue-900">
                                {result.claim ? `Claim: ${result.claim}` : ''}
                            </div>
                        </div>
                        <div className="card p-4 sm:p-8">
                            <div>
                                <ul className="mt-8 list-disc list-inside space-y-2">
                                    {result.sources && result.sources.map((src, idx) => (
                                        <li key={idx}>
                                            <span>{src.title} </span>
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
                        </div>
                    </section>
                )}
            </section>
        </div>
    );
}

