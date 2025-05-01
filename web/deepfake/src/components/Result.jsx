// prediction result in scores
function Result({ prediction }) {
    if (!prediction) return null;

    return (
        <section className="card p-8">
            <h2 className="text-3xl font-semibold text-center text-[#0D47A1]">Analysis Results</h2>
            <div className="text-center p-4 bg-[#E3F2FD] rounded-lg mt-4">
                <p className="text-2xl text-[#0D47A1]">
                    <strong>Prediction:</strong>
                    <span className="ml-2">{prediction.prediction} (Score: {(prediction.score * 100).toFixed(2)}%)</span>
                </p>
            </div>
        </section>
    );
};

export default Result;

