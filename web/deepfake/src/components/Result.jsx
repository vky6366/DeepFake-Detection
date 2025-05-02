import React from 'react';

const Result = ({ response }) => {
  return (
    <div className="mt-4 p-4 bg-gray-100 rounded text-sm space-y-2">
      <p>📢 <strong>Message:</strong> {response.message}</p>
      {response.prediction && (
        <p>🧠 <strong>Prediction:</strong> {response.prediction}</p>
      )}
      {response.score !== undefined && (
        <p>🎯 <strong>Confidence Score:</strong> {response.score.toFixed(4)}</p>
      )}
    </div>
  );
};

export default Result;
