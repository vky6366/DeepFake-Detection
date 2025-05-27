import React from 'react'

const Result = ({ data }) => {
  return (
    <>
      <div className="card p-8">
        <h2 className="text-3xl font-semibold text-center text-[#0D47A1]"> Prediction Results</h2>
        <div className="text-center p-4 bg-[#E3F2FD] rounded-lg mt-4">
          <p className="text-2xl text-[#0D47A1]">
            {/* <strong>Prediction:</strong> <span className="ml-2">{prediction}</span> */}
            <p><strong>Prediction:</strong> {data.prediction}</p>
            <p><strong>Score:</strong> {data.score}</p>
            <p><strong>Message:</strong> {data.message}</p>
          </p>
        </div>
      </div>
    </>
  )
}

export default Result;