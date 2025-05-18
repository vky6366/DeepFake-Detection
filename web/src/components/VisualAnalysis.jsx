import React from 'react'

const VisualAnlysis = () => {
  return (
    <>
      <div className="card p-8">
        <div className="space-y-4 text-center">
          <h3 className="font-semibold text-[#0D47A1]">Heatmap Analysis</h3>
          <div className="relative aspect-video bg-[#E3F2FD] rounded-lg overflow-hidden">
            {imageURL ? (
              <img src={imageURL} alt="Grad-CAM Heatmap" className="w-full h-full object-contain" />
            ) : (
              <div className="text-gray-500 py-20">No image to display</div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}

export default VisualAnlysis;