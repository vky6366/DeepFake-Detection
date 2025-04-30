import React, { useState } from 'react';
import Header from './components/Header';
import Upload from './components/Upload';
import Loading from './components/Loading';
import Result from './components/Result';
import VisualAnalysis from './components/VisualAnalysis';
import axios from './server/api';
// import axios from 'axios';

function App  () {
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [frameImg, setFrameImg] = useState(null);
  const [gradcamImg, setGradcamImg] = useState(null);

  const fetchVisualAnalysis = async () => {
    try {
      const frameResponse = await axios.get('/get_3rd_frame');
      const heatmapResponse = await axios.get(`/gradcam?timestamp=${new Date().getTime()}`);

      console.log(url);
      const frameBlob = new Blob([new Uint8Array(frameResponse.data.image_bytes)], { type: 'image/jpeg' });
      setFrameImg(URL.createObjectURL(frameBlob));

      const heatmapBlob = new Blob([new Uint8Array(heatmapResponse.data.heatmap_bytes)], { type: 'image/jpeg' });
      setGradcamImg(URL.createObjectURL(heatmapBlob));
    } catch (error) {
      console.error('Error fetching visual analysis:', error);
    }
  };


return (
  <div className="gradient-bg min-h-screen flex flex-col items-center">
    <Header />
    <main className="container mx-auto px-6 py-12 flex flex-col gap-8 w-full max-w-3xl">
      <Upload setPrediction={setPrediction} setLoading={setLoading} fetchVisualAnalysis={fetchVisualAnalysis} />
      {loading && <Loading/>}
      {prediction && <Result prediction={prediction} />}
      <VisualAnalysis frameImg={frameImg} gradcamImg={gradcamImg} />
    </main>
  </div>
);
};

export default App;