import React, { useState } from 'react';
import Api from '../server/api';
import Header from './Header';
import Upload from './Upload';
import Loading from './Loading';
import Result from './Result';
import VisualAnalysis from './VisualAnalysis';

const Display = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [imageURL, setImageURL] = useState(null);

  const handleUpload = async (videoFile) => {
    if (!videoFile) return;

    const formData = new FormData();
    formData.append('video', videoFile);

    try {
      setLoading(true);
      const response = await Api.post('/upload', formData);
      const resData = response.data;

      setData(resData);
      if (resData.prediction.toLowerCase() === 'fake') {
        await fetchVisual('gradcam');
      } else {
        setImageURL(null);
      }
    } catch (error) {
      alert('Error processing the video. Please try again.');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const fetchVisual = async (type) => {
    try {
      const res = await Api.get(`/${type}`, { responseType: 'arraybuffer' });
      const blob = new Blob([res.data], { type: 'image/jpeg' });
      setImageURL(URL.createObjectURL(blob));
    } catch (error) {
      console.warn('Error fetching visual analysis image:', error);
    }
  };

  return (
    <div className="gradient-bg min-h-screen flex flex-col items-center">
      <Header />
      <main className="container mx-auto px-6 py-12 flex flex-col gap-8 w-full max-w-3xl">
        <Upload onUpload={handleUpload} />
        {loading && <Loading />}
        {data && <Result data={data} />}
        {imageURL && <VisualAnalysis imageURL={imageURL} />}
      </main>
    </div>
  );
};

export default Display;
