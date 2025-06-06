import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Header from "./components/Header";
import Home from "./Home";
import VideoPage from "./components/VideoPage";
import AudioPage from "./components/AudioPage";
import TextPage from "./components/TextPage";
import ImagePage from "./components/ImagePage";

export default function App() {
  return (
    <div className="gradient-bg min-h-screen flex flex-col items-center">
      <Header />
      <Router>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/video" element={<VideoPage />} />
          <Route path="/audio" element={<AudioPage />} />
          <Route path="/text" element={<TextPage />} />
          <Route path="/image" element={<ImagePage />} />
        </Routes>
      </Router>
    </div>
  );
}