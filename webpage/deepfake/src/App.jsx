// rendering all the routes

// importing required modules
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Header from "./components/Header";
import Home from "./components/Home";
import Video from "./components/Videos";
import Audio from "./components/Audios";
import Image from "./components/Images";
import News from "./components/News";

// app component
export default function App() {
    return (
        // routing
        <Router>
            <div className="min-h-screen bg-[#F3F6FB]">
                {/* header */}
                <Header />
                {/* navigation back and forward */}
                <nav className="max-w-6xl mx-auto px-4 py-3 flex gap-4 text-sm">
                    <Link to="/" className="text-[#0D47A1] hover:underline">Home</Link>
                    <Link to="/video" className="text-[#0D47A1] hover:underline">Video</Link>
                    <Link to="/audio" className="text-[#0D47A1] hover:underline">Audio</Link>
                    <Link to="/image" className="text-[#0D47A1] hover:underline">Image</Link>
                    <Link to="/news" className="text-[#0D47A1] hover:underline">News</Link>
                </nav>

                {/* content clickable area  */}
                <div className="max-w-6xl mx-auto px-4 pb-16">
                    <Routes>
                        <Route path="/" element={<Home />} />
                        <Route path="/video" element={<Video />} />
                        <Route path="/audio" element={<Audio />} />
                        <Route path="/image" element={<Image />} />
                        <Route path="/news" element={<News />} />
                    </Routes>
                </div>
            </div>
        </Router>
    );
}



