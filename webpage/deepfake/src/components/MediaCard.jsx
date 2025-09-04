// media card for navigation from route to route 

import { useNavigate } from "react-router-dom";

export default function MediaCard({ title, path, color, icon }) {
    const navigate = useNavigate();
    return (
        <div
            onClick={() => navigate(path)}
            className={`card ${color} cursor-pointer flex flex-col items-center justify-center w-60 h-60 text-3xl font-bold text-white shadow-lg transform hover:scale-105 transition duration-200`}
        >
            <span className="text-6xl mb-4">{icon}</span>
            {title}
        </div>
    );
}