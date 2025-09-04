// home page for all the routes where user can select the routes/section for uploading their content

import { Link } from "react-router-dom";

function Home() {
    return (
        <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-8">

            <h1 className="text-3xl font-bold text-gray-800 mb-10">
                🛡️ Deepfake Detection Portal
            </h1>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-4xl">

                {/* the section for upldoading Videos */}
                <Link
                    to="/video"
                    className="bg-white hover:bg-gray-100 transition p-8 rounded-2xl shadow-md text-center flex flex-col items-center"
                >
                    <span className="text-5xl mb-3">🎥</span>
                    <h2 className="text-xl font-semibold text-gray-700">Video Detection</h2>
                    <p className="text-gray-500 mt-2">Upload Videos For Deepfake Analysis</p>
                </Link>

                {/* the section for uploading Audios */}
                <Link
                    to="/audio"
                    className="bg-white hover:bg-gray-100 transition p-8 rounded-2xl shadow-md text-center flex flex-col items-center"
                >
                    <span className="text-5xl mb-3">🎵</span>
                    <h2 className="text-xl font-semibold text-gray-700">Audio Detection</h2>
                    <p className="text-gray-500 mt-2">Check Audio If Manipulated</p>
                </Link>

                {/* the section for uploading Images */}
                <Link
                    to="/image"
                    className="bg-white hover:bg-gray-100 transition p-8 rounded-2xl shadow-md text-center flex flex-col items-center"
                >
                    <span className="text-5xl mb-3">🖼️</span>
                    <h2 className="text-xl font-semibold text-gray-700">Image Detection</h2>
                    <p className="text-gray-500 mt-2">Analyze Images For Authenticity</p>
                </Link>

                {/*the section for uploading News */}
                <Link
                    to="/news"
                    className="bg-white hover:bg-gray-100 transition p-8 rounded-2xl shadow-md text-center flex flex-col items-center"
                >
                    <span className="text-5xl mb-3">📰</span>
                    <h2 className="text-xl font-semibold text-gray-700">News Detection</h2>
                    <p className="text-gray-500 mt-2">Verify Text/News Content</p>
                </Link>

            </div>

        </div>
    );
}

export default Home;

