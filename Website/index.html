<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deepfake Detector</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        :root {
            --primary: #6366f1;
            --secondary: #4f46e5;
            --accent: #f43f5e;
        }
        
        .custom-gradient {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }

        .glass-effect {
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(148, 163, 184, 0.1);
        }

        .result-transition {
            transition: all 0.5s ease-in-out;
            transform-origin: top;
        }

        .result-hidden {
            transform: scaleY(0);
            height: 0;
            opacity: 0;
        }

        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            h1 {
                font-size: 1.875rem !important;
            }
            h2 {
                font-size: 1.5rem !important;
            }
        }

        @media (max-width: 480px) {
            .container {
                padding: 0.5rem;
            }
            .upload-container {
                padding: 1rem !important;
            }
        }
    </style>
</head>

<body class="custom-gradient min-h-screen text-gray-100 pb-20">
    <!-- Header with new design -->
    <div class="text-center py-16 glass-effect border-b border-indigo-500/10">
        <h1 class="text-6xl font-bold tracking-tight">
            <span class="bg-clip-text text-transparent bg-gradient-to-r from-indigo-500 to-pink-500">DEEP</span>
            <span class="bg-clip-text text-transparent bg-gradient-to-r from-pink-500 to-indigo-500">SHIELD</span>
        </h1>
        <p class="text-indigo-200/60 mt-4 text-xl font-light">Advanced Deepfake Detection System</p>
    </div>

    <div class="container mx-auto px-4 py-12 max-w-6xl">
        <!-- Upload section with improved layout -->
        <div class="max-w-2xl mx-auto glass-effect rounded-3xl p-10 shadow-2xl mb-12">
            <h2 class="text-3xl font-semibold mb-10 text-center bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-pink-400">
                Upload a Video
            </h2>

            <form id="uploadForm" enctype="multipart/form-data" class="flex flex-col gap-8 items-center">
                <div class="w-full">
                    <label for="videoInput"
                        class="cursor-pointer flex flex-col items-center gap-6 p-10 border-2 border-dashed border-indigo-500/30 rounded-2xl hover:border-pink-500/50 transition-all duration-300 bg-indigo-950/20">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-16 w-16 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                        <input type="file" id="videoInput" name="video" accept="video/*" required class="hidden">
                        <span id="fileName" class="text-indigo-200/60 text-lg text-center">Drop your video here or click to browse</span>
                    </label>
                </div>
                <button type="submit"
                    class="bg-gradient-to-r from-indigo-600 to-pink-600 text-white font-medium px-12 py-4 rounded-xl hover:from-indigo-500 hover:to-pink-500 transition-all duration-300 flex items-center gap-3 transform hover:scale-105 hover:shadow-xl hover:shadow-indigo-500/20">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                    </svg>
                    Analyze Video
                </button>
            </form>

            <!-- Loading indicator with new design -->
            <div id="loading" class="hidden mt-8 text-center">
                <div class="flex items-center justify-center gap-4">
                    <div class="animate-spin rounded-full h-10 w-10 border-b-2 border-t-2 border-indigo-500"></div>
                    <span class="text-indigo-400 font-medium text-lg">Analyzing video...</span>
                </div>
            </div>
        </div>

        <!-- Results section with transition -->
        <div id="results" class="result-transition result-hidden max-w-2xl mx-auto glass-effect rounded-3xl p-10 shadow-2xl mb-12">
            <h2 class="text-3xl font-semibold mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-pink-400">
                Analysis Results
            </h2>
            <div class="text-center p-8 bg-indigo-950/30 rounded-2xl border border-indigo-500/10">
                <p class="text-4xl">
                    <strong class="text-indigo-200">Prediction:</strong> 
                    <span id="prediction" class="ml-3 font-light bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-pink-400">--</span>
                </p>
            </div>
        </div>

        <!-- Visual Analysis section with improved layout -->
        <div id="visualAnalysis" class="result-transition result-hidden max-w-5xl mx-auto glass-effect rounded-3xl p-10 shadow-2xl">
            <h2 class="text-3xl font-semibold mb-10 text-center bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-pink-400">
                Visual Analysis
            </h2>

            <div class="grid md:grid-cols-2 gap-10">
                <div class="space-y-6">
                    <h3 class="text-xl font-medium text-center text-indigo-200">Original Frame</h3>
                    <div class="relative aspect-video bg-indigo-950/30 rounded-2xl overflow-hidden shadow-lg border border-indigo-500/10">
                        <img id="frameImg" class="w-full h-full object-contain hidden" alt="Original Frame">
                    </div>
                </div>

                <div class="space-y-6">
                    <h3 class="text-xl font-medium text-center text-indigo-200">Heatmap Analysis</h3>
                    <div class="relative aspect-video bg-indigo-950/30 rounded-2xl overflow-hidden shadow-lg border border-indigo-500/10">
                        <img id="gradcamImg" class="w-full h-full object-contain hidden" alt="Grad-CAM Heatmap">
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize state management
        const state = {
            isProcessing: false,
            resetUI: function() {
                document.getElementById('results').classList.add('result-hidden');
                document.getElementById('visualAnalysis').classList.add('result-hidden');
                document.getElementById('frameImg').classList.add('hidden');
                document.getElementById('gradcamImg').classList.add('hidden');
            }
        };

        // File input handler with validation
        document.getElementById('videoInput').addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                if (!file.type.startsWith('video/')) {
                    alert('Please upload a valid video file.');
                    this.value = '';
                    return;
                }
                document.getElementById('fileName').textContent = file.name;
            } else {
                document.getElementById('fileName').textContent = "Drop your video here or click to browse";
            }
            state.resetUI();
        });

        // Form submission handler
        document.getElementById('uploadForm').addEventListener('submit', async function(event) {
            event.preventDefault();
            if (state.isProcessing) return;

            const fileInput = document.getElementById('videoInput');
            if (fileInput.files.length === 0) {
                alert('Please select a video before uploading.');
                return;
            }

            try {
                state.isProcessing = true;
                state.resetUI();
                
                // Show loading state
                document.getElementById('loading').classList.remove('hidden');
                
                // Prepare and send form data
                const formData = new FormData();
                formData.append('video', fileInput.files[0]);

                // Main API call
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }

                const data = await response.json();
                
                // Update prediction result
                document.getElementById('prediction').textContent = data.prediction;
                document.getElementById('results').classList.remove('result-hidden');

                // Fetch additional analysis data
                await Promise.all([
                    fetch3rdFrame(),
                    fetchGradCAM()
                ]);

                // Show visual analysis section
                document.getElementById('visualAnalysis').classList.remove('result-hidden');
            } catch (error) {
                console.error('Error:', error);
                alert('Error processing the video. Please try again.');
            } finally {
                state.isProcessing = false;
                document.getElementById('loading').classList.add('hidden');
            }
        });

        // Modified fetch functions with better error handling
        async function fetch3rdFrame() {
            try {
                const response = await fetch('http://127.0.0.1:5000/get_3rd_frame');
                if (!response.ok) throw new Error('Failed to fetch frame');
                
                const data = await response.json();
                const imageData = new Uint8Array(data.image_bytes);
                const blob = new Blob([imageData], { type: "image/jpeg" });
                
                const frameImg = document.getElementById('frameImg');
                frameImg.src = URL.createObjectURL(blob);
                frameImg.classList.remove('hidden');
            } catch (error) {
                console.error("Failed to load frame:", error);
                throw error;
            }
        }

        async function fetchGradCAM() {
            try {
                const response = await fetch("http://127.0.0.1:5000/gradcam");
                if (!response.ok) throw new Error('Failed to fetch heatmap');
                
                const data = await response.json();
                if (!data.heatmap_bytes?.length) throw new Error("Invalid heatmap data");
                
                const imageData = new Uint8Array(data.heatmap_bytes);
                const blob = new Blob([imageData], { type: "image/jpeg" });
                
                const gradcamImg = document.getElementById("gradcamImg");
                gradcamImg.src = URL.createObjectURL(blob);
                gradcamImg.classList.remove("hidden");
            } catch (error) {
                console.error("Failed to load heatmap:", error);
                throw error;
            }
        }
    </script>

</body>

</html>
