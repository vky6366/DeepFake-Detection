
const API_BASE_URL = window.location.hostname === 'localhost'
  ? 'http://localhost:5000'
  : `http://${window.location.hostname}:5000`;

const instance = axios.create({
  baseURL: API_BASE_URL,
});


// Drag and drop functionality
document.getElementById('videoInput').addEventListener('change', function () {
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
});

// Form submission handler
document.getElementById('uploadForm').addEventListener('submit', async function (event) {
    event.preventDefault();

    // Check if a file is selected
    const fileInput = document.getElementById('videoInput');
    if (fileInput.files.length === 0) {
        alert('Please select a video before uploading.');
        return;
    }

    try {
        document.getElementById('loading').classList.remove('hidden');

        const formData = new FormData();
        formData.append('video', fileInput.files[0]);

        // Updated to use API_BASE_URL
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        // predication response in json format
        const data = await response.json();
        document.getElementById('prediction').textContent = `${data.prediction} (Score: ${(data.score * 100).toFixed(2)}%)`;
        document.getElementById('results').classList.remove('hidden');

        await Promise.all([fetch3rdFrame(), fetchGradCAM()]);

    } catch (error) {
        console.error('Error:', error);
        alert('Error processing the video. Please try again.');
    } finally {
        document.getElementById('loading').classList.add('hidden');
    }
});

// Function to fetch the 3rd frame from the server
async function fetch3rdFrame() {
    try {
        // Updated to use API_BASE_URL
        const response = await fetch(`${API_BASE_URL}/get_3rd_frame`);
        if (!response.ok) throw new Error('Failed to fetch 3rd frame');

        const data = await response.json();
        if (!data.image_bytes) throw new Error('Invalid image data');

        const blob = new Blob([new Uint8Array(data.image_bytes)], { type: "image/jpeg" });
        const frameImg = document.getElementById('frameImg');
        frameImg.src = URL.createObjectURL(blob);
        frameImg.classList.remove('hidden');
    } catch (error) {
        console.error('Failed to load 3rd frame:', error);
    }
}

// Function to fetch the Grad-CAM heatmap from the server
async function fetchGradCAM() {
    try {
        // Updated to use API_BASE_URL
        const response = await fetch(`${API_BASE_URL}/gradcam?timestamp=${new Date().getTime()}`);
        if (!response.ok) throw new Error('Failed to fetch heatmap');

        const data = await response.json();
        console.log('Server response:', data);

        if (!data.heatmap_bytes || data.heatmap_bytes.length === 0) throw new Error('Invalid heatmap data');

        const blob = new Blob([new Uint8Array(data.heatmap_bytes)], { type: "image/jpeg" });
        console.log('Blob created:', blob);

        const gradcamImg = document.getElementById('gradcamImg');
        if (!gradcamImg) throw new Error('gradcamImg element not found');

        // Revoke any previous object URL to prevent memory leaks
        if (gradcamImg.src) {
            URL.revokeObjectURL(gradcamImg.src);
        }

        gradcamImg.src = URL.createObjectURL(blob);
        gradcamImg.classList.remove('hidden');
        console.log('Heatmap displayed successfully');
    } catch (error) {
        console.error('Failed to load heatmap:', error);
        alert('Error retrieving the heatmap. Please try again.');
    }
}
