console.log('Script loaded!');

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

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        // prediction response in json format
        const data = await response.json();
        document.getElementById('prediction').textContent = `${data.prediction}`;
        document.getElementById('results').classList.remove('hidden');

        // Call heatmap analysis and handle any errors silently
        try {
            await fetchGradCAM();
        } catch (error) {
            console.warn('Heatmap generation completed with warnings:', error);
            // Don't show alert here, just log the warning
        }

    } catch (error) {
        console.error('Error:', error);
        alert('Error processing the video. Please try again.');
    } finally {
        document.getElementById('loading').classList.add('hidden');
    }
});

// Function to fetch the Grad-CAM heatmap from the server
async function fetchGradCAM() {
    try {
        const response = await fetch(`/gradcam?timestamp=${new Date().getTime()}`);
        if (!response.ok) {
            console.warn('Non-OK response from heatmap endpoint:', response.status);
            return; // Silent fail
        }

        const data = await response.json();
        console.log('Heatmap response received');

        if (!data.heatmap_bytes || data.heatmap_bytes.length === 0) {
            console.warn('Empty heatmap data received');
            return; // Silent fail
        }

        const blob = new Blob([new Uint8Array(data.heatmap_bytes)], { type: "image/jpeg" });
        const gradcamImg = document.getElementById('gradcamImg');
        if (!gradcamImg) {
            console.warn('gradcamImg element not found');
            return; // Silent fail
        }

        // Revoke any previous object URL to prevent memory leaks
        if (gradcamImg.src) {
            URL.revokeObjectURL(gradcamImg.src);
        }

        gradcamImg.src = URL.createObjectURL(blob);
        gradcamImg.classList.remove('hidden');
        console.log('Heatmap displayed successfully');
    } catch (error) {
        console.warn('Heatmap generation warning:', error);
        // Don't show alert, just log the warning
    }
}