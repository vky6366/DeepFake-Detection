console.log('Script loaded!');
// Drag and drop functionality
// Form submission handler
document.getElementById('uploadForm').addEventListener('submit', async function (event) {
    event.preventDefault();

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

        const data = await response.json();
        document.getElementById('prediction').textContent = `${data.prediction}`;
        document.getElementById('results').classList.remove('hidden');

        
        if (data.prediction.toLowerCase() === 'fake') {
            await fetchGradCAM();
        } else {
            await fetchOriginalFrame();
        }

    } catch (error) {
        console.error('Error:', error);
        alert('Error processing the video. Please try again.');
    } finally {
        document.getElementById('loading').classList.add('hidden');
    }
});

async function fetchOriginalFrame() {
    try {
        const response = await fetch(`/frame?timestamp=${new Date().getTime()}`);
        if (!response.ok) {
            console.warn('Non-OK response from frame endpoint:', response.status);
            return;
        }

        const data = await response.json();
        console.log('Frame response received');

        if (!data.frame_bytes || data.frame_bytes.length === 0) {
            console.warn('Empty frame data received');
            return;
        }

        const blob = new Blob([new Uint8Array(data.frame_bytes)], { type: "image/jpeg" });
        const gradcamImg = document.getElementById('gradcamImg'); // reuse the same img tag
        if (!gradcamImg) {
            console.warn('gradcamImg element not found');
            return;
        }

        if (gradcamImg.src) {
            URL.revokeObjectURL(gradcamImg.src);
        }

        gradcamImg.src = URL.createObjectURL(blob);
        gradcamImg.classList.remove('hidden');
        console.log('Original frame displayed successfully');
    } catch (error) {
        console.warn('Original frame loading warning:', error);
    }
}
