document.getElementById('uploadForm').addEventListener('submit', async function (event) {
    event.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById('videoInput');
    if (fileInput.files.length === 0) {
        alert('Please select a file before submitting.');
        return;
    }

    formData.append('file', fileInput.files[0]);

    // Show loading animation
    document.getElementById('loading').style.display = 'block';

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 sec timeout

        const response = await fetch('http://127.0.0.1:5000/upload', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        const data = await response.json();
        console.log("Response from server:", data);

        if (response.ok) {
            document.getElementById('prediction').textContent = data.prediction;
        } else {
            alert(data.error || 'An error occurred while processing your request.');
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            alert('Request timed out. Try again later.');
        } else {
            console.error("Error:", error);
            alert('An error occurred. Please try again later.');
        }
    } finally {
        // Hide loading animation once response is received
        document.getElementById('loading').style.display = 'none';
    }
});


