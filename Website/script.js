document.addEventListener('DOMContentLoaded', () => {
    const baseUrl = window.location.origin;
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadProgress = document.getElementById('upload-progress');
    const progressBar = uploadProgress.querySelector('.progress-bar');
    const results = document.getElementById('results');

    // Drag and drop handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        });
    });

    // Handle file drop
    dropZone.addEventListener('drop', handleDrop);
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        handleFile(file);
    }

    function handleFileSelect(e) {
        const file = e.target.files[0];
        handleFile(file);
    }

    function showError(message, isWarning = false) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert ${isWarning ? 'alert-warning' : 'alert-danger'} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        document.querySelector('.upload-container').insertAdjacentElement('afterend', alertDiv);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            alertDiv.classList.remove('show');
            setTimeout(() => alertDiv.remove(), 150);
        }, 5000);
    }

    async function handleFile(file) {
        if (!file || !file.type.startsWith('video/')) {
            showError('Please upload a valid video file');
            return;
        }

        // Clear previous alerts
        document.querySelectorAll('.alert').forEach(alert => alert.remove());
        
        uploadProgress.classList.remove('d-none');
        progressBar.style.width = '0%';
        results.classList.add('d-none');

        const formData = new FormData();
        formData.append('video', file);

        try {
            // Upload video and get prediction
            progressBar.style.width = '30%';
            const response = await fetch(`${baseUrl}/upload`, {  // Add baseUrl
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            const predictionData = await response.json();
            progressBar.style.width = '60%';

            // Initialize result data structure
            let resultData = {
                prediction: predictionData,
                heatmap: null,
                analysis: null
            };

            try {
                // Get GradCAM heatmap
                const heatmapResponse = await fetch(`${baseUrl}/gradcam`);  // Add baseUrl
                if (heatmapResponse.ok) {
                    resultData.heatmap = await heatmapResponse.json();
                } else {
                    showError('Warning: Could not generate heatmap visualization', true);
                }
            } catch (error) {
                console.error('Heatmap fetch error:', error);
                showError('Warning: Could not generate heatmap visualization', true);
            }

            progressBar.style.width = '80%';

            try {
                // Get facial analysis
                const analysisResponse = await fetch(`${baseUrl}/facial_analysis`);  // Add baseUrl
                if (analysisResponse.ok) {
                    resultData.analysis = await analysisResponse.json();
                } else {
                    showError('Warning: Could not perform facial analysis', true);
                }
            } catch (error) {
                console.error('Analysis fetch error:', error);
                showError('Warning: Could not perform facial analysis', true);
            }

            progressBar.style.width = '100%';
            
            // Display results even if some parts failed
            displayResults(resultData);
        } catch (error) {
            console.error('Error:', error);
            showError('Error processing video: ' + error.message);
        } finally {
            setTimeout(() => {
                uploadProgress.classList.add('d-none');
                progressBar.style.width = '0%';
            }, 500);
        }
    }

    function displayResults(data) {
        // Display prediction result
        const predictionResult = document.getElementById('prediction-result');
        const confidenceScore = document.getElementById('confidence-score');
        
        predictionResult.textContent = data.prediction.prediction;
        predictionResult.className = data.prediction.prediction === 'FAKE' ? 
            'prediction-fake' : 'prediction-real';
        
        confidenceScore.textContent = `${(data.prediction.score * 100).toFixed(2)}%`;

        // Display original frame if available
        const originalFrame = document.getElementById('original-frame');
        const originalFrameContainer = originalFrame.closest('.col-md-6');
        if (data.frame && data.frame.image_bytes) {
            originalFrame.src = `data:image/jpeg;base64,${arrayBufferToBase64(data.frame.image_bytes)}`;
            originalFrameContainer.classList.remove('d-none');
        } else {
            originalFrameContainer.classList.add('d-none');
        }

        // Display heatmap if available
        const heatmapImg = document.getElementById('heatmap');
        const heatmapContainer = heatmapImg.closest('.col-md-6');
        if (data.heatmap && data.heatmap.heatmap_bytes) {
            heatmapImg.src = `data:image/jpeg;base64,${arrayBufferToBase64(data.heatmap.heatmap_bytes)}`;
            heatmapContainer.classList.remove('d-none');
        } else {
            heatmapContainer.classList.add('d-none');
        }

        // Display facial analysis if available
        const facialAnalysisSection = document.getElementById('facial-analysis');
        const focusedRegions = document.getElementById('focused-regions');
        if (data.analysis && data.analysis.focused_regions) {
            focusedRegions.innerHTML = '';
            Object.entries(data.analysis.focused_regions).forEach(([region, score]) => {
                const regionElement = document.createElement('div');
                regionElement.className = 'region-item';
                regionElement.textContent = `${region}: ${score}%`;
                focusedRegions.appendChild(regionElement);
            });
            facialAnalysisSection.classList.remove('d-none');
        } else {
            facialAnalysisSection.classList.add('d-none');
        }

        results.classList.remove('d-none');
    }

    function arrayBufferToBase64(buffer) {
        let binary = '';
        const bytes = new Uint8Array(buffer);
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }
});