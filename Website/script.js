// // // document.getElementById('uploadForm').addEventListener('submit', async function (event) {
// // //     event.preventDefault();

// // //     const formData = new FormData();
// // //     const fileInput = document.getElementById('videoInput');
// // //     if (fileInput.files.length === 0) {
// // //         alert('Please select a file before submitting.');
// // //         return;
// // //     }

// // //     formData.append('video', fileInput.files[0]); 

// // //     // Show loading animation
// // //     document.getElementById('loading').style.display = 'block';

// // //     try {
// // //         const controller = new AbortController();
// // //         const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 sec timeout

// // //         const response = await fetch('http://127.0.0.1:5000/upload', {
// // //             method: 'POST',
// // //             body: formData,
// // //             signal: controller.signal
// // //         });

// // //         clearTimeout(timeoutId);

// // //         const data = await response.json();
// // //         console.log("Response from server:", data);

// // //         if (response.ok) {
// // //             document.getElementById('prediction').textContent =data.prediction;
// // //         } else {
// // //             alert(data.error || 'An error occurred while processing your request.');
// // //         }
// // //     } catch (error) {
// // //         if (error.name === 'AbortError') {
// // //             alert('Request timed out. Try again later.');
// // //         } else {
// // //             console.error("Error:", error);
// // //             alert('An error occurred. Please try again later.');
// // //         }
// // //     } finally {
// // //         // Hide loading animation once response is received
// // //         document.getElementById('loading').style.display = 'none';
// // //     }
// // // });

// // document.getElementById('uploadForm').addEventListener('submit', async function (event) {
// //     event.preventDefault();

// //     const formData = new FormData();
// //     const fileInput = document.getElementById('videoInput');

// //     if (fileInput.files.length === 0) {
// //         alert('Please select a file before submitting.');
// //         return;
// //     }

// //     formData.append('video', fileInput.files[0]); // Updated to match Flask route

// //     // Show loading animation
// //     document.getElementById('loading').style.display = 'block';

// //     try {
// //         const controller = new AbortController();
// //         const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 sec timeout

// //         const response = await fetch('http://127.0.0.1:5000/upload', {
// //             method: 'POST',
// //             body: formData,
// //             signal: controller.signal
// //         });

// //         clearTimeout(timeoutId);

// //         const data = await response.json();
// //         console.log("Response from server:", data);

// //         if (response.ok) {
// //             // Show results section
// //             document.getElementById('results').style.display = 'block';

// //             // Update the prediction message
// //             document.getElementById('prediction').textContent = data.prediction;
// //             document.getElementById('score').textContent = data.score.toFixed(4); // Format score to 4 decimals
// //         } else {
// //             alert(data.error || 'An error occurred while processing your request.');
// //         }
// //     } catch (error) {
// //         if (error.name === 'AbortError') {
// //             alert('Request timed out. Try again later.');
// //         } else {
// //             console.error("Error:", error);
// //             alert('An error occurred. Please try again later.');
// //         }
// //     } finally {
// //         // Hide loading animation once response is received
// //         document.getElementById('loading').style.display = 'none';
// //     }
// // });

// document.getElementById('uploadForm').addEventListener('submit', async function (event) {
//     event.preventDefault();

//     const formData = new FormData();
//     const fileInput = document.getElementById('videoInput');
//     if (fileInput.files.length === 0) {
//         alert('Please select a file before submitting.');
//         return;
//     }

//     formData.append('video', fileInput.files[0]); // Updated to match the Flask route

//     // Show loading animation
//     document.getElementById('loading').style.display = 'block';

//     try {
//         const controller = new AbortController();
//         const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 sec timeout

//         const response = await fetch('http://127.0.0.1:5000/upload', {
//             method: 'POST',
//             body: formData,
//             signal: controller.signal
//         });

//         clearTimeout(timeoutId);

//         const data = await response.json();
//         console.log("Response from server:", data);

//         if (response.ok) {
//             document.getElementById('prediction').textContent = Prediction. $data.prediction, score.$data.score.toFixed(4);
//         } else {
//             alert(data.error || 'An error occurred while processing your request.');
//         }
//     } catch (error) {
//         if (error.name === 'AbortError') {
//             alert('Request timed out. Try again later.');
//         } else {
//             console.error("Error:", error);
//             alert('An error occurred. Please try again later.');
//         }
//     }
    
    
//     finally {
//         // Hide loading animation once response is received
//         document.getElementById('loading').style.display = 'none';
//     }
// });

document.getElementById('uploadForm').addEventListener('submit', async function (event) {
    event.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById('videoInput');

    if (fileInput.files.length === 0) {
        alert('Please select a file before submitting.');
        return;
    }

    formData.append('video', fileInput.files[0]); // Ensure it matches Flask's expected key "video"

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
            // Display results
            document.getElementById('results').style.display = 'block';
            document.getElementById('prediction').textContent = data.prediction;
            document.getElementById('score').textContent = data.score.toFixed(4); // Format to 4 decimals
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
        // Hide loading animation after response
        document.getElementById('loading').style.display = 'none';
    }
});
