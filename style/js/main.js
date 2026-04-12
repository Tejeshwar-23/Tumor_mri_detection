document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const imageInput = document.getElementById('image-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const resultsContainer = document.getElementById('results-container');

    // UI Elements for results
    const imgOriginal = document.getElementById('img-original');
    const imgDenoised = document.getElementById('img-denoised');
    const imgEnhanced = document.getElementById('img-enhanced');
    const imgProcessed = document.getElementById('img-processed');

    const resPrediction = document.getElementById('res-prediction');
    const resProbability = document.getElementById('res-probability');
    const resConfidence = document.getElementById('res-confidence');

    // Select file on click
    dropZone.onclick = () => imageInput.click();

    // Show selected file name
    imageInput.onchange = (e) => {
        const file = e.target.files[0];
        if (file) {
            dropZone.querySelector('p').textContent = `Selected: ${file.name}`;
        }
    };

    // Analyze button click
    analyzeBtn.onclick = async () => {
        const file = imageInput.files[0];
        if (!file) {
            alert("Please select an image first.");
            return;
        }

        const formData = new FormData();
        formData.append('image', file);

        // UI State: Loading
        analyzeBtn.disabled = true;
        loadingOverlay.classList.remove('hidden');
        resultsContainer.classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            // Handle Server Errors
            if (data.error) {
                alert(data.error);
                return;
            }

            // Populate Images
            imgOriginal.src = `data:image/png;base64,${data.images.original}`;
            imgDenoised.src = `data:image/png;base64,${data.images.denoised}`;
            imgEnhanced.src = `data:image/png;base64,${data.images.enhanced}`;
            imgProcessed.src = `data:image/png;base64,${data.images.processed}`;

            // Populate Text Results
            resPrediction.textContent = data.prediction;
            resPrediction.className = 'value ' + (data.prediction === 'Tumor' ? 'tumor' : 'no-tumor');
            
            resProbability.textContent = `${data.probability} (${(data.probability * 100).toFixed(2)}%)`;
            resConfidence.textContent = `${data.confidence}%`;

            // Display Results
            resultsContainer.classList.remove('hidden');
            resultsContainer.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error("Fetch Error:", error);
            alert("Connection error or server failure. Please check if the backend is running.");
        } finally {
            // ALWAYS re-enable button and hide loading
            analyzeBtn.disabled = false;
            loadingOverlay.classList.add('hidden');
        }
    };
});
