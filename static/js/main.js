document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const imageInput = document.getElementById('image-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const consoleSection = document.getElementById('console-section');
    const processingLog = document.getElementById('processing-log');
    const resultsContainer = document.getElementById('results-container');
    const originalContainer = document.getElementById('original-container');

    // Result UI
    const resTag = document.getElementById('res-tag');
    const resConfidenceText = document.getElementById('res-confidence');
    const gaugeProgress = document.getElementById('gauge-progress');
    const resProbability = document.getElementById('res-probability');

    const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    const addLog = async (text, delay = 500) => {
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.textContent = text;
        processingLog.appendChild(entry);
        processingLog.scrollTop = processingLog.scrollHeight;
        await sleep(delay);
    };

    dropZone.onclick = () => imageInput.click();
    imageInput.onchange = (e) => {
        const file = e.target.files[0];
        if (file) dropZone.querySelector('p').textContent = `DATA_LOADED: ${file.name}`;
    };

    analyzeBtn.onclick = async () => {
        const file = imageInput.files[0];
        if (!file) {
            alert("UPLOAD_ERROR: No MRI data detected.");
            return;
        }

        const formData = new FormData();
        formData.append('image', file);

        // RESET UI
        analyzeBtn.disabled = true;
        consoleSection.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
        processingLog.innerHTML = "";
        originalContainer.classList.remove('scanning');

        try {
            await addLog("[SYS] INITIALIZING_NEURAL_SUBSYSTEM...", 800);
            await addLog("[SYS] LOADING_INFERENCE_MODEL: brain_tumor_model.h5", 600);
            
            // Start Fetch
            const responsePromise = fetch('/predict', { method: 'POST', body: formData });
            
            await addLog("[DIP] ACQUIRING_DATA_STREAM...", 400);
            await addLog("[DIP] APPLYING_GAUSSIAN_DENOISING...", 700);
            await addLog("[DIP] PERFORMING_HISTOGRAM_EQUALIZATION...", 700);
            await addLog("[DIP] NORMALIZING_TENSOR_INPUT_128X128...", 500);

            const response = await responsePromise;
            const data = await response.json();

            if (data.error) {
                await addLog(`[ERR] PIPELINE_CRASH: ${data.error}`, 0);
                alert(data.error);
                return;
            }

            await addLog("[AI] SCANNING_FOR_MALIGNANCY...", 1000);
            
            // Start Scanning Animation
            resultsContainer.classList.remove('hidden');
            originalContainer.classList.add('scanning');
            document.getElementById('img-original').src = `data:image/png;base64,${data.images.original}`;
            document.getElementById('img-denoised').src = `data:image/png;base64,${data.images.denoised}`;
            document.getElementById('img-enhanced').src = `data:image/png;base64,${data.images.enhanced}`;
            document.getElementById('img-processed').src = `data:image/png;base64,${data.images.processed}`;

            await sleep(2000); // Let scanner run for a bit

            // Finalize Results
            await addLog("[AI] DATA_SYNTHESIS_COMPLETE. CALCULATING_CONFIDENCE...", 500);
            
            // Update Confidence Gauge
            const dashArray = 440;
            const offset = dashArray - (dashArray * data.confidence) / 100;
            gaugeProgress.style.strokeDashoffset = offset;
            resConfidenceText.textContent = `${data.confidence}%`;

            // Update Prediction Tag
            resTag.textContent = data.prediction;
            resTag.className = 'result-tag ' + (data.prediction === 'Tumor' ? 'tumor' : 'no-tumor');
            
            resProbability.textContent = `${data.probability}`;

            await addLog("[SYS] ANALYSIS_COMPLETE. STANDBY_FOR_PHYSICIAN_REVIEW.", 0);
            document.getElementById('log-status').textContent = "SEQUENCE_COMPLETE";
            document.getElementById('log-status').style.color = "#4ade80";

            resultsContainer.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error(error);
            await addLog("[ERR] SYSTEM_OFFLINE_OR_NETWORK_FAILURE", 0);
            alert("SYSTEM_ERROR: Check server console for logs.");
        } finally {
            analyzeBtn.disabled = false;
        }
    };
});
