import os
import io
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


# --- CONFIGURATION ---
# 5MB Payload Limit
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 

# --- MODEL LOADING ---
model = None
try:
    # Ensure model is loaded from current working directory
    model_path = os.path.join(os.getcwd(), "brain_tumor_model.h5")
    model = load_model(model_path)
    print(" Model loaded successfully from:", model_path)
except Exception as e:
    print(" Model loading failed:", e)

# --- UTILS ---
def encode_image(img):
    """Encodes a CV2 image to base64 string."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

# --- ERROR HANDLERS ---
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({"error": "File too large (max 5MB)"}), 413

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Validation
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['image']
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({"error": "Unsupported file format. Please upload JPG or PNG."}), 400
    
    # 2. Check if model is loaded
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    try:
        # 3. Read and Decode (Enforce Grayscale)
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, 0) # Grayscale

        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Preserve original for display
        original_base64 = encode_image(img)

        # 4. PREPROCESSING PIPELINE (DIP) - MUST FOLLOW EXACTLY
        # Denoising
        denoised = cv2.GaussianBlur(img, (5, 5), 0)
        # Enhancement
        enhanced = cv2.equalizeHist(denoised)
        # Resizing
        processed = cv2.resize(enhanced, (128, 128))

        # 5. Prepare for Prediction
        norm = processed / 255.0
        final_tensor = norm.reshape(1, 128, 128, 1)

        # 6. INFERENCE - MUST FOLLOW EXACTLY
        pred_raw = model.predict(final_tensor)
        prob = float(pred_raw[0][0])
        
        # Clamp probability
        prob = max(0.0, min(1.0, prob))
        
        # Numerical Polish
        prob_rounded = round(prob, 4)

        if prob > 0.5:
            result = "Tumor"
            confidence = prob
        else:
            result = "No Tumor"
            confidence = 1 - prob

        conf_rounded = round(confidence * 100, 2)

        # 7. Response
        return jsonify({
            "prediction": result,
            "probability": prob_rounded,
            "confidence": conf_rounded,
            "images": {
                "original": original_base64,
                "denoised": encode_image(denoised),
                "enhanced": encode_image(enhanced),
                "processed": encode_image(processed)
            }
        })

    except Exception as e:
        print("Backend Processing Error:", e)
        return jsonify({"error": "Image processing failed"}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)
