import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Set page config for a clean look
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

st.title("🧠 Brain Tumor Detection System")
st.write("An educational tool for automated MRI analysis using Deep Learning.")

st.markdown("---")

# 1. Model Loading - MUST FOLLOW EXACTLY
@st.cache_resource
def get_model():
    try:
        return load_model("brain_tumor_model.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = get_model()

# 2. Image Upload
uploaded_file = st.file_uploader("Upload an MRI image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # PREPROCESSING (DIP) — MUST FOLLOW EXACTLY
    
    # Raw image decoding step
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 0)  # grayscale
    
    # Safety check
    if img is None:
        st.error("Invalid image uploaded")
        st.stop()
    
    # Store original for display (preserving untainted version)
    original_display = cv2.imdecode(file_bytes, 1) # decode with color for better original display
    original_display = cv2.cvtColor(original_display, cv2.COLOR_BGR2RGB)
    
    # Start Preprocessing Pipeline
    # Step 1: Gaussian Blur
    denoised = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Step 2: Histogram Equalization
    enhanced = cv2.equalizeHist(denoised)
    
    # Step 3: Resize (128x128)
    processed = cv2.resize(enhanced, (128, 128))
    
    # Prepare for model (Normalization and Reshape)
    final_input = processed / 255.0
    final_input = final_input.reshape(1, 128, 128, 1) # Mandatory Reshape
    
    # UI Layout: Side-by-Side Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(original_display, use_column_width=True)
        st.caption("The raw MRI scan uploaded by the user.")
        
    with col2:
        st.subheader("Processed Input")
        st.image(processed, use_column_width=True, clamp=True)
        st.caption("Resized for model input (128x128).")
        
    st.markdown("---")
    
    # DISPLAY PREPROCESSING STEPS
    st.subheader("🔍 Preprocessing Steps (Educational)")
    
    p_col1, p_col2 = st.columns(2)
    
    with p_col1:
        st.image(denoised, caption="Denoised Image", use_column_width=True)
        st.info("**Gaussian Blur** applied for noise reduction.")
        
    with p_col2:
        st.image(enhanced, caption="Enhanced Image", use_column_width=True)
        st.info("**Histogram Equalization** applied for contrast enhancement.")
        
    st.markdown("---")
    
    # PREDICTION LOGIC — MUST FOLLOW EXACTLY
    if model is not None:
        with st.spinner("Analyzing image..."):
            # prob = model.predict(img)[0][0] -> using final_input as it's the reshaped version
            pred = model.predict(final_input)
            prob = pred[0][0]
            
            if prob > 0.5:
                result = "Tumor"
                confidence = prob
            else:
                result = "No Tumor"
                confidence = 1 - prob
        
        # OUTPUT DISPLAY
        st.subheader("📊 Final Result")
        
        # Styling for high visibility
        if result == "Tumor":
            st.error(f"### Prediction: {result}")
        else:
            st.success(f"### Prediction: {result}")
            
        st.write(f"**Tumor Probability:** {prob:.4f} ({prob*100:.2f}%)")
        st.write(f"**Confidence:** {confidence*100:.2f}%")
        st.info(f"*Confidence: Model confidence in predicted class*")
    else:
        st.warning("Model not loaded correctly. Please check brain_tumor_model.h5")

else:
    st.info("Please upload an MRI scan to begin the detection process.")

# Footer
st.markdown("---")
st.caption("Developed for educational purposes. Always consult a medical professional for diagnosis.")
