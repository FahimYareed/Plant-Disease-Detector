
import base64
import os
import json
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
import streamlit as st

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

st.markdown("""
    <style>
    div[data-testid="stForm"] {
        border: none;
        box-shadow: none;
    }
    .stForm > div {
        display: flex;
        justify-content: center;
    }
    button[data-testid="baseButton-secondary"] {
        display: block;
        margin: 0 auto;
    }
    div[data-testid="stForm"] button {
        display: block;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/swin.h5" 

# loading the class labels
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# loading the descriptions
descriptions_path = os.path.join(working_dir, "disease_descriptions.json")
disease_descriptions = json.load(open(descriptions_path))

# loading the preventions
preventions_path = os.path.join(working_dir, "disease_preventions.json")
disease_preventions = json.load(open(preventions_path))

def predict_image_class(image_path, class_indices, confidence_threshold=0.3):
    model = tf.keras.models.load_model(f"{working_dir}/trained_model/swin.h5")
    image = Image.open(image_path).resize((224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    
    # Get the maximum confidence score
    max_confidence = np.max(predictions)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_indices[str(predicted_class_index)]
    
    # Check if confidence is below threshold
    if max_confidence < confidence_threshold:
        return "not_a_leaf", max_confidence, predicted_class_name
    
    return predicted_class_name, max_confidence, None

# Streamlit App
st.title('Plant Disease Detector')

uploaded_image = st.file_uploader("Upload an image")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    resized_img = image.resize((200, 200))

    buffer = BytesIO()
    resized_img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    # Centered image
    st.markdown(f"""
        <div style='display: flex; justify-content: center;'>
            <div style='max-width: 120px;'>
                <img src='data:image/png;base64,{img_b64}'
                     style='width: 100%; border-radius: 8px; margin: 10px auto;' />
        </div>
    </div>
""", unsafe_allow_html=True)

    with st.form("classify_form", clear_on_submit=False):
        submit = st.form_submit_button("Predict")

        if submit:
            uploaded_image.seek(0)

            prediction, confidence, attempted_class = predict_image_class(uploaded_image, class_indices)
            
            if prediction == "not_a_leaf":
                st.markdown(f"""
                    <div style='
                        border: 2px solid #dc2626;
                        border-radius: 10px;
                        padding: 10px;
                        margin: 20px auto;
                        width: 80%;
                        max-width: 400px;
                        background-color: #fef2f2;
                        color: #991b1b;
                        text-align: center;
                    '>
                        <p style='margin: 10px auto'>This doesn't appear to be a leaf image.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align: center;'><h4>Predicted: {prediction}</h4></div>", unsafe_allow_html=True)

                description = disease_descriptions.get(prediction, "No description available for this disease.")
                st.markdown(f"<div style='text-align: center; color: gray;'><p>{description}</p></div>", unsafe_allow_html=True)
               
                prevention = disease_preventions.get(prediction, ["No prevention tips available."])
                prevention_html = "<br>".join([f"• {item}" for item in prevention])           
                st.markdown(f"""
                    <div style='
                        border: 2px solid #2d6a4f;
                        border-radius: 10px;
                        padding: 15px;
                        margin: 20px auto;
                        width: 80%;
                        max-width: 500px;
                        background-color: #f0fdf4;
                        color: #1b4332;
                        text-align: center;
                    '>
                        <h5 style='margin-bottom: 10px; color: #2d6a4f;'>Preventive Measures</h5>
                        <div style='margin: 0; display: inline-block; text-align: left;'>{prevention_html}</div>
                    </div>
                """, unsafe_allow_html=True)


