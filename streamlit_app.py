import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import *
from PIL import Image


# Load the pre-trained face classification model
model = tf.keras.models.load_model('fine_tuned_best_model.h5')

class_labels = ['animal fish',
 'animal fish bass',
 'fish sea_food black_sea_sprat',
 'fish sea_food gilt_head_bream',
 'fish sea_food hourse_mackerel',
 'fish sea_food red_mullet',
 'fish sea_food red_sea_bream',
 'fish sea_food sea_bass',
 'fish sea_food shrimp',
 'fish sea_food striped_red_mullet',
 'fish sea_food trout']


# Function to make predictions
def predict_fish(img):
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = tf.image.convert_image_dtype(np.array(img), dtype=tf.float32)  # Normalize pixel values to [0, 1]
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)[0]
    predicted_class = np.argmax(predictions)  # Get class index
    confidence = predictions[predicted_class]  # Confidence score

    return class_labels[predicted_class], confidence


# Streamlit UI
st.title("Fish Classification App")
st.write("Upload a fish image to classify its species.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        label, confidence = predict_fish(image)
        st.success(f"Predicted Category: **{label}**")
        st.info(f"Confidence Score: {confidence:.2f}")

