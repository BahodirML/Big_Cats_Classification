import streamlit as st
import tensorflow
from PIL import Image
import numpy as np
import pathlib
from PIL import Image as PILImage
import plotly as px
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import ResNet50V2

images_size = 224
TL_Models =[
    ResNet50V2(input_shape=(images_size, images_size, 3), weights='imagenet', include_top=False)
]

# Define all the TL models names. This will be later used during visualization
TL_Models_NAMES = [
    'ResNet50V2'
]

# Fine tuning 
for tl_model in TL_Models:
    tl_model.trainable = False

model = keras.Sequential([
        tl_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

# title 
st.title('Animal Classification Model')

# Get user's name
name = st.text_input("Enter your name: ")

# Check if name is provided
if name:
    st.write(f"Hi {name}, Welcome to Our Streamlit App!")
else:
    st.write("Please enter your name above.")

# uploading
file = st.file_uploader("Upload picture", type=['png', 'jpeg', 'gif', 'svg'])


if file:
    # image
    st.image(file)

    img = PILImage.open(file)

    # model
    # Load the weights
    model.load_weights('7obj-model.weights.h5')
    if model:
        # Preprocess the image (e.g., resize, normalize)
        img = img.resize((224, 224))  # Example resizing to match the model's input shape
        img_array = np.array(img) / 255.0  # Example normalization
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make predictions using the model
        pred_probs = model.predict(img_array)[0]

        # Get the predicted class and its probability
        pred_id = np.argmax(pred_probs)
        pred = model.dls.vocab[pred_id]
        prob = pred_probs[pred_id]

        # Display predictions and probabilities
        st.success(f"Prediction: {pred}")
        st.info(f"Probability: {prob * 100:.1f}%")
