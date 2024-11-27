import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Load the pre-trained model
model = load_model("my_trained_model.h5")

# Function to preprocess the image (similar to the model's training preprocessing)
def preprocess_image(image):
    # Convert image to grayscale
    image = image.convert("L")  # Convert to grayscale
    # Resize to 28x28 pixels (MNIST format)
    image = image.resize((28, 28))
    # Normalize pixel values to the range [0, 1]
    image = np.array(image) / 255.0
    # Invert the image (MNIST has white digits on black background)
    image = 1 - image
    # Flatten the image to a 1D array (784 pixels)
    image = image.flatten()
    # Reshape to match model input
    image = image.reshape(1, 28, 28, 1)
    return image

# Function to predict the digit
def predict_digit(image):
    image = preprocess_image(image)
    # Get the model's prediction
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction, axis=1)
    return predicted_label[0]

# Streamlit UI
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9) and the model will predict the digit.")

# Image upload widget
uploaded_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# If an image is uploaded, display the image and make a prediction
if uploaded_image is not None:
    # Open the uploaded image using PIL
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Predict the digit
    predicted_digit = predict_digit(image)
    
    # Display the result
    st.write(f"Predicted Digit: {predicted_digit}")
