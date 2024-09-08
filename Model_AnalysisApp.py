import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report

# Load the pre-trained model and history
with open('cnn_mnist_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('history.pkl', 'rb') as file_pi:
    history = pickle.load(file_pi)

# Title of the app
st.title("CNN MNIST Model - Streamlit App")

# Section 1: Model Performance Analysis
st.header("Model Analysis")

# Plot Accuracy
st.subheader("Accuracy over Epochs")
plt.figure(figsize=(10, 5))
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
st.pyplot(plt)

# Plot Loss
st.subheader("Loss over Epochs")
plt.figure(figsize=(10, 5))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
st.pyplot(plt)

# Section 2: Predict on User Input
st.header("Generate Predictions")

# User input section for digit prediction
st.write("Input a digit (0-9) for prediction:")

# Load the MNIST test set for use in the prediction
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0  # Preprocessing

# Generate a random digit from the test set
if st.button("Generate Random Image"):
    idx = np.random.randint(0, len(x_test))
    random_image = x_test[idx]
    
    # Display the image
    st.image(random_image.reshape(28, 28), caption="Randomly Selected Image", width=150)
    
    # Make a prediction
    prediction = model.predict(np.expand_dims(random_image, axis=0))
    predicted_class = np.argmax(prediction)
    
    st.write(f"Predicted Class: {predicted_class}")

# Option to upload your own digit image for prediction
st.subheader("Upload your own 28x28 grayscale image:")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load the uploaded image
    from PIL import Image
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    image = np.array(image)
    
    # Preprocess the image
    image = image.reshape(1, 28, 28, 1) / 255.0
    
    # Display the uploaded image
    st.image(image.reshape(28, 28), caption="Uploaded Image", width=150)
    
    # Make a prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    
    st.write(f"Predicted Class: {predicted_class}")
