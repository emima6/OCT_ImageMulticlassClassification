import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import base64
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt

# Function to Set Background Image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# Set background (Replace 'background.jpg' with your image)
set_background("D:/PROJECT/OCT2/background.jpg")



#  Load Trained Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("oct10_hybrid_vgg16_resnet50.keras")

model = load_model()

# Load Pre-trained Feature Extractors (VGG16 & ResNet50)
vgg16 = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet50 = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add GlobalAveragePooling2D
vgg16 = tf.keras.Model(inputs=vgg16.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(vgg16.output))
resnet50 = tf.keras.Model(inputs=resnet50.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(resnet50.output))

#  Function to Preprocess Image
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (224, 224))  # Resize to model input size
    img = img.astype('float32') / 255.0  # Rescale
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Function to Extract Features from Image
def extract_features(img):
    vgg16_feature = vgg16.predict(img, verbose=0)  # Shape: (1, 512)
    resnet50_feature = resnet50.predict(img, verbose=0)  # Shape: (1, 2048)
    return np.concatenate([vgg16_feature, resnet50_feature], axis=-1)  # Shape: (1, 2560)

#  Function to Classify Image
def classify_image(model, image):
    processed_img = preprocess_image(image)
    combined_features = extract_features(processed_img)
    prediction = model.predict(combined_features)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100  # Convert to percentage
    return predicted_class, confidence, prediction

# Class Labels
class_labels = ["CNV", "DME", "DRUSEN", "NORMAL"]

#  Streamlit UI
st.title("🔍 OCT Image Classification")
st.write("Upload an OCT scan image to classify it into CNV, DME, DRUSEN, or NORMAL.")

# File Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Perform Classification
    predicted_class, confidence, prediction = classify_image(model, image)

    # Display Classification Results
    st.write(f"### 🏷 Predicted Class: **{class_labels[predicted_class]}**")
    st.write(f"### 🎯 Confidence Score: **{confidence:.2f}%**")

    # Display Confidence Distribution
    fig, ax = plt.subplots()
    ax.bar(class_labels, prediction[0], color=['blue', 'green', 'red', 'purple'])
    ax.set_ylabel("Confidence Score")
    ax.set_title("Classification Confidence Distribution")
    st.pyplot(fig)
