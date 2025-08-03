import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = load_model("model/face_classifier.h5")
class_names = ["fake_faces", "real_faces"]  # Make sure this matches your dataset

st.set_page_config(page_title="AI vs Real Face Classifier", layout="centered")
st.title("🧠 Real vs AI Face Classifier")
st.write("Upload a face image and the model will predict whether it's a real human or AI-generated.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    img = load_img(uploaded_file, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index] * 100
    predicted_label = class_names[predicted_index]

    st.success(f"Prediction: **{predicted_label}** ({confidence:.2f}% confidence)")
    