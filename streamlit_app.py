import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image



# ---- CONFIG ----
st.set_page_config(page_title="AI vs Real Face Classifier", layout="centered")

# ---- LOGO ----
st.image("assets/logo.png", width=150)  # Adjust path/size if needed

# ---- TITLE ----
st.markdown("## 🧠 Real vs AI Face Classifier")
st.write("Upload a face image and the model will predict whether it's a **real human** or **AI-generated**.")

# ---- LOAD MODEL ----
model = load_model("model/face_classifier.h5")
class_names = ["fake_faces", "real_faces"]

# ---- UPLOAD IMAGE ----
uploaded_file = st.file_uploader("📤 Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Show uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        img = image.resize((150, 150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        confidence = prediction[0][predicted_index] * 100
        predicted_label = class_names[predicted_index]

        # Display result
        st.markdown(
            f"### ✅ Prediction: `{predicted_label}` \nConfidence: `{confidence:.2f}%`"
        )
    except Exception as e:
        st.error(f"⚠️ Error processing the image: {e}")
else:
    st.info("👆 Upload an image to start...")
