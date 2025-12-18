import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

IMG_SIZE = 32

st.set_page_config(page_title="GTSRB Traffic Sign Recognition")

st.title("🚦 German Traffic Sign Recognition System")
st.write("Upload a traffic sign image")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gtsrb_model.h5")

model = load_model()

with open("class_names.json", "r") as f:
    class_names = json.load(f)

uploaded_file = st.file_uploader(
    "Upload Traffic Sign Image",
    type=["jpg", "png", "jpeg", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    st.subheader("Prediction Result")
    st.write(f"**Traffic Sign:** {class_names[str(predicted_class)]}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

