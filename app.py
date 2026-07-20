import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_cropper import st_cropper

# Load model
model = load_model("face-mask-detector.keras")

st.title("😷 Face Mask Detection Project")

option = st.selectbox("Select Input", ["Image", "Capture"])

uploaded_file = None
cropped_img = None

# Upload Image
if option == "Image":
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

# Capture Image
else:
    camera_image = st.camera_input("Capture Image")

    if camera_image is not None:
        image = Image.open(camera_image)

        cropped_img = st_cropper(
            image,
            realtime_update=True,
            box_color="red",
            aspect_ratio=(1,1)
        )

        st.image(cropped_img, caption="Captured Image", width=350)

# Detect Button
if st.button("Detect"):

    image = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

    elif cropped_img is not None:
        image = cropped_img

    else:
        st.warning("Please upload or capture an image.")

    if image is not None:

        # Show Uploaded Image
        st.image(image, caption="Uploaded Image", width=350)

        # Preprocess
        img = image.convert("RGB")
        img = img.resize((150,150))

        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array, verbose=0)
        score = float(prediction[0][0])

        st.write(f"### Prediction Score : {score:.4f}")

        # Prediction
        if score > 0.5:
            st.success("😷 Person is WITH MASK")
        else:
            st.error("❌ Person is WITHOUT MASK")
