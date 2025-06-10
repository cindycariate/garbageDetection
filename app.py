import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO



# Page config
st.set_page_config(page_title="Garbage Detection", layout="centered")

# Add this at the top of your app
st.markdown(
    """
    <style>
    /* Change background */
    .stApp {
        background-color: #e6f7e6; /* light green */
    }

    /* Optional: Center the title text and style it */
    h1 {
        color: #2d6a4f;
        text-align: center;
    }

    /* Optional: Style sidebar */
    .css-1d391kg { 
        background-color: #b7e4c7 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Title with emoji
st.markdown("<h1 style='text-align: center;'> Garbage Detection 🗑️</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Load YOLOv8 model
model = YOLO("best.pt")

# Sidebar
with st.sidebar:
    st.header("Options")
    option = st.radio("Choose input method:", ("📁 Upload Image", "📷 Use Camera"))
    st.info("Select how you want to input the image for detection.")

# Image input
image = None
if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif option == "📷 Use Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        image = Image.open(camera_image)

# Show input image
if image:
    st.image(image, caption="📸 Input Image", use_container_width=True)
    st.markdown("---")

    # Detection button
    if st.button("Detect Garbage"):
        with st.spinner("Detecting... Please wait."):
            results = model.predict(source=np.array(image), conf=0.25)
            annotated_img = results[0].plot()
        st.success("✅ Detection complete!")
        st.image(annotated_img, caption="🧾 Detection Results", use_container_width=True)
