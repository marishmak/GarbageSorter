# app.py
import streamlit as st
import requests


# FastAPI endpoint
FASTAPI_URL = "http://fastapi:8001/predict"

# Title and instructions for the app
st.title("Garbage Sorter App")
st.write("""
This app predicts the garbage class based on input image.""")

uploaded_file = st.file_uploader("Choose an image...", type="png")

# Button to trigger prediction
if uploaded_file is not None:

    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.')

    # Send the image to the FastAPI backend
    files = {"file": uploaded_file.getvalue()}

    
    # Send a request to the FastAPI prediction endpoint
    response = requests.post(url=FASTAPI_URL, files=files)
    caption = response.json()["caption"]

    st.success(f"The garbage is classified as: {caption}")

