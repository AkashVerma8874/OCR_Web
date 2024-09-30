import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np

# Function to extract text from an image
def extract(im):
    # Convert the image to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Use pytesseract to do OCR on the images
    text = pytesseract.image_to_string(gray)
    return text

# Function to highlight keywords in text
def highlight_keywords(text, keywords):
    highlighted_text = text
    for keyword in keywords:
        if keyword:
            highlighted_text = highlighted_text.replace(keyword, f"<mark>{keyword}</mark>")
    return highlighted_text

# Streamlit app
st.title("OCR Image Text Extraction and Keyword Search")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Convert image to a format suitable for OCR
    image_cv = np.array(image)
    
    # Extract text from the image
    extracted_text = extract(image_cv)
    
    # Display extracted text
    st.subheader("Extracted Text:")
    st.text_area("Text", extracted_text, height=300)

    # Input for keywords to search
    keywords_input = st.text_input("Enter keywords to search (comma-separated):")
    keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]

    # Highlight keywords in the extracted text
    if keywords:
        highlighted_text = highlight_keywords(extracted_text, keywords)
        st.subheader("Search Results:")
        st.markdown(highlighted_text, unsafe_allow_html=True)