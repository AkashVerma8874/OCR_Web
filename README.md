# Project Report

OCR Image Text Extraction and Keyword Search Application
Overview
This project involves the development of a web application using Streamlit that allows users to upload images and extract text from them using Optical Character Recognition (OCR). The application also enables users to search for specific keywords in the extracted text and highlights these keywords for better visibility.




## Features
Image Upload: Users can upload images in JPEG or PNG formats.
Text Extraction: The application uses Tesseract OCR to extract text from the uploaded images.
Keyword Search: Users can input keywords, and the application will highlight these keywords in the extracted text.
User-Friendly Interface: Built with Streamlit for a seamless user experience.
Technology Stack
Python: Programming language used for backend development.
Streamlit: A Python library to create web applications quickly and easily.
Pytesseract: Python wrapper for Google’s Tesseract-OCR Engine.
OpenCV: A library for computer vision tasks.
Pillow (PIL): Python Imaging Library to handle image processing.
NumPy: Library for numerical operations on arrays.
Code Overview
The main components of the code are as follows:
## Importing Libraries

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
## Importing Libraries

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
## Conclusion

This project demonstrates the effective use of Python and various libraries to create an interactive web application for OCR text extraction and keyword search. The user-friendly interface and robust functionality make it a useful tool for anyone needing to process and analyze text from images.
## Future Enhancements
Support for Aditional Languages: Implementing multi-language support in Tesseract.
Image Preprocessing: Adding preprocessing steps to improve OCR accuracy (e.g., image resizing, denoising).
Exporting Results: Allowing users to download the extracted text and highlighted results.
Mobile Responsiveness: Improving the app's design for better accessibility on mobile devices.
