# OCR_web
# Develop and deploy a web-based prototype that demonstrates the ability to perform Optical
# Character Recognition (OCR) on an uploaded image (in picture format) containing text in both
# Hindi and English. The web application should also implement a basic keyword search
#functionality based on the extracted text. The prototype must be accessible via a live URL.
#Set up the Environment

#Install the required libraries using pip.

#pip install pytesseract opencv-python pillow numpy matplotlib
#For Tesseract, you'll need to install the Tesseract executable separately, as it is not a Python library.

#Linux: Install via package manager
#sudo apt install tesseract-ocr
#After installation, configure the Tesseract executable in your Python script:

#python
#Copy code
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
#Read and Preprocess the Image Use OpenCV to read the image and perform preprocessing like resizing, thresholding, etc., to improve OCR results.
import cv2

# Load image
image = cv2.imread('sample_image.png')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding (optional, but often improves results)
_, threshold_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

# Show the image (for debugging purposes)
cv2.imshow('Threshold Image', threshold_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
Run OCR using Tesseract After preprocessing, run OCR on the image using Tesseract.

python
Copy code
from PIL import Image
import pytesseract

# Convert the image to a PIL format for Tesseract
pil_image = Image.fromarray(threshold_image)

# Perform OCR
text = pytesseract.image_to_string(pil_image)
print("Extracted Text:", text)
Post-Processing the OCR Output You can clean up the extracted text by removing unwanted characters, formatting, or applying any NLP techniques.

python
Copy code
import re

# Simple text cleanup
cleaned_text = re.sub(r'\W+', ' ', text)  # Remove non-word characters
print("Cleaned Text:", cleaned_text)
Enhancements (Optional)

Noise Removal: Use filters like Gaussian blur to remove noise from images.
python
Copy code
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
Edge Detection: You can use edge detection techniques like Canny edge detection for preprocessing:
python
Copy code
edges = cv2.Canny(gray_image, 100, 200)
Sample Full Code
Here is a sample code that integrates all the steps mentioned:

python
Copy code
import cv2
import pytesseract
from PIL import Image
import re

# Configure Tesseract path (only needed for Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load and preprocess the image
image = cv2.imread('sample_image.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, threshold_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

# Convert to PIL image for Tesseract OCR
pil_image = Image.fromarray(threshold_image)

# Perform OCR
text = pytesseract.image_to_string(pil_image)

# Post-process the OCR result
cleaned_text = re.sub(r'\W+', ' ', text)

# Display the extracted text
print("Extracted Text:", cleaned_text)
Additional Libraries
If you need more advanced features like PDF to image conversion or deep learning-based OCR, you can add the following libraries:

pdf2image: For converting PDF files to images
bash
Copy code
pip install pdf2image
keras-ocr: For deep learning-based OCR
bash
Copy code
pip install keras-ocr
