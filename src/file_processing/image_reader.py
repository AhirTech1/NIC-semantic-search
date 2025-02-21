import pytesseract
from PIL import Image

def extract_text_from_image(image_path):
    """
    Extract text from an image using OCR.
    :param image_path: Path to the image file
    :return: Extracted text as a string
    """
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
    except Exception as e:
        print(f"Error processing image: {e}")
        text = ""
    return text
