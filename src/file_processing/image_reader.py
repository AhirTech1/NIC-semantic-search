from PIL import Image  # For image processing
import pytesseract  # For OCR
import re  # For cleaning text
import tempfile  # To save the cleaned text temporarily


# Function to clean and preprocess extracted text
def clean_text(raw_text):
    # Remove standalone numbers (like table values)
    cleaned_text = re.sub(r'\n\d+\n', '\n', raw_text)

    # Remove multiple newlines for better formatting
    cleaned_text = re.sub(r'\n{2,}', '\n\n', cleaned_text)

    # Remove common headers or unwanted patterns
    cleaned_text = re.sub(r'Page \d+', '', cleaned_text)
    cleaned_text = re.sub(r'NSS Report no\. \d+:.*?\n', '', cleaned_text)

    # Lowercase the text for consistency
    cleaned_text = cleaned_text.lower()

    # Strip unnecessary spaces
    cleaned_text = cleaned_text.strip()

    return cleaned_text


# Function to extract and clean text from an image using OCR
def extract_and_clean_text_from_image(image_path):
    # Open the image
    image = Image.open(image_path)

    # Extract text using OCR
    extracted_text = pytesseract.image_to_string(image)

    # Clean the extracted text
    cleaned_text = clean_text(extracted_text)

    # Save the cleaned text temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as temp_file:
        temp_file.write(cleaned_text)
        return temp_file.name  # Return the path to the cleaned text file


# Example usage
image_file = '1.png'  # Replace with your image path
cleaned_text_file = extract_and_clean_text_from_image(image_file)
print(f"Cleaned text from image ready for NLP model at: {cleaned_text_file}")
