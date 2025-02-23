import fitz  # PyMuPDF
import tempfile
import re
from PIL import Image  # For image processing
import io  # To handle image byte streams
import pytesseract  # For OCR (Extracting text from images)


# Function to clean and preprocess extracted text
def clean_text(raw_text):
    cleaned_text = re.sub(r'\n\d+\n', '\n', raw_text)
    cleaned_text = re.sub(r'\n{2,}', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'Page \d+', '', cleaned_text)
    cleaned_text = re.sub(r'NSS Report no\. \d+:.*?\n', '', cleaned_text)
    cleaned_text = cleaned_text.lower()
    cleaned_text = cleaned_text.strip()
    return cleaned_text


# Function to extract images from PDF
def extract_images_from_pdf(pdf_path):
    image_paths = []
    with fitz.open(pdf_path) as doc:
        for page_number in range(len(doc)):
            page = doc[page_number]
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Save image temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image_ext}", mode='wb') as img_file:
                    img_file.write(image_bytes)
                    image_paths.append(img_file.name)

    return image_paths


# Function to extract text from images using OCR
def extract_text_from_images(image_paths):
    extracted_text = ""
    for img_path in image_paths:
        image = Image.open(img_path)
        text = pytesseract.image_to_string(image)
        extracted_text += text + "\n"
    return extracted_text


# Extract, clean text and process images from PDF
def extract_and_prepare_text_with_images(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()

    # Clean the extracted text
    cleaned_text = clean_text(text)

    # Extract images and their text
    image_paths = extract_images_from_pdf(pdf_path)
    image_text = extract_text_from_images(image_paths)

    # Combine PDF text and extracted image text
    combined_text = cleaned_text + "\n" + image_text

    # Save the combined text temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as temp_file:
        temp_file.write(combined_text)
        return temp_file.name  # Return path to the clean temp file


# Example usage
pdf_file = 'AnnualReport.pdf'
final_cleaned_file = extract_and_prepare_text_with_images(pdf_file)
print(f"Cleaned text with image data ready for NLP model at: {final_cleaned_file}")
