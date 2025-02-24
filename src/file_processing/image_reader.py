from PIL import Image
import pytesseract
import re
import tempfile
import os  # For handling directories


# Function to clean and preprocess extracted text
def clean_text(raw_text):
    cleaned_text = re.sub(r'\n\d+\n', '\n', raw_text)
    cleaned_text = re.sub(r'\n{2,}', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'Page \d+', '', cleaned_text)
    cleaned_text = re.sub(r'NSS Report no\. \d+:.*?\n', '', cleaned_text)
    cleaned_text = cleaned_text.lower()
    cleaned_text = cleaned_text.strip()
    return cleaned_text


# Function to extract and clean text from a single image
def extract_and_clean_text_from_image(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    cleaned_text = clean_text(extracted_text)

    # Save cleaned text temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as temp_file:
        temp_file.write(cleaned_text)
        return temp_file.name  # Return file path


# Function to process multiple images from a list or directory
def process_multiple_images(image_paths):
    processed_files = []
    for image_path in image_paths:
        processed_file = extract_and_clean_text_from_image(image_path)
        processed_files.append(processed_file)
        print(f"Processed and cleaned Image: {image_path} -> {processed_file}")
    return processed_files


# Function to combine all processed image files into one
def combine_processed_image_files(processed_files, output_filename="combined_image_output.txt"):
    combined_text = ""
    for file_path in processed_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            combined_text += file.read() + "\n\n"  # Add spacing between files

    # Save combined text into a single file
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(combined_text)

    print(f"All processed image files have been combined into: {output_filename}")
    return output_filename


# Example usage for multiple images
image_directory = 'images'  # Folder containing images
image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
processed_image_files = process_multiple_images(image_files)
# Combine all processed image files into one
final_combined_image_file = combine_processed_image_files(processed_image_files)
