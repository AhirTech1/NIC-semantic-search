import fitz  # PyMuPDF
import tempfile
import re
import os  # For handling directories
from transformers import LongformerTokenizer, LongformerForMaskedLM
import torch


# Initialize Longformer tokenizer and model
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to clean text using basic regex first
def basic_clean_text(raw_text):
    cleaned_text = re.sub(r'\n\d+\n', '\n', raw_text)
    cleaned_text = re.sub(r'\n{2,}', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'Page \d+', '', cleaned_text)
    cleaned_text = re.sub(r'NSS Report no\. \d+:.*?\n', '', cleaned_text)
    cleaned_text = cleaned_text.lower()
    cleaned_text = cleaned_text.strip()
    return cleaned_text

# Function to clean text using Longformer and handle device correctly
def clean_text_with_longformer(text, device):
    inputs = tokenizer(text, return_tensors="pt", max_length=4096, truncation=True).to(device)  # Move inputs to GPU/CPU
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    cleaned_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return cleaned_text

# Function to extract text and clean using Longformer from a single PDF
def extract_and_clean_text_from_pdf(pdf_path, device):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()

    # Basic text cleaning
    basic_cleaned_text = basic_clean_text(text)

    # Further clean using Longformer
    longformer_cleaned_text = clean_text_with_longformer(basic_cleaned_text, device)

    # Save cleaned text temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as temp_file:
        temp_file.write(longformer_cleaned_text)
        return temp_file.name  # Return file path

# Function to process multiple PDFs using Longformer
def process_multiple_pdfs(pdf_paths, device):
    processed_files = []
    for pdf_path in pdf_paths:
        processed_file = extract_and_clean_text_from_pdf(pdf_path, device)
        processed_files.append(processed_file)
        print(f"Processed and cleaned PDF with Longformer: {pdf_path} -> {processed_file}")
    return processed_files

# Function to combine all processed files into one
def combine_processed_files(processed_files, output_filename="combined_output.txt"):
    combined_text = ""
    for file_path in processed_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            combined_text += file.read() + "\n\n"  # Add spacing between files

    # Save combined text into a single file
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(combined_text)

    print(f"All processed files have been combined into: {output_filename}")
    return output_filename

# Example usage for multiple PDFs
pdf_directory = 'pdfs'  # Folder containing PDFs
pdf_files = [os.path.join(pdf_directory, file) for file in os.listdir(pdf_directory) if file.endswith('.pdf')]
processed_pdf_files = process_multiple_pdfs(pdf_files, device)
# Combine all processed files into one
final_combined_file = combine_processed_files(processed_pdf_files)
print("Final output file : ",final_combined_file)