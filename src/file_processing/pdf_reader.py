import fitz  # PyMuPDF for PDF extraction
import tempfile
import re  # For initial cleaning
from transformers import LongformerTokenizer, LongformerForMaskedLM
import torch

# Load Longformer tokenizer and model
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Function to clean and preprocess extracted text (basic)
def clean_text(raw_text):
    cleaned_text = re.sub(r'\n\d+\n', '\n', raw_text)  # Remove standalone numbers
    cleaned_text = re.sub(r'\n{2,}', '\n\n', cleaned_text)  # Remove multiple newlines
    cleaned_text = re.sub(r'Page \d+', '', cleaned_text)  # Remove page numbers
    cleaned_text = re.sub(r'NSS Report no\. \d+:.*?\n', '', cleaned_text)  # Remove report references
    cleaned_text = cleaned_text.lower()  # Lowercase text
    cleaned_text = cleaned_text.strip()  # Remove unnecessary spaces
    return cleaned_text


# Function to split large text into smaller manageable chunks
def split_text(text, max_length=4000):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length):
        chunk = ' '.join(words[i:i + max_length])
        chunks.append(chunk)
    return chunks


# Function to further process and clean text using Longformer
def clean_text_with_longformer(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=4096, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
    cleaned_text = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
    return cleaned_text


# Extract, clean, and prepare text from PDF using Longformer
def extract_and_prepare_text_for_nlp(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()

    # Basic initial cleaning
    cleaned_text = clean_text(text)

    # Split into manageable chunks for Longformer
    text_chunks = split_text(cleaned_text)

    # Process each chunk using Longformer for deeper cleaning
    final_cleaned_chunks = [clean_text_with_longformer(chunk) for chunk in text_chunks]

    # Combine all cleaned chunks back together
    final_cleaned_text = ' '.join(final_cleaned_chunks)

    # Save the cleaned text to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as temp_file:
        temp_file.write(final_cleaned_text)
        return temp_file.name  # Return path to the cleaned temporary file


# Example usage
cleaned_file_path = extract_and_prepare_text_for_nlp('AnnualReport.pdf')
print(f"Final cleaned text ready for NLP model at: {cleaned_file_path}")
