import os
import re
import torch
from sentence_transformers import SentenceTransformer, util

# Define paths to extracted text files
pdf_text_file = "/home/garuda/PycharmProjects/NIC-semantic-search/src/file_processing/combined_output.txt"  # From pdf_reader.py
image_text_file = "/home/garuda/PycharmProjects/NIC-semantic-search/src/file_processing/combined_image_output.txt"  # From image_reader.py

# Load BERT model
print("Loading BERT model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("BERT model loaded successfully.")

# Function to clean text (removes unwanted characters)
def clean_text(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII characters
    text = re.sub(r"[\uFFFD]", " ", text)  # Remove unknown characters (�)
    text = re.sub(r"\s+", " ", text)  # Normalize spaces
    text = text.replace("", "").replace("", "").replace("o", "of")  # Fix common OCR issues
    return text.strip()

# Function to read and clean text files
def read_text_file(file_path):
    if os.path.exists(file_path):
        print(f"Reading text from: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            return [clean_text(line.strip()) for line in file if line.strip()]  # Clean each line
    else:
        print(f"Warning: {file_path} not found.")
        return []

# Load text as a list of lines
pdf_text = read_text_file(pdf_text_file)
image_text = read_text_file(image_text_file)

# Merge both text sources
all_text = pdf_text + image_text

if not all_text:
    print("No text data available. Exiting program.")
    exit()

# Encode each text line
print("Encoding text in chunks...")
text_embeddings = model.encode(all_text, convert_to_tensor=True)
print(f"Encoded {len(all_text)} text chunks.")

# Get user query
query = input("\nEnter your search query: ")
query_embedding = model.encode(query, convert_to_tensor=True)

# Compute similarity scores
print("Computing similarity scores...")
similarities = util.pytorch_cos_sim(query_embedding, text_embeddings)[0]
top_match_index = torch.argmax(similarities).item()

# Retrieve surrounding lines for better context
window_size = 2  # Number of surrounding lines to include
start_idx = max(0, top_match_index - window_size)
end_idx = min(len(all_text), top_match_index + window_size + 1)

# Display most relevant text with context
print("\nMost relevant text based on your query (cleaned):")
print("\n".join(all_text[start_idx:end_idx]))
