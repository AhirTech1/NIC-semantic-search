import os
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Load BERT QA Model (for extracting answers)
print("🔄 Loading QA model...")
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
print("✅ QA model loaded.")

# Load BERT-based sentence transformer (for similarity search)
print("🔄 Loading similarity model...")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lighter model for faster execution
print("✅ Similarity model loaded.")

# Paths to extracted text files
pdf_text_path = "/home/garuda/PycharmProjects/NIC-semantic-search/src/file_processing/combined_output.txt"
image_text_path = "/home/garuda/PycharmProjects/NIC-semantic-search/src/file_processing/combined_image_output.txt"


def read_text_file(file_path):
    """ Reads a text file and returns its content as a string """
    print(f"📂 Reading text file: {file_path}")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().strip()
            print(f"📜 Loaded {len(content)} characters from {file_path}")
            return content
    else:
        print(f"❌ File not found: {file_path}")
        return ""


def chunk_text(text, max_length=1000):
    """ Splits text into larger chunks to improve retrieval accuracy """
    print(f"✂️ Chunking text (Max length per chunk: {max_length} characters)")
    sentences = text.split(". ")
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    print(f"✅ Created {len(chunks)} chunks.")
    return chunks


def retrieve_relevant_context(query, text, top_k=3, min_similarity=0.4):
    """ Retrieves the most relevant paragraph using semantic similarity """
    print("\n🔍 Retrieving relevant context...")
    chunks = chunk_text(text)  # Split text into manageable chunks

    if not chunks:
        print("⚠️ No chunks available to search.")
        return ""

    print("🔄 Encoding query...")
    query_embedding = similarity_model.encode(query, convert_to_tensor=True)

    print("🔄 Encoding document chunks...")
    chunk_embeddings = similarity_model.encode(chunks, convert_to_tensor=True)

    print("🔄 Calculating similarities...")
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]

    # Get top_k most relevant chunks
    top_indices = torch.topk(similarities, top_k).indices.tolist()

    # Filter out chunks with low similarity scores
    filtered_chunks = []
    for idx in top_indices:
        score = similarities[idx].item()
        if score >= min_similarity:
            print(f"✅ Chunk {idx} selected (Similarity: {score:.2f})")
            filtered_chunks.append(chunks[idx])
        else:
            print(f"⚠️ Chunk {idx} ignored (Low similarity: {score:.2f})")

    if not filtered_chunks:
        print("❌ No relevant context found above similarity threshold.")
        return ""

    return " ".join(filtered_chunks)


def answer_question(question, context):
    """ Uses the BERT QA model to extract the answer from the most relevant context """
    print("\n🤖 Answering the question using the QA model...")
    response = qa_pipeline(question=question, context=context)
    print(f"✅ Model response: {response}")
    return response["answer"]


def main():
    print("\n🔄 Loading text data...")

    # Read extracted text from files
    pdf_text = read_text_file(pdf_text_path)
    image_text = read_text_file(image_text_path)

    # Combine both sources
    combined_text = pdf_text + "\n" + image_text

    if not combined_text.strip():
        print("❌ No text data found! Please check if your text extraction process worked correctly.")
        return

    print("✅ Text data loaded successfully. Ready for queries.")

    while True:
        query = input("\nEnter your search query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("🚪 Exiting...")
            break

        print("\n🔍 Finding relevant information...")

        # Retrieve the most relevant context
        try:
            context = retrieve_relevant_context(query, combined_text)
            if not context.strip():
                print("⚠️ No relevant context found. Try rephrasing your question.")
                continue

            print("\n📖 Most Relevant Context:\n", context)

            # Answer the question using the QA model
            answer = answer_question(query, context)
            print("\n💡 Answer:", answer)

        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
