import os
import ollama

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

def get_llm_answer(query, text):
    """ Sends the question along with text data to an Ollama LLM """
    print("\n🤖 Asking Ollama LLM...")
    response = ollama.chat(
        model="mistral",  # You can use "gemma", "llama2", or "custom_model"
        messages=[
            {"role": "system", "content": "You are an AI assistant that answers questions based on the provided document."},
            {"role": "user", "content": f"Based on the following document, answer the question:\n\n{text}\n\nQuestion: {query}"}
        ]
    )
    return response["message"]["content"]  # Extract the answer

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

        try:
            # Send the full document to the LLM and get an answer
            answer = get_llm_answer(query, combined_text)
            print("\n💡 Answer:", answer)

        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
