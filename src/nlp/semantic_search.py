from sentence_transformers import SentenceTransformer, util

# Load a pre-trained BERT-based model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_text_embedding(text):
    """
    Convert input text into an embedding using BERT.
    :param text: Text to be converted
    :return: Numerical embedding of the text
    """
    return model.encode(text, convert_to_tensor=True)

def find_most_similar_section(query_embedding, document_embeddings, document_sections):
    """
    Compare the query embedding with document embeddings and return the most relevant section.
    :param query_embedding: Embedding of the user's input query
    :param document_embeddings: List of embeddings from document sections
    :param document_sections: List of corresponding document sections
    :return: Most relevant section based on similarity
    """
    similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)
    best_match_idx = similarities.argmax().item()
    return document_sections[best_match_idx]
