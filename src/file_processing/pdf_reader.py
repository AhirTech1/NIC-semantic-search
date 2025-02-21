import pdfplumber

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file.
    :param file_path: Path to the PDF file
    :return: Extracted text as a string
    """
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error processing PDF: {e}")
    return text ##save to a new text file
