import os
from src.loader import load_pdf
from src.classifier import classify_document


def load_all_pdfs(folder_path="data"):

    all_docs = []

    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            path = os.path.join(folder_path, file)

            docs = load_pdf(path)

            for doc in docs:
                doc.metadata["candidate"] = file
                doc.metadata["source"] = path

            all_docs.extend(docs)

    return all_docs


def classify_documents(documents):

    """
    Analyze each candidate resume using the LLM classifier
    and return structured candidate profiles.
    """

    candidate_texts = {}

    # Group pages belonging to the same candidate
    for doc in documents:

        candidate = doc.metadata["candidate"]

        if candidate not in candidate_texts:
            candidate_texts[candidate] = ""

        candidate_texts[candidate] += doc.page_content + "\n"

    classifications = {}

    for candidate, text in candidate_texts.items():

        print(f"Classifying {candidate}...")

        profile = classify_document(text)

        classifications[candidate] = profile

    return classifications