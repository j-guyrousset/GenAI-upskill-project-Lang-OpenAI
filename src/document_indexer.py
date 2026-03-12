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
                doc.metadata["source"] = file

            all_docs.extend(docs)

    return all_docs


def classify_documents(documents):

    file_map = {}

    for doc in documents:

        file_name = doc.metadata["source"]

        if file_name not in file_map:
            file_map[file_name] = doc.page_content

    classifications = {}

    for file_name, text in file_map.items():

        result = classify_document(text)

        classifications[file_name] = result

    return classifications