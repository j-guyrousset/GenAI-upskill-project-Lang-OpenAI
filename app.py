from dotenv import load_dotenv
#load environment variables from .env file
load_dotenv()
import os
from pathlib import Path

from src.document_indexer import load_all_pdfs, classify_documents
from src.splitter import split_documents
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store
from src.qa_chain import build_qa_chain
from src.ranker import rank_documents


def main():
    print("\n--- PDF Analyzer Started ---\n")

    #file_path = "data/sample_01.pdf"

    # 1 Load all PDFs in the data folder (default arg in load_all_pdfs)
    print("\nLoading pdf files...")
    documents = load_all_pdfs("data")

    if not documents:
        print("No PDF files found in /data folder.")
        return

    print(f"{len(documents)} pages loaded.\n")

    # 2 Classify documents based on user criteria
    print("\nDocument classifications:")
    classifications = classify_documents(documents)

    for file_name, classification in classifications.items():
        print(f"\n{file_name}")
        print(classification)

    # 3 Split Documents into chunks
    print("\nSplitting documents...")
    chunks = split_documents(documents)

    # 4 Create embeddings
    print("Creating embeddings...")
    embeddings = create_embeddings()

    # 5 Create vector store
    print("Building vector database...")
    vector_store = create_vector_store(chunks, embeddings)

    # 6 Build RAG chain
    rag_chain = build_qa_chain(vector_store)

    # 7️ Main interaction loop
    while True:

        print("\n-----------------------------------")
        print("Options:")
        print("1 - Find best resume for .Net developer role")
        print("exit - Quit")
        print("-----------------------------------")

        choice = input("\nSelect option: ")

        if choice.lower() == "exit":
            break

        # Option 1: Document ranking
        if choice == "1":

            criteria ="\nFind the best .NET developer resume\n"

            result = rank_documents(criteria, classifications)

            print("\nBest document match:\n")
            print(result)

        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()