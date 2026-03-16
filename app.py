from dotenv import load_dotenv
import os
#load environment variables from .env file
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

from pathlib import Path

from src.document_indexer import load_all_pdfs, classify_documents
from src.splitter import split_documents
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store
from src.job_analyzer import analyze_job
from src.qa_chain import build_qa_chain
from src.ranker import rank_documents


def main():
    
    print("\n=== AI Resume Analyzer ===\n")

    # --------------------------------------------------
    # Step 1: Load resumes
    # --------------------------------------------------

    print("Loading resumes from /data folder...\n")

    documents = load_all_pdfs("data")

    if not documents:
        print("No PDF files found in /data folder.")
        return

    print(f"{len(documents)} pages loaded.\n")

    # --------------------------------------------------
    # Step 2: Extract candidate profiles
    # --------------------------------------------------

    print("Analyzing candidate resumes...\n")

    classifications = classify_documents(documents)

    print("Candidate profiles extracted:\n")

    for candidate, profile in classifications.items():
        print(f"{candidate}")
        print(profile)
        print("")

    # --------------------------------------------------
    # Step 3: Build RAG system for resume Q&A
    # --------------------------------------------------

    print("Preparing document search system...\n")

    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vector_store = create_vector_store(chunks, embeddings)
    rag_chain = build_qa_chain(vector_store)

    print("System ready.\n")

    # --------------------------------------------------
    # Main user interaction loop
    # --------------------------------------------------

    while True:

        print("\n----------------------------------")
        print("Options:")
        print("1 - Rank candidates for a position")
        print("2 - Ask a question about resumes")
        print("exit - Quit")
        print("----------------------------------")

        choice = input("\nSelect option: ")

        if choice.lower() == "exit":
            print("\nGoodbye.\n")
            break

        # --------------------------------------------------
        # Option 1: Rank candidates for a job
        # --------------------------------------------------

        if choice == "1":

            position = input(
                "\nWhich candidate best fits the position?\n"
            )

            print("\nAnalyzing job requirements...\n")

            job_profile = analyze_job(position)

            print("Job profile:")
            print(job_profile)
            print("")

            print("Ranking candidates...\n")

            ranking = rank_documents(
                position,
                job_profile,
                classifications
            )

            print("\n=== Candidate Ranking ===\n")

            print(ranking)

        # --------------------------------------------------
        # Option 2: Ask questions about resumes
        # --------------------------------------------------

        elif choice == "2":

            question = input(
                "\nEnter your question about the candidates:\n"
            )

            response = rag_chain.invoke({"input": question})

            print("\nAnswer:\n")

            print(response["answer"])

        else:

            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()