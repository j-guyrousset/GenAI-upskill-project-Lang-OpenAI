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
from src.job_skill_extractor import extract_job_skills
from src.qa_chain import build_qa_chain
from src.ranker import rank_documents


def initialize_pipeline():

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in .env")

    print("Loading resumes...")

    documents = load_all_pdfs("data")

    print("Classifying candidates...")

    classifications = classify_documents(documents)

    print("Building vector database...")

    chunks = split_documents(documents)

    embeddings = create_embeddings()

    vector_store = create_vector_store(chunks, embeddings)

    rag_chain = build_qa_chain(vector_store)

    return {
        "documents": documents,
        "classifications": classifications,
        "vector_store": vector_store,
        "rag_chain": rag_chain
    }


def rank_candidates(position, pipeline):

    job_profile = analyze_job(position)

    skills = extract_job_skills(position)

    query = " ".join(skills)

    retrieved_docs = pipeline["vector_store"].similarity_search(query, k=20)

    candidate_set = set()

    for doc in retrieved_docs:
        candidate_set.add(doc.metadata["candidate"])

    filtered_profiles = {
        k: v for k, v in pipeline["classifications"].items()
        if k in candidate_set
    }

    ranking = rank_documents(
        position,
        job_profile,
        filtered_profiles
    )

    return ranking


def ask_question(question, pipeline):

    response = pipeline["rag_chain"].invoke({"input": question})

    return response["answer"]