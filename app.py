from dotenv import load_dotenv

from src.loader import load_pdf
from src.splitter import split_documents
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store
from src.qa_chain import build_qa_chain


load_dotenv()


def main():

    file_path = "data/sample.pdf"

    # 1 Load PDF
    documents = load_pdf(file_path)

    # 2 Split text
    chunks = split_documents(documents)

    # 3 Create embeddings
    embeddings = create_embeddings()

    # 4 Create vector store
    vector_store = create_vector_store(chunks, embeddings)

    # 5 Build RAG chain
    rag_chain = build_qa_chain(vector_store)

    while True:

        question = input("\nAsk a question about the PDF (type exit to quit): ")

        if question.lower() == "exit":
            break

        response = rag_chain.invoke({"input": question})

        print("\nAnswer:\n", response["answer"])


if __name__ == "__main__":
    main()