from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain


def build_qa_chain(vector_store):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant.

Use the provided context to answer the question.

If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{input}
"""
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retrieval_chain = create_retrieval_chain(
        vector_store.as_retriever(),
        document_chain
    )

    return retrieval_chain