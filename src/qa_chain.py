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
You are assisting a recruitment team evaluating resumes
for a .NET developer position.

Use the provided resume context to answer the recruiter’s question.

Focus on:
- .NET technologies
- years of experience
- project experience
- Azure or cloud experience
- seniority or leadership roles

If the information is not in the resumes, say you do not know.

Context:
{context}

Recruiter question:
{input}

Answer clearly and concisely.
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