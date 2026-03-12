from langchain_openai import ChatOpenAI


def classify_document(text):


    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
Classify this document according to:

- Document Type
- Main Topic
- Industry
- Risk Level (Low, Medium, High)

Return JSON.

Document:
{text[:3000]}
"""

    response = llm.invoke(prompt)

    return response.content