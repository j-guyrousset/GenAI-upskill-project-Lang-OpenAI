from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def rank_documents(criteria, classifications):

    prompt = f"""
User criteria:
{criteria}

Available documents:
{classifications}

Return the best matching file name and explanation.
"""

    response = llm.invoke(prompt)

    return response.content