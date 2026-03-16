from langchain_openai import ChatOpenAI


def extract_job_skills(position):

    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = f"""
Extract the main technical skills required for this job.

Position:
{position}

Return a JSON list of skills.

Example:
["C#", ".NET Core", "Azure"]
"""

    response = llm.invoke(prompt)

    return response.content