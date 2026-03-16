from langchain_openai import ChatOpenAI


def analyze_job(position):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
You are helping a recruitment team.

Define evaluation criteria for the following position:

Position:
{position}

Return a JSON with:

- key_technologies
- complementary_skills
- typical_experience_level
- important_project_types

Example format:

{{
"key_technologies": [],
"complementary_skills": [],
"experience_expectation": "...",
"project_types": []
}}
"""

    response = llm.invoke(prompt)

    return response.content