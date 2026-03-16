from langchain_openai import ChatOpenAI


def classify_document(text):


    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
You are analyzing a resume.

Extract structured information about the candidate.

Focus on:
- Programming languages
- Frameworks
- Cloud technologies
- Databases
- DevOps tools
- Total years of experience
- Major projects
- Seniority indicators (lead, architect, manager)

Return ONLY valid JSON with this format:

{{
"candidate_name": "...",
"years_experience": number,
"technical_skills": [],
"frameworks": [],
"cloud_technologies": [],
"databases": [],
"devops_tools": [],
"key_projects": "...",
"seniority_indicators": "...",
"summary": "short summary of the candidate"
}}

Resume:
{text[:4000]}
"""

    response = llm.invoke(prompt)

    return response.content