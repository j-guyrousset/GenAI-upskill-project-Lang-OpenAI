from langchain_openai import ChatOpenAI


def classify_document(text):


    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
You are assisting a recruitment team.

Analyze the following resume and extract structured information
about the candidate for a .NET developer role.

Evaluate the candidate on the following criteria:

1. .NET Technical Skills
Look for:
C#, .NET Core, ASP.NET, Web API, Entity Framework.

Score from 0 to 5.

2. Years of Experience
Estimate total professional experience relevant to software development.

Score from 0 to 5:
0 = less than 1 year
1 = 1-2 years
2 = 2-4 years
3 = 4-6 years
4 = 6-10 years
5 = more than 10 years

3. Number of .NET Projects
Estimate number of projects using .NET technologies.

Score from 0 to 5.

4. Complementary Skills
Look for:
Azure, SQL Server, Docker, Microservices, CI/CD, DevOps.

Score from 0 to 5.

5. Seniority Indicators
Look for:
Architecture responsibilities, leadership, mentoring, technical lead roles.

Score from 0 to 5.

Return ONLY valid JSON in the following format:

{{
"candidate_name": "...",
"net_skills_score": number,
"years_experience_score": number,
"dotnet_projects_score": number,
"complementary_skills_score": number,
"seniority_score": number,
"summary": "short explanation"
}}

Resume:
{text[:4000]}
"""

    response = llm.invoke(prompt)

    return response.content