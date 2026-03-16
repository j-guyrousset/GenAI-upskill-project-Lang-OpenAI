from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def rank_documents(position, job_profile, classifications):

    prompt = f"""
You are assisting a recruitment team.

The goal is to identify the best candidate for this position:

{position}

Job requirements:
{job_profile}

Candidate profiles:
{classifications}

Evaluate candidates based on:

1. Match between candidate technologies and required technologies
2. Relevant experience
3. Relevant project experience
4. Complementary skills
5. Seniority

Score each candidate from 0 to 100.

Then rank the candidates from best to worst.

Return the result in this format:

1. Candidate Name - Score - Explanation
2. Candidate Name - Score - Explanation
3. Candidate Name - Score - Explanation
"""

    response = llm.invoke(prompt)

    return response.content