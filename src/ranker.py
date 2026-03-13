from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def rank_documents(criteria, classifications):

    prompt = f"""
You are assisting a recruitment team selecting a .NET developer.

The candidates have already been analyzed and scored.

Use the following weighting model:

.NET skills weight = 35%
Experience weight = 25%
.NET projects weight = 20%
Complementary skills weight = 10%
Seniority weight = 10%

Calculate a final score for each candidate using:

Total Score =
(.NET Skills * 0.35) +
(Experience * 0.25) +
(Projects * 0.20) +
(Complementary Skills * 0.10) +
(Seniority * 0.10)

Then rank all candidates from best to worst.

Explain briefly why the top candidate is the best match.

Recruitment criteria:
{criteria}

Candidate data:
{classifications}

Return your answer in this format:

1. Candidate Name - Score - Explanation
2. Candidate Name - Score - Explanation
3. Candidate Name - Score - Explanation
"""

    response = llm.invoke(prompt)

    return response.content