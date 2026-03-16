import streamlit as st
from app import initialize_pipeline, rank_candidates, ask_question

st.set_page_config(
    page_title="AI Resume Analyzer",
    layout="wide"
)

st.title("AI Resume Analyzer")

st.write(
    "Analyze resumes and rank candidates for a job position."
)

# Initialize pipeline only once
if "pipeline" not in st.session_state:

    with st.spinner("Initializing AI pipeline..."):

        st.session_state.pipeline = initialize_pipeline()

    st.success("System ready")


pipeline = st.session_state.pipeline


# -----------------------------
# Candidate Ranking Section
# -----------------------------

st.header("Candidate Ranking")

position = st.text_input(
    "Enter the position you want to hire for"
)

if st.button("Rank Candidates"):

    with st.spinner("Analyzing candidates..."):

        ranking = rank_candidates(position, pipeline)

    st.subheader("Ranking Result")

    st.write(ranking)


# -----------------------------
# Resume Q&A Section
# -----------------------------

st.header("Ask Questions About Candidates")

question = st.text_input(
    "Example: Which candidate has the most Azure experience?"
)

if st.button("Ask"):

    with st.spinner("Searching resumes..."):

        answer = ask_question(question, pipeline)

    st.write(answer)