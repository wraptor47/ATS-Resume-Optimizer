import streamlit as st
import pandas as pd
import os
from ats_core import ATSOptimizer, extract_text_from_pdf, extract_skills_from_text  # Use actual module name

# Load resume data and initialize model
@st.cache(allow_output_mutation=True)
def load_model_and_data():
    ats = ATSOptimizer()
    resume_data = pd.read_csv("resume_data_new.csv")
    ats.prepare_data(resume_data)
    ats.cluster_skills(n_clusters=10)
    ats.create_resume_vectors(resume_data)
    return ats

ats = load_model_and_data()

st.title("📄 ATS Resume Scorer")

# Upload resume
uploaded_resume = st.file_uploader("Upload your Resume (PDF)", type=['pdf'])

# Input job description
jd_text = st.text_area("Paste Job Description Here")

if uploaded_resume and jd_text:
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_resume.read())
    resume_text = extract_text_from_pdf("temp_resume.pdf")
    resume_skills = extract_skills_from_text(resume_text, known_skills=ats.unique_skills)
    jd_skills = extract_skills_from_text(jd_text, known_skills=ats.unique_skills)

    result = ats.process_single_resume(resume_skills, jd_skills)

    st.subheader("📊 ATS Match Score")
    st.metric("Resume Score", f"{result['confidence_score']:.2f}")
    st.text(f"Predicted Domain: {result['domain']}")
    st.success(result['message'])

    st.subheader("✅ Extracted Skills")
    st.write("**From Resume:**", resume_skills)
    st.write("**From JD:**", jd_skills)
