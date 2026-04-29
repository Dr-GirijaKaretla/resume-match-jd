import os
import io
import streamlit as st
import PyPDF2
import docx
import google.generativeai as genai

# -----------------------------------------------------------
# 1. Configure Gemini
# -----------------------------------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def call_llm(prompt, temperature=0.3):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


# -----------------------------------------------------------
# 2. File parsing helpers
# -----------------------------------------------------------
def extract_text_from_pdf(file_obj):
    reader = PyPDF2.PdfReader(file_obj)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def extract_text_from_docx(file_obj):
    doc = docx.Document(file_obj)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text(uploaded_file):
    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()
    data = uploaded_file.read()
    file_obj = io.BytesIO(data)

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file_obj)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file_obj)
    else:
        return data.decode("utf-8", errors="ignore")


# -----------------------------------------------------------
# 3. Main pipeline
# -----------------------------------------------------------
def process(resume_file, jd_file):
    if resume_file is None or jd_file is None:
        return "Please upload both files.", "", ""

    resume_text = extract_text(resume_file)
    jd_text = extract_text(jd_file)

    # A. Recruiter analysis
    analysis_prompt = f"""
You are a senior technical recruiter.

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"

CANDIDATE RESUME:
\"\"\"{resume_text}\"\"\"

Tasks:
- Give a MATCH SCORE (0–100)
- Provide justification
- List strengths
- List gaps
- Suggest improvements
"""
    analysis = call_llm(analysis_prompt)

    # B. Tailored resume
    resume_prompt = f"""
Rewrite the resume to align with the job description.
Do NOT invent experience.
Output ATS‑friendly sections.

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"

ORIGINAL RESUME:
\"\"\"{resume_text}\"\"\"
"""
    tailored_resume = call_llm(resume_prompt)

    # C. Cover letter
    cover_prompt = f"""
Write a professional cover letter based on:

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"

TAILORED RESUME:
\"\"\"{tailored_resume}\"\"\"
"""
    cover_letter = call_llm(cover_prompt)

    return analysis, tailored_resume, cover_letter


# -----------------------------------------------------------
# 4. Streamlit UI
# -----------------------------------------------------------
st.title("📄 Resume–Job Match Assistant (Gemini)")

st.write("Upload your resume and job description to generate:")
st.write("- Recruiter analysis")
st.write("- Tailored resume")
st.write("- Cover letter")

resume_file = st.file_uploader("Upload Resume (PDF/DOCX/TXT)")
jd_file = st.file_uploader("Upload Job Description (PDF/DOCX/TXT)")

if st.button("🚀 Analyze & Generate"):
    analysis, tailored_resume, cover_letter = process(resume_file, jd_file)

    st.subheader("🔍 Recruiter Analysis")
    st.write(analysis)

    st.subheader("📝 Tailored Resume")
    st.text_area("", tailored_resume, height=300)

    st.subheader("✉️ Cover Letter")
    st.text_area("", cover_letter, height=300)
