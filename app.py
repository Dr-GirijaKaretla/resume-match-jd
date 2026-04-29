import os
import io
import gradio as gr
import PyPDF2
import docx
from groq import Groq

# ============================================================
# 1. LOAD API KEY (Colab Secrets or Environment Variable)
# ============================================================

# In Colab, you will set:
# os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ============================================================
# 2. LLM CALL FUNCTION (Groq – Llama3‑70B)
# ============================================================

def call_llm(prompt, temperature=0.3):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


# ============================================================
# 3. FILE PARSING HELPERS
# ============================================================

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

    with open(uploaded_file.name, "rb") as f:
        data = f.read()

    file_obj = io.BytesIO(data)

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file_obj)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file_obj)
    else:
        return data.decode("utf-8", errors="ignore")


# ============================================================
# 4. MAIN PIPELINE: ANALYSIS → TAILORED RESUME → COVER LETTER
# ============================================================

def process_resume_and_jd(resume_file, jd_file):
    if resume_file is None or jd_file is None:
        return (
            "⚠️ Please upload both resume and job description.",
            "",
            "",
        )

    resume_text = extract_text(resume_file)
    jd_text = extract_text(jd_file)

    # --------------------------------------------------------
    # A. Recruiter Analysis
    # --------------------------------------------------------
    analysis_prompt = f"""
You are a senior technical recruiter.

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"

CANDIDATE RESUME:
\"\"\"{resume_text}\"\"\"

Tasks:
- Give an overall MATCH SCORE from 0 to 100.
- Provide a short justification.
- List KEY STRENGTHS.
- List SKILL / EXPERIENCE GAPS.
- Provide 5–10 resume improvement suggestions.

Format:

MATCH SCORE: <number>/100

JUSTIFICATION:
- ...

STRENGTHS:
- ...

GAPS:
- ...

RECOMMENDED IMPROVEMENTS:
- ...
"""
    analysis = call_llm(analysis_prompt)

    # --------------------------------------------------------
    # B. Tailored Resume
    # --------------------------------------------------------
    tailored_resume_prompt = f"""
You are a senior recruiter and resume writer.

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"

ORIGINAL RESUME:
\"\"\"{resume_text}\"\"\"

Rewrite the resume so it aligns strongly with the job description.
Rules:
- Do NOT invent fake experience.
- You may reorganize, rephrase, emphasize relevant skills.
- Remove irrelevant details.
- Output an ATS‑friendly resume with sections:
  SUMMARY, SKILLS, EXPERIENCE, EDUCATION, PROJECTS (optional)

Output only the rewritten resume.
"""
    tailored_resume = call_llm(tailored_resume_prompt)

    # --------------------------------------------------------
    # C. Cover Letter
    # --------------------------------------------------------
    cover_letter_prompt = f"""
You are a senior recruiter and expert cover letter writer.

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"

TAILORED RESUME:
\"\"\"{tailored_resume}\"\"\"

Write a professional one‑page cover letter:
- Address to "Dear Hiring Manager,"
- Highlight the most relevant strengths
- Do not repeat the resume line‑by‑line
- Use a confident, natural tone

Output only the cover letter.
"""
    cover_letter = call_llm(cover_letter_prompt)

    return analysis, tailored_resume, cover_letter


# ============================================================
# 5. GRADIO UI
# ============================================================

with gr.Blocks(title="Resume–Job Match Assistant (Groq)") as demo:
    gr.Markdown(
        """
# 🔍 Resume–Job Match Assistant  
Powered by **Groq Llama3‑70B**

Upload your **resume** and **job description**.  
The system will:
- Analyze match as a **senior recruiter**
- Generate a **tailored resume**
- Create a **custom cover letter**
        """
    )

    with gr.Row():
        resume_input = gr.File(label="Upload Resume (PDF/DOCX/TXT)")
        jd_input = gr.File(label="Upload Job Description (PDF/DOCX/TXT)")

    run_button = gr.Button("🚀 Analyze & Generate")

    analysis_output = gr.Markdown(label="Recruiter Analysis")
    tailored_resume_output = gr.Textbox(label="Tailored Resume", lines=20)
    cover_letter_output = gr.Textbox(label="Cover Letter", lines=20)

    run_button.click(
        fn=process_resume_and_jd,
        inputs=[resume_input, jd_input],
        outputs=[analysis_output, tailored_resume_output, cover_letter_output],
    )

if __name__ == "__main__":
    demo.launch(share=True)
