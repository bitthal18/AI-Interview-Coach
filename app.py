import streamlit as st
import os

from src.document_loader import load_pdf
from src.text_splitter import split_documents
from src.vector_store import create_vector_store
from src.matcher import calculate_match_percentage
from src.interview import generate_question
from src.evaluator import evaluate_answer

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎯",
    layout="wide"
)



# =========================================================
# CUSTOM UI STYLING
# =========================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 55%, #172554 100%);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 24px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 18px 50px rgba(0,0,0,0.25);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.05rem;
    }

    .section-card {
        padding: 1.25rem 1.35rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        margin-bottom: 1rem;
    }

    .question-card {
        padding: 1.5rem 1.7rem;
        border-radius: 20px;
        background: rgba(59,130,246,0.10);
        border: 1px solid rgba(96,165,250,0.25);
        margin: 1rem 0 1.2rem 0;
    }

    .question-label {
        color: #93c5fd;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.6rem;
    }

    .question-text {
        font-size: 1.35rem;
        line-height: 1.55;
        font-weight: 650;
    }

    .skill-pill {
        display: inline-block;
        padding: 0.42rem 0.75rem;
        margin: 0.2rem 0.25rem 0.2rem 0;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.10);
        color: #e2e8f0;
        font-size: 0.88rem;
    }

    .upload-note {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.09);
        padding: 1rem;
        border-radius: 16px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        min-height: 2.8rem;
        font-weight: 650;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .report-card {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        line-height: 1.65;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LLM
# =========================================================

llm = OllamaLLM(model="llama3.2")


# =========================================================
# FINAL REPORT PROMPT
# =========================================================


report_prompt = PromptTemplate(
    template="""You are an expert technical interview evaluator.

Analyze the following interview evaluations:

{evaluations}

Create a SHORT and actionable final interview report.

Include ONLY these sections:

1. Overall Score
2. Performance
3. Strengths
4. Weak Areas
5. Key Improvements

Rules:
- Keep the entire report under 120 words.
- Give ONE overall score out of 10.
- Strengths: maximum 2 bullet points.
- Weak Areas: maximum 2 bullet points.
- Key Improvements: maximum 2 bullet points.
- Each bullet must be short and specific.
- Focus only on the candidate's actual performance.
- Do not repeat the questions or candidate answers.
- Do not give lengthy explanations.
- Do not add any extra sections.

Final Report:""",
    input_variables=["evaluations"]
)


def generate_final_report(evaluations):

    evaluation_text = "\n\n".join(
        f"Question {i + 1} Evaluation:\n{evaluation}"
        for i, evaluation in enumerate(evaluations)
    )

    final_prompt = report_prompt.format(
        evaluations=evaluation_text
    )

    report = llm.invoke(final_prompt)

    return report.strip()


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "analysis_done": False,
    "interview_started": False,
    "interview_completed": False,
    "question_number": 0,
    "current_question": "",
    "previous_questions": [],
    "evaluations": [],
    "answers": [],
    "match_percentage": 0,
    "matched_skills": [],
    "missing_skills": [],
    "jd_text": "",
    "resume_text": "",
    "submitted": False,
    "current_evaluation": ""
}


for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# TITLE
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">🎯 AI Interview Coach</div>
    <div class="hero-subtitle">
        Resume analysis • Job matching • Personalized technical interview practice
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# FILE UPLOAD
# =========================================================

st.markdown('<div class="section-card"><h3>📄 Upload Resume</h3><div class="upload-note">PDF format • Your resume is used to personalize the interview</div>', unsafe_allow_html=True)

resume_file = st.file_uploader(
    "Choose resume PDF",
    type=["pdf"],
    key="resume_upload"
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card"><h3>💼 Upload Job Description</h3><div class="upload-note">PDF format • Used for skill matching and interview relevance</div>', unsafe_allow_html=True)

jd_file = st.file_uploader(
    "Choose job description PDF",
    type=["pdf"],
    key="jd_upload"
)
st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# ANALYSIS
# =========================================================

if resume_file and jd_file:

    if st.button("🚀 Analyze Resume & Job", type="primary"):

        os.makedirs("data", exist_ok=True)

        resume_path = "data/uploaded_resume.pdf"
        jd_path = "data/uploaded_jd.pdf"

        # Save Resume
        with open(resume_path, "wb") as f:
            f.write(resume_file.getbuffer())

        # Save JD
        with open(jd_path, "wb") as f:
            f.write(jd_file.getbuffer())


        with st.spinner(
            "Analyzing Resume and Job Description..."
        ):

            # Load PDFs
            resume_documents = load_pdf(
                resume_path
            )

            jd_documents = load_pdf(
                jd_path
            )


            # Extract Resume Text
            resume_text = "\n\n".join(
                doc.page_content
                for doc in resume_documents
            )


            # Extract JD Text
            jd_text = "\n\n".join(
                doc.page_content
                for doc in jd_documents
            )


            # Split Resume
            resume_chunks = split_documents(
                resume_documents
            )


            # Create Vector Store
            create_vector_store(
                resume_chunks
            )


            # Match Resume with JD
            percentage, matched_skills, missing_skills = (
                calculate_match_percentage(
                    resume_text,
                    jd_text
                )
            )


        # Save results
        st.session_state["resume_text"] = resume_text
        st.session_state["jd_text"] = jd_text

        st.session_state["match_percentage"] = percentage

        st.session_state["matched_skills"] = (
            matched_skills
        )

        st.session_state["missing_skills"] = (
            missing_skills
        )


        # Reset Interview
        st.session_state["analysis_done"] = True

        st.session_state["interview_started"] = False

        st.session_state["interview_completed"] = False

        st.session_state["question_number"] = 0

        st.session_state["current_question"] = ""

        st.session_state["previous_questions"] = []

        st.session_state["evaluations"] = []

        st.session_state["answers"] = []

        st.session_state["submitted"] = False

        st.session_state["current_evaluation"] = ""


        st.success(
            "Analysis completed successfully! 🎉"
        )


# =========================================================
# RESUME-JD RESULT
# =========================================================

if st.session_state["analysis_done"]:

    st.divider()

    st.header("📊 Resume–Job Analysis")


    percentage = st.session_state[
        "match_percentage"
    ]

    matched_skills = st.session_state[
        "matched_skills"
    ]

    missing_skills = st.session_state[
        "missing_skills"
    ]


    # Metrics
    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Match Percentage",
            f"{percentage}%"
        )


    with col2:

        st.metric(
            "Matched Skills",
            len(matched_skills)
        )


    with col3:

        st.metric(
            "Missing Skills",
            len(missing_skills)
        )


    # Matched Skills
    st.subheader("✅ Matched Skills")

    if matched_skills:
        pills = "".join(
            f'<span class="skill-pill">{skill}</span>'
            for skill in matched_skills
        )
        st.markdown(pills, unsafe_allow_html=True)
    else:
        st.caption("None")


    # Missing Skills
    st.subheader("⚠️ Missing Skills")

    if missing_skills:
        pills = "".join(
            f'<span class="skill-pill">{skill}</span>'
            for skill in missing_skills
        )
        st.markdown(pills, unsafe_allow_html=True)
    else:
        st.success("All required skills are matched! 🎉")


# =========================================================
# AI INTERVIEW
# =========================================================

if st.session_state["analysis_done"]:

    st.divider()

    st.header("🎤 AI Technical Interview")
    st.caption("Answer naturally. The AI will evaluate correctness, depth, and clarity.")

    # =====================================================
    # START INTERVIEW
    # =====================================================

    if not st.session_state["interview_started"]:

        st.write(
            "Start your personalized technical interview."
        )

        if st.button("▶️ Start AI Interview"):

            with st.spinner(
                "AI is preparing your first question..."
            ):

                question = generate_question(
                    st.session_state["jd_text"],
                    st.session_state["missing_skills"],
                    []
                )

            st.session_state["interview_started"] = True

            st.session_state["question_number"] = 1

            st.session_state["current_question"] = question

            st.session_state["previous_questions"] = [
                question
            ]

            st.session_state["submitted"] = False

            st.rerun()


    # =====================================================
    # INTERVIEW IN PROGRESS
    # =====================================================

    elif not st.session_state["interview_completed"]:

        question_number = st.session_state[
            "question_number"
        ]

        st.markdown(f"""
        <div class="question-card">
            <div class="question-label">Question {question_number}</div>
            <div class="question-text">
                {st.session_state["current_question"]}
            </div>
        </div>
        """, unsafe_allow_html=True)


        # =================================================
        # ANSWER NOT SUBMITTED
        # =================================================

        if not st.session_state["submitted"]:

            answer = st.text_area(
                "💬 Your Answer",

                height=180,
                key=f"answer_{question_number}"
            )

            if st.button(
                "📤 Submit Answer",
                key=f"submit_{question_number}"
            ):

                if not answer.strip():

                    st.warning(
                        "Please enter your answer."
                    )

                else:

                    with st.spinner(
                        "AI is evaluating your answer..."
                    ):

                        evaluation = evaluate_answer(
                            st.session_state[
                                "current_question"
                            ],
                            answer
                        )

                    st.session_state[
                        "answers"
                    ].append(answer)

                    st.session_state[
                        "evaluations"
                    ].append(evaluation)

                    st.session_state[
                        "current_evaluation"
                    ] = evaluation

                    st.session_state[
                        "submitted"
                    ] = True

                    st.rerun()


        # =================================================
        # EVALUATION + NEXT / END
        # =================================================

        else:

            st.subheader("🤖 AI Evaluation")
            st.markdown(
                f'<div class="report-card">{st.session_state["current_evaluation"]}</div>',
                unsafe_allow_html=True
            )

            st.divider()

            col1, col2 = st.columns(2)

            # ---------------------------------------------
            # NEXT QUESTION
            # ---------------------------------------------

            with col1:

                if st.button(
                    "➡️ Next Question",
                    key=f"next_{question_number}"
                ):

                    with st.spinner(
                        "AI is preparing the next question..."
                    ):

                        next_question = generate_question(
                            st.session_state["jd_text"],
                            st.session_state[
                                "missing_skills"
                            ],
                            st.session_state[
                                "previous_questions"
                            ]
                        )

                    st.session_state[
                        "question_number"
                    ] += 1

                    st.session_state[
                        "current_question"
                    ] = next_question

                    st.session_state[
                        "previous_questions"
                    ].append(next_question)

                    st.session_state[
                        "submitted"
                    ] = False

                    st.session_state[
                        "current_evaluation"
                    ] = ""

                    st.rerun()


            # ---------------------------------------------
            # END INTERVIEW
            # ---------------------------------------------

            with col2:

                if st.button(
                    "🛑 End Interview",
                    key=f"end_{question_number}"
                ):

                    st.session_state[
                        "interview_completed"
                    ] = True

                    st.rerun()

# =========================================================
# FINAL REPORT
# =========================================================

if st.session_state["interview_completed"]:

    st.divider()

    st.header("🏆 Final Interview Report")

    with st.spinner("Generating final performance report..."):
        final_report = generate_final_report(
            st.session_state["evaluations"]
        )

    st.markdown(
        f'<div class="report-card">{final_report}</div>',
        unsafe_allow_html=True
    )

    st.success("🎉 Interview completed successfully!")