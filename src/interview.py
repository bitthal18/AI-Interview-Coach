from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from src.retriever import create_retriever


llm = OllamaLLM(model="llama3.2")

retriever = create_retriever()


question_prompt = PromptTemplate(
    template="""You are conducting a technical interview for a fresher or junior candidate.

Candidate Resume:
{resume_context}

Job Description:
{jd}

Missing Skills:
{missing_skills}

Topics already asked:
{previous_questions}

Your task:

1. Identify the important technical skills, technologies, concepts,
   tools, and project technologies from the resume.
2. Select ONE topic that has NOT been tested in the previous questions.
3. Generate ONE interview question about that topic.

Rules:
- Questions must be based on the candidate's resume.
- Prioritize technical skills and concepts.
- Projects can be used as supporting context.
- Do not make every question project-related.
- Do not focus repeatedly on one programming language.
- Select a different topic whenever possible.
- Do not repeat or rephrase previous questions.
- Difficulty should be EASY to MODERATE.
- Ask only ONE concept at a time.
- Keep the question within 15 words.
- Do not ask about unsupported technologies.
- Return ONLY the question.
- Do not write "Question:".
- Do not provide an answer or explanation.

Interview Question:""",
    input_variables=[
        "resume_context",
        "jd",
        "missing_skills",
        "previous_questions"
    ]
)


def normalize_question(question):
    return (
        question.lower()
        .replace("?", "")
        .replace(".", "")
        .replace(",", "")
        .strip()
    )


def generate_question(
    jd,
    missing_skills,
    previous_questions=None
):

    if previous_questions is None:
        previous_questions = []

    # Retrieve broader resume context
    documents = retriever.invoke(
        "skills technologies programming languages frameworks "
        "databases tools concepts projects"
    )

    resume_context = "\n\n".join(
        doc.page_content for doc in documents
    )

    missing_text = (
        ", ".join(missing_skills)
        if missing_skills
        else "None"
    )

    previous_text = (
        "\n".join(previous_questions)
        if previous_questions
        else "None"
    )

    final_prompt = question_prompt.format(
        resume_context=resume_context,
        jd=jd,
        missing_skills=missing_text,
        previous_questions=previous_text
    )

    for attempt in range(5):

        question = llm.invoke(final_prompt).strip()

        # Remove unwanted prefix
        if question.lower().startswith("question:"):
            question = question[len("question:"):].strip()

        # Keep only first line
        question = question.split("\n")[0].strip()

        normalized_question = normalize_question(question)

        # Normalize previous questions
        normalized_previous = [
            normalize_question(q)
            for q in previous_questions
        ]

        word_count = len(question.split())

        # Validate
        if (
            question
            and word_count <= 15
            and normalized_question not in normalized_previous
        ):
            return question

        # Tell LLM to generate something different
        final_prompt += """

The generated question was not acceptable.

Generate a completely different question.

Choose a different technical topic from the candidate's resume.

Do NOT repeat any previous topic or question.

Return ONLY ONE question under 15 words.
"""

    return question