from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import json
import re


llm = OllamaLLM(model="llama3.2")


skill_extraction_prompt = PromptTemplate(
    template="""You are a job description analyzer.

Extract the technical skills, technologies, tools, frameworks,
databases, programming languages, and important technical concepts
explicitly required in the following job description.

Job Description:
{jd}

Rules:
- Extract only skills explicitly mentioned in the job description.
- Do not invent skills.
- Do not include soft skills.
- Do not include job titles.
- Return only a JSON list of strings.

Example format:
["Python", "React", "MongoDB"]

Skills:""",
    input_variables=["jd"]
)


recommendation_prompt = PromptTemplate(
    template="""You are an AI career assistant.

Matched Skills:
{matched}

Missing Skills:
{missing}

Give preparation recommendations.

Rules:
- If Missing Skills is "None", DO NOT mention any missing skill.
- Do not recommend learning or revising a skill that is already matched.
- Only recommend preparation for skills explicitly listed in Missing Skills.
- For matched skills, only suggest ways to demonstrate or strengthen practical knowledge.
- Do not invent any skills.
- Keep the answer concise.

Preparation Recommendations:""",
    input_variables=["matched", "missing"]
)


def normalize_text(text):
    return re.sub(r"[^a-z0-9+#.]", " ", text.lower())


def extract_skills_from_jd(jd):

    final_prompt = skill_extraction_prompt.format(
        jd=jd
    )

    result = llm.invoke(final_prompt).strip()

    # Remove markdown code fences if LLM adds them
    result = result.replace("```json", "")
    result = result.replace("```", "")
    result = result.strip()

    try:
        skills = json.loads(result)

        if isinstance(skills, list):
            return [
                str(skill).strip()
                for skill in skills
                if str(skill).strip()
            ]

    except json.JSONDecodeError:
        pass

    return []


def calculate_match_percentage(resume, jd):

    resume_text = normalize_text(resume)

    # Extract skills dynamically from JD
    required_skills = extract_skills_from_jd(jd)

    matched_skills = []
    missing_skills = []

    for skill in required_skills:

        normalized_skill = normalize_text(skill).strip()

        if normalized_skill and normalized_skill in resume_text:
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    if len(required_skills) == 0:
        percentage = 0
    else:
        percentage = round(
            (len(matched_skills) / len(required_skills)) * 100
        )

    return percentage, matched_skills, missing_skills


def match_resume_with_jd(resume, jd):

    percentage, matched, missing = calculate_match_percentage(
        resume,
        jd
    )

    matched_text = ", ".join(matched) if matched else "None"
    missing_text = ", ".join(missing) if missing else "None"

    if missing:

        final_prompt = recommendation_prompt.format(
            matched=matched_text,
            missing=missing_text
        )

        recommendations = llm.invoke(final_prompt)

    else:

        recommendations = (
            "All required skills are matched. "
            "Focus on strengthening practical implementation "
            "and interview communication."
        )

    result = f"""
==============================
    RESUME-JD MATCHING
==============================

1. Matching Skills

{matched_text}

2. Missing Skills

{missing_text}

3. Overall Match Percentage

{percentage}%

4. Preparation Recommendations

{recommendations.strip()}
"""

    return result.strip()