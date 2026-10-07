from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


llm = OllamaLLM(model="llama3.2")


evaluation_prompt = PromptTemplate(
    template="""You are an experienced technical interviewer.

Evaluate the candidate's answer to the interview question.

Question:
{question}

Candidate's Answer:
{answer}

Evaluate the answer on:

1. Correctness
2. Technical Depth
3. Clarity

Give a score out of 10 for each.

Then provide:
- Strengths
- Weaknesses
- Suggestions for improvement
- Overall Score out of 10

Be fair and concise.

Evaluation:""",
    input_variables=["question", "answer"]
)


def evaluate_answer(question, answer):

    final_prompt = evaluation_prompt.format(
        question=question,
        answer=answer
    )

    evaluation = llm.invoke(final_prompt)

    return evaluation