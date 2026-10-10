import json
import re

from google import genai


MODEL_NAME = "gemini-3.5-flash-lite"


def load_gemini_client(api_key):
    return genai.Client(api_key=api_key)


def generate_answer(client, question, retrieved_chunks):
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"Source: {chunk['source']}, Page {chunk['page']}\n"
            f"{chunk['text']}"
        )

    context = "\n\n--- SOURCE ---\n\n".join(context_parts)

    prompt = f"""
You are an academic study assistant.

Answer the student's question using the retrieved
study material below.

If the answer is not available in the supplied material,
clearly say that it was not found.

Use simple, exam-friendly language.

QUESTION:
{question}

STUDY MATERIAL:
{context}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text or "No answer was generated."


def evaluate_student_answer(
    client,
    question,
    student_answer,
    retrieved_chunks
):
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"Source: {chunk['source']}, Page {chunk['page']}\n"
            f"{chunk['text']}"
        )

    context = "\n\n--- SOURCE ---\n\n".join(context_parts)

    prompt = f"""
You are a fair academic answer evaluator.

Evaluate the student's answer using the question
and the supplied study material.

Give marks according to this fixed rubric:

1. conceptual_correctness: 0 to 3 marks
   Are the main ideas correct?

2. key_point_coverage: 0 to 3 marks
   How many important points from the material are included?

3. explanation: 0 to 2 marks
   Is the explanation clear and logically complete?

4. relevance: 0 to 1 mark
   Does the answer address the actual question?

5. evidence_alignment: 0 to 1 mark
   Is the answer consistent with the supplied material?

IMPORTANT RULES:
- Follow the maximum marks for each criterion.
- Accept correct explanations written in different words.
- Do not penalize grammar unless it changes the meaning.
- Consider what the question actually asks.
- Do not expect examples if they are not needed.
- Do not invent facts that are absent from the material.
- If the supplied material is insufficient to judge an important
  point, mention that limitation in the feedback.
- Give a short reason for the marks awarded in each criterion.
- Identify correct points, missing points, and concepts to revise.
- Suggest one follow-up question about a missing concept.
- Return ONLY valid JSON. Do not use Markdown code fences.

Return this exact structure:
{{
    "criteria_scores": {{
        "conceptual_correctness": 0,
        "key_point_coverage": 0,
        "explanation": 0,
        "relevance": 0,
        "evidence_alignment": 0
    }},
    "criteria_reasons": {{
        "conceptual_correctness": "Reason",
        "key_point_coverage": "Reason",
        "explanation": "Reason",
        "relevance": "Reason",
        "evidence_alignment": "Reason"
    }},
    "correct_points": ["Correct point"],
    "missing_points": ["Missing point"],
    "feedback": "Overall feedback",
    "weak_concepts": ["Concept to revise"],
    "suggested_followup_question": "Follow-up question"
}}

QUESTION:
{question}

STUDENT ANSWER:
{student_answer}

RETRIEVED STUDY MATERIAL:
{context}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    raw_text = (response.text or "").strip()

    try:
        cleaned_text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw_text,
            flags=re.IGNORECASE
        ).strip()

        result = json.loads(cleaned_text)

        criteria = result["criteria_scores"]
        reasons = result.get("criteria_reasons", {})

        # Maximum marks for each criterion
        limits = {
            "conceptual_correctness": 3,
            "key_point_coverage": 3,
            "explanation": 2,
            "relevance": 1,
            "evidence_alignment": 1
        }

        # Validate every criterion before calculating the total
        checked_scores = {}

        for name, maximum in limits.items():
            value = criteria[name]

            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise ValueError("Invalid criterion score")

            if not 0 <= value <= maximum:
                raise ValueError("Criterion score is out of range")

            if value != int(value):
                raise ValueError("Criterion score must be a whole number")

            checked_scores[name] = int(value)

        # Python calculates the final score
        total_score = sum(checked_scores.values())

        return {
            "score": total_score,
            "criteria_scores": checked_scores,
            "criteria_reasons": reasons,
            "correct_points": result.get("correct_points", []),
            "missing_points": result.get("missing_points", []),
            "feedback": result.get("feedback", ""),
            "weak_concepts": result.get("weak_concepts", []),
            "suggested_followup_question": result.get(
                "suggested_followup_question", ""
            )
        }

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {
            "score": None,
            "criteria_scores": {},
            "criteria_reasons": {},
            "correct_points": [],
            "missing_points": [],
            "feedback": (
                "The evaluation response could not be checked. "
                "Please try evaluating the answer again."
            ),
            "weak_concepts": [],
            "suggested_followup_question": ""
        }


def generate_adaptive_question(
    client,
    weak_concept,
    retrieved_chunks
):
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"Source: {chunk['source']}, Page {chunk['page']}\n"
            f"{chunk['text']}"
        )

    context = "\n\n--- SOURCE ---\n\n".join(context_parts)

    prompt = f"""
You are an academic tutor.

Generate ONE clear practice question about:
{weak_concept}

Use the supplied study material.
Focus on understanding the concept.
Do not provide the answer.
Do not invent facts beyond the material.
Return only the question.

STUDY MATERIAL:
{context}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return (response.text or "").strip()