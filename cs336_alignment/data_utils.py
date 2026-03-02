from __future__ import annotations

from typing import Any, Mapping


def _extract_gsm8k_final_answer(answer: str) -> str:
    """Return the final answer segment from a GSM8K answer string."""
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()


def extract_question_and_gt(example: Mapping[str, Any]) -> tuple[str, str]:
    """Extract (question, ground_truth_answer) from MATH or GSM8K schema.

    Supported schemas:
    - MATH-format: {"problem": str, "answer": str, ...}
    - GSM8K-format: {"question": str, "answer": str}
    """
    if not isinstance(example, Mapping):
        raise TypeError("example must be a mapping/dict-like object.")

    if "problem" in example:
        question = str(example.get("problem", "")).strip()
        answer = str(example.get("answer", "")).strip()
    elif "question" in example:
        question = str(example.get("question", "")).strip()
        answer = _extract_gsm8k_final_answer(str(example.get("answer", "")))
    else:
        raise KeyError("example must contain either 'problem' (MATH) or 'question' (GSM8K).")

    if not question:
        raise ValueError("extracted question is empty.")
    if not answer:
        raise ValueError("extracted ground-truth answer is empty.")

    return question, answer
