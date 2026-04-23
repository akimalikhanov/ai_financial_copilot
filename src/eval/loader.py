from __future__ import annotations

import json
from pathlib import Path

from src.eval.schemas import EvalQuestion


def load(path: str | Path) -> list[EvalQuestion]:
    with open(path) as f:
        raw: dict = json.load(f)
    questions: list[EvalQuestion] = []
    for i, (question_text, data) in enumerate(raw.items()):
        qid = f"q{i:03d}"
        questions.append(
            EvalQuestion(
                qid=qid,
                question=question_text,
                kind=data["kind"],
                answers=data["answers"],
                reference_pools=data.get("reference_pools", []),
            )
        )
    return questions
