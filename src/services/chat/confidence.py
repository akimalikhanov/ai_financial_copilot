import re

FACT_PATTERN = re.compile(r"\$[\d,.]+|\d+\.?\d*%|\b(19|20)\d{2}\b|\d[\d,]{2,}")
REF_PATTERN = re.compile(r"\[S\d+\]")


def compute_confidence(top_score: float | None, num_chunks: int) -> str:
    if num_chunks == 0:
        return "none"
    if top_score is None:
        return "medium"
    if top_score >= 0.7:
        return "high"
    if top_score >= 0.25:
        return "medium"
    return "low"


def has_ungrounded_claims(text: str) -> bool:
    for sentence in text.split("."):
        if FACT_PATTERN.search(sentence) and not REF_PATTERN.search(sentence):
            return True
    return False
