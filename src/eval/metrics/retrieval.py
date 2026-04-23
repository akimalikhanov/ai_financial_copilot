from __future__ import annotations

import math
from collections.abc import Sequence

from src.eval.matching import page_key
from src.schemas.retrieval import RAGContext


def context_to_page_keys(rag_context: RAGContext) -> list[str]:
    """Rank-ordered list of page_keys from RAGContext items.

    A ContextItem covering multiple pages expands to one page_key per page,
    preserving the item's rank order.
    """
    keys: list[str] = []
    for item in rag_context.items:
        name = item.citation.document_name
        for p in item.citation.page_numbers:
            keys.append(page_key(name, p))
    return keys


def _precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for key in top if key in relevant)
    return hits / k


def _recall_at_k(retrieved: Sequence[str], pools: Sequence[set[str]], k: int) -> float:
    if not pools:
        return 0.0
    top = set(retrieved[:k])
    hit = sum(1 for pool in pools if top & pool)
    return hit / len(pools)


def _mrr(retrieved: Sequence[str], pools: Sequence[set[str]]) -> float:
    if not pools:
        return 0.0
    union = set().union(*pools)
    for i, key in enumerate(retrieved, start=1):
        if key in union:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not retrieved:
        return 0.0
    dcg = 0.0
    for i, key in enumerate(retrieved[:k], start=1):
        if key in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg


def compute_retrieval_metrics(
    rag_context: RAGContext,
    expanded_pools: list[set[str]],
    k_values: tuple[int, ...] = (5, 10),
) -> dict:
    """Compute P@K, R@K, MRR, NDCG@K with pool semantics (OR within, AND across)."""
    retrieved = context_to_page_keys(rag_context)
    relevant: set[str] = set().union(*expanded_pools) if expanded_pools else set()

    metrics: dict = {}
    for k in k_values:
        metrics[f"precision@{k}"] = _precision_at_k(retrieved, relevant, k)
        metrics[f"recall@{k}"] = _recall_at_k(retrieved, expanded_pools, k)
        metrics[f"ndcg@{k}"] = _ndcg_at_k(retrieved, relevant, k)
    metrics["mrr"] = _mrr(retrieved, expanded_pools)
    return metrics
