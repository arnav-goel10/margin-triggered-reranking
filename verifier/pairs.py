"""Verifier inputs: a question and a context window with one span marked.

The verifier is a cross-encoder that reads ``question`` against the context
with the candidate span wrapped in marker tokens, and outputs a logit for "this
span is the answer". Training pairs take the gold span as the positive and the
reader's own wrong candidates as hard negatives, which is the confusion the
switch has to resolve: the reader's second choice against its first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verifier.squad import Question, normalize_answer

START, END = "[ANS]", "[/ANS]"
WINDOW_CHARS = 600


@dataclass(frozen=True)
class Pair:
    qid: str
    question: str
    marked_context: str
    label: int


def mark_span(context: str, start: int, end: int, window: int = WINDOW_CHARS) -> str:
    """Return the context around ``[start, end)`` with the span wrapped in markers.

    The window keeps the span inside the encoder's length limit for long
    paragraphs; ellipses show where the context was cut.
    """
    if not 0 <= start < end <= len(context):
        raise ValueError(f"bad span [{start}, {end}) for context of {len(context)}")
    left = max(0, start - window)
    right = min(len(context), end + window)
    prefix = "... " if left > 0 else ""
    suffix = " ..." if right < len(context) else ""
    return (
        f"{prefix}{context[left:start]}{START} {context[start:end]} {END}"
        f"{context[end:right]}{suffix}"
    )


def locate(context: str, candidate: Mapping[str, Any]) -> tuple[int, int] | None:
    """Character offsets of a reader candidate, or None if they do not match its text."""
    start, end = int(candidate.get("start", -1)), int(candidate.get("end", -1))
    text = candidate["text"]
    if 0 <= start < end <= len(context) and context[start:end] == text:
        return start, end
    found = context.find(text) if text else -1
    if found >= 0:
        return found, found + len(text)
    return None


def training_pairs(
    questions: Sequence[Question],
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    negatives_per_question: int = 2,
) -> list[Pair]:
    """Gold span as positive, the highest-ranked wrong reader candidates as negatives."""
    pairs: list[Pair] = []
    for q in questions:
        if not q.answers:
            continue
        gold_start = q.answer_starts[0]
        gold_end = gold_start + len(q.answers[0])
        if q.context[gold_start:gold_end] != q.answers[0]:
            continue  # a handful of SQuAD offsets are off; skip rather than guess
        pairs.append(Pair(q.qid, q.question, mark_span(q.context, gold_start, gold_end), 1))
        golds = {normalize_answer(a) for a in q.answers}
        taken = 0
        for cand in candidates.get(q.qid, []):
            if taken == negatives_per_question:
                break
            if normalize_answer(cand["text"]) in golds:
                continue
            span = locate(q.context, cand)
            if span is None:
                continue
            pairs.append(Pair(q.qid, q.question, mark_span(q.context, *span), 0))
            taken += 1
    return pairs


def scoring_inputs(
    questions: Sequence[Question],
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    top_k: int = 2,
) -> list[tuple[str, int, str, str]]:
    """(qid, rank, question, marked context) for the top ``top_k`` candidates."""
    rows = []
    for q in questions:
        for rank, cand in enumerate(candidates.get(q.qid, [])[:top_k]):
            span = locate(q.context, cand)
            if span is None:
                # keep the rank aligned; an unlocatable span scores as the bare question
                rows.append((q.qid, rank, q.question, f"{START} {cand['text']} {END}"))
            else:
                rows.append((q.qid, rank, q.question, mark_span(q.context, *span)))
    return rows
