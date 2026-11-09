"""SQuAD v1.1 loading, the article-level train split, and the official metrics.

The split is by article, not by question, so no paragraph appears on both
sides. Articles are assigned by a hash of their title and a fixed salt, which
makes the split reproducible without storing it and stable if the file is
re-ordered.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import string
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPLITS = ("reader_train", "reader_val", "heldout")
# Shares of SQuAD train articles. reader_val selects the reader and verifier
# checkpoints; heldout is where the switch is tuned. The dev set is neither.
DEFAULT_SHARES = {"reader_train": 0.85, "reader_val": 0.05, "heldout": 0.10}
SALT = "margin-verifier-v1"


@dataclass(frozen=True)
class Question:
    qid: str
    title: str
    context: str
    question: str
    answers: tuple[str, ...]
    answer_starts: tuple[int, ...]


def load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def iter_questions(squad: Mapping[str, Any]) -> Iterator[Question]:
    for article in squad["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                yield Question(
                    qid=qa["id"],
                    title=article["title"],
                    context=paragraph["context"],
                    question=qa["question"],
                    answers=tuple(a["text"] for a in qa["answers"]),
                    answer_starts=tuple(int(a["answer_start"]) for a in qa["answers"]),
                )


def assign_split(title: str, shares: Mapping[str, float] = DEFAULT_SHARES) -> str:
    """Map an article title to a split, deterministically."""
    if set(shares) != set(SPLITS):
        raise ValueError(f"shares must name exactly {SPLITS}")
    if abs(sum(shares.values()) - 1.0) > 1e-9:
        raise ValueError("shares must sum to 1")
    digest = hashlib.sha256(f"{SALT}:{title}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    edge = 0.0
    for name in SPLITS:
        edge += shares[name]
        if u < edge:
            return name
    return SPLITS[-1]


def split_articles(
    squad: Mapping[str, Any], shares: Mapping[str, float] = DEFAULT_SHARES
) -> dict[str, dict[str, Any]]:
    """Return one SQuAD-format dict per split, articles kept whole."""
    out: dict[str, dict[str, Any]] = {
        name: {"version": squad.get("version", "1.1"), "data": []} for name in SPLITS
    }
    for article in squad["data"]:
        out[assign_split(article["title"], shares)]["data"].append(article)
    return out


def question_ids(squad: Mapping[str, Any]) -> set[str]:
    return {q.qid for q in iter_questions(squad)}


# Official SQuAD answer normalisation and metrics (same definitions as
# src/evaluate-v2.0.py, which tests/test_squad.py checks against).
def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(gold: str, prediction: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(prediction))


def f1_score(gold: str, prediction: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    pred_tokens = normalize_answer(prediction).split()
    if not gold_tokens or not pred_tokens:
        return float(gold_tokens == pred_tokens)
    common = collections.Counter(gold_tokens) & collections.Counter(pred_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_over_golds(golds: Iterable[str], prediction: str) -> tuple[int, float]:
    golds = list(golds) or [""]
    return (
        max(exact_match(g, prediction) for g in golds),
        max(f1_score(g, prediction) for g in golds),
    )


def score_predictions(
    questions: Sequence[Question], predictions: Mapping[str, str]
) -> dict[str, float]:
    """EM and F1 in percent over ``questions``, as the official script reports."""
    if not questions:
        raise ValueError("no questions")
    em = f1 = 0.0
    for q in questions:
        e, f = best_over_golds(q.answers, predictions.get(q.qid, ""))
        em += e
        f1 += f
    return {"exact": 100 * em / len(questions), "f1": 100 * f1 / len(questions)}
