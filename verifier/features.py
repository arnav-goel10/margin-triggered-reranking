"""Top-two candidate records and the features the switch rules see.

A record holds what is known about one question once the reader and the
verifier have run: the reader's top two spans with their scores, the verifier's
logit for each, and (for tuning and scoring only) whether each span is correct
under SQuAD exact match. Rules never read the correctness fields.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from verifier.squad import Question, best_over_golds

FEATURE_NAMES = (
    "gap",
    "log_p1",
    "log_p2",
    "v1",
    "v2",
    "v_diff",
    "gap_x_v_diff",
    "len1",
    "len2",
    "nested",
)


@dataclass(frozen=True)
class TopTwo:
    qid: str
    text1: str
    text2: str
    p1: float
    p2: float
    v1: float
    v2: float
    c1_correct: bool
    c2_correct: bool

    @property
    def gap(self) -> float:
        return self.p1 - self.p2

    @property
    def informative(self) -> bool:
        """Exactly one of the two is right, so switching changes EM."""
        return self.c1_correct != self.c2_correct


def build_records(
    questions: Sequence[Question],
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    verifier_logits: Mapping[str, Sequence[float]],
) -> list[TopTwo]:
    """Join reader candidates, verifier logits and gold answers per question.

    Questions with fewer than two candidates get a copy of the first as the
    second, so no rule can switch them.
    """
    records: list[TopTwo] = []
    for q in questions:
        cands = list(candidates.get(q.qid, []))
        logits = list(verifier_logits.get(q.qid, []))
        if not cands:
            raise ValueError(f"no candidates for {q.qid}")
        if len(logits) < min(len(cands), 2):
            raise ValueError(f"missing verifier logits for {q.qid}")
        first = cands[0]
        second = cands[1] if len(cands) > 1 else cands[0]
        v1 = float(logits[0])
        v2 = float(logits[1]) if len(cands) > 1 else v1
        records.append(
            TopTwo(
                qid=q.qid,
                text1=first["text"],
                text2=second["text"],
                p1=float(first["score"]),
                p2=float(second["score"]),
                v1=v1,
                v2=v2,
                c1_correct=bool(best_over_golds(q.answers, first["text"])[0]),
                c2_correct=bool(best_over_golds(q.answers, second["text"])[0]),
            )
        )
    return records


def _log(p: float) -> float:
    return math.log(max(p, 1e-12))


def feature_matrix(records: Sequence[TopTwo]) -> np.ndarray:
    """One row per record, columns in ``FEATURE_NAMES`` order."""
    rows = []
    for r in records:
        t1, t2 = r.text1.strip().lower(), r.text2.strip().lower()
        nested = float(bool(t1) and bool(t2) and (t1 in t2 or t2 in t1))
        v_diff = r.v2 - r.v1
        rows.append(
            (
                r.gap,
                _log(r.p1),
                _log(r.p2),
                r.v1,
                r.v2,
                v_diff,
                r.gap * v_diff,
                float(len(r.text1.split())),
                float(len(r.text2.split())),
                nested,
            )
        )
    return np.asarray(rows, dtype=float).reshape(len(records), len(FEATURE_NAMES))
