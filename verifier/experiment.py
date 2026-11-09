"""Tuning on the held-out slice and the single dev read, as plain functions.

``tune`` runs k-fold cross-validation on held-out records for every rule,
then refits each rule on all of them. ``apply_rules`` replays the fitted rules
on another set of records (dev) without refitting anything.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from verifier.features import TopTwo
from verifier.switch import (
    LearnedSwitch,
    Outcome,
    ThresholdRule,
    fit_learned_switch,
    folds,
    oracle,
    outcome,
    tune_threshold_rule,
)

TAUS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75)
DELTAS = tuple(float(d) for d in np.arange(0.0, 6.01, 0.5))
RULES = ("reader", "verifier", "gated", "learned", "oracle")


def _fit(rule: str, records: Sequence[TopTwo]) -> Any:
    if rule == "verifier":
        return tune_threshold_rule(records, (math.inf,), DELTAS)[0]
    if rule == "gated":
        return tune_threshold_rule(records, TAUS, DELTAS)[0]
    if rule == "learned":
        return fit_learned_switch(records)
    return None


def _decide(rule: str, fitted: Any, records: Sequence[TopTwo]) -> list[bool]:
    if rule == "reader":
        return [False] * len(records)
    if rule == "oracle":
        return oracle(records)
    return list(fitted.decide(records))


def _outcome_dict(result: Outcome) -> dict[str, float]:
    return {
        "switched": result.switched,
        "fixed": result.fixed,
        "broken": result.broken,
        "net": result.net,
        "em_delta": round(result.em_delta, 4),
    }


def describe(rule: str, fitted: Any) -> dict[str, Any]:
    if isinstance(fitted, ThresholdRule):
        return {"tau": None if math.isinf(fitted.tau) else fitted.tau, "delta": fitted.delta}
    if isinstance(fitted, LearnedSwitch):
        return {
            "threshold": fitted.threshold,
            "mean": fitted.mean.tolist(),
            "scale": fitted.scale.tolist(),
            "weights": fitted.weights.tolist(),
        }
    return {}


def restore(rule: str, params: Mapping[str, Any]) -> Any:
    if rule in ("verifier", "gated"):
        tau = params["tau"]
        return ThresholdRule(math.inf if tau is None else float(tau), float(params["delta"]))
    if rule == "learned":
        return LearnedSwitch(
            np.asarray(params["mean"]),
            np.asarray(params["scale"]),
            np.asarray(params["weights"]),
            float(params["threshold"]),
        )
    return None


def tune(records: Sequence[TopTwo], *, k: int = 5, seed: int = 0) -> dict[str, Any]:
    """Cross-validated outcome per rule, then each rule refit on all records."""
    parts = folds(len(records), k, seed)
    cv: dict[str, Any] = {}
    for rule in RULES:
        fixed = broken = switched = 0
        per_fold = []
        for i, test_idx in enumerate(parts):
            train_idx = np.concatenate([p for j, p in enumerate(parts) if j != i])
            train = [records[t] for t in train_idx]
            test = [records[t] for t in test_idx]
            result = outcome(test, _decide(rule, _fit(rule, train), test))
            fixed, broken, switched = (
                fixed + result.fixed,
                broken + result.broken,
                switched + result.switched,
            )
            per_fold.append(round(result.em_delta, 4))
        pooled = Outcome(switched, fixed, broken, len(records))
        cv[rule] = {**_outcome_dict(pooled), "per_fold_em_delta": per_fold}
    fitted = {rule: _fit(rule, records) for rule in RULES}
    in_sample = {
        rule: _outcome_dict(outcome(records, _decide(rule, fitted[rule], records)))
        for rule in RULES
    }
    methods = [r for r in RULES if r not in ("reader", "oracle")]
    selected = max(methods, key=lambda r: (cv[r]["net"], -cv[r]["switched"]))
    return {
        "questions": len(records),
        "folds": k,
        "seed": seed,
        "cross_validated": cv,
        "refit_in_sample": in_sample,
        "selected": selected,
        "params": {rule: describe(rule, fitted[rule]) for rule in RULES},
    }


def apply_rules(
    records: Sequence[TopTwo], params: Mapping[str, Mapping[str, Any]]
) -> dict[str, list[bool]]:
    return {rule: _decide(rule, restore(rule, params[rule]), records) for rule in RULES}


def diagnostics(records: Sequence[TopTwo]) -> dict[str, Any]:
    """How well the gap flags reader errors, and who is right by gap band."""
    c1_wrong = np.array([not r.c1_correct for r in records])
    gaps = np.array([r.gap for r in records])
    # AUROC of "small gap" for "candidate 1 is wrong", by the rank statistic.
    order = np.argsort(-gaps, kind="mergesort")
    ranks = np.empty(len(gaps))
    ranks[order] = np.arange(1, len(gaps) + 1)
    pos, neg = c1_wrong.sum(), (~c1_wrong).sum()
    auroc = (
        float((ranks[c1_wrong].sum() - pos * (pos + 1) / 2) / (pos * neg)) if pos and neg else None
    )
    bands = []
    edges = (0.0, 0.05, 0.10, 0.20, 0.40, 1.0001)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        band = [r for r in records if lo <= r.gap < hi and r.informative]
        verifier_right = sum(1 for r in band if (r.v2 > r.v1) == r.c2_correct)
        bands.append(
            {
                "gap_from": lo,
                "gap_to": min(hi, 1.0),
                "informative": len(band),
                "candidate2_right": sum(1 for r in band if r.c2_correct),
                "verifier_picks_right": verifier_right,
            }
        )
    return {"gap_auroc_for_reader_error": auroc, "informative_by_gap_band": bands}
